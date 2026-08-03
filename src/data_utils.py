"""DiFrauD acquisition, validation, PU partitioning, and audit features."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATASET_ID = "difraud/difraud"
DOMAINS = (
    "phishing", "fake_news", "political_statements", "product_reviews",
    "job_scams", "sms", "twitter_rumours",
)
SPLITS = ("train", "validation", "test")


def dataset_revision(repo_id: str = DATASET_ID) -> str:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo_id).sha


def _candidate_files(files: Iterable[str], domain: str, split: str) -> list[str]:
    aliases = {"validation", "valid", "val", "dev"} if split == "validation" else {split}
    return [f for f in files if f.startswith(domain + "/") and Path(f).suffix.lower() in {".jsonl", ".json"}
            and any(a in Path(f).stem.lower() for a in aliases)]


def load_difraud(cache_dir: str | Path = "data/raw", revision: str | None = None) -> tuple[pd.DataFrame, str]:
    """Download official JSON/JSONL split files without executing remote code."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.dataset_info(DATASET_ID, revision=revision)
    sha = info.sha
    files = [s.rfilename for s in info.siblings]
    frames: list[pd.DataFrame] = []
    for domain in DOMAINS:
        for split in SPLITS:
            matches = _candidate_files(files, domain, split)
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {domain}/{split} JSON file, found {matches}.")
            local = hf_hub_download(DATASET_ID, matches[0], repo_type="dataset", revision=sha,
                                    cache_dir=str(Path(cache_dir) / "hf"))
            records = []
            with open(local, encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSON in {matches[0]} line {line_no}") from exc
                    records.extend(value if isinstance(value, list) else [value])
            frame = pd.DataFrame(records)
            required = {"text", "label"}
            if not required.issubset(frame.columns):
                raise ValueError(f"{matches[0]} lacks required columns {required}.")
            frame = frame[["text", "label"]].copy()
            frame["domain"], frame["split"], frame["source_file"] = domain, split, matches[0]
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["text"] = data["text"].astype("string")
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    invalid = set(data["label"].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Unexpected labels: {sorted(invalid)}")
    return data, sha


def normalize_text(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    return re.sub(r"\s+", " ", value.casefold()).strip()


def add_text_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    text = out["text"].fillna("").astype(str)
    out["normalized_text"] = text.map(normalize_text)
    out["text_hash"] = out["normalized_text"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    out["char_count"] = text.str.len()
    out["word_count"] = text.str.findall(r"\b\w+\b").str.len()
    out["sentence_count"] = text.str.count(r"[.!?]+")
    out["url_count"] = text.str.count(r"(?i)https?://|www\.")
    out["digit_count"] = text.str.count(r"\d")
    out["punctuation_count"] = text.str.count(r"[^\w\s]")
    letters = text.str.count(r"[A-Za-z]").clip(lower=1)
    out["uppercase_ratio"] = text.str.count(r"[A-Z]") / letters
    out["non_ascii_count"] = text.map(lambda x: sum(ord(c) > 127 for c in x))
    # Python regex is used because Arrow/RE2 does not support backreferences.
    out["repeated_punctuation"] = text.map(lambda x: len(re.findall(r"([!?.,])\1+", x)))
    out["html_residue"] = text.map(lambda x: len(re.findall(r"<[^>]+>|&(?:amp|lt|gt|nbsp);", x, re.I)))
    out["replacement_chars"] = text.str.count("�")
    return out


def make_pu_partition(train: pd.DataFrame, positive_fraction: float, seed: int) -> pd.DataFrame:
    """Reveal a reproducible fraction of training positives; never relabel U as negative."""
    if not 0 < positive_fraction <= 1:
        raise ValueError("positive_fraction must lie in (0, 1].")
    if set(train["label"].unique()) - {0, 1}:
        raise ValueError("PU simulation requires labels encoded as 0/1.")
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(train["label"].to_numpy() == 1)
    n_reveal = max(1, int(round(len(positives) * positive_fraction)))
    revealed = set(rng.choice(positives, n_reveal, replace=False).tolist())
    out = train.copy()
    out["pu_status"] = ["P" if i in revealed else "U" for i in range(len(out))]
    # hidden label is retained solely for evaluation/auditing, never passed to fit functions.
    out["hidden_label_eval_only"] = out["label"]
    return out


def training_view(partition: pd.DataFrame) -> pd.DataFrame:
    return partition.drop(columns=["label", "hidden_label_eval_only"], errors="ignore").copy()


def exact_leakage_table(data: pd.DataFrame) -> pd.DataFrame:
    keyed = add_text_features(data) if "text_hash" not in data else data
    rows = []
    for text_hash, group in keyed.groupby("text_hash", sort=False):
        if len(group) > 1:
            rows.append({"text_hash": text_hash, "copies": len(group),
                         "domains": group.domain.nunique(), "splits": group.split.nunique(),
                         "conflicting_labels": group.label.nunique() > 1})
    return pd.DataFrame(rows)

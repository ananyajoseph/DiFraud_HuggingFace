# DiFrauD PU–Mean Teacher Deception Detection

Research-quality, leakage-aware experiments on the [official DiFrauD dataset](https://huggingface.co/datasets/difraud/difraud). The primary artifact is the fully executable notebook [`difraud_pu_semisupervised_analysis.ipynb`](difraud_pu_semisupervised_analysis.ipynb).

## Research questions

- How much deception-detection performance remains when only 1–20% of positive training examples are labeled?
- Does valid non-negative positive–unlabeled (nnPU) learning beat the invalid shortcut of calling all unlabeled examples negative?
- Does teacher–student consistency improve a PU classifier?
- How much performance survives domain shift, and how much apparent signal comes from dataset provenance or shallow artifacts?

## Dataset

DiFrauD combines English binary text-classification data from seven domains: phishing, fake news, political statements, product reviews, job scams, SMS, and Twitter rumours. This project preserves the official train/validation/test files and records the immutable Hugging Face revision used. `label=1` is deceptive and `label=0` is non-deceptive. Downloads live under `data/raw/`, which Git ignores.

The notebook audits completeness, malformed and empty text, imbalance, text integrity, duplicates and split leakage, near duplicates, shallow shortcuts, domain separability, suspected label issues from out-of-fold predictions, vocabulary shift, and domain-level Jensen–Shannon divergence. “Suspected label issue” is diagnostic language, not a claim of annotation error.

### Citation

Use the citation supplied on the [DiFrauD dataset card](https://huggingface.co/datasets/difraud/difraud) and cite the original constituent datasets when publishing derived results. The notebook captures the card and data revision so the exact snapshot is reproducible.

## Methods

In each PU simulation, only a seeded fraction of positive **training** examples is revealed. Every negative and every remaining (hidden) positive enters the unlabeled pool. Hidden labels exist only in the experiment harness for evaluation and never reach the training function.

The nnPU objective is the non-negative correction of the unbiased PU risk:

`π E_P[ℓ(+f)] + max(0, E_U[ℓ(-f)] − π E_P[ℓ(-f)])`.

The **PU–Mean Teacher Deception Detector** adds a confidence-masked consistency loss between a student receiving stronger embedding dropout and an exponential-moving-average teacher receiving a weaker view. The consistency weight ramps up, and all thresholds and early-stopping choices use validation data only. It differs from ordinary self-training because unlabeled examples are never assigned negative training targets and supervised classification remains a valid PU risk.

## Repository layout

```text
difraud_pu_semisupervised_analysis.ipynb  primary report and executable study
src/data_utils.py                         download, validation, audit, PU split
src/pu_loss.py                            nnPU risk and diagnostics
src/models.py                             MLP and Mean Teacher utilities
src/evaluation.py                         thresholding and imbalanced metrics
tests/test_pu_loss.py                     nnPU correctness/safety tests
requirements.txt                          reproducible Python dependencies
```

## Install and run

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute difraud_pu_semisupervised_analysis.ipynb \
  --output difraud_pu_semisupervised_analysis.ipynb --ExecutePreprocessor.timeout=7200
pytest -q
```

The central notebook configuration defaults to `MODE="quick"`. Quick mode is CPU-oriented: it stratifies capped samples, freezes a compact sentence-transformer representation, executes all four label fractions for one seed, and runs reduced leave-one-domain-out tests for job scams, product reviews, and phishing. `MODE="full"` removes the cap, expands repeats and all seven held-out domains, and enables the larger training schedule; transformer fine-tuning is an explicitly unexecuted extension unless a GPU is available.

## Reproducibility and leakage controls

- The Hugging Face commit SHA, dependency versions, configuration, seed, and device are printed into notebook output.
- Official splits are preserved. Training sees train only; validation selects thresholds and settings; test is final evaluation only.
- Leave-one-domain-out excludes the held-out domain from training, prior estimation, early stopping, calibration, and threshold selection.
- The notebook clearly labels oracle-prior results as a non-deployable reference and compares a training-only Elkan–Noto estimate plus prior-sensitivity values.
- Local data, caches, checkpoints, embeddings, and experiment outputs are excluded by `.gitignore`.

## Tests

`pytest -q` checks finite scalar loss values, gradients, diagnostic shapes, correction behavior, invalid priors/encodings, empty batches, and NaN safeguards.

## Results

Only outputs stored in the executed notebook are results. Do not infer full-dataset performance from quick mode. The notebook’s execution manifest distinguishes executed cells and configurations from planned full-mode experiments.

## Ethical and scientific limitations

Deception classification is not factual verification. Labels inherit the assumptions and provenance of constituent datasets; domain, source, formatting, and duplication artifacts may dominate semantic deception cues. Simulated PU labeling is cleaner than real reporting processes, estimated class priors can be wrong, and high-confidence predictions are not proof of intent. False positives can harm people—especially job applicants, political speakers, reviewers, or message senders—so this work is unsuitable for autonomous enforcement. Deployments require domain-specific validation, human review, calibrated alert budgets, subgroup audits, privacy review, drift monitoring, and clear appeal mechanisms.


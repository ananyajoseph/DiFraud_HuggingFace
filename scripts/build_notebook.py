"""Build the self-contained research notebook from reviewable cell sources."""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# DiFrauD under scarce positive labels
## A leakage-resistant nnPU and Mean Teacher study of domain-independent deception detection

**Abstract.** This executable study asks whether textual deception transfers across seven heterogeneous domains when only a small fraction of deceptive training examples is labeled. It combines the non-negative positive–unlabeled (nnPU) risk estimator with a confidence-filtered exponential-moving-average teacher. Alongside leakage-resistant in-domain and leave-one-domain-out (LODO) evaluation, it audits composition, integrity, duplication, source shortcuts, label ambiguity, and distribution shift. Quick mode is a real, CPU-oriented representative experiment; full mode expands seeds, samples, and held-out domains. Results are not claims of factual verification or intent.

**Research questions.** Can deception be detected across heterogeneous domains? What remains with 1%, 5%, 10%, or 20% labeled positives? Does nnPU beat pseudo-negative training? Does consistency help? Does the model learn deception or dataset provenance? Does it transfer to an unseen domain?

**Proposed method.** The *PU–Mean Teacher Deception Detector* trains a student on valid nnPU classification risk and consistency against a confidence-filtered EMA teacher. Unlabeled examples are never declared negative.
""")

md("## 1. Environment and reproducibility")
code(r"""
from pathlib import Path
import os, re, random, platform, importlib.metadata as im, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic("matplotlib", "inline")
from scipy.spatial.distance import jensenshannon
from scipy.stats import bootstrap
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
    f1_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.data_utils import (DOMAINS, add_text_features, exact_leakage_table,
    load_difraud, make_pu_partition, training_view)
from src.pu_loss import nnpu_loss
from src.models import (MLPClassifier, consistency_loss, ema_update,
    make_teacher, mask_embedding, sigmoid_rampup)
from src.evaluation import choose_threshold, classification_metrics

MODE = os.environ.get("DIFRAUD_MODE", "quick")
CONFIG = {
    "mode": MODE, "seed": 42,
    "domains": list(DOMAINS), "positive_fractions": [0.01, 0.05, 0.10, 0.20],
    "seeds": [42] if MODE == "quick" else [42, 73, 101],
    "batch_size": 128 if MODE == "quick" else 64,
    "epochs": 6 if MODE == "quick" else 20,
    "learning_rate": 2e-3 if MODE == "quick" else 2e-5,
    "confidence_threshold": 0.80, "ema_decay": 0.99,
    "max_text_length": 256,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "sample_per_domain_split": 500 if MODE == "quick" else None,
    "lodo_domains": ["job_scams", "product_reviews", "phishing"] if MODE == "quick" else list(DOMAINS),
    "output_dir": "outputs", "data_dir": "data/raw",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(CONFIG["seed"])
versions = {p: im.version(p) for p in ["numpy", "pandas", "scikit-learn", "torch",
    "sentence-transformers", "huggingface-hub"]}
print({"python": platform.python_version(), "device": str(DEVICE), **versions})
print(CONFIG)
""")

md(r"""
## 2. Official dataset acquisition and validation

The loader queries the Hub file manifest and downloads the official JSON/JSONL files at an immutable commit SHA. It deliberately does **not** execute the repository loading script or enable arbitrary remote code. Each official split is retained. Quick mode makes a label-stratified cap *within each domain/split* after validation; it does not invent new splits.
""")
code(r"""
raw, DATASET_REVISION = load_difraud(CONFIG["data_dir"])
print("DiFrauD revision:", DATASET_REVISION)
validation = pd.Series({
    "rows": len(raw), "missing_text": int(raw.text.isna().sum()),
    "empty_text": int(raw.text.fillna("").str.strip().eq("").sum()),
    "missing_label": int(raw.label.isna().sum()),
    "invalid_labels": int((~raw.label.isin([0, 1])).sum()),
    "domains": raw.domain.nunique(), "splits": raw.split.nunique(),
    "source_files": raw.source_file.nunique(),
})
display(validation.to_frame("value"))
summary_full = raw.groupby(["domain", "split", "label"]).size().unstack(fill_value=0)
summary_full.columns = ["non_deceptive", "deceptive"]
summary_full["total"] = summary_full.sum(axis=1)
summary_full["positive_prevalence"] = summary_full.deceptive / summary_full.total
display(summary_full)

def stratified_cap(group, cap, seed=42):
    if cap is None or len(group) <= cap: return group
    pieces = []
    for _, part in group.groupby("label"):
        n = max(1, round(cap * len(part) / len(group)))
        pieces.append(part.sample(min(n, len(part)), random_state=seed))
    return pd.concat(pieces).sample(frac=1, random_state=seed).head(cap)

data = (raw.groupby(["domain", "split"], group_keys=False)
        .apply(lambda g: stratified_cap(g, CONFIG["sample_per_domain_split"]), include_groups=False)
        .reset_index(drop=True))
# pandas include_groups=False removes grouping columns; restore safely from retained source path.
if "domain" not in data:
    data = pd.concat([stratified_cap(g, CONFIG["sample_per_domain_split"])
        for _, g in raw.groupby(["domain", "split"], sort=False)], ignore_index=True)
data = add_text_features(data)
print(f"Analysis rows: {len(data):,} (full official rows: {len(raw):,})")
""")

md("## 3. Creative dataset-quality audit")
md(r"""
### 3.1 Composition, imbalance, and text integrity

Wilson 95% intervals describe domain/split prevalence uncertainty. Integrity features target parsing and source artifacts: length, sentences, URLs, digits, punctuation, capitalization, non-ASCII text, repeated punctuation, HTML residue, and replacement characters. Outliers use the robust 1.5×IQR rule within domains.
""")
code(r"""
from statsmodels.stats.proportion import proportion_confint
audit = data.groupby(["domain", "split"]).agg(n=("label","size"), positives=("label","sum"))
audit["prevalence"] = audit.positives / audit.n
ci = [proportion_confint(k, n, method="wilson") for k,n in zip(audit.positives,audit.n)]
audit[["prevalence_ci_low","prevalence_ci_high"]] = ci
audit["imbalance_ratio_majority_to_minority"] = audit[["positives","n"]].apply(
    lambda r: max(r.positives, r.n-r.positives) / max(1, min(r.positives, r.n-r.positives)), axis=1)
display(audit.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=data, y="domain", hue="label", ax=axes[0]); axes[0].set_title("Quick-mode composition by domain and label")
prev = data.groupby("domain").label.mean().sort_values()
prev.plot.barh(ax=axes[1]); axes[1].set_title("Deceptive prevalence by domain"); axes[1].set_xlabel("Fraction label=1")
plt.tight_layout(); plt.show()

integrity_cols = ["char_count","word_count","sentence_count","url_count","digit_count",
                  "punctuation_count","uppercase_ratio","non_ascii_count","repeated_punctuation",
                  "html_residue","replacement_chars"]
display(data.groupby("domain")[integrity_cols].median().round(2))
q1, q3 = data.groupby("domain").char_count.transform("quantile", .25), data.groupby("domain").char_count.transform("quantile", .75)
extreme = data.char_count > q3 + 1.5*(q3-q1)
print({"empty": int(data.normalized_text.eq("").sum()), "extremely_short_le_3_words": int((data.word_count<=3).sum()),
       "length_outliers": int(extreme.sum()), "html_rows": int((data.html_residue>0).sum()),
       "encoding_replacement_rows": int((data.replacement_chars>0).sum())})
""")

md(r"""
### 3.2 Exact duplicates, conflicting labels, and near-duplicate split leakage

Exact normalized hashes are exhaustive. Near-duplicate search uses character 3–5 gram TF–IDF and cosine neighbors on the quick sample; only cross-split pairs above 0.92 are counted. Text excerpts are truncated to 120 characters to limit disclosure.
""")
code(r"""
dupes = exact_leakage_table(data)
display(pd.Series({"duplicate_groups": len(dupes),
    "cross_domain_groups": int((dupes.domains>1).sum()) if len(dupes) else 0,
    "cross_split_groups": int((dupes.splits>1).sum()) if len(dupes) else 0,
    "conflicting_label_groups": int(dupes.conflicting_labels.sum()) if len(dupes) else 0}).to_frame("count"))

char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=30000)
X_char = char_vec.fit_transform(data.normalized_text)
nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1).fit(X_char)
dist, idx = nn.kneighbors(X_char)
pairs = pd.DataFrame({"i":np.arange(len(data)), "j":idx[:,1], "similarity":1-dist[:,1]})
pairs = pairs[(pairs.i < pairs.j) & (pairs.similarity >= .92)]
pairs["cross_split"] = [data.iloc[i].split != data.iloc[j].split for i,j in zip(pairs.i,pairs.j)]
print({"near_duplicate_pairs_ge_0.92":len(pairs), "cross_split_near_pairs":int(pairs.cross_split.sum())})
examples=[]
for _,r in pairs[pairs.cross_split].head(5).iterrows():
    a,b=data.iloc[int(r.i)],data.iloc[int(r.j)]
    examples.append({"similarity":r.similarity,"a":a.text[:120],"a_split":a.split,"b":b.text[:120],"b_split":b.split})
display(pd.DataFrame(examples))
""")

md(r"""
### 3.3 Shortcut learning, domain separability, and suspected label issues

The domain-prior baseline uses training-domain prevalence only. The metadata model uses no text. A domain classifier and SVD map quantify provenance signal. Label-quality screening uses leakage-safe out-of-fold text predictions on training rows; disagreement indicates ambiguity or shortcut failure—not a confirmed error.
""")
code(r"""
train = data[data.split=="train"].reset_index(drop=True)
val = data[data.split=="validation"].reset_index(drop=True)
test = data[data.split=="test"].reset_index(drop=True)
metadata_cols = ["char_count","word_count","sentence_count","url_count","digit_count",
                 "punctuation_count","uppercase_ratio","non_ascii_count","repeated_punctuation","html_residue"]
meta_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
meta_model.fit(train[metadata_cols], train.label)
meta_p = meta_model.predict_proba(test[metadata_cols])[:,1]
domain_prior = train.groupby("domain").label.mean()
prior_p = test.domain.map(domain_prior).fillna(train.label.mean()).to_numpy()
print("Metadata-only", classification_metrics(test.label, meta_p, choose_threshold(val.label, meta_model.predict_proba(val[metadata_cols])[:,1])))
print("Domain-prior", classification_metrics(test.label, prior_p, choose_threshold(val.label, val.domain.map(domain_prior).fillna(train.label.mean()))))

tfidf_domain = TfidfVectorizer(min_df=3, max_features=15000, ngram_range=(1,2), sublinear_tf=True)
Xd = tfidf_domain.fit_transform(train.text.fillna("")); Xdt = tfidf_domain.transform(test.text.fillna(""))
domain_clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xd, train.domain)
print("Domain classifier accuracy:", domain_clf.score(Xdt, test.domain))
svd = TruncatedSVD(n_components=2, random_state=CONFIG["seed"])
coords = svd.fit_transform(TfidfVectorizer(min_df=3,max_features=15000).fit_transform(data.text.fillna("")))
fig,axes=plt.subplots(1,2,figsize=(15,5))
sns.scatterplot(x=coords[:,0],y=coords[:,1],hue=data.domain,alpha=.55,s=18,ax=axes[0]); axes[0].set_title("TF–IDF/SVD representation colored by domain")
sns.scatterplot(x=coords[:,0],y=coords[:,1],hue=data.label.astype(str),alpha=.55,s=18,ax=axes[1]); axes[1].set_title("Same representation colored by recorded label")
plt.tight_layout(); plt.show()

oof_pipe = make_pipeline(TfidfVectorizer(min_df=3,max_features=20000,ngram_range=(1,2)),
                         LogisticRegression(max_iter=1000,class_weight="balanced"))
cv = StratifiedKFold(5, shuffle=True, random_state=CONFIG["seed"])
oof = cross_val_predict(oof_pipe, train.text.fillna(""), train.label, cv=cv, method="predict_proba", n_jobs=-1)[:,1]
suspect = train.assign(oof_probability=oof, disagreement=np.where(train.label.eq(1),1-oof,oof))
display(suspect.nlargest(8,"disagreement")[["domain","label","oof_probability","text"]].assign(text=lambda x:x.text.str[:160]))
""")

md(r"""
### 3.4 Distribution shift and quality scorecard

Pairwise Jensen–Shannon distance compares smoothed domain unigram distributions (0=same, 1=maximally separated with base 2). Vocabulary overlap is Jaccard overlap among each domain's 1,000 most frequent terms. The scorecard intentionally avoids an arbitrary overall score.
""")
code(r"""
v = TfidfVectorizer(max_features=10000, use_idf=False, norm="l1", binary=False)
M = v.fit_transform(data.text.fillna("")); domains=list(DOMAINS)
means={d:np.asarray(M[data.domain.eq(d).to_numpy()].mean(axis=0)).ravel()+1e-12 for d in domains}
js=pd.DataFrame(index=domains,columns=domains,dtype=float)
for a in domains:
    for b in domains: js.loc[a,b]=jensenshannon(means[a],means[b],base=2)
plt.figure(figsize=(8,6)); sns.heatmap(js,annot=True,fmt=".2f",cmap="mako"); plt.title("Domain token-distribution Jensen–Shannon distance"); plt.tight_layout(); plt.show()

top={d:set(np.argsort(means[d])[-1000:]) for d in domains}
jacc=pd.DataFrame([[len(top[a]&top[b])/len(top[a]|top[b]) for b in domains] for a in domains],index=domains,columns=domains)
display(jacc.round(2))
scorecard=pd.DataFrame({
 "dimension":["Completeness","Uniqueness","Label consistency","Leakage risk","Class balance","Domain shift","Representativeness","Documentation","Reproducibility"],
 "evidence":[f"{validation.missing_text} missing; {validation.empty_text} empty",f"{len(dupes)} duplicate normalized-text groups",
             f"{int(dupes.conflicting_labels.sum()) if len(dupes) else 0} conflicting duplicate groups",
             f"{int((dupes.splits>1).sum()) if len(dupes) else 0} exact cross-split groups; {int(pairs.cross_split.sum())} sampled near pairs",
             f"domain prevalence range {data.groupby('domain').label.mean().min():.2f}–{data.groupby('domain').label.mean().max():.2f}",
             f"mean off-diagonal JS distance {js.values[np.triu_indices(7,1)].mean():.2f}",
             "Seven heterogeneous English source domains; not population representative",
             "Dataset card plus constituent provenance; audit original sources before deployment",
             f"Hub SHA {DATASET_REVISION[:12]}, official splits, seeded execution"],
 "interpretation":["quantitative","risk flag","risk flag","risk flag","domain-dependent","substantial if high","limited","review needed","strong"]})
display(scorecard)
""")

md(r"""
## 4. Leakage-safe design, baselines, and PU protocol

**Setting A (in-domain):** train only on official training data, select thresholds on official validation data, and evaluate official test once. **Setting B (LODO):** remove an entire domain before any training or prior estimate; select thresholds on validation rows from remaining domains; the held-out domain's labels are used only for final scoring.

For each labeled-positive fraction, a seeded subset of positive training examples becomes `P`; all negatives and remaining positives become `U`. The training view removes both recorded and hidden labels. Oracle prevalence is a non-deployable reference. A deployable Elkan–Noto estimate is computed from training-only out-of-fold probabilities as `mean(s(x)|P)` and clipped for stability. Sensitivity uses 0.75π, π, and 1.25π.

The naive U-as-negative baseline is included explicitly as an invalid PU comparator—not called PU learning. Majority, domain-prior, metadata, supervised TF–IDF/logistic regression, and fully supervised neural classifiers provide anchors.
""")
code(r"""
text_model = make_pipeline(TfidfVectorizer(min_df=3,max_features=30000,ngram_range=(1,2),sublinear_tf=True),
                           LogisticRegression(max_iter=1500,class_weight="balanced"))
text_model.fit(train.text.fillna(""),train.label)
val_text_p=text_model.predict_proba(val.text.fillna(""))[:,1]
test_text_p=text_model.predict_proba(test.text.fillna(""))[:,1]
thr=choose_threshold(val.label,val_text_p)
baseline_rows=[{"method":"majority","fraction":1.0,**classification_metrics(test.label,np.full(len(test),train.label.mean()),.5)},
               {"method":"supervised_tfidf_upper_bound","fraction":1.0,**classification_metrics(test.label,test_text_p,thr)}]
print(pd.DataFrame(baseline_rows)[["method","f1","macro_f1","pr_auc","roc_auc","mcc"]])

def elkan_noto_prior(partition):
    # training-only: estimate c=P(s=1|y=1) by OOF prediction of observed selection s.
    y_s=partition.pu_status.eq("P").astype(int)
    if y_s.sum()<2: return float(np.clip(partition.pu_status.eq("P").mean()*5,.02,.98))
    folds=min(3,int(y_s.sum()))
    model=make_pipeline(TfidfVectorizer(min_df=2,max_features=10000),LogisticRegression(max_iter=1000,class_weight="balanced"))
    oof_s=cross_val_predict(model,partition.text.fillna(""),y_s,cv=StratifiedKFold(folds,shuffle=True,random_state=42),method="predict_proba")[:,1]
    c=max(oof_s[y_s.eq(1)].mean(),1e-3)
    return float(np.clip(y_s.mean()/c,.01,.99))
""")

md(r"""
## 5. Correct nnPU and PU–Mean Teacher

For score function $f$ and logistic loss $\ell$, the unbiased PU risk is

$$R_{PU}=\pi\,\mathbb{E}_{P}[\ell(f,+1)] + \mathbb{E}_{U}[\ell(f,-1)]-\pi\,\mathbb{E}_{P}[\ell(f,-1)].$$

The last two terms estimate negative risk. nnPU prevents pathological negative empirical risk; when $\hat R_N<-\beta$, the implementation uses the Kiryo gradient correction $-\gamma\hat R_N$, otherwise $\pi\hat R_P+\hat R_N$. Mini-batches contain separate P and U draws and return all risk components.

The combined objective is $R_{nnPU}+\lambda(t)L_{consistency}$. Weak and strong feature dropout create meaning-preserving views of frozen sentence embeddings. Teacher probabilities on weak views are retained only above confidence $\tau$; the student matches them on strong views. $\lambda(t)$ has a sigmoid ramp-up and teacher weights use EMA. Confidence masking, ramp-up, validation early stopping, and a valid PU classification term reduce confirmation bias. Unlike ordinary self-training, no hard pseudo-label replaces PU risk.
""")
code(r"""
from sentence_transformers import SentenceTransformer
encoder=SentenceTransformer(CONFIG["embedding_model"],device=str(DEVICE))
encoder.max_seq_length=CONFIG["max_text_length"]
emb=encoder.encode(data.text.fillna("").tolist(),batch_size=CONFIG["batch_size"],show_progress_bar=True,normalize_embeddings=True)
E={split:torch.tensor(emb[data.split.eq(split).to_numpy()],dtype=torch.float32) for split in ["train","validation","test"]}

def predict_neural(model,X,batch=512):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),batch): out.append(torch.sigmoid(model(X[i:i+batch].to(DEVICE))).cpu())
    return torch.cat(out).numpy()

def train_pu_teacher(X, part, prior, seed, use_consistency=True):
    seed_everything(seed); student=MLPClassifier(X.shape[1],128,.25).to(DEVICE); teacher=make_teacher(student)
    opt=torch.optim.AdamW(student.parameters(),lr=CONFIG["learning_rate"],weight_decay=1e-4)
    p_idx=np.flatnonzero(part.pu_status.to_numpy()=="P"); u_idx=np.flatnonzero(part.pu_status.to_numpy()=="U")
    steps=max(1,int(np.ceil(max(len(p_idx),len(u_idx))/CONFIG["batch_size"])))
    history=[]; best=None; best_score=-np.inf; patience=0; global_step=0
    for epoch in range(CONFIG["epochs"]):
        student.train(); rng=np.random.default_rng(seed+epoch)
        for _ in range(steps):
            pi=rng.choice(p_idx,CONFIG["batch_size"],replace=len(p_idx)<CONFIG["batch_size"])
            ui=rng.choice(u_idx,CONFIG["batch_size"],replace=len(u_idx)<CONFIG["batch_size"])
            xp,xu=X[pi].to(DEVICE),X[ui].to(DEVICE)
            risk,diag=nnpu_loss(student(xp),student(xu),prior)
            cons=torch.tensor(0.,device=DEVICE); coverage=torch.tensor(0.,device=DEVICE)
            if use_consistency:
                weak=mask_embedding(xu,.05); strong=mask_embedding(xu,.20)
                with torch.no_grad(): tlog=teacher(weak)
                cons,coverage=consistency_loss(student(strong),tlog,CONFIG["confidence_threshold"])
            weight=sigmoid_rampup(global_step,max(1,CONFIG["epochs"]*steps//2))
            loss=risk+weight*cons
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),5); opt.step()
            ema_update(teacher,student,CONFIG["ema_decay"]); global_step+=1
        vp=predict_neural(teacher if use_consistency else student,E["validation"])
        score=average_precision_score(val.label,vp); history.append((epoch,float(loss),float(risk),float(cons),float(coverage),score))
        if score>best_score+1e-4:
            best_score=score; best={k:v.detach().cpu().clone() for k,v in (teacher if use_consistency else student).state_dict().items()}; patience=0
        else: patience+=1
        if patience>=3: break
    final=make_teacher(student) if use_consistency else student
    final.load_state_dict(best); return final.to(DEVICE),history
""")

md("### 5.1 Label-efficiency experiment and prior sensitivity")
code(r"""
results=list(baseline_rows); histories={}; prior_rows=[]
for seed in CONFIG["seeds"]:
  for fraction in CONFIG["positive_fractions"]:
    part=make_pu_partition(train,fraction,seed)
    assert "label" not in training_view(part).columns and "hidden_label_eval_only" not in training_view(part).columns
    oracle=float(train.label.mean()); estimated=elkan_noto_prior(part)
    prior_rows.append({"seed":seed,"fraction":fraction,"oracle_prior_reference":oracle,"elkan_noto_training_only":estimated})
    # Invalid comparator: observed P=1, all U=0.
    naive=make_pipeline(TfidfVectorizer(min_df=2,max_features=20000,ngram_range=(1,2)),LogisticRegression(max_iter=1000,class_weight="balanced"))
    naive.fit(part.text.fillna(""),part.pu_status.eq("P").astype(int))
    nv=naive.predict_proba(val.text.fillna(""))[:,1]; nt=naive.predict_proba(test.text.fillna(""))[:,1]
    results.append({"method":"naive_U_as_negative","seed":seed,"fraction":fraction,**classification_metrics(test.label,nt,choose_threshold(val.label,nv))})
    for method,cons in [("nnPU_only",False),("PU_Mean_Teacher",True)]:
      model,hist=train_pu_teacher(E["train"],part,estimated,seed,cons)
      vp=predict_neural(model,E["validation"]); tp=predict_neural(model,E["test"])
      results.append({"method":method,"seed":seed,"fraction":fraction,"prior":estimated,**classification_metrics(test.label,tp,choose_threshold(val.label,vp))})
      histories[(method,seed,fraction)]=hist
    # Oracle and misspecification references run PU-only at 10% to limit quick-mode cost.
    if fraction==.10:
      for tag,pi in [("oracle",oracle),("prior_0.75x",estimated*.75),("prior_1.25x",min(.99,estimated*1.25))]:
        model,_=train_pu_teacher(E["train"],part,float(np.clip(pi,.01,.99)),seed,False)
        vp=predict_neural(model,E["validation"]);tp=predict_neural(model,E["test"])
        results.append({"method":f"nnPU_{tag}","seed":seed,"fraction":fraction,"prior":pi,**classification_metrics(test.label,tp,choose_threshold(val.label,vp))})
results=pd.DataFrame(results); display(pd.DataFrame(prior_rows)); display(results.sort_values(["fraction","method"])[["method","seed","fraction","f1","macro_f1","pr_auc","roc_auc","mcc","brier","ece"]])
""")

md("### 5.2 Semi-supervised-only ablation and fully supervised neural upper bound")
code(r"""
def train_bce_teacher(X,y,seed,teacher_mode):
    seed_everything(seed); student=MLPClassifier(X.shape[1],128,.25).to(DEVICE); teacher=make_teacher(student)
    opt=torch.optim.AdamW(student.parameters(),lr=CONFIG["learning_rate"])
    Xt=X.to(DEVICE); yt=torch.tensor(np.asarray(y),dtype=torch.float32,device=DEVICE)
    best=None;best_score=-1
    for epoch in range(CONFIG["epochs"]):
        student.train(); logits=student(mask_embedding(Xt,.20)); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yt)
        if teacher_mode:
            with torch.no_grad(): tlog=teacher(mask_embedding(Xt,.05))
            cons,_=consistency_loss(student(mask_embedding(Xt,.20)),tlog,CONFIG["confidence_threshold"])
            loss=loss+sigmoid_rampup(epoch,CONFIG["epochs"]//2)*cons
        opt.zero_grad();loss.backward();opt.step();ema_update(teacher,student,CONFIG["ema_decay"])
        candidate=teacher if teacher_mode else student;vp=predict_neural(candidate,E["validation"]);score=average_precision_score(val.label,vp)
        if score>best_score:best_score=score;best={k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}
    final=make_teacher(student) if teacher_mode else student; final.load_state_dict(best);return final.to(DEVICE)

# "SSL-only" has supervised labeled P and sampled true labeled negatives; hidden U gets consistency only.
part10=make_pu_partition(train,.10,CONFIG["seed"]); labeled=part10.pu_status.eq("P")|((train.label.eq(0))&(np.arange(len(train))%10==0))
ssl=train_bce_teacher(E["train"][labeled.to_numpy()],train.label[labeled],CONFIG["seed"],True)
full=train_bce_teacher(E["train"],train.label,CONFIG["seed"],False)
for name,model in [("semi_supervised_only",ssl),("fully_supervised_neural_upper_bound",full)]:
    vp=predict_neural(model,E["validation"]);tp=predict_neural(model,E["test"])
    row={"method":name,"seed":CONFIG["seed"],"fraction":.10 if "semi" in name else 1.,**classification_metrics(test.label,tp,choose_threshold(val.label,vp))}
    results=pd.concat([results,pd.DataFrame([row])],ignore_index=True)
display(results.groupby(["method","fraction"],dropna=False)[["f1","macro_f1","pr_auc","mcc"]].agg(["mean","std"]).round(3))
""")

md(r"""
## 6. Leave-one-domain-out evaluation

Quick mode executes job scams (severe imbalance), product reviews (approximately balanced), and phishing (security-message domain). For every rotation, the held-out domain is absent from fitting, prior estimation, early stopping, calibration, and threshold selection. Its official test labels appear only in the final metric call. Full mode rotates all domains.
""")
code(r"""
lodo=[]
for held in CONFIG["lodo_domains"]:
    tr=data[(data.split=="train")&data.domain.ne(held)].reset_index(drop=True)
    va=data[(data.split=="validation")&data.domain.ne(held)].reset_index(drop=True)
    te=data[(data.split=="test")&data.domain.eq(held)].reset_index(drop=True)
    assert held not in set(tr.domain) and held not in set(va.domain)
    pipe=clone(text_model).fit(tr.text.fillna(""),tr.label)
    vp=pipe.predict_proba(va.text.fillna(""))[:,1];tp=pipe.predict_proba(te.text.fillna(""))[:,1]
    lodo.append({"held_out_domain":held,"method":"supervised_tfidf_upper_bound",**classification_metrics(te.label,tp,choose_threshold(va.label,vp))})
    # Combined PU-MT on locally encoded frozen text features; held-out labels never enter.
    vec=TfidfVectorizer(min_df=2,max_features=5000,ngram_range=(1,2)); Xtr=vec.fit_transform(tr.text.fillna(""));Xva=vec.transform(va.text.fillna(""));Xte=vec.transform(te.text.fillna(""))
    sv=TruncatedSVD(n_components=min(128,Xtr.shape[1]-1),random_state=42); Atr=torch.tensor(sv.fit_transform(Xtr),dtype=torch.float32);Ava=torch.tensor(sv.transform(Xva),dtype=torch.float32);Ate=torch.tensor(sv.transform(Xte),dtype=torch.float32)
    part=make_pu_partition(tr,.10,CONFIG["seed"]);pi=elkan_noto_prior(part)
    oldE,oldval=E,val;E={"validation":Ava,"test":Ate};val=va
    model,_=train_pu_teacher(Atr,part,pi,CONFIG["seed"],True)
    vp=predict_neural(model,Ava);tp=predict_neural(model,Ate)
    lodo.append({"held_out_domain":held,"method":"PU_Mean_Teacher_10pct",**classification_metrics(te.label,tp,choose_threshold(va.label,vp))})
    E,val=oldE,oldval
lodo=pd.DataFrame(lodo);display(lodo[["held_out_domain","method","f1","macro_f1","pr_auc","roc_auc","mcc","false_positives_per_1000_negatives"]])
""")

md("## 7. Curves, calibration, confusion matrices, and uncertainty")
code(r"""
plot=results[results.method.isin(["naive_U_as_negative","nnPU_only","PU_Mean_Teacher"])]
plt.figure(figsize=(8,5));sns.lineplot(data=plot,x="fraction",y="pr_auc",hue="method",marker="o",errorbar=None);plt.title("Label efficiency on official test split (quick-mode seed)");plt.ylabel("PR-AUC");plt.xlabel("Fraction of positives revealed in training");plt.tight_layout();plt.show()

best_frac=.10;part=make_pu_partition(train,best_frac,CONFIG["seed"]);pi=elkan_noto_prior(part)
best_model,_=train_pu_teacher(E["train"],part,pi,CONFIG["seed"],True)
vp=predict_neural(best_model,E["validation"]);tp=predict_neural(best_model,E["test"]);threshold=choose_threshold(val.label,vp)
from sklearn.metrics import PrecisionRecallDisplay
fig,axes=plt.subplots(1,3,figsize=(16,4))
PrecisionRecallDisplay.from_predictions(test.label,tp,ax=axes[0]);axes[0].set_title("PU–Mean Teacher precision–recall")
prob_true,prob_pred=calibration_curve(test.label,tp,n_bins=10,strategy="quantile");axes[1].plot(prob_pred,prob_true,"o-");axes[1].plot([0,1],[0,1],"--",color="gray");axes[1].set(title="Calibration",xlabel="Mean predicted probability",ylabel="Observed deceptive fraction")
ConfusionMatrixDisplay.from_predictions(test.label,tp>=threshold,ax=axes[2],display_labels=["non-deceptive","deceptive"]);axes[2].set_title("Validation-selected threshold")
plt.tight_layout();plt.show()

rng=np.random.default_rng(42);values=[]
for _ in range(500):
    ix=rng.integers(0,len(test),len(test));values.append(f1_score(test.label.iloc[ix],tp[ix]>=threshold,zero_division=0))
print("Test F1 bootstrap percentile 95% CI:",np.quantile(values,[.025,.975]))
""")

md("## 8. Error analysis and cautious interpretability")
code(r"""
errors=test.assign(probability=tp,prediction=(tp>=threshold).astype(int))
errors["error_type"]=np.select([(errors.label==0)&(errors.prediction==1),(errors.label==1)&(errors.prediction==0)],["false_positive","false_negative"],default="correct")
display(errors[errors.error_type!="correct"].groupby(["domain","error_type"]).size().unstack(fill_value=0))
errors["length_bin"]=pd.qcut(errors.char_count.rank(method="first"),4,labels=["short","medium-short","medium-long","long"])
display(errors.assign(error=errors.error_type.ne("correct")).groupby(["length_bin","domain"],observed=True).error.mean().unstack().round(2))
display(errors.assign(has_url=errors.url_count.gt(0),error=errors.error_type.ne("correct")).groupby(["domain","has_url"]).error.mean().unstack().round(2))
display(errors[errors.error_type!="correct"].assign(confidence=lambda x:abs(x.probability-.5),text=lambda x:x.text.str[:180]).nlargest(12,"confidence")[["domain","error_type","label","probability","text"]])

# Classical coefficients are directional associations, not causal explanations.
lr=text_model.named_steps["logisticregression"];vec=text_model.named_steps["tfidfvectorizer"];names=vec.get_feature_names_out();coef=lr.coef_[0]
display(pd.DataFrame({"toward_deceptive":names[np.argsort(coef)[-15:][::-1]],"coefficient":np.sort(coef)[-15:][::-1]}))
display(pd.DataFrame({"toward_non_deceptive":names[np.argsort(coef)[:15]],"coefficient":np.sort(coef)[:15]}))
print("Inspect these for URL artifacts, political names, review/job templates, source vocabulary, and formatting shortcuts; coefficients do not prove semantic importance.")
""")

md(r"""
## 9. Conclusions, limitations, and execution manifest

Interpret the tables above, not anticipated outcomes. Compare `nnPU_only` with `naive_U_as_negative`, and `PU_Mean_Teacher` with `nnPU_only`, at matched fractions. One quick-mode seed cannot support significance claims; standard deviations are undefined and the bootstrap interval quantifies sampling—not training—uncertainty. Full mode supplies three paired seeds and seven LODO rotations; a paired permutation or Wilcoxon test is appropriate only after those runs exist.

Major limitations include simulated rather than naturally occurring PU selection; uncertain and domain-varying positive priors; source provenance, duplicates, and shallow artifacts; English-only heterogeneous constituent data; capped quick-mode samples; and limited hyperparameter search. Deception is not factual falsity, and attribution is not intent. False positives have real costs. No autonomous enforcement is justified.

Future work should de-duplicate before training, estimate domain-conditional priors, run all seeds and LODO domains, fine-tune a compact transformer on GPU, add nested validation for augmentation/threshold choices, audit subgroups and temporal drift, trace every constituent dataset license/provenance record, and externally validate on post-collection domains.
""")
code(r"""
EXECUTION_MANIFEST = pd.DataFrame([
 {"experiment":"Official seven-domain acquisition and quality audit","executed":True,"scope":f"{MODE}: {len(data)} rows; official SHA {DATASET_REVISION[:12]}"},
 {"experiment":"In-domain baselines and PU fractions","executed":True,"scope":f"fractions={CONFIG['positive_fractions']}, seeds={CONFIG['seeds']}"},
 {"experiment":"PU–Mean Teacher and ablations","executed":True,"scope":"frozen MiniLM embeddings; validation early stopping"},
 {"experiment":"LODO","executed":True,"scope":str(CONFIG["lodo_domains"])},
 {"experiment":"Full transformer fine-tuning","executed":False,"scope":"configuration/design only; GPU required"},
 {"experiment":"Full three-seed/all-domain experiment","executed":MODE=="full","scope":"not represented by quick-mode outputs"},
])
display(EXECUTION_MANIFEST)
print("No test labels were used for fitting, prior estimation, early stopping, calibration, or threshold selection.")
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.write(nb, Path(__file__).resolve().parents[1] / "difraud_pu_semisupervised_analysis.ipynb")

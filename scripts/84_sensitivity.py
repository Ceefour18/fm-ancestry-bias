"""Preregistered sensitivity analyses (PREREGISTRATION Part 5). Same H1 model, BH across 7 methods and
4-category decision rule as script 80. Primary metric: per-donor coarse macro-F1, seed 2026."""
import os, numpy as np, pandas as pd, anndata, statsmodels.formula.api as smf
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests
M = 0.02; METHODS = ["pca", "scvi", "celltypist", "geneformer", "uce", "scgpt_human", "scfoundation"]

def donor_f1(p, true="label_coarse", pred="pred_coarse"):
    out = {}
    for d, g in p.groupby("subject_id"):
        keep = g[true].value_counts().loc[lambda s: s >= 10].index; h = g[g[true].isin(keep)]
        out[d] = (f1_score(h[true], h[pred], labels=list(keep), average="macro", zero_division=0), len(g))
    return pd.DataFrame(out, index=["macro_f1", "n_cells"]).T

def h1(df, x="AA"):
    c = f"{x} + log_cells"
    if df.age.notna().all() and df.age.nunique() > 1: c += " + age"
    if df.disease.nunique() > 1: c += " + C(disease)"
    f = smf.ols(f"macro_f1 ~ {c}", df).fit()
    lo, hi = f.conf_int(0.05).loc[x]; lo9, hi9 = f.conf_int(0.10).loc[x]
    return {"gap": 100 * f.params[x], "ci95_lo": 100 * lo, "ci95_hi": 100 * hi, "ci90_lo": 100 * lo9,
            "ci90_hi": 100 * hi9, "p": f.pvalues[x], "n_donors": len(df), "covariates": c}

def verdict(r):
    detect = r["p_bh"] < 0.05 and (r["ci95_hi"] < 0 or r["ci95_lo"] > 0)
    if detect and r["gap"] <= -2: return "meaningful gap"
    if detect and abs(r["gap"]) < 2: return "detectable but negligible"
    if r["ci90_lo"] > -2 and r["ci90_hi"] < 2: return "equivalent"
    return "inconclusive"

D = pd.read_csv("results/metrics/donor_metrics.csv")
D["disease"] = D["classification"].fillna("Control").astype(str) if "classification" in D else "Control"
D.loc[D.dataset == "randolph", "disease"] = "Control"
cov = D[(D.method == "pca") & (D.seed == 2026)].set_index(["dataset", "subject_id"])[["ancestry", "age", "YRI", "disease"]]
analyses = {}

# primary (reference), fine labels, prostate patients / controls
P = D[D.seed == 2026].copy()
analyses["primary (reference)"] = P
analyses["fine labels"] = P[P.method != "celltypist"].assign(macro_f1=P.fine_macro_f1)
analyses["prostate patients only"] = P[(P.dataset == "prostate") & (P.disease != "Control")]
analyses["prostate controls only"] = P[(P.dataset == "prostate") & (P.disease == "Control")]

# composition-matched: each donor resampled to the dataset's mean cell-type mix; 5 draws averaged
rows = []
for d in ["randolph", "prostate"]:
    obs = anndata.read_h5ad(f"data/processed/{d}_labelled.h5ad", backed="r").obs[["label_coarse"]].astype(str)
    for m in METHODS:
        p = pd.read_csv(f"results/preds/{d}_primary_{m}_s2026.csv.gz").set_index("cell_id").join(obs)
        target = p.groupby("subject_id").label_coarse.value_counts(normalize=True).groupby(level=1).mean()
        draws = []
        for rep in range(5):
            rng = np.random.default_rng(2026 + rep); parts = []
            for donor, g in p.groupby("subject_id"):
                n = g.label_coarse.value_counts().reindex(target.index, fill_value=0)
                tg = target[n > 0] / target[n > 0].sum(); s = (n[n > 0] / tg).min()
                for t, k in (tg * s).round().astype(int).items():
                    if k > 0: parts.append(g[g.label_coarse == t].sample(k, random_state=int(rng.integers(1e9))))
            draws.append(donor_f1(pd.concat(parts)).macro_f1)
        f = pd.concat(draws, axis=1).mean(1)
        for donor, v in f.items():
            rows.append({"dataset": d, "method": m, "subject_id": donor, "macro_f1": v,
                         "n_cells": int(p[p.subject_id == donor].shape[0])})
CM = pd.DataFrame(rows).merge(cov.reset_index(), on=["dataset", "subject_id"])
analyses["composition-matched"] = CM

# influenza cells (Randolph), if the flu kNN run has finished
if all(os.path.exists(f"results/preds/randolph_flu_{m}_s2026.csv.gz") for m in METHODS):
    obs = anndata.read_h5ad("data/processed/randolph_labelled.h5ad", backed="r").obs[["label_coarse"]].astype(str)
    rows = []
    for m in METHODS:
        f = donor_f1(pd.read_csv(f"results/preds/randolph_flu_{m}_s2026.csv.gz").set_index("cell_id").join(obs))
        rows += [{"dataset": "randolph", "method": m, "subject_id": k, **v} for k, v in f.to_dict("index").items()]
    analyses["influenza cells"] = pd.DataFrame(rows).merge(cov.reset_index(), on=["dataset", "subject_id"])
else:
    print("NOTE: flu predictions not found yet; influenza analysis skipped (rerun when job finishes).")

out = []
for name, A in analyses.items():
    A = A.copy(); A["AA"] = (A.ancestry == "AA").astype(int); A["log_cells"] = np.log(A.n_cells.astype(float))
    for d in A.dataset.unique():
        for m in METHODS:
            sub = A[(A.dataset == d) & (A.method == m)]
            if sub.AA.nunique() == 2: out.append({"analysis": name, "dataset": d, "method": m, **h1(sub)})
    if name in ["primary (reference)", "influenza cells"]:                      # continuous YRI, Randolph
        for m in METHODS:
            sub = A[(A.dataset == "randolph") & (A.method == m)].dropna(subset=["YRI"])
            out.append({"analysis": f"continuous YRI ({name.split(' (')[0]})", "dataset": "randolph", "method": m, **h1(sub, "YRI")})
R = pd.DataFrame(out)
R["p_bh"] = R.groupby(["analysis", "dataset"]).p.transform(lambda x: multipletests(x, method="fdr_bh")[1])
R["verdict"] = R.apply(verdict, axis=1)
R.to_csv("results/metrics/sensitivity_h1.csv", index=False)

pd.set_option("display.width", 200)
R["cell"] = R.gap.map("{:+.2f}".format) + " (" + R.verdict.str.replace("equivalent", "equiv").str.replace("inconclusive", "inconcl") + ")"
for d in ["randolph", "prostate"]:
    print(f"\n=== {d}: AA - EA gap in F1 points (verdict); YRI rows = 0% -> 100% African ancestry ===")
    t = R[R.dataset == d].pivot(index="analysis", columns="method", values="cell")
    print(t[[m for m in METHODS if m in t.columns]].to_string())
print("\nDonors per analysis:"); print(R.groupby(["analysis", "dataset"]).n_donors.first().to_string())
"""POST HOC (exploratory): marker-based re-annotation of the NK/CD8_T boundary in Randolph NI cells,
then re-scoring of the seed-2026 predictions against those labels. Rule fixed before scoring:
author NK/CD8_T cells -> CD8_T if mean(T genes) > mean(NK genes), NK if lower, author label if tied."""
import glob, numpy as np, pandas as pd, scanpy as sc, statsmodels.formula.api as smf
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests
T_GENES = ["TRAC", "CD3D", "CD3E", "CD3G", "CD8B"]; NK_GENES = ["KLRF1", "NCAM1", "SH2D1B"]
a = sc.read_h5ad("data/processed/randolph_labelled.h5ad"); a = a[a.obs.condition == "NI"].copy(); a.var_names_make_unique()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
tg = [g for g in T_GENES if g in a.var_names]; ng = [g for g in NK_GENES if g in a.var_names]
print("T genes used:", tg, "| NK genes used:", ng)
t = np.asarray(a[:, tg].X.mean(1)).ravel(); n = np.asarray(a[:, ng].X.mean(1)).ravel()
obs = a.obs[["subject_id", "ancestry", "age", "label_coarse"]].copy(); obs["label_coarse"] = obs.label_coarse.astype(str)
box = obs.label_coarse.isin(["NK", "CD8_T"]).values
obs["label_marker"] = obs.label_coarse
obs.loc[box & (t > n), "label_marker"] = "CD8_T"; obs.loc[box & (n > t), "label_marker"] = "NK"
obs.to_csv("results/metrics/posthoc_marker_labels_randolph.csv.gz")

print("\n=== Author label -> marker label (% of author label), by ancestry ===")
b = obs[box]
print((pd.crosstab([b.ancestry, b.label_coarse], b.label_marker, normalize="index") * 100).round(1))
print("cells tied (author label kept):", int((box & (t == n)).sum()))

rows = []
for f in glob.glob("results/preds/randolph_primary_*_s2026.csv.gz"):
    m = f.split("primary_")[1].rsplit("_s", 1)[0]
    p = pd.read_csv(f).set_index("cell_id").join(obs[["label_coarse", "label_marker"]])
    for d, g in p.groupby("subject_id"):
        r = {"method": m, "subject_id": d, "n_cells": len(g)}
        for lab in ["label_coarse", "label_marker"]:
            keep = g[lab].value_counts().loc[lambda s: s >= 10].index; h = g[g[lab].isin(keep)]
            r[lab] = f1_score(h[lab], h.pred_coarse, labels=list(keep), average="macro", zero_division=0)
        rows.append(r)
cov = obs.drop_duplicates("subject_id").set_index("subject_id")[["ancestry", "age"]]
D = pd.DataFrame(rows).merge(cov, left_on="subject_id", right_index=True)
D["AA"] = (D.ancestry == "AA").astype(int); D["log_cells"] = np.log(D.n_cells)
D.to_csv("results/metrics/posthoc_marker_relabel_donor_f1.csv", index=False)

M7 = ["pca", "scvi", "geneformer", "uce", "scgpt_human", "scfoundation", "scgpt_cp"]
print("\n=== H1: AA - EA gap (F1 points), author labels vs marker labels; BH across 7 methods ===")
res = []
for lab in ["label_coarse", "label_marker"]:
    for m in M7:
        r = smf.ols(f"{lab} ~ AA + age + log_cells", D[D.method == m]).fit(); lo, hi = r.conf_int().loc["AA"]
        res.append({"labels": lab, "method": m, "gap": 100 * r.params["AA"], "lo": 100 * lo, "hi": 100 * hi, "p": r.pvalues["AA"]})
R = pd.DataFrame(res); R["p_bh"] = R.groupby("labels").p.transform(lambda x: multipletests(x, method="fdr_bh")[1])
R["cell"] = R.gap.map("{:+.2f}".format) + " [" + R.lo.map("{:+.2f}".format) + ", " + R.hi.map("{:+.2f}".format) + "] p_bh=" + R.p_bh.round(3).astype(str)
print(R.pivot(index="method", columns="labels", values="cell").loc[M7].to_string())

H = ["pca", "scvi", "geneformer", "uce", "scgpt_human", "scfoundation"]
print("\n=== H1b: extra gap vs PCA (F1 points), author vs marker labels ===")
for lab in ["label_coarse", "label_marker"]:
    P = D[D.method.isin(H)].copy(); P["method"] = pd.Categorical(P.method, categories=H)
    r = smf.mixedlm(f"{lab} ~ AA * C(method) + age + log_cells", P, groups=P.subject_id).fit(reml=True)
    print(f"[{lab}]")
    for term in [x for x in r.params.index if x.startswith("AA:")]:
        lo, hi = r.conf_int().loc[term]
        print(f"  {term:30s} {100*r.params[term]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] p={r.pvalues[term]:.3f}")
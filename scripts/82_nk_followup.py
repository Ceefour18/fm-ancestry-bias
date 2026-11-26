"""H3 follow-up, Randolph NI cells.
A (preregistered): pseudobulk counts per donor for NK and CD8_T -> DESeq2 in R (script 82b).
B (exploratory, post hoc): are CD8_T cells predicted NK real T cells with NK-like features?"""
import os, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp, statsmodels.formula.api as smf
os.makedirs("results/h3_nk", exist_ok=True)
a = sc.read_h5ad("data/processed/randolph_labelled.h5ad")
a = a[a.obs.condition == "NI"].copy(); a.var_names_make_unique()

# A. pseudobulk: sum raw counts per donor (>=20 cells of that type), genes with >=10 counts in >=5 donors
for t in ["NK", "CD8_T"]:
    s = a[a.obs.label_coarse == t]
    n = s.obs.subject_id.astype(str).value_counts(); keep = n[n >= 20].index
    s = s[s.obs.subject_id.astype(str).isin(keep).values]
    d = s.obs.subject_id.astype(str).values; u, inv = np.unique(d, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(d)), (inv, np.arange(len(d)))), shape=(len(u), len(d)))
    pb = pd.DataFrame((G @ sp.csr_matrix(s.X)).toarray().T.round().astype(int), index=s.var_names, columns=u)
    pb = pb[(pb >= 10).sum(1) >= 5]
    meta = s.obs.drop_duplicates("subject_id").set_index(s.obs.drop_duplicates("subject_id").subject_id.astype(str))
    meta = meta.loc[u, ["ancestry", "age"]]; meta["n_cells"] = n.loc[u].values
    pb.to_csv(f"results/h3_nk/pb_{t}_counts.csv.gz"); meta.to_csv(f"results/h3_nk/pb_{t}_meta.csv")
    print(f"pseudobulk {t}: {pb.shape[0]} genes x {pb.shape[1]} donors "
          f"(AA {int((meta.ancestry == 'AA').sum())}, EA {int((meta.ancestry == 'EA').sum())})", flush=True)

# B. marker check on confused cells (exploratory)
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
TCELL = ["CD3E", "CD3D", "CD8A", "CD8B"]
NKLIKE = ["NKG7", "GNLY", "KLRD1", "FCGR3A", "KLRF1", "KLRC2", "B3GAT1", "GZMB", "PRF1", "KIR2DL1", "KIR3DL1"]
genes = [g for g in TCELL + NKLIKE if g in a.var_names]
p = pd.read_csv("results/preds/randolph_primary_uce_s2026.csv.gz").set_index("cell_id").reindex(a.obs_names)
grp = pd.Series("other", index=a.obs_names)
grp[(a.obs.label_coarse == "CD8_T") & (p.pred_coarse == "CD8_T")] = "CD8 called CD8"
grp[(a.obs.label_coarse == "CD8_T") & (p.pred_coarse == "NK")] = "CD8 called NK"
grp[(a.obs.label_coarse == "NK") & (p.pred_coarse == "NK")] = "NK called NK"
grp[(a.obs.label_coarse == "NK") & (p.pred_coarse == "CD8_T")] = "NK called CD8"
X = pd.DataFrame(a[:, genes].X.toarray(), index=a.obs_names, columns=genes)
X["group"] = grp.values; X["ancestry"] = a.obs.ancestry.astype(str).values
X = X[X.group != "other"]
print("\n=== B. Mean log-expression by group (UCE predictions); cells per group in last column ===")
tab = X.groupby(["group", "ancestry"])[genes].mean().round(2); tab["cells"] = X.groupby(["group", "ancestry"]).size()
pd.set_option("display.width", 220); print(tab.to_string())
print("\n% of cells expressing CD3E (>0):")
print((X.assign(cd3=X.CD3E > 0).groupby(["group", "ancestry"]).cd3.mean() * 100).round(1).unstack())

# C. per donor: fraction of CD8 T cells that are NK-like (exploratory)
cd8 = a[a.obs.label_coarse == "CD8_T"].copy()
sc.tl.score_genes(cd8, [g for g in ["NKG7", "GNLY", "KLRD1", "FCGR3A", "KLRC2", "B3GAT1", "KIR2DL1", "KIR3DL1"] if g in cd8.var_names],
                  score_name="nklike", random_state=2026)
thr = np.quantile(cd8.obs.nklike[cd8.obs.ancestry == "EA"], 0.90)          # top 10% of EA CD8 cells
cd8.obs["is_nklike"] = cd8.obs.nklike > thr
for g, col in [("KLRC2", "nkg2c"), ("B3GAT1", "cd57")]:
    if g in cd8.var_names: cd8.obs[col] = np.asarray(cd8[:, g].X.toarray()).ravel() > 0
dd = cd8.obs.groupby("subject_id", observed=True).agg(ancestry=("ancestry", "first"), age=("age", "first"),
        frac_nklike=("is_nklike", "mean"), frac_nkg2c=("nkg2c", "mean"), frac_cd57=("cd57", "mean")).reset_index()
dd["AA"] = (dd.ancestry == "AA").astype(int); dd.to_csv("results/h3_nk/cd8_nklike_by_donor.csv", index=False)
print("\n=== C. Per-donor fraction of CD8 T cells (median %), and AA-EA difference adjusted for age ===")
for v in ["frac_nklike", "frac_nkg2c", "frac_cd57"]:
    f = smf.ols(f"{v} ~ AA + age", dd).fit(); lo, hi = f.conf_int().loc["AA"]
    med = (dd.groupby("ancestry")[v].median() * 100).round(1).to_dict()
    print(f"{v:12s} median AA {med.get('AA')}% | EA {med.get('EA')}% | diff {100*f.params['AA']:+.1f} pts "
          f"[{100*lo:+.1f}, {100*hi:+.1f}], p = {f.pvalues['AA']:.4f}")
"""H4: ancestry mixing per cell type (iLISI, silhouette), each method vs scVI, paired Wilcoxon across cell types.
Same cells for every method: up to 5,000 per cell type, seed 2026."""
import numpy as np, pandas as pd, scanpy as sc, anndata, harmonypy
from sklearn.metrics import silhouette_score
from scipy.stats import wilcoxon
N = 5000; rng = np.random.default_rng(2026)
METHODS = ["pca", "scvi", "geneformer", "uce", "scgpt_human", "scfoundation", "scgpt_cp"]
def emb_files(d): return {"scvi": f"emb_scvi_{d}", "geneformer": f"emb_geneformer_{d}", "uce": f"emb_uce_{d}_L33_s2026",
    "scgpt_human": f"emb_scgpt_{d}_scGPT_human", "scgpt_cp": f"emb_scgpt_{d}_scGPT_CP", "scfoundation": f"emb_scfoundation_{d}"}
rows = []
for d in ["randolph", "prostate"]:
    obs = anndata.read_h5ad(f"data/processed/{d}_labelled.h5ad", backed="r").obs
    if "condition" in obs: obs = obs[obs.condition == "NI"]                     # primary cells, as H1
    pick = []
    for t, g in obs.groupby("label_coarse", observed=True):
        if g.ancestry.nunique() == 2: pick += list(rng.choice(g.index, min(N, len(g)), replace=False))
    S = obs.loc[pick, ["label_coarse", "ancestry"]].astype(str)
    print(f"{d}: {len(S)} cells sampled across {S.label_coarse.nunique()} cell types", flush=True)
    for m in METHODS:
        if m == "pca":                                                          # same recipe as scripts 45/60
            x = sc.read_h5ad(f"data/processed/{d}_labelled.h5ad")[obs.index].copy()
            sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
            sc.pp.highly_variable_genes(x, n_top_genes=2000); sc.pp.pca(x, n_comps=50, mask_var="highly_variable")
            X = pd.DataFrame(x.obsm["X_pca"], index=x.obs_names).loc[S.index].to_numpy(np.float32)
        else:
            f = emb_files(d)[m]; E = np.load(f"data/processed/{f}.npy", mmap_mode="r")
            ids = pd.read_csv(f"data/processed/{f}_cells.csv").iloc[:, 0].astype(str)
            idx = pd.Index(ids).get_indexer(S.index.astype(str)); assert (idx >= 0).all()
            X = np.asarray(E[np.sort(idx)], dtype=np.float32)[np.argsort(np.argsort(idx))]
        for t in S.label_coarse.unique():
            k = (S.label_coarse == t).values; meta = S[k][["ancestry"]].reset_index(drop=True)
            ilisi = harmonypy.compute_lisi(X[k], meta, ["ancestry"], perplexity=30).mean()
            sil = silhouette_score(X[k], meta.ancestry, metric="euclidean")
            rows.append({"dataset": d, "method": m, "cell_type": t, "n": int(k.sum()),
                         "pct_AA": round(100 * (meta.ancestry == "AA").mean(), 1), "ilisi": ilisi, "silhouette": sil})
        print(f"  {d} {m} done", flush=True)
H = pd.DataFrame(rows); H.to_csv("results/metrics/h4_mixing.csv", index=False)
print("\n=== iLISI by cell type (higher = AA and EA more mixed; max 2) ===")
print(H.pivot_table(index=["dataset", "cell_type"], columns="method", values="ilisi").round(3).to_string())
print("\n=== silhouette by ancestry (higher = more separated by ancestry) ===")
print(H.pivot_table(index=["dataset", "cell_type"], columns="method", values="silhouette").round(3).to_string())
print("\n=== H4: each method vs scVI, paired Wilcoxon across cell types (both datasets) ===")
out = []
for m in [x for x in METHODS if x != "scvi"]:
    a = H[H.method == m].set_index(["dataset", "cell_type"]); b = H[H.method == "scvi"].set_index(["dataset", "cell_type"]).loc[a.index]
    for v in ["ilisi", "silhouette"]:
        diff = a[v] - b[v]
        out.append({"method": m, "metric": v, "median_diff_vs_scvi": diff.median(), "n_celltypes": len(diff),
                    "n_more_separated": int((diff < 0).sum() if v == "ilisi" else (diff > 0).sum()),
                    "p_wilcoxon": wilcoxon(a[v], b[v]).pvalue})
W = pd.DataFrame(out); W.to_csv("results/metrics/h4_wilcoxon.csv", index=False)
print(W.round(4).to_string(index=False))
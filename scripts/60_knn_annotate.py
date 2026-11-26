"""Stage 6: kNN (k=15) annotation, 5-fold donor CV, trained on EA donors of other folds only.
Same rules as the blinded pilot (script 45). Saves per-cell predictions; no metrics, no ancestry summaries."""
import argparse, json, os, time, numpy as np, pandas as pd, scanpy as sc, anndata
from sklearn.neighbors import NearestNeighbors
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--methods", default="pca,scvi,geneformer,uce,scgpt_human,scgpt_cp,scfoundation")
p.add_argument("--seeds", default="2026")                 # 2027-2030 = sensitivity
p.add_argument("--cells", default="primary", choices=["primary", "flu", "all"])
a = p.parse_args()
d = a.dataset
EMB = {"scvi": f"emb_scvi_{d}", "geneformer": f"emb_geneformer_{d}", "uce": f"emb_uce_{d}_L33_s2026",
       "scgpt_human": f"emb_scgpt_{d}_scGPT_human", "scgpt_cp": f"emb_scgpt_{d}_scGPT_CP",
       "scfoundation": f"emb_scfoundation_{d}"}
FINE_SET = {"B_naive", "B_memory", "CD4_naive", "CD4_memory", "CD8_T", "DC", "Monocyte", "NK"}

obs = anndata.read_h5ad(f"data/processed/{d}_labelled.h5ad", backed="r").obs.copy()
if "condition" in obs:                                        # Randolph: NI = primary (prereg line 46)
    if a.cells == "primary": obs = obs[obs.condition == "NI"]
    elif a.cells == "flu":   obs = obs[obs.condition == "flu"]
fine_col = [c for c in ["label_fine_gated", "label_fine"] if c in obs and set(obs[c].astype(str).unique()) <= FINE_SET]
assert fine_col, {c: sorted(obs[c].astype(str).unique()) for c in ["label_fine", "label_fine_gated"] if c in obs}
fine_col = fine_col[0]
y_c = obs.label_coarse.astype(str).values; y_f = obs[fine_col].astype(str).values
print(f"{d} [{a.cells}]: {len(obs)} cells, {obs.subject_id.nunique()} donors, fine labels = {fine_col}", flush=True)

def embedding(m):
    if m == "pca":                                            # identical to pilot script 45
        x = sc.read_h5ad(f"data/processed/{d}_labelled.h5ad")[obs.index].copy()
        sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
        sc.pp.highly_variable_genes(x, n_top_genes=2000)
        sc.pp.pca(x, n_comps=50, mask_var="highly_variable")
        return x.obsm["X_pca"].astype(np.float32)
    E = np.load(f"data/processed/{EMB[m]}.npy", mmap_mode="r")
    ids = pd.read_csv(f"data/processed/{EMB[m]}_cells.csv").iloc[:, 0].astype(str)
    idx = pd.Index(ids).get_indexer(obs.index.astype(str))
    assert (idx >= 0).all(), f"{m}: {(idx < 0).sum()} cells missing from embedding"
    return np.asarray(E[idx], dtype=np.float32)

def vote(nbr, y):                                             # majority of 15; ties -> alphabetical first (= sklearn)
    cls, code = np.unique(y, return_inverse=True)
    counts = np.zeros((nbr.shape[0], len(cls)), dtype=np.int16)
    for j in range(nbr.shape[1]): np.add.at(counts, (np.arange(nbr.shape[0]), code[nbr[:, j]]), 1)
    return cls[counts.argmax(1)]

splits = json.load(open("results/splits.json"))
os.makedirs("results/preds", exist_ok=True)
for m in a.methods.split(","):
    t0 = time.time(); X = embedding(m)
    for seed in a.seeds.split(","):
        out = pd.DataFrame({"cell_id": obs.index, "subject_id": obs.subject_id.astype(str).values,
                            "fold": -1, "pred_coarse": "", "pred_fine": ""})
        for k, donors in splits[d]["cv"][seed].items():
            test = obs.subject_id.isin(donors).values
            train = (~test) & (obs.ancestry == "EA").values
            nn = NearestNeighbors(n_neighbors=15, n_jobs=8).fit(X[train])
            nbr = nn.kneighbors(X[test], return_distance=False)
            out.loc[test, "fold"] = int(k)
            out.loc[test, "pred_coarse"] = vote(nbr, y_c[train])
            out.loc[test, "pred_fine"] = vote(nbr, y_f[train])
        assert (out.fold >= 0).all(), "some cells were never in a test fold"
        f = f"results/preds/{d}_{a.cells}_{m}_s{seed}.csv.gz"; out.to_csv(f, index=False)
        print(f"  {m} seed {seed}: {X.shape[1]} dims -> {f}", flush=True)
    print(f"  {m} done in {round(time.time() - t0)} s", flush=True)
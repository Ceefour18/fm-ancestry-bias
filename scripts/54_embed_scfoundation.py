import argparse, os, subprocess, sys, time, numpy as np, pandas as pd, anndata, scipy.sparse as sp
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True); p.add_argument("--n_cells", type=int, default=0)
a = p.parse_args()
ROOT = os.getcwd(); MD = f"{ROOT}/models/scFoundation/model"
sub = f"{a.dataset}_test{a.n_cells}" if a.n_cells else a.dataset
work = f"{ROOT}/data/processed/scfoundation/{sub}"; os.makedirs(work, exist_ok=True); t0 = time.time()

ad = anndata.read_h5ad(f"data/processed/uce/{sub}/cells.h5ad")   # raw counts, gene symbols
genes = pd.read_csv(f"{MD}/OS_scRNA_gene_index.19264.tsv", sep="\t")["gene_name"].astype(str).tolist()
col = {g: i for i, g in enumerate(genes)}
ours = ad.var_names.astype(str)
keep = np.array([g in col for g in ours])
X = sp.csr_matrix(ad.X)[:, keep].tocoo()
target = np.array([col[g] for g in ours[keep]])
M = sp.csr_matrix((X.data.astype(np.float32), (X.row, target[X.col])), shape=(ad.n_obs, len(genes)))
print(f"genes matched: {keep.sum()} of {len(genes)} scFoundation genes ({keep.sum()/len(genes):.1%}); "
      f"UMI kept: {M.sum()/ad.X.sum():.1%}", flush=True)
npz = f"{work}/input.npz"; sp.save_npz(npz, M)

cmd = [sys.executable, "get_embedding.py", "--task_name", sub, "--input_type", "singlecell",
       "--output_type", "cell", "--pool_type", "all", "--tgthighres", "t4", "--data_path", npz,
       "--save_path", work + "/", "--pre_normalized", "F", "--version", "ce"]
subprocess.run(cmd, cwd=MD, check=True)

E = np.load(f"{work}/{sub}_01B-resolution_singlecell_cell_embedding_t4_resolution.npy").astype(np.float32)
assert E.shape[0] == ad.n_obs, f"row mismatch {E.shape[0]} vs {ad.n_obs}"
np.save(f"data/processed/emb_scfoundation_{sub}.npy", E)
pd.Series(ad.obs_names).to_csv(f"data/processed/emb_scfoundation_{sub}_cells.csv", index=False)
print("Embeddings:", E.shape, "| NaN:", int(np.isnan(E).sum()), "| total", round(time.time() - t0), "s")
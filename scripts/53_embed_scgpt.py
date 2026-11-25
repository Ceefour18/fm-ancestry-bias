import argparse, os, time, random, numpy as np, pandas as pd, anndata, torch
import scgpt as scg
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True); p.add_argument("--n_cells", type=int, default=0)
p.add_argument("--model", default="scGPT_human"); p.add_argument("--batch", type=int, default=64)
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
sub = f"{a.dataset}_test{a.n_cells}" if a.n_cells else a.dataset
tag = f"{sub}_{a.model}"
t0 = time.time()
ad = anndata.read_h5ad(f"data/processed/uce/{sub}/cells.h5ad")   # counts-only input made by 51_embed_uce.py --stage prep
ad.var["gene_name"] = ad.var_names.astype(str)
n_in = ad.n_obs; ids = ad.obs_names.copy()
out = scg.tasks.embed_data(ad, f"models/{a.model}", gene_col="gene_name", max_length=1200,
                           batch_size=a.batch, device="cuda", use_fast_transformer=False, return_new_adata=False)
assert out.n_obs == n_in and (out.obs_names == ids).all(), "scGPT dropped or reordered cells"
E = np.asarray(out.obsm["X_scGPT"], dtype=np.float32)
np.save(f"data/processed/emb_scgpt_{tag}.npy", E)
pd.Series(out.obs_names).to_csv(f"data/processed/emb_scgpt_{tag}_cells.csv", index=False)
print("Embeddings:", E.shape, "| NaN:", int(np.isnan(E).sum()), "| total", round(time.time() - t0), "s")
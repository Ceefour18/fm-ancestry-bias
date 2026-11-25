import argparse, os, time, numpy as np, pandas as pd, scanpy as sc
from geneformer import TranscriptomeTokenizer, EmbExtractor

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)                 # randolph | prostate
p.add_argument("--n_cells", type=int, default=0)           # 0 = all cells; >0 = random test subset
p.add_argument("--model", default="models/Geneformer/Geneformer-V2-104M")
p.add_argument("--batch", type=int, default=64)
a = p.parse_args()

tag = f"{a.dataset}_test{a.n_cells}" if a.n_cells else a.dataset
work = f"data/processed/geneformer/{tag}"; os.makedirs(f"{work}/input", exist_ok=True)
t0 = time.time()

ad = sc.read_h5ad(f"data/processed/{a.dataset}_labelled.h5ad")
if a.n_cells:
    ad = ad[np.random.default_rng(2026).choice(ad.n_obs, a.n_cells, replace=False)].copy()
ad.var["ensembl_id"] = ad.var["gene_ids"].astype(str).values
ad.obs["n_counts"] = np.asarray(ad.X.sum(axis=1)).ravel()
ad.obs["cell_id"] = ad.obs_names.astype(str)
ad.write_h5ad(f"{work}/input/cells.h5ad")
print(f"Input: {ad.n_obs} cells x {ad.n_vars} genes")

tk = TranscriptomeTokenizer(custom_attr_name_dict={"cell_id": "cell_id"}, nproc=16, model_version="V2")
if not os.path.exists(f"{work}/tokens.dataset"): tk.tokenize_data(f"{work}/input", f"{work}", "tokens", file_format="h5ad")
print(f"Tokenised in {time.time()-t0:.0f}s")

ex = EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode="cell", max_ncells=None,
                  emb_layer=-1, emb_label=["cell_id"], forward_batch_size=a.batch, nproc=16,
                  model_version="V2")
embs = ex.extract_embs(a.model, f"{work}/tokens.dataset", f"{work}", "embs")
embs = embs.set_index("cell_id").loc[ad.obs_names]                 # back to original cell order
X = embs.to_numpy(dtype=np.float32)
np.save(f"data/processed/emb_geneformer_{tag}.npy", X)
pd.Series(embs.index).to_csv(f"data/processed/emb_geneformer_{tag}_cells.csv", index=False)
print(f"Embeddings: {X.shape} | NaN: {np.isnan(X).sum()} | total {time.time()-t0:.0f}s")

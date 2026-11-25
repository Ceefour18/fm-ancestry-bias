import argparse, time, numpy as np, pandas as pd, scanpy as sc, scvi
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--batch_key", default="none")   # "none" or an obs column
p.add_argument("--n_latent", type=int, default=30)
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
scvi.settings.seed = a.seed; t0 = time.time()
ad = sc.read_h5ad(f"data/processed/{a.dataset}_labelled.h5ad")
ad.layers["counts"] = ad.X.copy()
bk = None if a.batch_key == "none" else a.batch_key
sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3", layer="counts", batch_key=bk, subset=True)
scvi.model.SCVI.setup_anndata(ad, layer="counts", batch_key=bk)
m = scvi.model.SCVI(ad, n_latent=a.n_latent, n_layers=2)
m.train(early_stopping=True)
E = m.get_latent_representation().astype(np.float32)
tag = a.dataset
np.save(f"data/processed/emb_scvi_{tag}.npy", E)
pd.Series(ad.obs_names).to_csv(f"data/processed/emb_scvi_{tag}_cells.csv", index=False)
m.save(f"checkpoints/scvi_{tag}", overwrite=True)
print("Embeddings:", E.shape, "| NaN:", int(np.isnan(E).sum()), "| batch_key:", bk,
      "| epochs:", len(m.history["elbo_train"]), "| total", round(time.time() - t0), "s")

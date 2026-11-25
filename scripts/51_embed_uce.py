import argparse, os, sys, time, random, runpy
import numpy as np, pandas as pd, scipy.sparse as sp, anndata
p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["prep", "run"], required=True)
p.add_argument("--dataset", required=True); p.add_argument("--n_cells", type=int, default=0)
p.add_argument("--nlayers", type=int, default=4); p.add_argument("--batch", type=int, default=100)
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
ROOT = os.getcwd(); UCE = f"{ROOT}/models/UCE"
sub = f"{a.dataset}_test{a.n_cells}" if a.n_cells else a.dataset
work = f"{ROOT}/data/processed/uce/{sub}"; os.makedirs(work, exist_ok=True)
inp = f"{work}/cells.h5ad"

if a.stage == "prep":   # fm-core: read labelled file, write minimal counts-only h5ad
    ad = anndata.read_h5ad(f"data/processed/{a.dataset}_labelled.h5ad")
    if a.n_cells:  # subset fixed by 2026, independent of --seed
        ad = ad[np.sort(np.random.default_rng(2026).choice(ad.n_obs, a.n_cells, replace=False))]
    X = sp.csr_matrix(ad.X, dtype=np.float32)
    assert np.allclose(X.data[:100000], np.round(X.data[:100000])), "not raw counts"
    m = anndata.AnnData(X=X, obs=pd.DataFrame(index=ad.obs_names.astype(str).values),
                        var=pd.DataFrame(index=ad.var_names.astype(str).values))
    m.obs.index.name = None; m.var.index.name = None; m.var_names_make_unique()
    m.write_h5ad(inp); print("prepared", m.shape, inp)
    sys.exit(0)

import torch            # fm-uce: run UCE with fixed seeds, all cells kept
tag = f"{sub}_L{a.nlayers}_s{a.seed}"
out_dir = f"{work}/L{a.nlayers}_s{a.seed}/"; os.makedirs(out_dir, exist_ok=True); t0 = time.time()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
argv = ["eval_single_anndata.py", "--adata_path", inp, "--dir", out_dir, "--species", "human",
        "--filter", "", "--batch_size", str(a.batch), "--nlayers", str(a.nlayers)]
if a.nlayers == 33:
    mfile = f"{UCE}/model_files/33l_8ep_1024t_1280.torch"
    assert os.path.exists(mfile), "download the 33-layer model first"
    argv += ["--model_loc", mfile]
os.chdir(UCE); sys.path.insert(0, UCE); sys.argv = argv
runpy.run_path("eval_single_anndata.py", run_name="__main__")
os.chdir(ROOT)
ref = anndata.read_h5ad(inp, backed="r").obs_names
out = anndata.read_h5ad(f"{out_dir}cells_uce_adata.h5ad")
assert out.n_obs == len(ref) and (out.obs_names == ref).all(), "UCE dropped or reordered cells"
E = np.asarray(out.obsm["X_uce"], dtype=np.float32)
np.save(f"data/processed/emb_uce_{tag}.npy", E)
pd.Series(out.obs_names).to_csv(f"data/processed/emb_uce_{tag}_cells.csv", index=False)
s = time.time() - t0
print("shape", E.shape, "| NaN", int(np.isnan(E).sum()), "| seconds", round(s))
if a.n_cells: print("projected GPU-hours, both datasets:", round(s / a.n_cells * (235150 + 293287) / 3600, 1))

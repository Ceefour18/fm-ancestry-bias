"""H2 donor sets (fixed before training). Test = fixed split 2026 test donors. Validation = 5 EA donors
from the training pool (learning-rate choice only). Regimes have equal donor counts N = remaining EA donors."""
import json, numpy as np, anndata
s = json.load(open("results/splits.json"))["randolph"]["fixed"]["2026"]
o = anndata.read_h5ad("data/processed/randolph_labelled.h5ad", backed="r").obs
anc = o.drop_duplicates("subject_id").set_index("subject_id").ancestry.astype(str)
ea = sorted(d for d in s["train"] if anc[d] == "EA"); aa = sorted(d for d in s["train"] if anc[d] == "AA")
val = sorted(np.random.default_rng(2026).choice(ea, 5, replace=False).tolist())
pool_ea = [d for d in ea if d not in val]; N = len(pool_ea)
H = {"test": s["test"], "val": val, "N": N, "regimes": {}}
for seed in [2026, 2027, 2028]:
    r = np.random.default_rng(seed)
    H["regimes"][str(seed)] = {
        "ea_only": pool_ea,
        "balanced": sorted(r.choice(pool_ea, N // 2, replace=False).tolist() + r.choice(aa, N - N // 2, replace=False).tolist()),
        "random": sorted(r.choice(pool_ea + aa, N, replace=False).tolist())}
json.dump(H, open("results/h2_splits.json", "w"), indent=1)
print(f"test {len(H['test'])} donors (AA {sum(anc[d]=='AA' for d in H['test'])}) | val {len(val)} EA | N = {N}")
for seed, R in H["regimes"].items():
    print(seed, {k: f"{len(v)} donors, AA {sum(anc[d]=='AA' for d in v)}" for k, v in R.items()})
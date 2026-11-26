"""CellTypist (Immune_All_Low) on ALL donors (pretrained: no training on our data, so no CV).
Per-cell predictions in the script-60 format so scripts 70/80 score it identically.
Fine labels not scored: the model merges naive and central-memory T cells."""
import pandas as pd, scanpy as sc, celltypist
from celltypist import models
model = models.Model.load("Immune_All_Low.pkl")

def to_coarse(lbl):                                     # corrected 2026-11-26: all helper/Treg -> CD4_T, macrophage-like -> Monocyte
    l = lbl.lower()
    if "cd4" in l or "helper" in l or "treg" in l or "regulatory t" in l: return "CD4_T"
    if "cd8" in l or "mait" in l or "gamma-delta" in l or "cytotoxic t" in l: return "CD8_T"
    if "nk" in l or "ilc" in l: return "NK"
    if "b cell" in l or "plasma" in l or "naive b" in l or "memory b" in l: return "B"
    if "monocyte" in l or "macrophage" in l or "mono-mac" in l or "kupffer" in l: return "Monocyte"
    if "dc" in l or "dendritic" in l: return "DC"
    return "Other"

for d, cells in [("randolph", "primary"), ("randolph", "flu"), ("prostate", "primary")]:
    a = sc.read_h5ad(f"data/processed/{d}_labelled.h5ad")
    if "condition" in a.obs: a = a[a.obs.condition == ("NI" if cells == "primary" else "flu")].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    raw = celltypist.annotate(a, model=model, majority_voting=False).predicted_labels.predicted_labels.astype(str)
    out = pd.DataFrame({"cell_id": a.obs_names, "subject_id": a.obs.subject_id.astype(str).values, "fold": -1,
                        "pred_coarse": raw.map(to_coarse).values, "pred_fine": "NA", "pred_raw": raw.values})
    out.to_csv(f"results/preds/{d}_{cells}_celltypist_s2026.csv.gz", index=False)
    print(f"{d} {cells}: {len(out)} cells | {out.pred_coarse.value_counts().to_dict()}", flush=True)
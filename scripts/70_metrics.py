"""Stage 7: per-donor metrics from kNN predictions. Same scoring rules as pilot script 45.
Writes donor-level tables for script 80; prints only pooled (not ancestry-split) checks."""
import glob, os, numpy as np, pandas as pd, anndata
from sklearn.metrics import f1_score, recall_score

def score(g, true, pred):
    keep = g[true].value_counts().loc[lambda s: s >= 10].index        # >=10 cells of that type in this donor
    h = g[g[true].isin(keep)]
    L = list(keep)
    return (f1_score(h[true], h[pred], labels=L, average="macro", zero_division=0),
            recall_score(h[true], h[pred], labels=L, average="macro", zero_division=0),   # balanced accuracy
            f1_score(h[true], h[pred], labels=L, average=None, zero_division=0), L, len(h))

donor_rows, type_rows = [], []
for d in ["randolph", "prostate"]:
    obs = anndata.read_h5ad(f"data/processed/{d}_labelled.h5ad", backed="r").obs
    covs = [c for c in ["ancestry", "age", "YRI", "batch", "classification", "condition"] if c in obs]
    for f in sorted(glob.glob(f"results/preds/{d}_primary_*_s*.csv.gz")):
        method = f.split(f"{d}_primary_")[1].rsplit("_s", 1)[0]; seed = int(f.rsplit("_s", 1)[1][:4])
        p = pd.read_csv(f).set_index("cell_id"); p["pred_fine"] = p.pred_fine.fillna("NA").astype(str)
        g_all = p.join(obs[["label_coarse", "label_fine"]].astype(str))
        assert g_all.label_coarse.notna().all(), f"{f}: cells missing from labelled file"
        for donor, g in g_all.groupby("subject_id"):
            mf1, bacc, per, L, n = score(g, "label_coarse", "pred_coarse")
            ff1 = score(g, "label_fine", "pred_fine")[0]
            row = {"dataset": d, "method": method, "seed": seed, "subject_id": donor,
                   "n_cells": len(g), "n_cells_scored": n, "n_types": len(L),
                   "macro_f1": mf1, "bal_acc": bacc, "fine_macro_f1": ff1}
            row.update(obs.loc[obs.subject_id.astype(str) == str(donor), covs].iloc[0].to_dict())
            donor_rows.append(row)
            if seed == 2026:
                type_rows += [{"dataset": d, "method": method, "subject_id": donor, "cell_type": t, "f1": v}
                              for t, v in zip(L, per)]
        print(f"scored {d} {method} seed {seed}", flush=True)

D = pd.DataFrame(donor_rows); T = pd.DataFrame(type_rows)
os.makedirs("results/metrics", exist_ok=True)
D.to_csv("results/metrics/donor_metrics.csv", index=False)
T.to_csv("results/metrics/donor_celltype_f1.csv", index=False)
print("\nPOOLED checks (both ancestries together; no ancestry comparison):")
print(D[D.seed == 2026].groupby(["dataset", "method"]).agg(donors=("subject_id", "nunique"),
      macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std")).round(3))
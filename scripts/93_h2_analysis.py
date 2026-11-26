"""H2 analysis (preregistered): f1 ~ AA * regime + (1 | donor), EA-only = reference, per model.
Per-donor coarse macro-F1 on the 40 fixed test donors (>=10 cells rule), 3 regimes x 3 seeds."""
import glob, re, numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.metrics import f1_score
cov = (pd.read_csv("results/metrics/donor_metrics.csv").query("dataset == 'randolph'")
       .drop_duplicates("subject_id").set_index("subject_id")[["ancestry"]])
rows, nk = [], []
for f in sorted(glob.glob("results/preds_h2/*_lr1e-05.csv.gz")):
    m = re.match(r".*/(geneformer|scgpt)_(ea_only|balanced|random)_s(\d+)_lr1e-05\.csv\.gz$", f)
    if not m: continue
    model, regime, seed = m.group(1), m.group(2), int(m.group(3))
    p = pd.read_csv(f)
    for d, g in p.groupby("subject_id"):
        keep = g.label.value_counts().loc[lambda s: s >= 10].index; h = g[g.label.isin(keep)]
        rows.append({"model": model, "regime": regime, "seed": seed, "subject_id": str(d),
                     "f1": f1_score(h.label, h.pred_coarse, labels=list(keep), average="macro", zero_division=0)})
        if "NK" in keep:
            nk.append({"model": model, "regime": regime, "seed": seed, "subject_id": str(d),
                       "f1_nk": f1_score(h.label, h.pred_coarse, labels=["NK"], average="macro", zero_division=0)})
D = pd.DataFrame(rows).join(cov, on="subject_id"); D["AA"] = (D.ancestry == "AA").astype(int)
N = pd.DataFrame(nk).join(cov, on="subject_id")
D.to_csv("results/metrics/h2_donor_f1.csv", index=False)
print("Runs found:", D.groupby(["model", "regime"]).seed.nunique().to_dict(), "| test donors:", D.subject_id.nunique())

pd.set_option("display.width", 160)
print("\n=== Mean macro-F1 by regime and ancestry (x100), averaged over seeds and donors ===")
print((D.groupby(["model", "regime", "ancestry"]).f1.mean().unstack() * 100).round(2).assign(
      gap_AA_minus_EA=lambda t: (t.AA - t.EA).round(2)).to_string())
print("\n=== NK-cell F1 by regime and ancestry (x100) ===")
print((N.groupby(["model", "regime", "ancestry"]).f1_nk.mean().unstack() * 100).round(2).assign(
      gap_AA_minus_EA=lambda t: (t.AA - t.EA).round(2)).to_string())

print("\n=== PREREGISTERED H2 model: f1 ~ AA * regime + (1 | donor), EA-only reference (F1 points) ===")
out = []
for model, g in D.groupby("model"):
    g = g.copy(); g["regime"] = pd.Categorical(g.regime, categories=["ea_only", "balanced", "random"])
    r = smf.mixedlm("f1 ~ AA * C(regime)", g, groups=g.subject_id).fit(reml=True); ci = r.conf_int()
    for t in [x for x in r.params.index if x.startswith("AA")]:
        out.append({"model": model, "term": t, "est_points": 100 * r.params[t], "ci95_lo": 100 * ci.loc[t, 0],
                    "ci95_hi": 100 * ci.loc[t, 1], "p": r.pvalues[t], "converged": r.converged})
H2 = pd.DataFrame(out); H2.to_csv("results/metrics/h2_model.csv", index=False)
print(H2.round(3).to_string(index=False))
print("\nRead: 'AA' = AA gap under EA-only training; 'AA:C(regime)[T.balanced]' = change in that gap with")
print("balanced training (positive = gap shrinks); same for random.")
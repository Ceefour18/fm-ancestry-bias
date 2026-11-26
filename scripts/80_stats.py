"""Stage 8: preregistered tests (PREREGISTRATION.md Part 5).
H1: F1 ~ AA + age + log(cells) [+ disease group, prostate]; BH across 7 methods; 4-category decision rule.
H1b: f1 ~ AA*method + covs + (1|donor), PCA reference; confirmatory = pooled (+dataset, +disease); supportive = Randolph.
H3: per-cell-type gaps. Power re-reported with observed SD. Seed sensitivity 2027-2030. H4 = script 81."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
M = 0.02
METHODS = ["pca", "scvi", "geneformer", "uce", "scgpt_human", "scfoundation", "scgpt_cp"]   # 7 for BH
H1B = ["pca", "scvi", "geneformer", "uce", "scgpt_human", "scfoundation"]
D = pd.read_csv("results/metrics/donor_metrics.csv")
D["AA"] = (D.ancestry == "AA").astype(int); D["log_cells"] = np.log(D.n_cells)
D["disease"] = D["classification"].fillna("Control").astype(str) if "classification" in D else "none"
D.loc[D.dataset == "randolph", "disease"] = "Control"

def covs_for(df):
    c = "AA + log_cells"
    if df.age.notna().all() and df.age.nunique() > 1: c += " + age"
    if df.disease.nunique() > 1: c += " + C(disease)"
    return c

def h1(df, y="macro_f1"):
    c = covs_for(df); f = smf.ols(f"{y} ~ {c}", data=df).fit()
    lo95, hi95 = f.conf_int(0.05).loc["AA"]; lo90, hi90 = f.conf_int(0.10).loc["AA"]
    sd = np.sqrt(f.mse_resid); n1, n0 = int(df.AA.sum()), int((1 - df.AA).sum())
    power = TTestIndPower().power(effect_size=M / sd, nobs1=n1, ratio=n0 / n1, alpha=0.05)
    return {"gap_points": 100 * f.params["AA"], "ci95_lo": 100 * lo95, "ci95_hi": 100 * hi95,
            "ci90_lo": 100 * lo90, "ci90_hi": 100 * hi90, "p": f.pvalues["AA"],
            "resid_sd": sd, "power_2pts": power, "n_AA": n1, "n_EA": n0, "covariates": c}

def verdict(r):                                 # PREREGISTRATION Part 5, in its order
    detect = r["p_bh"] < 0.05 and (r["ci95_hi"] < 0 or r["ci95_lo"] > 0)
    if detect and r["gap_points"] <= -100 * M: return "meaningful gap"
    if detect and abs(r["gap_points"]) < 100 * M: return "detectable but negligible"
    if r["ci90_lo"] > -100 * M and r["ci90_hi"] < 100 * M: return "equivalent (no meaningful gap)"
    return "inconclusive"

rows = []
for y in ["macro_f1", "bal_acc", "fine_macro_f1"]:
    for d in ["randolph", "prostate"]:
        for s in sorted(D.seed.unique()):
            for m in METHODS:
                rows.append({"metric": y, "dataset": d, "seed": s, "method": m,
                             **h1(D[(D.dataset == d) & (D.method == m) & (D.seed == s)], y)})
R = pd.DataFrame(rows)
R["p_bh"] = R.groupby(["metric", "dataset", "seed"]).p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
R["verdict"] = R.apply(verdict, axis=1)
R.to_csv("results/metrics/h1_gaps.csv", index=False)

pd.set_option("display.width", 170)
show = ["method", "gap_points", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "p_bh", "power_2pts", "verdict"]
for d in ["randolph", "prostate"]:
    sub = R[(R.metric == "macro_f1") & (R.dataset == d) & (R.seed == 2026)]
    print(f"\n=== H1 PRIMARY: {d}, coarse macro-F1, seed 2026 | covariates: {sub.covariates.iloc[0]} "
          f"| n AA {sub.n_AA.iloc[0]}, EA {sub.n_EA.iloc[0]} ===")
    print(sub[show].round(3).to_string(index=False))

def h1b(df, label):
    df = df[df.method.isin(H1B)].copy(); df["method"] = pd.Categorical(df.method, categories=H1B)
    c = "AA * C(method) + " + covs_for(df).replace("AA + ", "")
    if df.dataset.nunique() > 1: c += " + C(dataset)"
    f = smf.mixedlm(f"macro_f1 ~ {c}", df, groups=df.subject_id).fit(reml=True)
    ci = f.conf_int()
    return pd.DataFrame([{"model": label, "converged": f.converged, "term": t, "est_points": 100 * f.params[t],
                          "ci95_lo": 100 * ci.loc[t, 0], "ci95_hi": 100 * ci.loc[t, 1], "p": f.pvalues[t]}
                         for t in f.params.index if t.startswith("AA")]), c
P = D[D.seed == 2026]
pooled, c1 = h1b(P, "CONFIRMATORY: pooled"); rand, c2 = h1b(P[P.dataset == "randolph"], "supportive: Randolph")
H1b = pd.concat([pooled, rand]); H1b.to_csv("results/metrics/h1b_interactions.csv", index=False)
print(f"\n=== H1b (PCA = reference; AA:method < 0 means a LARGER gap than PCA, F1 points) ===")
print(f"pooled formula: macro_f1 ~ {c1}\nRandolph formula: macro_f1 ~ {c2}")
print(H1b.round(3).to_string(index=False))

T = pd.read_csv("results/metrics/donor_celltype_f1.csv").merge(
    P[["dataset", "method", "subject_id", "AA", "log_cells", "age", "disease"]], on=["dataset", "method", "subject_id"])
h3 = []
for (d, m, t), g in T[T.method.isin(H1B)].groupby(["dataset", "method", "cell_type"]):
    if g.AA.nunique() == 2 and len(g) >= 10:
        h3.append({"dataset": d, "method": m, "cell_type": t, "n_donors": len(g), **h1(g.rename(columns={"f1": "y"}), "y")})
H3 = pd.DataFrame(h3)
H3["p_bh"] = H3.groupby(["dataset", "method"]).p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
H3.to_csv("results/metrics/h3_celltype_gaps.csv", index=False)
for d in ["randolph", "prostate"]:
    print(f"\n=== H3: {d} per-cell-type gap, F1 points (BH p in brackets), seed 2026 ===")
    h = H3[H3.dataset == d].copy(); h["cell"] = h.gap_points.round(2).astype(str) + " (" + h.p_bh.round(3).astype(str) + ")"
    print(h.pivot(index="cell_type", columns="method", values="cell").to_string())

print("\n=== Sensitivity: Randolph primary gap by CV seed ===")
print(R[(R.metric == "macro_f1") & (R.dataset == "randolph")].pivot(index="method", columns="seed", values="gap_points").round(2))
print("\n=== Secondary metrics, seed 2026 (verdicts) ===")
print(R[(R.metric != "macro_f1") & (R.seed == 2026)].pivot_table(index=["dataset", "method"], columns="metric",
      values="verdict", aggfunc="first").to_string())
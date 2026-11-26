"""Manuscript figures 1-5 (static, print). Run from ~/fm-ancestry-bias in the fm-core environment:
    python scripts/95_figures.py
Writes figures/fig1_pipeline ... fig5_h2 as .pdf (vector, for the journal) and .png (300 dpi, for review).

Colour encodes ONE thing across the whole paper: ancestry (AA orange, EA blue; validated pair,
CVD dE 26.6). Method type is encoded by marker shape/fill (foundation model = filled circle,
baseline = hollow square), so nothing relies on colour alone."""
import glob, os, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import statsmodels.formula.api as smf
from sklearn.metrics import f1_score

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "pdf.fonttype": 42, "savefig.dpi": 300})
AA_C, EA_C = "#f47230", "#257de0"
INK, MID, LIGHT, BAND = "#222222", "#6f6f6f", "#c4c4c4", "#efefef"
NAME = {"pca": "PCA", "scvi": "scVI", "celltypist": "CellTypist", "geneformer": "Geneformer", "uce": "UCE",
        "scgpt_human": "scGPT", "scfoundation": "scFoundation", "scgpt_cp": "scGPT-CP"}
ORDER = ["pca", "scvi", "celltypist", "geneformer", "uce", "scgpt_human", "scfoundation"]
FM = {"geneformer", "uce", "scgpt_human", "scfoundation", "scgpt_cp"}
REGIME = {"ea_only": "EA-only", "balanced": "Balanced", "random": "Random"}


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name + ".pdf/.png")


def fmt(v):
    return "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "\u2212")


def letter(ax, s):
    ax.text(-0.06, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


def ci_mark(ax, x, y, lo, hi, fm, ms=4.8):
    """Point + 95% CI. Foundation model: filled black circle. Baseline: hollow grey square."""
    c = INK if fm else MID
    ax.plot([lo, hi], [y, y], color=c, lw=1.1, solid_capstyle="round", zorder=3)
    ax.plot(x, y, "o" if fm else "s", ms=ms, mfc=c if fm else "white", mec=c, mew=1.1, zorder=4)


def method_legend(ax, loc="lower left"):
    h = [Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4.8, label="Foundation model"),
         Line2D([], [], marker="s", ls="", mfc="white", mec=MID, mew=1.1, ms=4.8, label="Baseline")]
    ax.legend(handles=h, loc=loc, handletextpad=0.3, borderaxespad=0.2)


def ols_gap(df, y, x="AA"):
    f = smf.ols(f"{y} ~ {x} + age + log_cells", df).fit()
    lo, hi = f.conf_int().loc[x]
    return 100 * f.params[x], 100 * lo, 100 * hi, f.pvalues[x]


# ---------------------------------------------------------------- Figure 1: study pipeline
fig, ax = plt.subplots(figsize=(7.2, 2.5))
ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(0.05, 1.95)
steps = [(0.5, 1.5, "1. Data", "Randolph 90 + prostate 57\ndonors; AA and EA"),
         (1.5, 1.5, "2. QC and labels", "Reference: B, CD4 T, CD8 T,\nDC, monocyte, NK"),
         (2.5, 1.5, "3. Embed every cell", "Geneformer, scGPT, UCE,\nscFoundation, PCA, scVI"),
         (2.5, 0.5, "4. kNN (k = 15)", "Trained on EA donors only;\n5-fold donor cross-validation"),
         (1.5, 0.5, "5. Score each donor", "Macro-F1 over cell types\nwith at least 10 cells"),
         (0.5, 0.5, "6. Statistics", "H1 gap, H1b vs PCA, H3, H4;\nH2 fine-tuning")]
for x, y, t, s in steps:
    ax.add_patch(FancyBboxPatch((x - 0.44, y - 0.31), 0.88, 0.62, boxstyle="round,pad=0.0,rounding_size=0.05",
                                fc="white", ec=INK if t.startswith("3") else LIGHT, lw=1.1 if t.startswith("3") else 0.8))
    ax.text(x, y + 0.14, t, ha="center", va="center", fontweight="bold", fontsize=8)
    ax.text(x, y - 0.1, s, ha="center", va="center", fontsize=6.5, color="#333333", linespacing=1.3)
for a, b in [((0.95, 1.5), (1.05, 1.5)), ((1.95, 1.5), (2.05, 1.5)), ((2.5, 1.18), (2.5, 0.82)),
             ((2.05, 0.5), (1.95, 0.5)), ((1.05, 0.5), (0.95, 0.5))]:
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MID, lw=0.9, shrinkA=0, shrinkB=0))
ax.annotate("", xy=(1.5, 0.82), xytext=(1.5, 1.18),
            arrowprops=dict(arrowstyle="-|>", color=MID, lw=0.9, ls=(0, (3, 2)), shrinkA=0, shrinkB=0))
ax.text(1.54, 1.0, "CellTypist: pretrained classifier,\nlabels cells directly", fontsize=6.6, color=MID, va="center")
save(fig, "fig1_pipeline")

# ---------------------------------------------------------------- Figure 2: H1 gap per method
R = pd.read_csv("results/metrics/h1_gaps.csv")
R = R[(R.metric == "macro_f1") & (R.seed == 2026) & (R.method.isin(ORDER))]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
ys = {m: i for i, m in enumerate(ORDER[::-1])}
for ax, (d, title), L in zip(axes, [("randolph", "Randolph (n = 90; primary)"), ("prostate", "Prostate (n = 57)")], "ab"):
    ax.axvspan(-2, 2, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=LIGHT, lw=0.8, zorder=1)
    for m in ORDER:
        r = R[(R.dataset == d) & (R.method == m)].iloc[0]
        ci_mark(ax, r.gap_points, ys[m], r.ci95_lo, r.ci95_hi, m in FM)
        star = " *" if r.verdict == "meaningful gap" else ""
        ax.text(3.0, ys[m], f"{fmt(r.gap_points)}{star}", va="center", fontsize=7, color=INK)
    ax.set_yticks(list(ys.values())); ax.set_yticklabels([NAME[m] for m in ys])
    ax.set_xlim(-6.5, 4.4); ax.set_xticks([-6, -4, -2, 0, 2]); ax.set_ylim(-0.7, len(ORDER) - 0.1)
    ax.set_xlabel("Macro-F1 gap, AA − EA (points)")
    ax.set_title(title, loc="left", fontweight="bold"); letter(ax, L)
axes[0].text(0, len(ORDER) - 0.45, "±2-point margin", ha="center", fontsize=6.5, color=MID)
axes[1].legend(handles=[Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4.8, label="Foundation model"),
                        Line2D([], [], marker="s", ls="", mfc="white", mec=MID, mew=1.1, ms=4.8, label="Baseline")],
               loc="lower right", bbox_to_anchor=(1.0, 1.06), ncol=2, handletextpad=0.3, columnspacing=1.0)
fig.text(0.125, -0.1, "Points: age- and cell-count-adjusted gap with 95% CI. Shaded: ±2-point equivalence margin.\n"
         "* meaningful gap (preregistered rule: ≤ −2 points and CI excluding 0, BH-adjusted).", fontsize=6.5, color=MID)
save(fig, "fig2_h1_gaps")

# ---------------------------------------------------------------- Figure 3: H1b and label dependence
B = pd.read_csv("results/metrics/h1b_interactions.csv")
B = B[B.term.str.startswith("AA:")].copy()
B["method"] = B.term.str.extract(r"T\.(\w+)\]")[0]
mk = pd.read_csv("results/metrics/posthoc_marker_relabel_donor_f1.csv")
mg = pd.read_csv("results/metrics/posthoc_merged_cytotoxic_randolph.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.45})
# a: H1b extra gap vs PCA
ax = axes[0]; meths = [m for m in ORDER if m != "pca"]; yy = {m: i for i, m in enumerate(meths[::-1])}
ax.axvline(0, color=LIGHT, lw=0.8)
for model, off, filled in [("CONFIRMATORY: pooled", 0.16, True), ("supportive: Randolph", -0.16, False)]:
    for m in meths:
        r = B[(B.model == model) & (B.method == m)].iloc[0]
        c = INK if m in FM else MID
        ax.plot([r.ci95_lo, r.ci95_hi], [yy[m] + off] * 2, color=c, lw=1.0)
        ax.plot(r.est_points, yy[m] + off, "o", ms=4.3, mfc=c if filled else "white", mec=c, mew=1.0)
ax.set_yticks(list(yy.values())); ax.set_yticklabels([NAME[m] for m in yy])
ax.set_xlabel("Extra AA gap relative to PCA (points)")
ax.set_title("H1b: gap beyond PCA", loc="left", fontweight="bold"); letter(ax, "a")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4.3, label="Pooled (confirmatory)"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=4.3, label="Randolph (supportive)")],
          loc="upper center", bbox_to_anchor=(0.45, -0.2), handletextpad=0.3)
# b: Randolph H1 gap under three labelings (post hoc)
ax = axes[1]; m6 = ["pca", "scvi", "geneformer", "uce", "scgpt_human", "scfoundation"]; y6 = {m: i for i, m in enumerate(m6[::-1])}
ax.axvspan(-2, 2, color=BAND, lw=0, zorder=0); ax.axvline(0, color=LIGHT, lw=0.8, zorder=1)
schemes = [("Author labels", mk, "label_coarse", "o", INK, INK, 0.22),
           ("NK/CD8 relabelled by markers", mk, "label_marker", "o", "white", INK, 0.0),
           ("NK and CD8 T merged", mg, "f1", "D", MID, MID, -0.22)]
for lab, df, col, mkr, fc, ec, off in schemes:
    for m in m6:
        g, lo, hi, _ = ols_gap(df[df.method == m], col)
        ax.plot([lo, hi], [y6[m] + off] * 2, color=ec, lw=1.0, zorder=3)
        ax.plot(g, y6[m] + off, mkr, ms=4.0, mfc=fc, mec=ec, mew=1.0, zorder=4)
ax.set_yticks(list(y6.values())); ax.set_yticklabels([NAME[m] for m in y6]); ax.set_xlim(-3, 2.2)
ax.set_xlabel("Macro-F1 gap, AA − EA (points)")
ax.set_title("Randolph gap by reference labelling (post hoc)", loc="left", fontweight="bold"); letter(ax, "b")
ax.legend(handles=[Line2D([], [], marker=s[3], ls="", mfc=s[4], mec=s[5], ms=4.0, label=s[0]) for s in schemes],
          loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, handletextpad=0.3)
save(fig, "fig3_h1b_labels")

# ---------------------------------------------------------------- Figure 4: the NK/CD8 boundary
H3 = pd.read_csv("results/metrics/h3_celltype_gaps.csv")
H3 = H3[(H3.dataset == "randolph") & (H3.method.isin(ORDER))]
types = [t for t in ["B", "CD4_T", "CD8_T", "Monocyte", "DC", "NK"] if t in set(H3.cell_type)]
M = H3.pivot(index="cell_type", columns="method", values="gap_points").reindex(index=types, columns=ORDER)
obs = pd.read_csv("data/processed/randolph_obs.csv.gz", index_col=0)
conf = []
for m in ["pca", "uce", "celltypist"]:
    p = pd.read_csv(f"results/preds/randolph_primary_{m}_s2026.csv.gz").set_index("cell_id").join(obs[["ancestry", "label_coarse"]])
    for a, g in p[p.pred_coarse == "NK"].groupby("ancestry"):
        conf.append({"method": m, "ancestry": a, "pct": 100 * (g.label_coarse == "CD8_T").mean()})
conf = pd.DataFrame(conf)
cd8 = pd.read_csv("results/h3_nk/cd8_nklike_by_donor.csv")
fig = plt.figure(figsize=(7.2, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.15, 0.7], wspace=0.75)
# a: heatmap of per-cell-type gaps
ax = fig.add_subplot(gs[0])
cmap = LinearSegmentedColormap.from_list("div", ["#b54d12", AA_C, "#f6f6f6", EA_C, "#1757ac"])
im = ax.imshow(M.values, cmap=cmap, norm=TwoSlopeNorm(0, -12, 12), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 6 else INK)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([NAME[m] for m in ORDER], rotation=45, ha="right")
ax.set_yticks(range(len(types))); ax.set_yticklabels([t.replace("_", " ") for t in types])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.outline.set_visible(False)
cb.set_label("AA − EA (points)", fontsize=6.5); cb.ax.tick_params(labelsize=6.5)
ax.set_title("Per-cell-type gap (Randolph)", loc="left", fontweight="bold"); letter(ax, "a")
# b: share of NK calls that the reference labels CD8 T
ax = fig.add_subplot(gs[1]); xs = np.arange(3); w = 0.36
for k, (a, c) in enumerate([("AA", AA_C), ("EA", EA_C)]):
    v = [conf[(conf.method == m) & (conf.ancestry == a)].pct.iloc[0] for m in ["pca", "uce", "celltypist"]]
    bars = ax.bar(xs + (k - 0.5) * (w + 0.03), v, w, color=c, label=a, edgecolor="white", linewidth=0.6)
    for b_, val in zip(bars, v):
        ax.text(b_.get_x() + b_.get_width() / 2, val + 0.4, f"{val:.1f}", ha="center", fontsize=6.3, color=INK)
ax.set_xticks(xs); ax.set_xticklabels(["PCA", "UCE", "CellTypist"])
ax.set_ylabel("Called NK but labelled\nCD8 T by reference (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, handlelength=1)
ax.set_title("NK ↔ CD8 T confusion", loc="left", fontweight="bold"); letter(ax, "b")
# c: NK-like CD8 T cells per donor
ax = fig.add_subplot(gs[2]); rng = np.random.default_rng(2026)
for k, (a, c) in enumerate([("AA", AA_C), ("EA", EA_C)]):
    v = 100 * cd8[cd8.ancestry == a].frac_nklike.values
    ax.scatter(k + rng.uniform(-0.16, 0.16, len(v)), v, s=9, color=c, alpha=0.8, linewidths=0)
    ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color=INK, lw=1.3)
f = smf.ols("frac_nklike ~ AA + age", cd8).fit()
ax.set_xticks([0, 1]); ax.set_xticklabels(["AA", "EA"]); ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("NK-like cells among CD8 T (%)")
ax.text(0.5, 1.0, f"p = {f.pvalues['AA']:.4f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=MID)
ax.set_title("Per donor", loc="left", fontweight="bold", pad=12); letter(ax, "c")
save(fig, "fig4_nk_boundary")

# ---------------------------------------------------------------- Figure 5: H2 fine-tuning
D2 = pd.read_csv("results/metrics/h2_donor_f1.csv")
anc = obs.drop_duplicates("subject_id").set_index("subject_id").ancestry.astype(str)
nk = []
for fpath in sorted(glob.glob("results/preds_h2/*_lr1e-05.csv.gz")):
    mt = re.match(r".*/(geneformer|scgpt)_(ea_only|balanced|random)_s(\d+)_lr1e-05\.csv\.gz$", fpath)
    if not mt: continue
    p = pd.read_csv(fpath)
    for d, g in p.groupby("subject_id"):
        if (g.label == "NK").sum() >= 10:
            nk.append({"model": mt.group(1), "regime": mt.group(2), "seed": int(mt.group(3)), "subject_id": str(d),
                       "f1": f1_score(g.label, g.pred_coarse, labels=["NK"], average="macro", zero_division=0)})
NK = pd.DataFrame(nk); NK["ancestry"] = NK.subject_id.map(anc)
H2m = pd.read_csv("results/metrics/h2_model.csv")


def seed_gaps(df):
    t = df.groupby(["model", "regime", "seed", "ancestry"]).f1.mean().unstack()
    return (100 * (t.AA - t.EA)).rename("gap").reset_index()


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.35})
regs = ["ea_only", "balanced", "random"]
for ax, (df, title, L) in zip(axes, [(D2, "Overall macro-F1 gap", "a"), (NK, "NK-cell F1 gap", "b")]):
    G = seed_gaps(df); ax.axhline(0, color=LIGHT, lw=0.8)
    if L == "a": ax.axhline(-2, color=LIGHT, lw=0.8, ls=(0, (3, 2)))
    for model, mkr, fc, off in [("geneformer", "o", INK, -0.1), ("scgpt", "^", "white", 0.1)]:
        means = []
        for i, r in enumerate(regs):
            v = G[(G.model == model) & (G.regime == r)].gap.values
            ax.scatter(np.full(len(v), i + off), v, s=7, color=LIGHT, zorder=2, linewidths=0)
            means.append(v.mean())
        ax.plot(np.arange(3) + off, means, color=INK, lw=0.9, zorder=3)
        ax.plot(np.arange(3) + off, means, mkr, ms=5, mfc=fc, mec=INK, mew=1.0, ls="", zorder=4,
                label={"geneformer": "Geneformer", "scgpt": "scGPT"}[model])
    ax.set_xticks(range(3)); ax.set_xticklabels([REGIME[r] for r in regs]); ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel("Fine-tuning donors (n = 20 each)"); ax.set_ylabel("AA − EA (points)")
    ax.set_title(title, loc="left", fontweight="bold"); letter(ax, L)
axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=2, handletextpad=0.3)
txt = []
for model in ["scgpt", "geneformer"]:
    r = H2m[(H2m.model == model) & (H2m.term.str.contains("balanced"))].iloc[0]
    txt.append(f"{ {'scgpt': 'scGPT', 'geneformer': 'Geneformer'}[model] }: balanced vs EA-only, gap change {r.est_points:+.2f} (p = {r.p:.3f})")
fig.text(0.125, -0.17, "Grey dots: individual seeds; markers: mean of 3 seeds; dashed line: −2-point threshold.\n"
         + "; ".join(txt) + ".", fontsize=6.5, color=MID)
save(fig, "fig5_h2_finetuning")
print("done")

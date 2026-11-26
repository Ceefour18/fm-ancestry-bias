# Deviations and clarifications log

## 2026-11-25 — scVI batch key (clarification, no results seen)
The preregistration does not specify a batch covariate for scVI. scVI is trained without a batch key
(batch_key=None) on each dataset, matching the foundation models and PCA, which receive no batch correction.
Rationale: donor/batch correction would artificially mix ancestries and bias H4 (ancestry separation vs scVI)
and H1b. Settings: 2,000 HVGs (seurat_v3, raw counts), n_latent=30, n_layers=2, early stopping, seed 2026.

## 2026-11-25 — UCE model files (clarification)
Figshare blocks server downloads (HTTP 202/403). UCE helper files were obtained from the Hugging Face mirror
minwoosun/uce-misc and verified byte-identical to the official figshare release by MD5
(results/UCE_FILE_PROVENANCE.txt). The 33-layer checkpoint is taken from the official figshare file 43423236
(MD5 eed62d4fe51873c87bcf9660faa07707), downloaded via browser and uploaded.
scVI run detail: max_epochs set by scvi-tools 1.3.3 default heuristic (Randolph 34, prostate 28);
early stopping did not trigger; training ran to max_epochs. Embedding dims 30; no NaNs.

## 2026-11-25 — UCE layer decision (pre-specified rule applied)
33-layer timing test on 10,000 Randolph cells: 1,337 s on one 48 GB GPU, batch 12 (README value of 25 is for 80 GB GPUs).
Projected 19.6 GPU-hours for both datasets (<=48) -> 33-layer model used, as preregistered.

## 2026-11-25 — UCE reproducibility check (pre-specified)
5,000 Randolph cells, seeds 2026 vs 2027: kNN (k=15, 5-fold) coarse-label agreement 0.971 (>=0.95, PASS).
Embeddings are seed-dependent (median per-cell cosine 0.887, min 0.717) due to UCE gene sampling;
all analyses use seed 2026.

## 2026-11-25 — scGPT runtime configuration (clarification)
scGPT 0.2.4 run with use_fast_transformer=False (flash-attn 1.x not buildable on RTX 6000 Ada);
pretrained flash-attention weights are remapped by scGPT on load. Weight-load sanity check on 2,000 Randolph
cells: kNN (k=15, 5-fold) coarse accuracy 0.929 vs 0.575 for random embeddings. max_length 1200, seed 2026.
Checkpoints: whole-human (primary) and continual-pretrained (sensitivity), official Google Drive links in the
scGPT README; SHA256 in results/SHA256SUMS_scgpt.txt.

## 2026-11-25 — scFoundation configuration (clarification)
Official checkpoint models.ckpt (SharePoint, modified 2025-02-04; SHA256 in results/SCFOUNDATION_FILE_PROVENANCE.txt),
repo commit 397631c. get_embedding.py with README defaults for cell embeddings: singlecell, output cell, pool all
(3,072 dims), tgthighres t4, pre_normalized F (raw counts), version ce; internal seed 0. Input mapped by gene
symbol to the 19,264-gene list (Randolph test: 18,643 genes matched, 89.0% of UMIs retained; unmatched genes
zero-filled). Weight-load check (2,000 Randolph cells): kNN accuracy 0.960 vs 0.577 random. torch 2.4.1+cu121.

## 2026-11-26 — UCE full runs complete
33-layer, seed 2026, batch 12. Randolph (234,727 x 1,280) 31,402 s; prostate (286,890 x 1,280) 38,831 s.
Total 19.5 GPU-hours (timing-test projection 19.6). No NaNs; cell order verified against input.
logs/scf_314.out:genes matched: 18643 of 19264 scFoundation genes (96.8%); UMI kept: 89.0%
logs/scf_315.out:genes matched: 18791 of 19264 scFoundation genes (97.5%); UMI kept: 94.3%
Prostate disease group: 15 of 57 donors have "lethal prostate_cancer_group: NA" in GEO GSE270387
(data/raw/GSE270387/donors.tsv); the study describes its cohort as prostate cancer patients and healthy
controls. NA is coded as "Control" (3 groups: Control, Nonlethal, Lethal) so all 57 donors enter the
models as preregistered. The preregistered patients-only sensitivity analysis removes this assumption.
Prostate disease group: 15 of 57 donors have "lethal prostate_cancer_group: NA" in GEO GSE270387
(data/raw/GSE270387/donors.tsv); the study describes its cohort as prostate cancer patients and healthy
controls. NA is coded as "Control" (3 groups: Control, Nonlethal, Lethal) so all 57 donors enter the
models as preregistered. The preregistered patients-only sensitivity analysis removes this assumption.
Pooled H1b: Randolph donors (healthy volunteers) are coded disease group "Control", sharing that level with
prostate controls; coding them as a separate level duplicated the dataset term (singular model).
Age is available for Randolph only, so pooled models adjust for log(cells), dataset and disease group.

## 2026-11-26 — H4 implementation detail
iLISI (harmonypy compute_lisi, perplexity 30) and ancestry silhouette (Euclidean) computed per cell type on
up to 5,000 cells per cell type (seed 2026), identical cells for every method; Randolph NI cells as in H1.
Paired two-sided Wilcoxon vs scVI across 12 dataset x cell-type units.

## 2026-11-26 — POST HOC exploratory analyses (not preregistered), prompted by H3
1. NK confusion matrices (UCE, PCA): NK<->CD8_T confusion 2-3x higher in AA donors.
2. Marker check (script 82): CD8_T-labelled cells predicted NK are transcriptionally NK-like
   (TRAC detected 12.4% AA / 20.7% EA vs 64-70% in correctly called CD8 T, 8.5% in NK); 434 AA vs 179 EA cells.
   Not MAIT (TRAV1-2 0%); TRDC at NK-cell background. NK-like share of CD8_T per donor: AA 13.1% vs EA 6.8%
   (+7.6 pts, p = 0.0006).
3. Merged NK+CD8_T into one class and re-scored seed-2026 predictions (Randolph): H1 gaps -0.14 to +0.22
   (all p > 0.4); H1b extra gap vs PCA shrinks UCE -0.93 -> -0.23, Geneformer -0.76 -> -0.14,
   scFoundation -0.60 -> -0.15; scVI -0.59 -> -0.26 (p = 0.029).
Interpretation (to be tested): reference-label ambiguity at the NK/CD8 boundary, greater in AA donors,
accounts for most of the foundation-model-specific deficit. Preregistered results are reported unchanged.
4. POST HOC marker re-annotation (script 83; rule fixed before scoring): author NK/CD8_T cells relabelled by
   mean(TRAC,CD3D,CD3E,CD3G,CD8B) vs mean(KLRF1,NCAM1,SH2D1B); ties (2,757 cells) keep author label.
   Author-CD8_T -> marker-NK: AA 5.5% vs EA 2.2%; author-NK -> marker-CD8_T: AA 10.6% vs EA 11.1%.
   H1 gaps with marker labels: PCA -0.73, scVI -1.33, Geneformer -1.09, UCE -1.24, scGPT -1.11,
   scFoundation -1.07, scGPT-CP -0.59 (all BH p > 0.1). H1b extra gap vs PCA with marker labels:
   UCE -0.41 (p 0.17), Geneformer -0.18, scFoundation -0.21, scVI -0.40, scGPT -0.26 (all p > 0.1).
   Interpretation: the AA deficit (~1 pt) is method-general and persists with marker labels; the
   foundation-model excess over PCA is label-dependent (PCA's gap grows from -0.21 to -0.73).
Reference-label provenance (Randolph et al. 2021, supplementary methods; Randolph PhD dissertation ch. III):
Seurat v3.1.5, SCTransform per condition (batch, % mito regressed), anchor-based integration, PCA (30 PCs),
SNN graph on PCs 1-20, FindClusters resolution 0.5; clusters labelled by canonical markers
(CD8 T: CD3D+ CD8A+; NK: CD3D- NKG7+ GNLY+). Labels are cluster-level and PCA-derived, consistent with the
PCA baseline's advantage under author labels and its loss under marker-based labels.

## 2026-11-26 — Sensitivity analyses: scope (decided before running them)
Run: influenza cells; composition-matched test sets; continuous YRI; fine labels; prostate patients-only and
controls-only; CV seeds 2027-2030. Not run at this stage: "all 59 prostate donors" and "5-MAD QC", because
both change the cell set and require re-embedding with all foundation models (~25+ GPU-hours); deferred.
Sensitivity results (script 84): no analysis yields a "meaningful gap" for any method (flu cells, continuous
YRI, fine labels, composition-matched, prostate patients-only/controls-only). Composition matching uses each
donor's present cell types (renormalised mean composition), so donors lacking a rare type (e.g. DC) are kept;
with this, composition-matched Randolph gaps are -0.25 to -0.92 and all methods are equivalent.

## 2026-11-26 — Correction: BH family and CellTypist
Earlier runs of scripts 80/84 applied BH across PCA, scVI, 4 foundation models and scGPT-continual-pretrained.
The preregistration's 7 methods are the 4 foundation models + PCA, scVI and CellTypist; scGPT-CP is a
sensitivity analysis. Corrected: BH over the preregistered 7; scGPT-CP reported separately. CellTypist
(Immune_All_Low, pretrained) now scored on all donors (script 41); coarse labels only (the model merges naive
and central-memory T cells, so fine-label F1 is not computed for CellTypist).
CellTypist label-mapping correction: the coarse mapping (inherited from the blinded pilot, script 40) sent
"Type 1/Type 17/Follicular helper T cells" and "Treg(diff)" to Other instead of CD4_T, and "Kupffer cells"/
"Mono-mac" to Other instead of Monocyte. Corrected in script 41; "Other" share was equal by ancestry before
the fix (AA 5.05%, EA 4.89%). CellTypist excluded from the fine-label sensitivity row (no fine labels).

## 2026-11-26 — H2 design (fixed before training)
Fixed split 2026: 40 test donors (never used for training/tuning); 5 EA validation donors removed from the
50-donor training pool (seed 2026) for learning-rate selection only. Regimes of equal size N (= remaining EA
donors): EA-only (all), balanced (N/2 EA + N/2 AA), random (N from pool); seeds 2026-2028 vary donor draws
and initialisation. Mock (NI) cells. Geneformer: BertForSequenceClassification on the V2-104M checkpoint,
3 epochs, batch 8, linear schedule with 10% warmup, bf16; learning rate chosen from {1e-5, 5e-5, 1e-4} by
pooled macro-F1 on the EA validation donors (EA-only regime, seed 2026).
scGPT H2: whole-human checkpoint + 6-class CLS head (159/169 tensors pretrained), 51 expression bins, up to
1,200 genes per cell, 5 epochs, batch 32, Adam (eps 1e-4), StepLR gamma 0.9, AMP, no flash-attention;
learning rate chosen from {1e-5, 5e-5, 1e-4} by pooled macro-F1 on the EA validation donors.
Note: the training pool has 25 AA / 20 EA donors, so the "random" regime is near-balanced by chance
(10-12 AA of 20); the informative contrast is EA-only vs balanced.
H2 learning rates (EA validation donors, EA-only regime, seed 2026; pooled macro-F1):
Geneformer 1e-5 0.971 | 5e-5 0.959 | 1e-4 0.960 -> 1e-5. scGPT 1e-5 0.961 | 5e-5 0.935 | 1e-4 0.956 -> 1e-5.
Best value is at the lower edge of the grid; not extended (differences ~1 point).
H2 Geneformer runs: 7 of 9 failed on neuron-gpu02 (RTX 6000 Ada; driver CUDA 12.8 cannot run the CUDA 13
torch build in fm-geneformer) while the A6000 node was down. All 9 final Geneformer runs were rerun on
neuron-gpu02 in fm-scf (same packages, torch 2.4.1+cu121) for consistency; the 2 earlier successful runs are
kept in results/preds_h2/old_torch215/ and not analysed. Learning-rate selection had used torch 2.15 on A6000.
fm-scf: accelerate pinned to 1.0.1 (newer release incompatible with torch 2.4.1) for H2 Geneformer reruns.

## 2026-11-26 — H2 results (preregistered)
f1 ~ AA * regime + (1 | donor), EA-only reference, 40 fixed test donors, 3 seeds per regime.
scGPT: AA gap under EA-only -2.25 [-3.45, -1.05]; balanced +0.45 [0.11, 0.78] p=0.009; random +0.72 p<0.001.
Geneformer (rerun in fm-scf): AA gap -1.54 [-2.56, -0.51]; balanced +0.17 p=0.20; random +0.14 p=0.29.
NK-cell AA gap persists in all regimes (-6.4 to -9.3).

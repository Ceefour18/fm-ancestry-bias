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

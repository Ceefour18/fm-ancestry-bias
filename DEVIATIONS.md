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

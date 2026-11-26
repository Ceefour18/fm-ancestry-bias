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

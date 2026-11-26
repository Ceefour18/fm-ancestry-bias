suppressMessages(library(DESeq2))
markers <- c("NKG7","GNLY","KLRD1","FCGR3A","KLRF1","KLRC2","B3GAT1","GZMB","PRF1","KIR2DL1","KIR3DL1","CD3E","CD8A","ZBTB16","NCAM1")
for (t in c("NK", "CD8_T")) {
  cts  <- read.csv(gzfile(sprintf("results/h3_nk/pb_%s_counts.csv.gz", t)), row.names = 1, check.names = FALSE)
  meta <- read.csv(sprintf("results/h3_nk/pb_%s_meta.csv", t), row.names = 1)[colnames(cts), ]
  meta$ancestry <- factor(meta$ancestry, levels = c("EA", "AA")); meta$age_z <- as.numeric(scale(meta$age))
  dds <- DESeq(DESeqDataSetFromMatrix(as.matrix(cts), meta, ~ age_z + ancestry), quiet = TRUE)
  res <- results(dds, name = "ancestry_AA_vs_EA"); res <- res[order(res$padj), ]
  write.csv(as.data.frame(res), sprintf("results/h3_nk/deseq2_%s_AA_vs_EA.csv", t))
  cat(sprintf("\n==== %s: %d donors, genes with padj < 0.05: %d ====\n", t, ncol(cts), sum(res$padj < 0.05, na.rm = TRUE)))
  print(round(head(as.data.frame(res)[, c("log2FoldChange", "padj")], 15), 4))
  cat("Marker panel:\n"); mk <- intersect(markers, rownames(res))
  print(round(as.data.frame(res[mk, c("log2FoldChange", "padj")]), 4))
}

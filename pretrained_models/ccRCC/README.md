# ccRCC pre-trained EcoNet model

Clear-cell renal cell carcinoma model (ccRCC-specific EcoTyper, 11 ecotypes
E1-E11). Use with `4.Prediction/config_ccRCC.yaml`.

| File | Description |
|------|-------------|
| `global_graph.pkl` | ccRCC intercellular regulatory network (NetworkX DiGraph) |
| `ecotype_model.pth` | GAT (fine-tuned on ccRCC): expression to ecotype abundance (2,514 genes) |
| `response_model.pth` | ResponsePredictor (fine-tuned): ecotype features to R/NR. Arch `[32, 16]` |
| `gene_selected.txt` | The 2,514 genes the model expects |
| `tcga_reference.tsv.gz` | TCGA-KIRC TPM (614 samples) trimmed to the model genes, for KNN imputation of genes missing from your data |

Ecotypes: **E1-E11**. Response classes: 0 = non-responder (SD/PD), 1 = responder (CR/PR).

To predict on your own bulk RNA-seq, see `../../4.Prediction/README.md`.

# Pan-cancer pre-trained EcoNet model

14-cancer consensus model (general Carcinoma EcoTyper, 10 ecotypes CE1-CE10).
Built from a minfreq5 pan-cancer consensus network. Use with
`4.Prediction/config_pancancer.yaml`.

| File | Description |
|------|-------------|
| `pan_cancer_network.pkl` | 14-cancer consensus regulatory network (NetworkX DiGraph) |
| `ecotype_model.pth` | Pretrained GAT: expression to ecotype abundance (2,198 genes) |
| `response_model.pth` | ResponsePredictor: ecotype features to R/NR. Arch `[32, 8]`, dropout 0.6 (portable-generalization tuning) |
| `gene_selected.txt` | The 2,198 genes the model expects |
| `tcga_reference.tsv.gz` | Pan-cancer TCGA TPM trimmed to the model genes, for KNN imputation of genes missing from your data |

Ecotypes: **CE1-CE10** (output columns are labeled `E1`-`E10` by the pipeline).
Response classes: 0 = non-responder (SD/PD), 1 = responder (CR/PR).

The response predictor uses the portable single-MLP configuration selected for
cross-cohort generalization (mean leave-one-cohort-out AUC ~ 0.72).

To predict on your own bulk RNA-seq, see `../../4.Prediction/README.md`.

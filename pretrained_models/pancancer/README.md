# Pan-cancer pre-trained EcoNet model

Pan-cancer model over the immune-active carcinoma set (PC9-imm), general
Carcinoma EcoTyper, 10 ecotypes CE1-CE10. Use with
`4.Prediction/config_pancancer.yaml`.

| File | Description |
|------|-------------|
| `pan_cancer_network.pkl` | Pan-cancer regulatory network (NetworkX DiGraph) |
| `ecotype_model.pth` | Pretrained GAT: expression to ecotype abundance (1,951 genes) |
| `response_model.pth` | ResponsePredictor: ecotype features to R/NR. Arch `[32, 8]`, dropout 0.6 (do0.6 portable-generalization winner) |
| `gene_selected.txt` | The 1,951 genes the model expects |
| `tcga_reference.tsv.gz` | Per-cancer-Z normalized TCGA TPM trimmed to the model genes, for KNN imputation |

Ecotypes: **CE1-CE10** (output columns are labeled `E1`-`E10` by the pipeline).
Response classes: 0 = non-responder (SD/PD), 1 = responder (CR/PR).

**Normalization**: this model was trained on per-cancer-Z expression, so the
bundled reference is already per-cancer-Z and the config sets
`normalize_reference: false` (the pipeline does not re-normalize it). Your input
is still raw TPM, which the pipeline z-scores automatically.

The response predictor uses the portable single-MLP configuration selected for
cross-cohort generalization (mean leave-one-cohort-out AUC ~ 0.72).

To predict on your own bulk RNA-seq, see `../../4.Prediction/README.md`.

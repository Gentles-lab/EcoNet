# EcoNet: Prediction

Predict immunotherapy response for a new dataset using trained models from Steps 1-3. Optionally evaluate against known clinical outcomes.

## Quick Start

```bash
# 1. Edit config.yaml to point to your data and trained models
# 2. Run
python run_pipeline.py --config config.yaml
```

## Pipeline

```
Trained models (Steps 1-3) + new expression data
       |
  [Preprocessing] Z-score normalization + KNN imputation (TCGA reference)
       |
  [Inference] GAT -> ecotype features -> ResponsePredictor -> R/NR
       |
  ecotype_predictions.txt + response_predictions.csv
       |
  [Evaluation] (optional, if clinical data provided)
       |
  metrics.txt + roc_curve.png
```

## Setup

```bash
conda activate env_econet
```

## Input Data

### Trained Models (from Steps 1-3)

| File | Source | Description |
|------|--------|-------------|
| `global_graph.pkl` | Step 1 | Regulatory network |
| `finetuned_ecotype_model.pth` | Step 3 | Fine-tuned GAT |
| `finetuned_response_model.pth` | Step 3 | Fine-tuned ResponsePredictor |

### TCGA Reference

| File | Description |
|------|-------------|
| Gene expression (TPM, TSV) | Reference for KNN imputation |

### New Dataset

| File | Required | Description |
|------|----------|-------------|
| Gene expression (TPM, TSV) | Yes | Genes as rows, samples as columns |
| Clinical data (TSV) | No | For evaluation — must contain sample ID and response columns |

Expression input is raw TPM. The pipeline applies per-gene z-score normalization automatically.

## Configuration

Edit `config.yaml`:

```yaml
# Trained models
graph_pkl: "path/to/global_graph.pkl"
ecotype_model_pth: "path/to/finetuned_ecotype_model.pth"
response_model_pth: "path/to/finetuned_response_model.pth"

# TCGA reference
tcga_expression_tsv: "path/to/tcga_expression.tsv"

# New dataset
expression_tsv: "path/to/new_expression.tsv"
clinical_tsv: "path/to/new_clinical.tsv"   # null for prediction only

# Clinical format (only if clinical_tsv is provided)
sample_id_column: "RNA_ID"
response_column: "Best.response.on.ICB"
response_mapping:
  CR: 1
  PR: 1
  SD: 0
  PD: 0

# Architecture (must match training)
num_ecotypes: 11
hidden_channels: 8
```

## Outputs

| File | Description |
|------|-------------|
| **`output/ecotype_predictions.txt`** | Predicted ecotype abundances (ecotypes x samples) |
| **`output/response_predictions.csv`** | Predicted response class, label, and probabilities per sample |
| `output/metrics.txt` | Evaluation metrics (only if clinical data provided) |
| `output/roc_curve.png` | ROC curve (only if clinical data provided, binary classification) |


## Quick test

`config_test.yaml` (ccRCC) and `config_pancancer_test.yaml` run prediction on a small test input. Download the test data from [Zenodo](https://doi.org/10.5281/zenodo.22034558) and set the `PATH/TO/test_data` path to your test inputs, then:

```bash
python run_pipeline.py --config config_test.yaml
```

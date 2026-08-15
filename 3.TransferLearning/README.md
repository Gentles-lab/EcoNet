# EcoNet: Transfer Learning & Response Prediction

Trains a response predictor on pan-cancer immunotherapy data using ecotype features from the pretrained GAT (Component 2), then optionally fine-tunes on cancer-specific data. Predicts immunotherapy response (R/NR) from gene expression.

## Quick Start

```bash
# 1. Edit config.yaml to point to your data
# 2. Run
python run_pipeline.py --config config.yaml
```

To run a specific step:

```bash
python run_pipeline.py --config config.yaml --step 1    # pre-train only
python run_pipeline.py --config config.yaml --step 2    # fine-tune only
```

## Pipeline

```
GAT model (Component 2) + pan-cancer expression + R/NR labels
       |
  [Preprocessing] Z-score normalization + KNN imputation (TCGA reference)
       |
  [Step 1] Pre-train ResponsePredictor on pan-cancer immunotherapy data
       |       (or load a provided pre-trained model)
       |
  [Step 2] Fine-tune on cancer-specific data (optional, auto-detected)
       |
  response_model.pth + fine-tuned models + predictions + attention scores
```

Intermediate results are cached -- re-running safely skips completed steps.

## Pre-training Options

Step 1 supports two modes:

| Config | Behavior |
|--------|----------|
| `response_model_pth: "path/to/model.pth"` | Skip training, use provided model directly |
| `response_model_pth: null` | Train from scratch using pan-cancer expression + clinical data |

Pre-trained model weights are provided with this release. Set `response_model_pth` to null only if you want to retrain from scratch with your own data.

## Fine-tuning Modes

The pipeline auto-detects the fine-tuning mode from which config fields are set:

| Config fields provided | Mode | Behavior |
|----------------------|------|----------|
| No fine-tune fields | Skip | Use pre-trained models only |
| `finetune_expression_tsv` + `finetune_clinical_tsv` | Response only | Freeze GAT, fine-tune ResponsePredictor |
| Above + `finetune_abundance_tsv` | Iterative | Fine-tune both GAT and ResponsePredictor |

## Setup

```bash
conda activate env_econet
```

Required packages (pre-installed in env_econet): `torch`, `torch_geometric`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `pyyaml`.

## Input Data

### From Components 1-2

| File | Source | Description |
|------|--------|-------------|
| `global_graph.pkl` | Component 1 | Regulatory network |
| `NN11GraphModel.pth` | Component 2 | Pretrained GAT model |

### TCGA Reference

| File | Description |
|------|-------------|
| Gene expression (TPM, TSV) | Reference for KNN imputation of missing genes |

### Pre-training Data (optional if `response_model_pth` is provided)

| File | Description |
|------|-------------|
| Gene expression (TPM, TSV) | Pan-cancer immunotherapy cohort (e.g., Brooks DB) |
| Clinical data (TSV) | Must contain sample ID and response (R/NR) columns |

### Fine-tuning Data (optional)

| File | Description |
|------|-------------|
| Gene expression (TPM, TSV) | Cancer-specific cohort (e.g., KIRC) |
| Clinical data (TSV) | Sample ID + response labels |
| Ecotype abundance (TSV) | Optional -- enables iterative fine-tuning of GAT |

All expression inputs are raw TPM. The pipeline applies per-gene z-score normalization automatically.

## Configuration

Edit `config.yaml`:

```yaml
# From Components 1-2
graph_pkl: "path/to/global_graph.pkl"
gat_model_pth: "path/to/NN11GraphModel.pth"

# TCGA reference
tcga_expression_tsv: "path/to/tcga_expression.tsv"

# Pre-trained response model (set to null to train from scratch)
response_model_pth: "path/to/response_model.pth"

# Pre-training data (only needed if response_model_pth is null)
pretrain_expression_tsv: "path/to/brooks_expression.txt"
pretrain_clinical_tsv: "path/to/brooks_clinical.txt"

# Fine-tuning (set to null to skip)
finetune_expression_tsv: "path/to/kirc_expression.txt"
finetune_clinical_tsv: "path/to/kirc_clinical.txt"
finetune_abundance_tsv: "path/to/kirc_ecotype_abundance.txt"  # null = freeze GAT

# Clinical format
sample_id_column: "ID"
response_column: "Response"
response_mapping:       # flexible — supports multi-class
  R: 1
  NR: 0

# Architecture must match Component 2
num_ecotypes: 11
hidden_channels: 8
```

The number of output classes is derived automatically from `response_mapping`. Class weights are computed from data proportions to handle imbalanced labels. All parameters have sensible defaults documented in the YAML.

## Outputs

### Step 1: Pre-training

| File | Description |
|------|-------------|
| **`output/response_model.pth`** | **Pre-trained ResponsePredictor weights** |
| `output/pretrain_ecotype_predictions.txt` | GAT ecotype predictions on pre-train data (only if trained from scratch) |

### Step 2: Fine-tuning

| File | Description |
|------|-------------|
| **`output/finetuned_ecotype_model.pth`** | **Fine-tuned GAT weights** |
| **`output/finetuned_response_model.pth`** | **Fine-tuned ResponsePredictor weights** |
| `output/finetune_predictions.csv` | Predictions on fine-tune cohort |
| `output/edge_attn.txt` | Edge attention scores (post fine-tuning) |
| `output/node_topk_sum_attn.txt` | Node top-k sum attention |

## Method

- **Preprocessing**: Each expression dataset is z-score normalized per gene across samples, aligned to the model's gene list (derived from the graph-expression intersection), and missing genes are filled via KNN imputation (k=16) using TCGA as reference.

- **Step 1 (Pre-training)**: If a pre-trained response model is provided, it is used directly. Otherwise, the pretrained GAT (frozen) maps each sample's expression to 11-dimensional ecotype features via softmax. A ResponsePredictor MLP (11 -> 32 -> 16 -> N classes) is trained on these features to classify immunotherapy response using cross-entropy loss with class weights computed from data proportions.

- **Step 2 (Fine-tuning)**: Two modes depending on available data:
  - **Response only**: GAT is frozen; only the ResponsePredictor is updated using cancer-specific R/NR labels.
  - **Iterative**: Each epoch alternates (1) updating the GAT on ecotype abundance loss (KL-divergence), then (2) updating the ResponsePredictor on response loss (cross-entropy). This adapts both the ecotype representation and response prediction to the target cancer type.

## Reproducibility Note

Neural network training results may vary slightly across different hardware (CPU vs GPU, different CUDA versions). The provided pre-trained model weights produce consistent inference results on any platform. If retraining from scratch, exact metrics may differ but should be comparable.

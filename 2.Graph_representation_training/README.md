# EcoNet: Graph Representation Training

Trains a Graph Attention Network on the intercellular regulatory network (from Step 1) to predict ecotype abundance from bulk gene expression. The output `NN11GraphModel.pth` is the pretrained model used as input for transfer learning / response prediction.

## Quick Start

```bash
# 1. Edit config.yaml to point to your data
# 2. Run
python run_pipeline.py --config config.yaml
```

To run a specific step:

```bash
python run_pipeline.py --config config.yaml --step 1    # cross-validation only
python run_pipeline.py --config config.yaml --step 2    # final training only
```

## Pipeline

```
global_graph.pkl + expression matrix (TPM) + ecotype abundance
       |
  [Preprocessing] Per-gene z-score normalization
       |
  [Step 1] K-fold cross-validation (evaluate model performance)
       |
  [Step 2] Train final model on all data + attention analysis
       |
  NN11GraphModel.pth + gene_selected.txt + attention scores
```

Intermediate results are cached -- re-running safely skips completed steps.

## Setup

```bash
conda activate env_econet
```

Required packages (pre-installed in env_econet): `torch`, `torch_geometric`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `pyyaml`.

## Input Data

### 1. Regulatory Network (`global_graph.pkl`)

Output from [1.TME_representation](../1.TME_representation/). A NetworkX DiGraph encoding gene-gene regulatory relationships across ecotypes.

### 2. Bulk Expression Matrix (TSV)

TPM gene expression, genes as rows, samples as columns. Tab-separated with gene names as row index. The pipeline automatically applies per-gene z-score normalization across samples.

| | Sample1 | Sample2 | ... |
|---|---------|---------|-----|
| Gene1 | 0.52 | -1.23 | ... |
| Gene2 | -0.81 | 0.44 | ... |

### 3. Ecotype Abundance (TSV)

Ecotype abundance per sample (from EcoTyper). Ecotypes as rows, samples as columns.

| | Sample1 | Sample2 | ... |
|---|---------|---------|-----|
| E1 | 0.12 | 0.05 | ... |
| E2 | 0.08 | 0.15 | ... |

## Configuration

Edit `config.yaml`:

```yaml
# Data paths
graph_pkl: "path/to/global_graph.pkl"
expression_tsv: "path/to/expression_matrix.tsv"
abundance_tsv: "path/to/ecotype_abundance.txt"

# Number of ecotypes (must match abundance data)
num_ecotypes: 11

# GAT architecture
hidden_channels: 8
gat_heads_1: 4
gat_heads_2: 1
fc_dim: 128
dropout: 0.2

# Training
learning_rate: 0.0005
num_epochs: 1000
batch_size: 8
num_folds: 5
```

All parameters have sensible defaults documented in the YAML.

## Outputs

| File | Description |
|------|-------------|
| `output/cv_results.txt` | K-fold CV metrics (accuracy, precision, recall, F1, MSE, MAE, R2) |
| **`output/NN11GraphModel.pth`** | **Pretrained GAT model weights (input for transfer learning)** |
| **`output/gene_selected.txt`** | **Gene list (intersection of graph nodes and expression genes)** |
| `output/edge_attn.txt` | Edge-level attention scores (gene--gene pairs) |
| `output/node_sum_attn.txt` | Total attention per gene (sum of incident edges) |
| `output/node_ave_attn.txt` | Average attention per gene (total / degree) |
| `output/node_topk_sum_attn.txt` | Sum of top-k edge attentions per gene |
| `output/node_topk_ave_attn.txt` | Average of top-k edge attentions per gene |

## Quick test

`config_test.yaml` runs a reduced smoke test (few epochs/folds on a small matrix). Download the test data from [Zenodo](https://doi.org/10.5281/zenodo.22034558) and set the `PATH/TO/test_data` paths to your test inputs, then:

```bash
python run_pipeline.py --config config_test.yaml
```

# EcoNet

**EcoNet** predicts immunotherapy response from bulk RNA-seq. It models the
tumor microenvironment as an ecosystem of cellular **ecotypes**, constructs an
intercellular **regulatory network**, and applies a **Graph Attention Network
(GAT)** over that network to derive ecotype-level features for response
prediction.

![EcoNet pipeline](pipeline.png)

The pipeline has four steps:

1. **Network Construction**: build an intercellular regulatory network from
   scRNA-seq, an EcoTyper model, and the NicheNet database.
2. **GAT Pretraining**: train a GAT to predict ecotype abundance from bulk
   expression over that network.
3. **Transfer Learning**: train (and optionally fine-tune) a response predictor
   on immunotherapy cohorts using the GAT's ecotype features.
4. **Prediction**: score a new cohort as responder or non-responder.

There are two ways to use EcoNet:

- **A. Predict with a provided model.** Bring only your bulk RNA-seq and use the
  shipped ccRCC or pan-cancer model (Step 4 only).
- **B. Build your own.** Train a network and model from your own scRNA-seq and
  cohorts (Steps 1 to 4).

## Installation

```bash
conda env create -f environment.yml
conda activate econet
```

`torch` and `torch_geometric` wheels are platform and CUDA specific. See
`environment.yml` for GPU notes.

## A. Predict with a provided model

Everything needed ships in this repo under `pretrained_models/`, with no
external downloads. Your only input is a bulk expression matrix (raw **TPM**,
genes as rows and samples as columns, tab-separated). The pipeline z-score
normalizes per gene automatically.

```bash
cd 4.Prediction

# Edit expression_tsv in the config to point at your matrix, then:
python run_pipeline.py --config config_ccRCC.yaml        # ccRCC model (RE1-RE11)
# or
python run_pipeline.py --config config_pancancer.yaml    # pan-cancer model (CE1-CE10)
```

Outputs (in `output_*/`):

| File | Description |
|------|-------------|
| `response_predictions.csv` | Per-sample class and probabilities (CR/PR = responder/1, SD/PD = non-responder/0) |
| `ecotype_predictions.txt` | Predicted ecotype abundances per sample |
| `metrics.txt`, `roc_curve.png` | Only if you also provide a clinical table for evaluation |

Genes missing from your matrix are filled by KNN imputation against a small TCGA
reference bundled with each model. Two models are provided:

| Model | Ecotypes | Response predictor | Bundle |
|-------|----------|--------------------|--------|
| ccRCC | RE1-RE11 | fine-tuned, `[32,16]` | `pretrained_models/ccRCC/` |
| pan-cancer | CE1-CE10 | portable, `[32,8]` do0.6 (LODO AUC ~0.72) | `pretrained_models/pancancer/` |

See each bundle's `README.md` for file details.

## B. Build your own network and model

Run the four steps in order. Each is driven by a `config.yaml` (paths
resolve relative to the config file) and caches intermediate results, so
re-running skips completed steps.

```bash
# 1. Network from your scRNA-seq + EcoTyper model + NicheNet DB
cd 1.NetworkConstruction
python run_pipeline.py --config config_ccRCC.yaml          # or config_carcinoma.yaml
#   (optional) merge per-cancer networks into a pan-cancer consensus:
#   python merge_pancancer.py

# 2. Pretrain the GAT on bulk expression + ecotype abundance
cd ../2.GATPretrain
python run_pipeline.py --config config.yaml

# 3. Train / fine-tune the response predictor on immunotherapy cohorts
cd ../3.TransferLearning
python run_pipeline.py --config config_ccRCC.yaml          # or config_pancancer.yaml

# 4. Predict on a new cohort
cd ../4.Prediction
python run_pipeline.py --config config.yaml
```

Each step's `README.md` documents its required inputs and output files.

## Repository structure

```
EcoNet/
├── pretrained_models/          # ready-to-use ccRCC + pan-cancer bundles
│   ├── ccRCC/
│   └── pancancer/
├── 1.NetworkConstruction/      # scRNA-seq + EcoTyper + NicheNet -> network
├── 2.GATPretrain/              # network + bulk expr -> GAT
├── 3.TransferLearning/         # GAT + cohorts -> response predictor
├── 4.Prediction/               # models + new cohort -> R/NR
└── environment.yml
```

## Maintainer

WANG Ruohan, ruohwang@stanford.edu

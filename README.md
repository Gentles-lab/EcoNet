# EcoNet

**EcoNet** predicts immunotherapy response from bulk RNA-seq. It models the
tumor microenvironment as an ecosystem of cellular **ecotypes**, constructs an
intercellular **regulatory network**, and applies a **Graph Attention Network
(GAT)** over that network to derive ecotype-level features for response
prediction.

![EcoNet pipeline](pipeline.png)

The pipeline has four steps:

1. **TME Representation**: build an intercellular regulatory network from
   scRNA-seq, an EcoTyper model, and the NicheNet database.
2. **Graph Representation Training**: train a GAT to predict ecotype abundance
   from bulk expression over that network.
3. **Response Prediction Training**: train (and optionally fine-tune) a response
   predictor on immunotherapy cohorts using the GAT's ecotype features.
4. **Prediction**: score a new cohort as responder or non-responder.

There are two ways to use EcoNet:

- **A. Predict with a provided model.** Run the shipped ccRCC or pan-cancer model on your bulk RNA-seq data (Step 4 only).
- **B. Build your own.** Build a signaling network and train a model from your own scRNA-seq and cohorts (Steps 1 to 4).

## Installation

```bash
conda env create -f environment.yml
conda activate econet
```

`torch` and `torch_geometric` wheels are platform and CUDA specific. See
`environment.yml` for GPU notes.

## A. Predict with a provided model

The repo includes pretrained models under pretrained\_models/. The only required input is a tab-separated bulk expression matrix of raw TPM values (genes as rows, samples as columns). Per-gene z-score normalization is applied automatically.

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

Two models are provided:

| Model | Ecotypes | Directory |
|-------|----------|--------|
| pan-cancer | CE1-CE10 | `pretrained_models/pancancer/` |
| ccRCC | RE1-RE11 | `pretrained_models/ccRCC/` |

See each model's `README.md` for file details.

## B. Build your own network and model

Run the four steps in order. Each is driven by a `config.yaml` (paths
resolve relative to the config file) and caches intermediate results, so
re-running skips completed steps.

```bash
# 1. Network from your scRNA-seq + EcoTyper model + NicheNet DB
cd 1.TME_representation
python run_pipeline.py --config config_ccRCC.yaml          # or config_carcinoma.yaml
#   (optional) merge per-cancer networks into a pan-cancer consensus:
#   python merge_pancancer.py

# 2. Pretrain the GAT on bulk expression + ecotype abundance
cd ../2.Graph_representation_training
python run_pipeline.py --config config.yaml

# 3. Train / fine-tune the response predictor on immunotherapy cohorts
cd ../3.Response_prediction_training
python run_pipeline.py --config config_ccRCC.yaml          # or config_pancancer.yaml

# 4. Predict on a new cohort
cd ../4.Prediction
python run_pipeline.py --config config.yaml
```

Each step's `README.md` documents its required inputs and output files.

## Test data

Small example inputs for a quick smoke test of each step are available on Zenodo:
https://doi.org/10.5281/zenodo.22034558. Download and unpack them, then point the
`PATH/TO/test_data` paths in each step's `config_test.yaml` at that folder.


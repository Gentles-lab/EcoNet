# EcoNet: Network Construction

Constructs an intercellular regulatory network from scRNA-seq data and an EcoTyper model. The output `global_graph.pkl` is a directed gene network used as input for GAT model training.

## Quick Start

```bash
# 1. Edit config.yaml to point to your data and EcoTyper model
# 2. Run
python run_pipeline.py --config config.yaml
```

The final output is `output/global_graph.pkl`.

To resume from a specific step:

```bash
python run_pipeline.py --config config.yaml --step 2 3
```

## Pipeline

```
scRNA-seq (h5ad) + EcoTyper model + NicheNet DB
       |
  [Step 1] Identify overexpressed ligand-receptor interactions per sample
       |
  [Step 2] Score ligand activities per ecotype cell-pair (iRegulon AUC)
       |
  [Step 3] Random walk -> extract subgraphs -> merge into global network
       |
  global_graph.pkl
```

Intermediate results are cached -- re-running safely skips completed steps.

## Setup

```bash
conda create -n econet python=3.10
conda activate econet
pip install torch torch_geometric scanpy networkx scipy statsmodels \
            scikit-learn joblib matplotlib seaborn pandas numpy pyyaml
```

## Input Data

### 1. scRNA-seq (`data/scRNA.h5ad`)

AnnData h5ad with log-normalized expression. Required `.obs` columns:

| Column | Example | Description |
|--------|---------|-------------|
| `CellType_State` | `CD8.T.cells_S01` | Cell type + state label from EcoTyper |
| `Sample` | `P55_scRNA` | Sample / patient ID |

Column names are configurable in `config.yaml`.

### 2. EcoTyper Model

Point `ecotyper_dir` to an [EcoTyper](https://github.com/digitalcytometry/ecotyper) output directory. The pipeline auto-discovers ecotypes, marker genes, and abundance from the standard EcoTyper output structure:

```
EcotyperModel/
  Carcinoma_Fractions/              # <- fractions_name in config
    Ecotypes/
      discovery/
        ecotypes.txt                # ecotype-to-cell-state mapping
        ecotype_abundance.txt       # discovery abundance
      recovery/
        my_scrna_dataset/           # <- recovery_dataset in config
          ecotype_abundance.txt     # links ecotypes to your scRNA samples
    Cell_States/
      discovery/
        B.cells/9/gene_info.txt     # marker genes per cell state (NMF output)
        CD8.T.cells/10/gene_info.txt
        ...
```

The pipeline automatically:
- **Discovers ecotype IDs** from `ecotypes.txt`
- **Extracts marker genes** from `Cell_States/discovery/` (auto-detects the optimal NMF rank per cell type)
- **Reads ecotype abundance** from `recovery/{dataset}/` (or `discovery/` if `recovery_dataset` is null)

No pre-processing of the EcoTyper output is needed.

### 3. NicheNet Database (`NicheNet_DB/`)

Download from [NicheNet](https://github.com/saeyslab/nichenetr). Required files:

- `lr_network.csv`
- `weighted_lr_sig.csv`
- `weighted_gr.csv`
- `ligand_target_matrix.csv`

## Configuration

Edit `config.yaml`:

```yaml
# Data paths
scrna_h5ad: "data/scRNA.h5ad"
nichenet_dir: "NicheNet_DB"

# EcoTyper model -- just point to the directory
ecotyper_dir: "data/EcotyperModel"
fractions_name: "Carcinoma_Fractions"       # subfolder name in ecotyper_dir
recovery_dataset: "my_scrna_dataset"        # recovery subfolder (null = use discovery)

# scRNA-seq columns
cell_state_column: "CellType_State"
sample_column: "Sample"
```

All pipeline parameters (thresholds, Hill equation, bootstrap, noisy gene filter) have sensible defaults documented in the YAML.

## Outputs

| File | Description |
|------|-------------|
| `output/overexpressed_genes_{sample}.csv` | Differentially expressed genes per cell state |
| `output/overexpressed_lri_{sample}.csv` | Significant ligand-receptor pairs |
| `output/communication_probabilities_{sample}.csv` | Cell-cell communication scores |
| `output/{ecotype}_{sender}_{receiver}.tsv` | Ranked ligand activities |
| `output/{ecotype}_{sender}_{receiver}.png` | Ligand-target heatmap |
| `output/{ecotype}_{sender}_{receiver}.pkl` | Per-pair regulatory subnetwork |
| `output/{ecotype}_graph.pkl` | Per-ecotype merged graph |
| **`output/global_graph.pkl`** | **Final network (input for GAT model)** |

## Applying to a New Dataset

1. Run [EcoTyper](https://github.com/digitalcytometry/ecotyper) on your data.
2. Place your h5ad and EcoTyper output directory under `data/`.
3. Copy NicheNet DB files into `NicheNet_DB/`.
4. Edit `config.yaml`:
   - Set `ecotyper_dir` to your EcoTyper output
   - Set `recovery_dataset` to the recovery run matching your scRNA-seq samples
   - Set column names to match your h5ad
   - Adjust `noisy_genes` for your cell types (or set to `{}`)
5. Run: `python run_pipeline.py --config config.yaml`

## Method

- **Step 1**: Wilcoxon rank-sum tests with Bonferroni correction identify overexpressed genes per cell state. Ligand-receptor pairs are filtered from the NicheNet database. Communication probabilities are computed via Hill equation with bootstrap permutation testing.

- **Step 2**: For each ecotype, the highest-abundance sample is selected. Ligand activity is scored per sender-receiver pair using iRegulon-style AUC against NicheNet's ligand-target matrix.

- **Step 3**: Random walks through NicheNet's weighted signaling network find paths (ligand -> mediator -> TF), constrained to expressed genes. Top mediator genes are extracted and all ecotype subgraphs are merged into a single global network.

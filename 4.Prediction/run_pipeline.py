#!/usr/bin/env python
"""
EcoNet Prediction Pipeline
============================
Predict immunotherapy response for a new dataset using trained models
from Steps 1-3.

Usage:
    python run_pipeline.py --config config.yaml
"""

import os
import sys
import yaml
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, auc)
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import GATConv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

class Config:
    def __init__(self, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(path))

        def resolve(p):
            if p is None:
                return None
            return p if os.path.isabs(p) else os.path.join(base, p)

        # Models from Steps 1-3
        self.graph_pkl = resolve(d["graph_pkl"])
        self.ecotype_model_pth = resolve(d["ecotype_model_pth"])
        self.response_model_pth = resolve(d["response_model_pth"])

        # TCGA reference
        self.tcga_expression_tsv = resolve(d["tcga_expression_tsv"])

        # New dataset
        self.expression_tsv = resolve(d["expression_tsv"])
        self.clinical_tsv = resolve(d.get("clinical_tsv"))

        # Output
        self.output_dir = resolve(d["output_dir"])

        # Clinical format
        self.sample_id_column = d.get("sample_id_column", "ID")
        self.response_column = d.get("response_column", "Response")
        self.response_mapping = d.get("response_mapping", {"R": 1, "NR": 0})
        self.num_classes = len(set(self.response_mapping.values()))

        # GAT architecture
        self.num_ecotypes = d["num_ecotypes"]
        self.in_channels = d.get("in_channels", 1)
        self.hidden_channels = d.get("hidden_channels", 8)
        self.gat_heads_1 = d.get("gat_heads_1", 4)
        self.gat_heads_2 = d.get("gat_heads_2", 1)
        self.fc_dim = d.get("fc_dim", 128)
        self.gat_dropout = d.get("gat_dropout", 0.2)

        # ResponsePredictor architecture
        self.response_hidden_dims = d.get("response_hidden_dims", [32, 16])
        self.response_dropout = d.get("response_dropout", 0.1)

        # KNN
        self.knn_neighbors = d.get("knn_neighbors", 16)

        # Other
        self.random_seed = d.get("random_seed", 123)
        self.batch_size = int(d.get("batch_size", 100))

    def validate(self):
        errors = []
        for name, path in [("Graph", self.graph_pkl),
                           ("Ecotype model", self.ecotype_model_pth),
                           ("Response model", self.response_model_pth),
                           ("TCGA expression", self.tcga_expression_tsv),
                           ("Expression", self.expression_tsv)]:
            if not os.path.exists(path):
                errors.append(f"{name} not found: {path}")
        if self.clinical_tsv and not os.path.exists(self.clinical_tsv):
            errors.append(f"Clinical not found: {self.clinical_tsv}")
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

    def summary(self):
        print(f"  Graph:          {self.graph_pkl}")
        print(f"  Ecotype model:  {self.ecotype_model_pth}")
        print(f"  Response model: {self.response_model_pth}")
        print(f"  TCGA reference: {self.tcga_expression_tsv}")
        print(f"  Expression:     {self.expression_tsv}")
        if self.clinical_tsv:
            print(f"  Clinical:       {self.clinical_tsv} (evaluation enabled)")
        else:
            print(f"  Clinical:       not provided (prediction only)")


# =============================================================================
# Models (must match Steps 2-3)
# =============================================================================

class EcotypeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes,
                 heads_1=4, heads_2=1, fc_dim=128, dropout=0.2):
        super(EcotypeClassifier, self).__init__()
        self.num_nodes = num_nodes
        self.dropout_rate = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads_1,
                            dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads_1, hidden_channels,
                            heads=heads_2, concat=False, dropout=dropout)
        self.fc1 = nn.Linear(num_nodes * hidden_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        batch_size, num_nodes, in_channels = x.size()
        x = x.reshape(-1, in_channels)
        num_edges = edge_index.size(1)
        edge_index_batch = edge_index.repeat(1, batch_size)
        offset = (torch.arange(batch_size, device=x.device)
                  .repeat_interleave(num_edges) * num_nodes)
        edge_index_batch = edge_index_batch + offset

        x, _ = self.gat1(x, edge_index_batch, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x, _ = self.gat2(x, edge_index_batch, return_attention_weights=True)

        x = x.view(batch_size, num_nodes, -1).reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResponsePredictor(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_dims=None,
                 dropout=0.1):
        super(ResponsePredictor, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        self.fc1 = nn.Linear(input_size, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_size)
        self.dropout_rate = dropout

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = torch.dropout(x, p=self.dropout_rate, train=self.training)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def zscore_normalize(df):
    means = df.mean(axis=1)
    stds = df.std(axis=1).replace(0, 1e-6)
    return df.sub(means, axis=0).div(stds, axis=0)


def derive_gene_list(graph_pkl, tcga_tsv):
    with open(graph_pkl, "rb") as f:
        graph = pickle.load(f)
    tcga = pd.read_csv(tcga_tsv, index_col=0, sep="\t")
    tcga_genes = set(tcga.index)
    gene_list = [g for g in graph.nodes() if g in tcga_genes]
    return graph, gene_list


def load_graph_edges(graph, gene_list, device):
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    edges = [(gene_to_idx[u], gene_to_idx[v])
             for u, v in graph.edges()
             if u in gene_to_idx and v in gene_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    return edges, edge_index


def impute_expression(target_tsv, gene_list, tcga_normed, knn_neighbors):
    target_raw = pd.read_csv(target_tsv, sep="\t", index_col=0).dropna(how="all")
    target_normed = zscore_normalize(target_raw)

    aligned = pd.DataFrame(index=gene_list, columns=target_normed.columns)
    genes_available = target_normed.index.intersection(gene_list)
    aligned.loc[genes_available] = target_normed.loc[genes_available]

    combined = pd.concat([tcga_normed, aligned.T], axis=0)
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    imputed = imputer.fit_transform(combined)
    imputed_df = pd.DataFrame(
        imputed, index=combined.index, columns=combined.columns
    ).loc[aligned.columns]
    return imputed_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EcoNet: Predict immunotherapy response for a new dataset")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg.validate()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("EcoNet Prediction Pipeline")
    print(f"Config: {args.config}")
    print(f"Output: {cfg.output_dir}")
    cfg.summary()

    set_seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Load graph and derive gene list ---
    print(f"\nDeriving gene list from graph + TCGA expression...")
    graph, gene_list = derive_gene_list(cfg.graph_pkl, cfg.tcga_expression_tsv)
    edges, edge_index = load_graph_edges(graph, gene_list, device)
    print(f"  Genes: {len(gene_list)}, Edges: {len(edges)}")

    # --- Prepare TCGA reference ---
    print("  Z-score normalizing TCGA reference...")
    tcga_raw = pd.read_csv(cfg.tcga_expression_tsv, index_col=0, sep="\t")
    tcga_normed = zscore_normalize(tcga_raw).reindex(gene_list).fillna(0).T

    # --- Preprocess new dataset ---
    print(f"\nPreprocessing new dataset...")
    imputed_df = impute_expression(
        cfg.expression_tsv, gene_list, tcga_normed, cfg.knn_neighbors)
    print(f"  Imputed: {imputed_df.shape[0]} samples x {imputed_df.shape[1]} genes")

    # --- Load clinical if provided ---
    clinical = None
    if cfg.clinical_tsv:
        clinical = pd.read_csv(cfg.clinical_tsv, sep="\t")
        clinical = clinical[
            clinical[cfg.response_column].isin(cfg.response_mapping.keys())]
        clinical["Label"] = clinical[cfg.response_column].map(
            cfg.response_mapping).astype(int)
        clinical.set_index(cfg.sample_id_column, inplace=True)

    # --- Align samples ---
    samples = imputed_df.index
    if clinical is not None:
        samples = samples.intersection(clinical.index)
    print(f"  Samples for prediction: {len(samples)}")

    X = torch.tensor(imputed_df.loc[samples].values,
                     dtype=torch.float32).to(device)

    # --- Load models ---
    print(f"\nLoading models...")
    ecotype_model = EcotypeClassifier(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.num_ecotypes,
        num_nodes=len(gene_list),
        heads_1=cfg.gat_heads_1,
        heads_2=cfg.gat_heads_2,
        fc_dim=cfg.fc_dim,
        dropout=cfg.gat_dropout,
    ).to(device)
    ecotype_model.load_state_dict(
        torch.load(cfg.ecotype_model_pth, map_location=device))
    ecotype_model.eval()

    response_model = ResponsePredictor(
        input_size=cfg.num_ecotypes,
        output_size=cfg.num_classes,
        hidden_dims=cfg.response_hidden_dims,
        dropout=cfg.response_dropout,
    ).to(device)
    response_model.load_state_dict(
        torch.load(cfg.response_model_pth, map_location=device))
    response_model.eval()
    print("  Models loaded.")

    # --- Predict ---
    print(f"\nPredicting...")
    all_eco_feats = []
    all_pred_probs = []

    with torch.no_grad():
        for i in range(0, len(X), cfg.batch_size):
            x_batch = X[i:i+cfg.batch_size].unsqueeze(2)
            logits = ecotype_model(x_batch, edge_index)
            eco_feat = F.softmax(logits, dim=1)
            pred_log = response_model(eco_feat)
            pred_probs = torch.exp(pred_log)
            all_eco_feats.append(eco_feat.cpu())
            all_pred_probs.append(pred_probs.cpu())

    eco_feats = torch.cat(all_eco_feats, dim=0).numpy()
    pred_probs = torch.cat(all_pred_probs, dim=0).numpy()
    pred_cls = pred_probs.argmax(axis=1)

    # --- Save ecotype predictions ---
    eco_path = os.path.join(cfg.output_dir, "ecotype_predictions.txt")
    eco_df = pd.DataFrame(
        eco_feats, index=samples,
        columns=[f"E{i+1}" for i in range(cfg.num_ecotypes)])
    eco_df.T.to_csv(eco_path, sep="\t")
    print(f"  Saved: {eco_path}")

    # --- Save response predictions ---
    pred_path = os.path.join(cfg.output_dir, "response_predictions.csv")
    # Build reverse mapping for readable labels
    label_names = {}
    for name, val in cfg.response_mapping.items():
        if val not in label_names:
            label_names[val] = name
    pred_dict = {
        "SampleID": list(samples),
        "PredClass": pred_cls,
        "PredLabel": [label_names.get(c, str(c)) for c in pred_cls],
    }
    for c in range(cfg.num_classes):
        pred_dict[f"Prob_class{c}"] = pred_probs[:, c]
    pd.DataFrame(pred_dict).to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    # --- Evaluate if clinical data provided ---
    if clinical is not None:
        y_true = clinical.loc[samples, "Label"].values
        pred_dict["TrueLabel"] = y_true
        pred_dict["TrueResponse"] = [
            clinical.loc[s, cfg.response_column] for s in samples]

        # Re-save with true labels
        pd.DataFrame(pred_dict).to_csv(pred_path, index=False)

        acc = accuracy_score(y_true, pred_cls)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred_cls, average="macro", zero_division=0)

        print(f"\n  Evaluation Results:")
        print(f"    Samples:   {len(samples)}")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {p:.4f}")
        print(f"    Recall:    {r:.4f}")
        print(f"    F1:        {f1:.4f}")

        # AUC
        try:
            if cfg.num_classes == 2:
                auc_score = roc_auc_score(y_true, pred_probs[:, 1])
            else:
                auc_score = roc_auc_score(y_true, pred_probs,
                                          multi_class="ovr")
            print(f"    AUC:       {auc_score:.4f}")
        except ValueError:
            auc_score = float("nan")
            print(f"    AUC:       N/A")

        # ROC curve (binary only)
        if cfg.num_classes == 2 and not np.isnan(auc_score):
            fpr, tpr, _ = roc_curve(y_true, pred_probs[:, 1])
            plt.figure(figsize=(4.5, 5))
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
            plt.plot([0, 1], [0, 1], "k:", label="Random")
            plt.xlabel("False Positive Rate", fontsize=14)
            plt.ylabel("True Positive Rate", fontsize=14)
            plt.tick_params(axis="both", which="major", labelsize=12)
            plt.legend(loc="lower right", fontsize=10, frameon=False)
            plt.tight_layout()
            roc_path = os.path.join(cfg.output_dir, "roc_curve.png")
            plt.savefig(roc_path, dpi=600)
            plt.close()
            print(f"  Saved: {roc_path}")

        # Save metrics
        metrics_path = os.path.join(cfg.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Samples: {len(samples)}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {p:.4f}\n")
            f.write(f"Recall: {r:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
            f.write(f"AUC: {auc_score:.4f}\n")
        print(f"  Saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

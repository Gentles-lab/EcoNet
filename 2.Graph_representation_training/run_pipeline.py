#!/usr/bin/env python
"""
EcoNet GAT Pretraining Pipeline
=================================
Train a Graph Attention Network on an intercellular regulatory network
to predict ecotype abundance from bulk gene expression.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --step 1      # cross-validation only
    python run_pipeline.py --config config.yaml --step 2      # final training only
"""

import os
import sys
import yaml
import random
import argparse
import itertools
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             mean_squared_error, mean_absolute_error, r2_score)
from torch_geometric.nn import GATConv
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr


# =============================================================================
# Configuration loader
# =============================================================================

class Config:
    """Load configuration from YAML."""

    def __init__(self, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(path))

        def resolve(p):
            return p if os.path.isabs(p) else os.path.join(base, p)

        # Input data
        self.graph_pkl = resolve(d["graph_pkl"])
        self.expression_tsv = resolve(d["expression_tsv"])
        self.abundance_tsv = resolve(d["abundance_tsv"])

        # Output
        self.output_dir = resolve(d["output_dir"])

        # GAT architecture
        self.num_ecotypes = d["num_ecotypes"]
        self.in_channels = d.get("in_channels", 1)
        self.hidden_channels = d.get("hidden_channels", 8)
        self.gat_heads_1 = d.get("gat_heads_1", 4)
        self.gat_heads_2 = d.get("gat_heads_2", 1)
        self.fc_dim = d.get("fc_dim", 128)
        self.dropout = d.get("dropout", 0.2)

        # Training
        self.learning_rate = d.get("learning_rate", 0.0005)
        self.num_epochs = d.get("num_epochs", 1000)
        self.batch_size = d.get("batch_size", 8)
        self.num_folds = d.get("num_folds", 5)
        self.random_seed = d.get("random_seed", 123)

        # Attention analysis
        self.top_k_edges = d.get("top_k_edges", 5)

    def validate(self, steps):
        """Check that required input files exist."""
        errors = []
        if not os.path.exists(self.graph_pkl):
            errors.append(f"Graph not found: {self.graph_pkl}")
        if not os.path.exists(self.expression_tsv):
            errors.append(f"Expression data not found: {self.expression_tsv}")
        if not os.path.exists(self.abundance_tsv):
            errors.append(f"Abundance data not found: {self.abundance_tsv}")
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

    def summary(self):
        """Print configuration summary."""
        print(f"  Graph:       {self.graph_pkl}")
        print(f"  Expression:  {self.expression_tsv}")
        print(f"  Abundance:   {self.abundance_tsv}")
        print(f"  Ecotypes:    {self.num_ecotypes}")
        print(f"  Architecture: GAT({self.in_channels} -> "
              f"{self.hidden_channels}x{self.gat_heads_1}h -> "
              f"{self.hidden_channels}x{self.gat_heads_2}h) -> "
              f"FC({self.fc_dim}) -> {self.num_ecotypes}")
        print(f"  Training:    lr={self.learning_rate}, epochs={self.num_epochs}, "
              f"batch={self.batch_size}, folds={self.num_folds}")


# =============================================================================
# GAT Model
# =============================================================================

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes,
                 heads_1=4, heads_2=1, fc_dim=128, dropout=0.2):
        super(GAT, self).__init__()
        self.num_nodes = num_nodes
        self.dropout_rate = dropout

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads_1,
                            dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads_1, hidden_channels,
                            heads=heads_2, concat=False, dropout=dropout)

        self.fc1 = nn.Linear(num_nodes * hidden_channels, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, out_channels)

    def forward(self, x, edge_index):
        # x: [batch_size, num_nodes, in_channels]
        batch_size, num_nodes, in_channels = x.size()
        x = x.view(-1, in_channels)  # [batch_size * num_nodes, in_channels]

        # Expand edge_index for batched graph
        num_edges = edge_index.size(1)
        edge_index_batch = edge_index.repeat(1, batch_size)
        offset = (torch.arange(batch_size, device=x.device)
                  .repeat_interleave(num_edges) * num_nodes)
        edge_index_batch = edge_index_batch + offset

        # GAT layers
        x, (attn_edge_index, attn_weight) = self.gat1(
            x, edge_index_batch, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x, (attn_edge_index, attn_weight) = self.gat2(
            x, edge_index_batch, return_attention_weights=True)

        # Match attention weights to input edges
        N = batch_size * num_nodes
        input_ids = edge_index_batch[0] * N + edge_index_batch[1]
        attn_ids = attn_edge_index[0] * N + attn_edge_index[1]
        attn_ids_sorted, sorted_idx = torch.sort(attn_ids)
        search_pos = torch.searchsorted(attn_ids_sorted, input_ids)
        in_range = search_pos < len(attn_ids_sorted)
        match = in_range & (attn_ids_sorted[
            search_pos.clamp(max=len(attn_ids_sorted) - 1)] == input_ids)
        attn_weight_sorted = attn_weight[sorted_idx]
        attn_weight_full = torch.zeros_like(
            input_ids, dtype=attn_weight.dtype, device=x.device)
        attn_weight_full[match] = attn_weight_sorted[
            search_pos[match]].squeeze()

        attn_weight_batch = attn_weight_full.view(batch_size, num_edges)

        x = x.view(batch_size, num_nodes, -1).reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x, attn_weight_batch


# =============================================================================
# Data loading
# =============================================================================

def zscore_normalize(df):
    """Per-gene z-score normalization across samples (genes as rows)."""
    means = df.mean(axis=1)
    stds = df.std(axis=1).replace(0, 1e-6)
    return df.sub(means, axis=0).div(stds, axis=0)


def load_data(cfg, device):
    """Load expression, abundance, and graph data; filter to shared genes."""
    # Load expression (genes x samples)
    expr_raw = pd.read_csv(cfg.expression_tsv, index_col=0, sep="\t")

    # Z-score normalize per gene across samples
    print("  Z-score normalizing expression data...")
    expr_df = zscore_normalize(expr_raw).T  # transpose to samples x genes

    # Load abundance (ecotypes x samples -> transpose to samples x ecotypes)
    abund_df = pd.read_csv(cfg.abundance_tsv, sep="\t", index_col=0).T

    # Align samples
    common_samples = expr_df.index.intersection(abund_df.index)
    if len(common_samples) == 0:
        print("Error: no common samples between expression and abundance data.")
        sys.exit(1)
    expr_df = expr_df.loc[common_samples]
    abund_df = abund_df.loc[common_samples]
    print(f"  Samples: {len(common_samples)}")

    # Load graph
    with open(cfg.graph_pkl, "rb") as f:
        graph = pickle.load(f)

    # Filter genes to intersection of graph nodes and expression columns
    gene_selected = [g for g in graph.nodes() if g in expr_df.columns]
    print(f"  Genes in graph: {len(graph.nodes())}, "
          f"in expression: {len(expr_df.columns)}, "
          f"intersection: {len(gene_selected)}")

    expr_selected = expr_df[gene_selected]

    # Build edge_index
    gene_to_idx = {g: i for i, g in enumerate(gene_selected)}
    edges = [(gene_to_idx[u], gene_to_idx[v])
             for u, v in graph.edges()
             if u in gene_to_idx and v in gene_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    print(f"  Edges: {len(edges)}")

    # Build tensors
    X = torch.tensor(expr_selected.values, dtype=torch.float32).to(device)
    y = torch.tensor(abund_df.values, dtype=torch.float32).to(device)

    return X, y, edge_index, gene_selected, edges


# =============================================================================
# Step 1: Cross-validation
# =============================================================================

def run_step1(cfg, X, y, edge_index, device):
    """K-fold cross-validation to evaluate model performance."""
    print(f"\n{'='*60}")
    print(f"Step 1: {cfg.num_folds}-fold cross-validation")
    print(f"{'='*60}")

    results_path = os.path.join(cfg.output_dir, "cv_results.txt")
    if os.path.exists(results_path):
        print(f"  Cached: {results_path} exists, skipping.")
        return

    num_nodes = X.shape[1]
    kf = KFold(n_splits=cfg.num_folds, shuffle=True,
               random_state=cfg.random_seed)

    fold_attentions = []
    metrics = {k: [] for k in
               ["accuracy", "precision", "recall", "f1", "mse", "mae", "r2"]}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{cfg.num_folds} ---")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = GAT(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            out_channels=cfg.num_ecotypes,
            num_nodes=num_nodes,
            heads_1=cfg.gat_heads_1,
            heads_2=cfg.gat_heads_2,
            fc_dim=cfg.fc_dim,
            dropout=cfg.dropout,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.KLDivLoss(reduction="batchmean")

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=cfg.batch_size, shuffle=True)

        # Train
        model.train()
        all_attn_weights = []
        for epoch in range(cfg.num_epochs):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2)
                optimizer.zero_grad()
                out, attn_weight = model(x_batch, edge_index)
                loss = criterion(F.log_softmax(out, dim=1), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if epoch == cfg.num_epochs - 1:
                    all_attn_weights.append(attn_weight.cpu().detach().numpy())

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{cfg.num_epochs}], "
                      f"Loss: {total_loss:.4f}")

        # Fold attention
        all_attn_np = np.concatenate(all_attn_weights, axis=0)
        fold_attentions.append(np.mean(all_attn_np, axis=0))

        # Evaluate
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=cfg.batch_size, shuffle=False)

        model.eval()
        y_preds, y_trues = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.unsqueeze(2)
                out, _ = model(x_batch, edge_index)
                preds = torch.exp(F.log_softmax(out, dim=1))
                y_preds.append(preds.cpu().numpy())
                y_trues.append(y_batch.cpu().numpy())

        y_pred = np.concatenate(y_preds, axis=0)
        y_true = np.concatenate(y_trues, axis=0)
        y_pred_cls = y_pred.argmax(axis=1)
        y_true_cls = y_true.argmax(axis=1)

        metrics["accuracy"].append(accuracy_score(y_true_cls, y_pred_cls))
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_cls, y_pred_cls, average="weighted", zero_division=0)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f1)
        metrics["mse"].append(mean_squared_error(y_true, y_pred))
        metrics["mae"].append(mean_absolute_error(y_true, y_pred))
        metrics["r2"].append(r2_score(y_true, y_pred))

    # Print and save results
    lines = []
    for name, vals in metrics.items():
        line = f"{name}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        print(f"  {line}")
        lines.append(line)

    # Attention correlation between folds
    lines.append("")
    lines.append("Attention correlation between folds:")
    print("\n  Attention correlation between folds:")
    for i, j in itertools.combinations(range(len(fold_attentions)), 2):
        corr, _ = pearsonr(fold_attentions[i], fold_attentions[j])
        line = f"  Fold {i+1} vs Fold {j+1}: {corr:.4f}"
        print(f"  {line}")
        lines.append(line)

    with open(results_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved: {results_path}")


# =============================================================================
# Step 2: Final training + attention analysis
# =============================================================================

def run_step2(cfg, X, y, edge_index, gene_selected, edges, device):
    """Train final model on all data; save weights and attention scores."""
    print(f"\n{'='*60}")
    print("Step 2: Train final model on all data")
    print(f"{'='*60}")

    model_path = os.path.join(cfg.output_dir, "NN11GraphModel.pth")
    if os.path.exists(model_path):
        print(f"  Cached: {model_path} exists, skipping.")
        return

    num_nodes = X.shape[1]

    # Save gene list
    gene_path = os.path.join(cfg.output_dir, "gene_selected.txt")
    with open(gene_path, "w") as f:
        for gene in gene_selected:
            f.write(gene + "\n")
    print(f"  Saved: {gene_path} ({len(gene_selected)} genes)")

    model = GAT(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.num_ecotypes,
        num_nodes=num_nodes,
        heads_1=cfg.gat_heads_1,
        heads_2=cfg.gat_heads_2,
        fc_dim=cfg.fc_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.KLDivLoss(reduction="batchmean")

    train_loader = DataLoader(
        TensorDataset(X, y),
        batch_size=cfg.batch_size, shuffle=True)

    all_attn_weights = []
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.unsqueeze(2)
            optimizer.zero_grad()
            out, attn_weight = model(x_batch, edge_index)
            loss = criterion(F.log_softmax(out, dim=1), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch == cfg.num_epochs - 1:
                all_attn_weights.append(attn_weight.cpu().detach().numpy())

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{cfg.num_epochs}], "
                  f"Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"  Saved: {model_path}")

    # --- Attention analysis ---
    print("\n  Computing attention scores...")
    all_attn_np = np.concatenate(all_attn_weights, axis=0)
    mean_attn = np.mean(all_attn_np, axis=0)

    edge_list_named = [(gene_selected[u], gene_selected[v]) for u, v in edges]

    # Node top-k sum attention (only attention output saved)
    node_edge_attn_map = defaultdict(list)
    for (u, v), a in zip(edge_list_named, mean_attn):
        node_edge_attn_map[u].append(a)
        node_edge_attn_map[v].append(a)

    k = cfg.top_k_edges
    node_topk_sum = {n: sum(sorted(al, reverse=True)[:k])
                     for n, al in node_edge_attn_map.items()}
    _write_node_attn(os.path.join(cfg.output_dir, "node_topk_sum_attn.txt"),
                     node_topk_sum, f"Top{k}EdgeAttentionSum")

    print(f"  Saved: node_topk_sum_attn.txt -> {cfg.output_dir}/")


def _write_node_attn(path, node_dict, col_name):
    """Write sorted node attention scores to a TSV file."""
    sorted_items = sorted(node_dict.items(), key=lambda x: x[1], reverse=True)
    with open(path, "w") as f:
        f.write(f"Node\t{col_name}\n")
        for node, val in sorted_items:
            f.write(f"{node}\t{val:.6f}\n")


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EcoNet: GAT pretraining on ecotype abundance")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--step", nargs="*", type=int, default=[1, 2],
                        help="Steps to run (default: 1 2)")
    args = parser.parse_args()

    cfg = Config(args.config)
    steps = sorted(set(args.step))
    cfg.validate(steps)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("EcoNet GAT Pretraining Pipeline")
    print(f"Config: {args.config}")
    print(f"Steps:  {steps}")
    print(f"Output: {cfg.output_dir}")
    cfg.summary()

    set_seed(cfg.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load data once (shared by both steps)
    print(f"\nLoading data...")
    X, y, edge_index, gene_selected, edges = load_data(cfg, device)

    if 1 in steps:
        run_step1(cfg, X, y, edge_index, device)
    if 2 in steps:
        run_step2(cfg, X, y, edge_index, gene_selected, edges, device)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

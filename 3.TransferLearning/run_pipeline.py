#!/usr/bin/env python
"""
EcoNet Transfer Learning & Response Prediction Pipeline
=========================================================
Pre-train a response predictor on pan-cancer immunotherapy data, then
optionally fine-tune on cancer-specific data.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --step 1      # pre-train only
    python run_pipeline.py --config config.yaml --step 2      # fine-tune only
"""

import os
import sys
import yaml
import random
import shutil
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, auc)
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import GATConv


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
            if p is None:
                return None
            return p if os.path.isabs(p) else os.path.join(base, p)

        # Inputs from Components 1-2
        self.graph_pkl = resolve(d["graph_pkl"])
        self.gat_model_pth = resolve(d["gat_model_pth"])

        # TCGA reference
        self.tcga_expression_tsv = resolve(d["tcga_expression_tsv"])

        # Pre-trained response model (optional — skips Step 1)
        self.response_model_pth = resolve(d.get("response_model_pth"))

        # Pre-training data (only needed if response_model_pth is null)
        self.pretrain_expression_tsv = resolve(d.get("pretrain_expression_tsv"))
        self.pretrain_clinical_tsv = resolve(d.get("pretrain_clinical_tsv"))

        # Fine-tuning data (optional)
        self.finetune_expression_tsv = resolve(d.get("finetune_expression_tsv"))
        self.finetune_clinical_tsv = resolve(d.get("finetune_clinical_tsv"))
        self.finetune_abundance_tsv = resolve(d.get("finetune_abundance_tsv"))

        # Output
        self.output_dir = resolve(d["output_dir"])

        # Clinical format
        self.sample_id_column = d.get("sample_id_column", "ID")
        self.response_column = d.get("response_column", "Response")
        self.response_mapping = d.get("response_mapping", {"R": 1, "NR": 0})

        # GAT architecture (must match Component 2)
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

        # Pre-training
        self.pretrain_epochs = int(d.get("pretrain_epochs", 100))
        self.pretrain_lr = float(d.get("pretrain_lr", 0.001))
        self.pretrain_batch_size = int(d.get("pretrain_batch_size", 32))
        self.pretrain_weight_decay = float(d.get("pretrain_weight_decay", 1e-4))

        # Fine-tuning
        self.finetune_epochs = int(d.get("finetune_epochs", 20))
        self.finetune_lr_ecotyper = float(d.get("finetune_lr_ecotyper", 1e-4))
        self.finetune_lr_response = float(d.get("finetune_lr_response", 1e-8))
        self.finetune_batch_size = int(d.get("finetune_batch_size", 4))
        self.finetune_weight_decay = float(d.get("finetune_weight_decay", 1e-8))

        # Number of output classes (derived from response mapping)
        self.num_classes = len(set(self.response_mapping.values()))

        # Other
        self.random_seed = d.get("random_seed", 123)
        self.top_k_edges = d.get("top_k_edges", 5)

    @property
    def finetune_mode(self):
        """Auto-detect fine-tuning mode from provided inputs."""
        has_expr = self.finetune_expression_tsv is not None
        has_clin = self.finetune_clinical_tsv is not None
        has_abund = self.finetune_abundance_tsv is not None
        if has_expr and has_clin and has_abund:
            return "iterative"
        elif has_expr and has_clin:
            return "response_only"
        else:
            return "none"

    def validate(self, steps):
        """Check that required input files exist."""
        errors = []
        if not os.path.exists(self.graph_pkl):
            errors.append(f"Graph not found: {self.graph_pkl}")
        if not os.path.exists(self.gat_model_pth):
            errors.append(f"GAT model not found: {self.gat_model_pth}")
        if not os.path.exists(self.tcga_expression_tsv):
            errors.append(f"TCGA expression not found: {self.tcga_expression_tsv}")
        if 1 in steps and self.response_model_pth is None:
            if self.pretrain_expression_tsv and not os.path.exists(self.pretrain_expression_tsv):
                errors.append(f"Pre-train expression not found: "
                              f"{self.pretrain_expression_tsv}")
            if self.pretrain_clinical_tsv and not os.path.exists(self.pretrain_clinical_tsv):
                errors.append(f"Pre-train clinical not found: "
                              f"{self.pretrain_clinical_tsv}")
        if self.response_model_pth and not os.path.exists(self.response_model_pth):
            errors.append(f"Response model not found: {self.response_model_pth}")
        if 2 in steps and self.finetune_mode != "none":
            if not os.path.exists(self.finetune_expression_tsv):
                errors.append(f"Fine-tune expression not found: "
                              f"{self.finetune_expression_tsv}")
            if not os.path.exists(self.finetune_clinical_tsv):
                errors.append(f"Fine-tune clinical not found: "
                              f"{self.finetune_clinical_tsv}")
            if (self.finetune_mode == "iterative" and
                    not os.path.exists(self.finetune_abundance_tsv)):
                errors.append(f"Fine-tune abundance not found: "
                              f"{self.finetune_abundance_tsv}")
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

    def summary(self):
        """Print configuration summary."""
        print(f"  Graph:           {self.graph_pkl}")
        print(f"  GAT model:       {self.gat_model_pth}")
        print(f"  TCGA reference:  {self.tcga_expression_tsv}")
        if self.response_model_pth:
            print(f"  Response model:  {self.response_model_pth} (skip pre-training)")
        else:
            print(f"  Pre-train expr:  {self.pretrain_expression_tsv}")
            print(f"  Pre-train clin:  {self.pretrain_clinical_tsv}")
        print(f"  Fine-tune mode:  {self.finetune_mode}")
        if self.finetune_mode != "none":
            print(f"  Fine-tune expr:  {self.finetune_expression_tsv}")
            print(f"  Fine-tune clin:  {self.finetune_clinical_tsv}")
            if self.finetune_mode == "iterative":
                print(f"  Fine-tune abund: {self.finetune_abundance_tsv}")


# =============================================================================
# Models
# =============================================================================

class EcotypeClassifier(nn.Module):
    """GAT model for ecotype abundance prediction (same architecture as Component 2)."""
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


class ResponsePredictor(nn.Module):
    """MLP for binary response prediction from ecotype features."""
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
# Data utilities
# =============================================================================

def zscore_normalize(df):
    """Per-gene z-score normalization across samples (genes as rows)."""
    means = df.mean(axis=1)
    stds = df.std(axis=1).replace(0, 1e-6)
    return df.sub(means, axis=0).div(stds, axis=0)


def compute_class_weights(labels, num_classes, device):
    """Compute inverse-frequency class weights to balance training."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0  # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def derive_gene_list(graph_pkl, tcga_tsv):
    """Derive gene list from intersection of graph nodes and TCGA genes."""
    with open(graph_pkl, "rb") as f:
        graph = pickle.load(f)
    tcga = pd.read_csv(tcga_tsv, index_col=0, sep="\t")
    tcga_genes = set(tcga.index)
    gene_list = [g for g in graph.nodes() if g in tcga_genes]
    return graph, gene_list


def load_graph_edges(graph, gene_list, device):
    """Build edge_index tensor from graph and gene list."""
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    edges = [(gene_to_idx[u], gene_to_idx[v])
             for u, v in graph.edges()
             if u in gene_to_idx and v in gene_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    return edges, edge_index


def impute_expression(target_tsv, gene_list, tcga_normed, knn_neighbors,
                      normalize=True):
    """Z-score normalize, align genes, and KNN-impute missing genes.

    Args:
        target_tsv: path to raw expression (genes x samples, TSV)
        gene_list: list of genes the model expects
        tcga_normed: z-score normalized TCGA reference (genes x samples -> transposed to samples x genes)
        knn_neighbors: number of KNN neighbors
        normalize: whether to z-score normalize target data

    Returns:
        imputed_df: DataFrame (samples x genes), imputed and aligned
    """
    target_raw = pd.read_csv(target_tsv, sep="\t", index_col=0).dropna(how="all")
    if normalize:
        target_normed = zscore_normalize(target_raw)
    else:
        target_normed = target_raw

    # Align to gene list (missing genes become NaN)
    aligned = pd.DataFrame(index=gene_list, columns=target_normed.columns)
    genes_available = target_normed.index.intersection(gene_list)
    aligned.loc[genes_available] = target_normed.loc[genes_available]

    # Combine with TCGA reference and impute
    combined = pd.concat([tcga_normed, aligned.T], axis=0)
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    imputed = imputer.fit_transform(combined)
    imputed_df = pd.DataFrame(
        imputed, index=combined.index, columns=combined.columns
    ).loc[aligned.columns]

    return imputed_df


def load_clinical(clinical_tsv, sample_id_col, response_col, response_mapping):
    """Load clinical data and map response labels to binary."""
    clinical = pd.read_csv(clinical_tsv, sep="\t", dtype=str)
    clinical = clinical[clinical[response_col].isin(response_mapping.keys())]
    clinical["Label"] = clinical[response_col].map(response_mapping).astype(int)
    clinical.set_index(sample_id_col, inplace=True)
    return clinical


def load_ecotype_model(cfg, num_nodes, device):
    """Load pretrained EcotypeClassifier."""
    model = EcotypeClassifier(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.num_ecotypes,
        num_nodes=num_nodes,
        heads_1=cfg.gat_heads_1,
        heads_2=cfg.gat_heads_2,
        fc_dim=cfg.fc_dim,
        dropout=cfg.gat_dropout,
    ).to(device)
    model.load_state_dict(torch.load(cfg.gat_model_pth, map_location=device))
    model.eval()
    return model


# =============================================================================
# Step 1: Pre-train ResponsePredictor
# =============================================================================

def run_step1(cfg, graph, gene_list, edges, edge_index, tcga_normed, device):
    """Pre-train ResponsePredictor on pan-cancer immunotherapy data."""
    print(f"\n{'='*60}")
    print("Step 1: Pre-train ResponsePredictor")
    print(f"{'='*60}")

    model_path = os.path.join(cfg.output_dir, "response_model.pth")
    if os.path.exists(model_path):
        print(f"  Cached: {model_path} exists, skipping.")
        return

    # If a pre-trained response model is provided, copy it and skip training
    if cfg.response_model_pth:
        shutil.copy2(cfg.response_model_pth, model_path)
        print(f"  Using pre-trained response model: {cfg.response_model_pth}")
        print(f"  Copied to: {model_path}")
        return

    # 1a. Impute pre-training expression
    print("\n  Imputing pre-training expression data...")
    imputed_df = impute_expression(
        cfg.pretrain_expression_tsv, gene_list, tcga_normed,
        cfg.knn_neighbors)
    print(f"  Imputed: {imputed_df.shape[0]} samples x {imputed_df.shape[1]} genes")

    # 1b. Load clinical labels
    clinical = load_clinical(
        cfg.pretrain_clinical_tsv, cfg.sample_id_column,
        cfg.response_column, cfg.response_mapping)
    common = imputed_df.index.intersection(clinical.index)
    print(f"  Samples with expression + response labels: {len(common)}")

    X = torch.tensor(imputed_df.loc[common].values,
                     dtype=torch.float32).to(device)
    y = torch.tensor(clinical.loc[common, "Label"].values,
                     dtype=torch.long).to(device)

    # 1c. Load pretrained GAT and get ecotype features
    print("  Loading pretrained GAT...")
    ecotype_model = load_ecotype_model(cfg, len(gene_list), device)

    print("  Computing ecotype features...")
    ecotype_model.eval()
    ecotype_features_list = []
    batch_size = cfg.pretrain_batch_size
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size].unsqueeze(2)
            logits, _ = ecotype_model(x_batch, edge_index)
            features = F.softmax(logits, dim=1)
            ecotype_features_list.append(features)
    ecotype_features = torch.cat(ecotype_features_list, dim=0)
    print(f"  Ecotype features: {ecotype_features.shape}")

    # Save ecotype predictions
    eco_pred_path = os.path.join(cfg.output_dir, "pretrain_ecotype_predictions.txt")
    eco_pred_df = pd.DataFrame(
        ecotype_features.cpu().numpy(),
        index=common,
        columns=[f"E{i+1}" for i in range(cfg.num_ecotypes)])
    eco_pred_df.T.to_csv(eco_pred_path, sep="\t")
    print(f"  Saved: {eco_pred_path}")

    # 1d. Train ResponsePredictor
    set_seed(cfg.random_seed)
    print(f"\n  Training ResponsePredictor ({cfg.pretrain_epochs} epochs)...")
    response_model = ResponsePredictor(
        input_size=cfg.num_ecotypes,
        output_size=cfg.num_classes,
        hidden_dims=cfg.response_hidden_dims,
        dropout=cfg.response_dropout,
    ).to(device)

    class_w = compute_class_weights(y.cpu().numpy(), cfg.num_classes, device)
    print(f"  Class weights: {class_w.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.Adam(response_model.parameters(), lr=cfg.pretrain_lr,
                           weight_decay=cfg.pretrain_weight_decay)

    dataset = TensorDataset(ecotype_features, y)
    loader = DataLoader(dataset, batch_size=cfg.pretrain_batch_size, shuffle=True)

    response_model.train()
    for epoch in range(cfg.pretrain_epochs):
        total_loss = 0
        for feat_batch, y_batch in loader:
            optimizer.zero_grad()
            out = response_model(feat_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1}/{cfg.pretrain_epochs}], "
                  f"Loss: {total_loss:.4f}")

    # Evaluate on training set
    response_model.eval()
    with torch.no_grad():
        pred_log = response_model(ecotype_features)
        pred_probs = torch.exp(pred_log)
        pred_cls = pred_probs.argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()
    acc = accuracy_score(y_np, pred_cls)
    auc_score = _compute_auc(y_np, pred_probs.cpu().numpy(), cfg.num_classes)
    print(f"\n  Pre-training results (train set):")
    print(f"    Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

    # Save
    torch.save(response_model.state_dict(), model_path)
    print(f"  Saved: {model_path}")


# =============================================================================
# Step 2: Fine-tune
# =============================================================================

def run_step2(cfg, graph, gene_list, edges, edge_index, tcga_normed, device):
    """Fine-tune models on cancer-specific data."""
    print(f"\n{'='*60}")
    print(f"Step 2: Fine-tune ({cfg.finetune_mode} mode)")
    print(f"{'='*60}")

    if cfg.finetune_mode == "none":
        print("  No fine-tune data provided, skipping.")
        return

    ft_model_path = os.path.join(cfg.output_dir, "finetuned_response_model.pth")
    if os.path.exists(ft_model_path):
        print(f"  Cached: {ft_model_path} exists, skipping.")
        return

    # 2a. Impute fine-tuning expression
    print("\n  Imputing fine-tune expression data...")
    imputed_df = impute_expression(
        cfg.finetune_expression_tsv, gene_list, tcga_normed,
        cfg.knn_neighbors)
    print(f"  Imputed: {imputed_df.shape[0]} samples x {imputed_df.shape[1]} genes")

    # 2b. Load clinical labels
    clinical = load_clinical(
        cfg.finetune_clinical_tsv, cfg.sample_id_column,
        cfg.response_column, cfg.response_mapping)
    common = imputed_df.index.intersection(clinical.index)

    # 2c. Load ecotype abundance if iterative mode
    if cfg.finetune_mode == "iterative":
        abund_df = pd.read_csv(cfg.finetune_abundance_tsv, sep="\t",
                               index_col=0).T
        common = common.intersection(abund_df.index)

    print(f"  Training samples: {len(common)}")

    X = torch.tensor(imputed_df.loc[common].values,
                     dtype=torch.float32).to(device)
    y_response = torch.tensor(clinical.loc[common, "Label"].values,
                              dtype=torch.long).to(device)

    # 2d. Load models
    print("  Loading pretrained models...")
    ecotype_model = load_ecotype_model(cfg, len(gene_list), device)

    response_model = ResponsePredictor(
        input_size=cfg.num_ecotypes,
        output_size=cfg.num_classes,
        hidden_dims=cfg.response_hidden_dims,
        dropout=cfg.response_dropout,
    ).to(device)
    response_model_path = os.path.join(cfg.output_dir, "response_model.pth")
    response_model.load_state_dict(
        torch.load(response_model_path, map_location=device))

    # 2e. Fine-tune
    if cfg.finetune_mode == "iterative":
        y_ecotype = torch.tensor(abund_df.loc[common].values,
                                 dtype=torch.float32).to(device)
        ecotype_model, response_model, attn_scores = _finetune_iterative(
            ecotype_model, response_model, edge_index,
            X, y_ecotype, y_response, cfg, device)
    else:
        ecotype_model, response_model, attn_scores = _finetune_response_only(
            ecotype_model, response_model, edge_index,
            X, y_response, cfg, device)

    # 2f. Save models
    ft_eco_path = os.path.join(cfg.output_dir, "finetuned_ecotype_model.pth")
    torch.save(ecotype_model.state_dict(), ft_eco_path)
    print(f"  Saved: {ft_eco_path}")

    torch.save(response_model.state_dict(), ft_model_path)
    print(f"  Saved: {ft_model_path}")

    # 2g. Save predictions on fine-tune set
    ecotype_model.eval()
    response_model.eval()
    with torch.no_grad():
        logits, _ = ecotype_model(X.unsqueeze(2), edge_index)
        eco_feat = F.softmax(logits, dim=1)
        pred_log = response_model(eco_feat)
        pred_probs = torch.exp(pred_log)
        pred_cls = pred_probs.argmax(dim=1).cpu().numpy()

    y_np = y_response.cpu().numpy()
    acc = accuracy_score(y_np, pred_cls)
    auc_score = _compute_auc(y_np, pred_probs.cpu().numpy(), cfg.num_classes)
    print(f"\n  Fine-tuning results (train set):")
    print(f"    Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

    pred_path = os.path.join(cfg.output_dir, "finetune_predictions.csv")
    pred_dict = {"SampleID": list(common), "TrueLabel": y_np, "PredClass": pred_cls}
    probs_np = pred_probs.cpu().numpy()
    for c in range(cfg.num_classes):
        pred_dict[f"Prob_class{c}"] = probs_np[:, c]
    pd.DataFrame(pred_dict).to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    # 2h. Save attention scores
    _save_attention_scores(cfg, gene_list, edges, attn_scores)


def _finetune_response_only(ecotype_model, response_model, edge_index,
                            X, y_response, cfg, device):
    """Freeze GAT, fine-tune ResponsePredictor only."""
    set_seed(cfg.random_seed)
    print(f"\n  Fine-tuning ResponsePredictor only "
          f"({cfg.finetune_epochs} epochs)...")

    ecotype_model.eval()  # frozen
    response_model.train()

    class_w = compute_class_weights(y_response.cpu().numpy(), cfg.num_classes, device)
    print(f"  Class weights: {class_w.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.Adam(response_model.parameters(),
                           lr=cfg.finetune_lr_response,
                           weight_decay=cfg.finetune_weight_decay)

    dataset = TensorDataset(X, y_response)
    loader = DataLoader(dataset, batch_size=cfg.finetune_batch_size,
                        shuffle=True)

    for epoch in range(cfg.finetune_epochs):
        total_loss = 0
        response_model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.unsqueeze(2)
            with torch.no_grad():
                eco_feat = F.softmax(
                    ecotype_model(x_batch, edge_index)[0], dim=1)
            optimizer.zero_grad()
            out = response_model(eco_feat)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch [{epoch+1}/{cfg.finetune_epochs}], "
                  f"Loss: {total_loss:.4f}")

    print("  Fine-tuning finished.")

    ecotype_model.eval()
    with torch.no_grad():
        _, attn_scores = ecotype_model(X.unsqueeze(2), edge_index)
        attn_scores = attn_scores.cpu().numpy()

    return ecotype_model, response_model, attn_scores


def _finetune_iterative(ecotype_model, response_model, edge_index,
                        X, y_ecotype, y_response, cfg, device):
    """Iteratively fine-tune both GAT and ResponsePredictor."""
    set_seed(cfg.random_seed)
    print(f"\n  Iterative fine-tuning of both models "
          f"({cfg.finetune_epochs} epochs)...")

    class_w = compute_class_weights(y_response.cpu().numpy(), cfg.num_classes, device)
    print(f"  Class weights: {class_w.cpu().tolist()}")
    criterion_eco = nn.KLDivLoss(reduction="batchmean")
    criterion_res = nn.CrossEntropyLoss(weight=class_w)

    optimizer_eco = optim.Adam(ecotype_model.parameters(),
                               lr=cfg.finetune_lr_ecotyper,
                               weight_decay=cfg.finetune_weight_decay)
    optimizer_res = optim.Adam(response_model.parameters(),
                               lr=cfg.finetune_lr_response,
                               weight_decay=cfg.finetune_weight_decay)

    dataset = TensorDataset(X, y_ecotype, y_response)
    loader = DataLoader(dataset, batch_size=cfg.finetune_batch_size,
                        shuffle=True)

    for epoch in range(cfg.finetune_epochs):
        ecotype_model.train()
        response_model.train()
        total_eco_loss = 0
        total_res_loss = 0

        for x_batch, y1_batch, y2_batch in loader:
            x_batch = x_batch.unsqueeze(2)

            # Step 1: update ecotype model
            optimizer_eco.zero_grad()
            logits, _ = ecotype_model(x_batch, edge_index)
            log_probs = F.log_softmax(logits, dim=1)
            loss_eco = criterion_eco(log_probs, y1_batch)
            loss_eco.backward()
            optimizer_eco.step()
            total_eco_loss += loss_eco.item()

            # Step 2: update response model
            with torch.no_grad():
                eco_feat = F.softmax(logits, dim=1)
            optimizer_res.zero_grad()
            out = response_model(eco_feat)
            loss_res = criterion_res(out, y2_batch)
            loss_res.backward()
            optimizer_res.step()
            total_res_loss += loss_res.item()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch [{epoch+1}/{cfg.finetune_epochs}], "
                  f"Ecotype Loss: {total_eco_loss:.4f}, "
                  f"Response Loss: {total_res_loss:.4f}")

    print("  Iterative fine-tuning finished.")

    ecotype_model.eval()
    with torch.no_grad():
        _, attn_scores = ecotype_model(X.unsqueeze(2), edge_index)
        attn_scores = attn_scores.cpu().numpy()

    return ecotype_model, response_model, attn_scores


# =============================================================================
# Attention score utilities
# =============================================================================

def _compute_auc(y_true, pred_probs, num_classes):
    """Compute AUC — binary or multi-class (one-vs-rest)."""
    try:
        if num_classes == 2:
            return roc_auc_score(y_true, pred_probs[:, 1])
        else:
            return roc_auc_score(y_true, pred_probs, multi_class="ovr")
    except ValueError:
        return float("nan")


def _save_attention_scores(cfg, gene_list, edges, attn_scores):
    """Save edge and node attention scores (sum, ave, top-k sum, top-k ave)."""
    print("\n  Computing attention scores...")
    mean_attn = attn_scores.mean(axis=0)
    edge_list_named = [(gene_list[u], gene_list[v]) for u, v in edges]

    # Edge attention
    edge_attn_list = [(f"{u}--{v}", a)
                      for (u, v), a in zip(edge_list_named, mean_attn)]
    edge_attn_list.sort(key=lambda x: x[1], reverse=True)
    edge_path = os.path.join(cfg.output_dir, "edge_attn.txt")
    with open(edge_path, "w") as f:
        f.write("Edge\tMeanAttention\n")
        for name, a in edge_attn_list:
            f.write(f"{name}\t{a:.6f}\n")

    # Node total (sum) attention
    node_attn = {}
    node_edge_count = {}
    for (u, v), a in zip(edge_list_named, mean_attn):
        node_attn[u] = node_attn.get(u, 0) + a
        node_attn[v] = node_attn.get(v, 0) + a
        node_edge_count[u] = node_edge_count.get(u, 0) + 1
        node_edge_count[v] = node_edge_count.get(v, 0) + 1

    _write_node_attn(os.path.join(cfg.output_dir, "node_sum_attn.txt"),
                     node_attn, "TotalEdgeAttention")

    # Node average attention
    node_ave = {n: node_attn[n] / node_edge_count[n] for n in node_attn}
    _write_node_attn(os.path.join(cfg.output_dir, "node_ave_attn.txt"),
                     node_ave, "AverageEdgeAttention")

    # Per-node edge-attention lists, used by the top-k summaries
    node_edge_attn_map = defaultdict(list)
    for (u, v), a in zip(edge_list_named, mean_attn):
        node_edge_attn_map[u].append(a)
        node_edge_attn_map[v].append(a)

    # Node top-k sum attention
    k = cfg.top_k_edges
    node_topk_sum = {n: sum(sorted(al, reverse=True)[:k])
                     for n, al in node_edge_attn_map.items()}
    _write_node_attn(os.path.join(cfg.output_dir, "node_topk_sum_attn.txt"),
                     node_topk_sum, f"Top{k}EdgeAttentionSum")

    # Node top-k average attention
    node_topk_ave = {}
    for n, al in node_edge_attn_map.items():
        top = sorted(al, reverse=True)[:k]
        node_topk_ave[n] = sum(top) / len(top)
    _write_node_attn(os.path.join(cfg.output_dir, "node_topk_ave_attn.txt"),
                     node_topk_ave, f"Top{k}EdgeAttentionAve")

    print(f"  Saved: attention scores -> {cfg.output_dir}/")


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
        description="EcoNet: Transfer learning & response prediction")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--step", nargs="*", type=int, default=[1, 2],
                        help="Steps to run (default: 1 2)")
    args = parser.parse_args()

    cfg = Config(args.config)
    steps = sorted(set(args.step))
    cfg.validate(steps)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("EcoNet Transfer Learning & Response Prediction Pipeline")
    print(f"Config: {args.config}")
    print(f"Steps:  {steps}")
    print(f"Output: {cfg.output_dir}")
    cfg.summary()

    set_seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load graph and derive gene list
    print(f"\nDeriving gene list from graph + TCGA expression...")
    graph, gene_list = derive_gene_list(cfg.graph_pkl, cfg.tcga_expression_tsv)
    print(f"  Genes: {len(gene_list)}")

    edges, edge_index = load_graph_edges(graph, gene_list, device)
    print(f"  Edges: {len(edges)}")

    # Prepare TCGA reference (z-score normalized, transposed to samples x genes)
    print(f"  Z-score normalizing TCGA reference...")
    tcga_raw = pd.read_csv(cfg.tcga_expression_tsv, index_col=0, sep="\t")
    tcga_normed = zscore_normalize(tcga_raw).reindex(gene_list).fillna(0).T

    if 1 in steps:
        run_step1(cfg, graph, gene_list, edges, edge_index, tcga_normed,
                  device)
    if 2 in steps:
        run_step2(cfg, graph, gene_list, edges, edge_index, tcga_normed,
                  device)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

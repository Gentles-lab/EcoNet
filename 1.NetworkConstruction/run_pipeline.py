#!/usr/bin/env python
"""
EcoNet Network Construction Pipeline
======================================
Build an intercellular regulatory network from scRNA-seq data and EcoTyper
ecotype definitions. Outputs a global directed gene graph (global_graph.pkl)
for downstream GAT model training.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --step 1      # run only step 1
    python run_pipeline.py --config config.yaml --step 2 3    # run steps 2 and 3
"""

import os
import re
import sys
import glob
import yaml
import argparse
import itertools
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from collections import defaultdict
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from scipy.stats import ranksums, trim_mean, fisher_exact
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration loader
# =============================================================================

class Config:
    """Load configuration from YAML; auto-discover EcoTyper model structure."""

    def __init__(self, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(path))

        def resolve(p):
            return p if os.path.isabs(p) else os.path.join(base, p)

        # Input data
        self.scrna_h5ad = resolve(d["scrna_h5ad"])

        nd = resolve(d["nichenet_dir"])
        self.lr_network = os.path.join(nd, "lr_network.csv")
        self.weighted_lr_sig = os.path.join(nd, "weighted_lr_sig.csv")
        self.weighted_gr = os.path.join(nd, "weighted_gr.csv")
        self.ligand_target_matrix = os.path.join(nd, "ligand_target_matrix.csv")

        self.output_dir = resolve(d["output_dir"])

        # Ecotype abundance (provided directly in config)
        self.ecotype_abundance = resolve(d["ecotype_abundance"])

        # Auto-discover EcoTyper model structure
        ecotyper_dir = resolve(d["ecotyper_dir"])
        self._parse_ecotyper(ecotyper_dir)

        # scRNA-seq columns
        self.cell_state_column = d.get("cell_state_column", "CellType_State")
        self.sample_column = d.get("sample_column", "Sample")

        # Pipeline parameters
        self.overexpr_thresh_p = d.get("overexpr_thresh_p", 100)
        self.lri_variable_both = d.get("lri_variable_both", False)
        self.n_boot = d.get("n_boot", 100)
        self.hill_kh = d.get("hill_kh", 0.5)
        self.hill_n = d.get("hill_n", 1)
        self.comm_thresh_prob = d.get("comm_thresh_prob", 0)
        self.comm_thresh_pval = d.get("comm_thresh_pval", 0.05)

        self.top_ligands_k = d.get("top_ligands_k", 5)
        self.expression_threshold = d.get("expression_threshold", 1)
        self.top_mediators_n = d.get("top_mediators_n", 2)

        self.n_jobs = d.get("n_jobs", -1)
        self.noisy_genes = d.get("noisy_genes", {})

    def _parse_ecotyper(self, ecotyper_dir):
        """Auto-discover ecotypes and marker gene sources from EcoTyper.

        Searches for Cell_States/ and Ecotypes/ either directly under
        ecotyper_dir or inside a subfolder (e.g. Carcinoma_Fractions/).
        Reads Ecotypes/discovery/ecotypes.txt for ecotype-to-cell-state
        mappings. Marker genes are extracted at runtime from Cell_States
        gene_info.txt files (auto-detecting the optimal NMF rank directory).
        """
        # Auto-discover: find the directory containing Cell_States/ and Ecotypes/
        self.ecotyper_dir = ecotyper_dir
        self._frac_dir = self._find_model_root(ecotyper_dir)
        if self._frac_dir is None:
            # EcoTyper model not found; defer to validate() for a clean error
            # instead of crashing here during Config construction.
            self._ecotypes_df = None
            self.ecotypes = []
            return

        # Read ecotypes.txt to get ecotype -> cell state mappings
        ecotypes_file = os.path.join(
            self._frac_dir, "Ecotypes", "discovery", "ecotypes.txt")
        self._ecotypes_df = pd.read_csv(ecotypes_file, sep="\t")

        # Both EcoTyper formats (Carcinoma and CCRCC) produce columns
        # CellType, State, ID, Ecotype after pandas auto-detects the
        # R-style row-name index column.
        self._eco_id_col = "ID"
        self._eco_celltype_col = "CellType"
        self._eco_state_col = "State"
        self._eco_ecotype_col = "Ecotype"

        # Discover unique ecotype IDs
        self.ecotypes = sorted(
            self._ecotypes_df[self._eco_ecotype_col].unique().tolist(),
            key=lambda x: (len(x), x))

    @staticmethod
    def _find_model_root(ecotyper_dir):
        """Find the directory containing Cell_States/ and Ecotypes/.

        Checks ecotyper_dir itself first, then any immediate subdirectory
        (e.g. Carcinoma_Fractions/).
        """
        if not os.path.isdir(ecotyper_dir):
            return None
        for candidate in [ecotyper_dir]:
            if (os.path.isdir(os.path.join(candidate, "Cell_States")) and
                    os.path.isdir(os.path.join(candidate, "Ecotypes"))):
                return candidate
        # Check one level of subdirectories
        for entry in os.listdir(ecotyper_dir):
            candidate = os.path.join(ecotyper_dir, entry)
            if (os.path.isdir(candidate) and
                    os.path.isdir(os.path.join(candidate, "Cell_States")) and
                    os.path.isdir(os.path.join(candidate, "Ecotypes"))):
                return candidate
        return None

    def extract_marker_genes(self):
        """Extract marker genes from EcoTyper Cell_States into output dir.

        For each cell type, auto-detects the optimal NMF rank directory
        (the one containing gene_info.txt), then extracts state-specific
        genes based on the ecotype-to-state mapping.

        Returns the path to the generated marker genes directory.
        """
        marker_dir = os.path.join(self.output_dir, "marker_genes")
        cell_states_dir = os.path.join(
            self._frac_dir, "Cell_States", "discovery")

        # Parse ecotypes_df into (cell_type, state, state_id, ecotype) tuples
        records = []
        for _, row in self._ecotypes_df.iterrows():
            cell_type = row[self._eco_celltype_col]
            state = row[self._eco_state_col]
            state_id = row[self._eco_id_col]
            ecotype = row[self._eco_ecotype_col]
            records.append((cell_type, state, state_id, ecotype))

        # Auto-detect optimal NMF rank dir per cell type:
        # it's the numbered subdirectory containing gene_info.txt
        cell_type_dirs = {}
        if os.path.isdir(cell_states_dir):
            for ct in os.listdir(cell_states_dir):
                ct_path = os.path.join(cell_states_dir, ct)
                if not os.path.isdir(ct_path):
                    continue
                for sub in os.listdir(ct_path):
                    gi = os.path.join(ct_path, sub, "gene_info.txt")
                    if os.path.isfile(gi):
                        cell_type_dirs[ct] = gi
                        break

        # Extract marker genes
        ecotype_genes = {}
        for cell_type, state, state_id, ecotype in records:
            gi_path = cell_type_dirs.get(cell_type)
            if gi_path is None:
                print(f"  Warning: no gene_info.txt for {cell_type}, skipping.")
                continue

            gi_df = pd.read_csv(gi_path, sep="\t", index_col=0)
            state_genes = gi_df[gi_df["State"] == state]["Gene"].values

            eco_dir = os.path.join(marker_dir, f"{ecotype}marker")
            os.makedirs(eco_dir, exist_ok=True)

            out_file = os.path.join(eco_dir, f"{state_id}.txt")
            with open(out_file, "w") as f:
                for gene in state_genes:
                    f.write(gene + "\n")

            ecotype_genes.setdefault(ecotype, set()).update(state_genes)

        # Write consolidated marker files
        for ecotype, genes in ecotype_genes.items():
            eco_dir = os.path.join(marker_dir, f"{ecotype}marker")
            with open(os.path.join(eco_dir, f"{ecotype}_markers.txt"), "w") as f:
                for gene in sorted(genes):
                    f.write(gene + "\n")

        self.marker_genes_dir = marker_dir
        total = sum(len(g) for g in ecotype_genes.values())
        print(f"  Extracted marker genes for {len(ecotype_genes)} ecotypes "
              f"({total} total gene entries) -> {marker_dir}")

    def validate(self, steps):
        """Check that required input files exist for the requested steps."""
        errors = []
        if getattr(self, "_frac_dir", None) is None:
            errors.append(
                f"EcoTyper model not found: could not locate Cell_States/ and "
                f"Ecotypes/ in {self.ecotyper_dir} or its subdirectories")
        if 1 in steps:
            if not os.path.exists(self.scrna_h5ad):
                errors.append(f"scRNA h5ad not found: {self.scrna_h5ad}")
            if not os.path.exists(self.lr_network):
                errors.append(f"LR network not found: {self.lr_network}")
        if 2 in steps or 3 in steps:
            if not os.path.exists(self.ecotype_abundance):
                errors.append(f"Ecotype abundance not found: "
                              f"{self.ecotype_abundance}")
        if 2 in steps:
            if not os.path.exists(self.ligand_target_matrix):
                errors.append(f"Ligand-target matrix not found: "
                              f"{self.ligand_target_matrix}")
        if 3 in steps:
            if not os.path.exists(self.weighted_lr_sig):
                errors.append(f"Weighted LR-sig not found: "
                              f"{self.weighted_lr_sig}")
            if not os.path.exists(self.weighted_gr):
                errors.append(f"Weighted GR not found: {self.weighted_gr}")
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

    def summary(self):
        """Print discovered configuration."""
        print(f"  Ecotype abundance:   {self.ecotype_abundance}")
        print(f"  Ecotypes discovered: {self.ecotypes}")


# =============================================================================
# Shared utilities
# =============================================================================

def get_representative_sample(ecotype, abundance_path):
    df = pd.read_csv(abundance_path, sep="\t", index_col=0)
    return df.idxmax(axis=1).loc[ecotype]


def get_cell_states(marker_dir, ecotype):
    d = os.path.join(marker_dir, f"{ecotype}marker")
    if not os.path.isdir(d):
        return []
    return [f.rstrip(".txt") for f in os.listdir(d)
            if f.endswith(".txt") and not f.startswith(f"{ecotype}_markers")]


def get_expressed_genes(adata, target_cell_type, group_by, threshold=1,
                        noisy_genes=None):
    subset = adata[adata.obs[group_by] == target_cell_type]
    linear = np.expm1(subset.X)
    mean_expr = np.asarray(np.mean(linear, axis=0)).flatten()
    norm_expr = np.log2(10 * mean_expr + 1)
    expressed = adata.var_names[norm_expr >= threshold].tolist()

    if noisy_genes:
        prefix = target_cell_type.split("_")[0]
        for gene in noisy_genes.get(prefix, []):
            if gene in expressed:
                expressed.remove(gene)
    return expressed


# =============================================================================
# Step 1: Identify overexpressed ligand-receptor interactions
# =============================================================================

class CommunicationPipeline:
    def __init__(self, adata, group_by, lr_network_path):
        self.adata = adata
        self.group_by = group_by
        self.lr_network_path = lr_network_path

    def identify_overexpressed_genes(self, thresh_p=0.05, n_jobs=-1):
        adata = self.adata
        labels = adata.obs[self.group_by]
        unique_groups = labels.unique()
        features = adata.var_names
        gene_expr = csr_matrix(adata[:, features].X)

        def process_group(group):
            mask = (labels == group).values
            g_expr = gene_expr[mask]
            o_expr = gene_expr[~mask]
            pct_g = (g_expr > 0).mean(axis=0).A1
            pct_o = (o_expr > 0).mean(axis=0).A1
            valid = (pct_g > 0) | (pct_o > 0)
            vf = np.array(features)[valid]
            ge, oe = g_expr[:, valid], o_expr[:, valid]
            mg = np.log1p(np.expm1(ge).mean(axis=0).A1)
            mo = np.log1p(np.expm1(oe).mean(axis=0).A1)
            fc = mg - mo
            pv = np.array([ranksums(ge[:, i].toarray().flatten(),
                                    oe[:, i].toarray().flatten())[1]
                           for i in range(ge.shape[1])])
            padj = multipletests(pv, method='bonferroni')[1]
            sig = padj < thresh_p
            return pd.DataFrame({'gene': vf[sig], 'logFC': fc[sig],
                                 'p_value': pv[sig], 'p_adj': padj[sig],
                                 'group': group})

        self.overexpressed_genes = pd.concat(
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_group)(g) for g in unique_groups))

    def identify_overexpressed_interactions(self, variable_both=True):
        valid_genes = set(self.adata.var_names)
        lr = pd.read_csv(self.lr_network_path, index_col=0)
        lr = lr[(lr['from'].isin(valid_genes)) & (lr['to'].isin(valid_genes))]
        oe_set = set(self.overexpressed_genes['gene'])
        if variable_both:
            mask = lr.apply(lambda r: r['from'] in oe_set and r['to'] in oe_set, axis=1)
        else:
            mask = lr.apply(lambda r: r['from'] in oe_set or r['to'] in oe_set, axis=1)
        self.overexpressed_lri = lr[mask]

    def compute_communication_probabilities(self, nboot=100, Kh=0.5, n=1,
                                            thresh_prob=0, thresh_pval=0.05,
                                            n_jobs=-1):
        np.random.seed(1)
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        data = X.T
        group = self.adata.obs[self.group_by].astype("category")
        ugroups = group.cat.categories
        nc = len(ugroups)

        def penta(x):
            return np.mean([np.percentile(x, p) for p in [2, 25, 50, 75, 98]])

        def proc_g(g):
            return np.apply_along_axis(penta, 1, data[:, group == g])

        data_avg = np.column_stack(
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(proc_g)(g) for g in ugroups))

        lr_pairs = self.overexpressed_lri
        li = [list(self.adata.var.index).index(l) for l in lr_pairs['from']]
        ri = [list(self.adata.var.index).index(r) for r in lr_pairs['to']]
        P_spatial = np.ones((nc, nc))

        prob = np.zeros((nc, nc, len(lr_pairs)))
        pval = np.zeros((nc, nc, len(lr_pairs)))
        perms = [np.random.permutation(group.values) for _ in range(nboot)]

        def boot_avg(perm):
            gi = {g: np.where(perm == g)[0] for g in ugroups}
            return np.array([data[:, idx].mean(axis=1)
                             for idx in gi.values()]).T

        dba = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(boot_avg)(p) for p in perms)

        for i, (lig, rec) in enumerate(zip(lr_pairs['from'], lr_pairs['to'])):
            dlr = np.outer(data_avg[li[i], :], data_avg[ri[i], :])
            P1 = dlr ** n / (Kh ** n + dlr ** n) * P_spatial
            if np.sum(P1) == 0:
                prob[:, :, i], pval[:, :, i] = P1, 1.0
                continue
            bp = np.array([
                np.outer(dba[b][li[i], :], dba[b][ri[i], :]) ** n /
                (Kh ** n + np.outer(dba[b][li[i], :], dba[b][ri[i], :]) ** n)
                for b in range(nboot)])
            pval[:, :, i] = np.mean(bp > P1, axis=0)
            prob[:, :, i] = P1

        rows = [{"group1": g1, "group2": g2, "ligand": lr[0], "receptor": lr[1],
                 "prob": prob[i, j, k], "pval": pval[i, j, k]}
                for i, g1 in enumerate(ugroups)
                for j, g2 in enumerate(ugroups)
                for k, lr in enumerate(zip(lr_pairs['from'], lr_pairs['to']))
                if g1 != g2 and prob[i, j, k] > thresh_prob
                and pval[i, j, k] < thresh_pval]
        self.commun_prob_df = pd.DataFrame(rows)


def run_step1(cfg, adata_full):
    print("\n" + "=" * 60)
    print("STEP 1: Identify overexpressed ligand-receptor interactions")
    print("=" * 60)

    samples = sorted(set(adata_full.obs[cfg.sample_column]))
    print(f"Processing {len(samples)} samples...")

    for sample in samples:
        out_prefix = os.path.join(cfg.output_dir,
                                  f"communication_probabilities_{sample}.csv")
        if os.path.exists(out_prefix):
            print(f"  {sample}: already done, skipping.")
            continue

        print(f"  {sample}...")
        adata = adata_full[adata_full.obs[cfg.sample_column] == sample].copy()
        pipe = CommunicationPipeline(adata, cfg.cell_state_column, cfg.lr_network)
        pipe.identify_overexpressed_genes(thresh_p=cfg.overexpr_thresh_p,
                                          n_jobs=cfg.n_jobs)
        pipe.identify_overexpressed_interactions(variable_both=cfg.lri_variable_both)
        pipe.compute_communication_probabilities(
            nboot=cfg.n_boot, Kh=cfg.hill_kh, n=cfg.hill_n,
            thresh_prob=cfg.comm_thresh_prob, thresh_pval=cfg.comm_thresh_pval,
            n_jobs=cfg.n_jobs)

        pipe.overexpressed_genes.to_csv(
            os.path.join(cfg.output_dir, f"overexpressed_genes_{sample}.csv"))
        pipe.overexpressed_lri.to_csv(
            os.path.join(cfg.output_dir, f"overexpressed_lri_{sample}.csv"))
        pipe.commun_prob_df.to_csv(
            os.path.join(cfg.output_dir,
                         f"communication_probabilities_{sample}.csv"))

    print("Step 1 complete.")


# =============================================================================
# Step 2: Predict ligand activity
# =============================================================================

class LigandActivityPredictor:
    def __init__(self, geneset, background_genes, ltm, ligands):
        self.geneset = set(geneset)
        self.ltm = ltm.loc[background_genes]
        self.ligands = ligands
        self.response = pd.Series(
            {g: g in self.geneset for g in background_genes})

    def _auc_iregulon(self, pred, resp, top_perc=0.03):
        rank = pd.Series(pred).rank(ascending=False, method="max")
        gs = [g for g, v in resp.items() if v]
        max_auc = int(top_perc * len(rank)) * len(gs)
        auc_true = sum(len(rank) - rank.loc[g] for g in gs if g in rank.index)
        rands = []
        for _ in range(100):
            sr = pd.Series(rank.values, index=np.random.permutation(rank.index))
            rands.append(sum(len(rank) - sr[g] for g in gs if g in sr.index))
        auc_rand = np.mean(rands)
        corr = (auc_true - auc_rand) / max_auc if max_auc > 0 else np.nan
        return {"auc_iregulon": auc_true / max_auc if max_auc > 0 else np.nan,
                "auc_iregulon_corrected": corr}

    def predict(self, sort_by="aupr_corrected", k=5):
        results = []
        for lig in self.ligands:
            if lig not in self.ltm.columns:
                continue
            pv = self.ltm[lig]
            rv = self.response
            m = {}
            if len(np.unique(rv)) > 1 and len(np.unique(pv)) > 1:
                m["auroc"] = roc_auc_score(rv, pv)
                pr, rc, _ = precision_recall_curve(rv, pv)
                m["aupr"] = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(rc, pr)
                m["aupr_corrected"] = m["aupr"] - np.mean(rv)
                m["pearson"] = np.corrcoef(rv, pv)[0, 1]
                m["spearman"] = pd.Series(rv).corr(pd.Series(pv), method='spearman')
            else:
                m = {k: np.nan for k in ["auroc", "aupr", "aupr_corrected",
                                          "pearson", "spearman"]}
            m.update(self._auc_iregulon(pv, rv))
            results.append({"ligand": lig, **m})
        df = pd.DataFrame(results).sort_values(by=sort_by, ascending=False)
        self.top_ligands = df.head(k)["ligand"].values
        return df

    def plot_heatmap(self, fig_name):
        avail_g = [g for g in self.geneset if g in self.ltm.index]
        avail_l = [l for l in self.top_ligands if l in self.ltm.columns]
        if not avail_g or not avail_l:
            return
        sub = self.ltm.loc[avail_g, avail_l]
        plt.figure(figsize=(10, 5))
        ax = sns.heatmap(sub.T, cmap=sns.cubehelix_palette(gamma=0.8, as_cmap=True),
                         cbar_kws={'label': 'Strength'})
        plt.xlabel("Target genes", fontsize=16)
        plt.ylabel("Ligands", fontsize=16)
        ax.tick_params(axis='x', labelsize=14, labelrotation=90)
        ax.tick_params(axis='y', labelsize=14, labelrotation=0)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()


def run_step2(cfg, adata_full):
    print("\n" + "=" * 60)
    print("STEP 2: Predict ligand activity per ecotype")
    print("=" * 60)

    ltm = pd.read_csv(cfg.ligand_target_matrix, index_col=0)

    for ecotype in cfg.ecotypes:
        sample_id = get_representative_sample(ecotype, cfg.ecotype_abundance)
        cell_states = get_cell_states(cfg.marker_genes_dir, ecotype)
        if not cell_states:
            print(f"  {ecotype}: no marker genes, skipping.")
            continue

        comm_path = os.path.join(
            cfg.output_dir, f"communication_probabilities_{sample_id}.csv")
        if not os.path.exists(comm_path):
            print(f"  {ecotype}: Step 1 output missing for {sample_id}, skipping.")
            continue

        print(f"  {ecotype} (sample={sample_id}, {len(cell_states)} cell states)")
        comm_df = pd.read_csv(comm_path, index_col=0)
        adata = adata_full[adata_full.obs[cfg.sample_column] == sample_id].copy()
        bg_all = ltm.index.tolist()
        ecotype_dir = os.path.join(cfg.marker_genes_dir, f"{ecotype}marker")

        for sender, receiver in itertools.permutations(cell_states, 2):
            tsv_out = os.path.join(
                cfg.output_dir, f"{ecotype}_{sender}_{receiver}.tsv")
            if os.path.exists(tsv_out):
                continue

            filt = comm_df[(comm_df['group1'] == sender) &
                           (comm_df['group2'] == receiver)]
            if filt.empty:
                continue

            valid_ligs = [l for l in filt['ligand'].unique()
                          if l in ltm.columns]
            # Filter noisy genes from sender's ligand candidates
            if cfg.noisy_genes:
                sender_prefix = sender.split("_")[0]
                noisy = set(cfg.noisy_genes.get(sender_prefix, []))
                valid_ligs = [l for l in valid_ligs if l not in noisy]
            if not valid_ligs:
                continue

            tf_file = os.path.join(ecotype_dir, f"{receiver}.txt")
            with open(tf_file) as f:
                tf_list = [l.strip() for l in f]
            tf_list = [g for g in tf_list if g in ltm.index]

            expressed = get_expressed_genes(
                adata, receiver, cfg.cell_state_column,
                threshold=cfg.expression_threshold)
            bg = [g for g in bg_all if g in expressed]
            bg.extend(g for g in tf_list if g not in bg)

            pred = LigandActivityPredictor(tf_list, bg, ltm, valid_ligs)
            res = pred.predict(k=cfg.top_ligands_k)
            res.to_csv(tsv_out, sep="\t")
            pred.plot_heatmap(
                os.path.join(cfg.output_dir,
                             f"{ecotype}_{sender}_{receiver}.png"))

    print("Step 2 complete.")


# =============================================================================
# Step 3: Build regulatory network
# =============================================================================

class CellularNetwork:
    def __init__(self, weighted_lr_sig, expressed_genes, n_jobs=-1):
        self.expressed_genes = expressed_genes
        self.n_jobs = n_jobs
        G = nx.DiGraph()
        for _, row in weighted_lr_sig.iterrows():
            if row['weight'] > 0:
                G.add_edge(row['from'], row['to'], weight=row['weight'])
        G.remove_nodes_from([n for n in G.nodes if n not in expressed_genes])
        self.graph = G

    def compute_path_genes(self, ligand_list, tf_list):
        def proc(lig):
            if lig not in self.graph:
                return lig, {}
            probs = defaultdict(float)
            for tf in tf_list:
                if tf not in self.graph:
                    continue
                for path in nx.all_simple_paths(self.graph, lig, tf, cutoff=2):
                    pp = 1.0
                    for i in range(len(path) - 1):
                        pp *= self.graph[path[i]][path[i+1]]['weight']
                    for g in path[1:-1]:
                        probs[g] += pp
            return lig, dict(probs)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(proc)(l) for l in ligand_list)
        self.path_gene = dict(results)
        return self.path_gene

    def extract_subgraph(self, ligand_list, tf_list, top_n=5):
        nodes = set(ligand_list + tf_list)
        for sub in self.path_gene.values():
            topk = sorted(sub.items(), key=lambda x: x[1], reverse=True)[:top_n]
            nodes.update(g for g, _ in topk)
        return self.graph.subgraph(nodes).copy()


def run_step3(cfg, adata_full):
    print("\n" + "=" * 60)
    print("STEP 3: Build regulatory network (random walk)")
    print("=" * 60)

    wlr = pd.read_csv(cfg.weighted_lr_sig, index_col=0)

    # Phase A: compute per-pair networks
    print("Phase A: Random walk network construction...")
    for ecotype in cfg.ecotypes:
        sample_id = get_representative_sample(ecotype, cfg.ecotype_abundance)
        cell_states = get_cell_states(cfg.marker_genes_dir, ecotype)
        if not cell_states:
            continue

        ecotype_dir = os.path.join(cfg.marker_genes_dir, f"{ecotype}marker")
        adata_sample = None  # lazy load

        for sender, receiver in itertools.permutations(cell_states, 2):
            tsv_path = os.path.join(
                cfg.output_dir, f"{ecotype}_{sender}_{receiver}.tsv")
            pkl_path = os.path.join(
                cfg.output_dir, f"{ecotype}_{sender}_{receiver}.pkl")

            if not os.path.exists(tsv_path) or os.path.exists(pkl_path):
                continue

            print(f"  {ecotype}: {sender} -> {receiver}")

            ligs = (pd.read_csv(tsv_path, sep="\t", index_col=0)
                    ["ligand"].tolist()[:cfg.top_ligands_k])
            with open(os.path.join(ecotype_dir, f"{receiver}.txt")) as f:
                tfs = [l.strip() for l in f]

            if adata_sample is None:
                adata_sample = adata_full[
                    adata_full.obs[cfg.sample_column] == sample_id].copy()

            expressed = (get_expressed_genes(
                adata_sample, receiver, cfg.cell_state_column,
                threshold=cfg.expression_threshold,
                noisy_genes=cfg.noisy_genes) + tfs + ligs)

            nc = CellularNetwork(wlr, expressed, n_jobs=cfg.n_jobs)
            nc.compute_path_genes(ligs, tfs)

            with open(pkl_path, "wb") as f:
                pickle.dump(nc, f)

    # Phase B: assemble graphs
    print("Phase B: Assembling ecotype and global graphs...")
    global_graph = nx.DiGraph()

    for ecotype in cfg.ecotypes:
        cell_states = get_cell_states(cfg.marker_genes_dir, ecotype)
        if not cell_states:
            continue

        ecotype_dir = os.path.join(cfg.marker_genes_dir, f"{ecotype}marker")
        big = nx.DiGraph()

        for sender, receiver in itertools.permutations(cell_states, 2):
            pkl_path = os.path.join(
                cfg.output_dir, f"{ecotype}_{sender}_{receiver}.pkl")
            tsv_path = os.path.join(
                cfg.output_dir, f"{ecotype}_{sender}_{receiver}.tsv")
            tf_file = os.path.join(ecotype_dir, f"{receiver}.txt")

            if os.path.exists(pkl_path) and os.path.exists(tsv_path):
                with open(pkl_path, "rb") as f:
                    nc = pickle.load(f)
                ligs = (pd.read_csv(tsv_path, sep="\t", index_col=0)
                        ["ligand"].tolist()[:cfg.top_ligands_k])
                with open(tf_file) as f:
                    tfs = [l.strip() for l in f]
                big = nx.compose(big,
                                 nc.extract_subgraph(ligs, tfs,
                                                     top_n=cfg.top_mediators_n))
            elif os.path.exists(tf_file):
                with open(tf_file) as f:
                    for line in f:
                        big.add_node(line.strip())

        print(f"  {ecotype}: {big.number_of_nodes()} nodes, "
              f"{big.number_of_edges()} edges")

        with open(os.path.join(cfg.output_dir, f"{ecotype}_graph.pkl"), "wb") as f:
            pickle.dump(big, f)

        global_graph = nx.compose(global_graph, big)

    out_path = os.path.join(cfg.output_dir, "global_graph.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(global_graph, f)

    print(f"\nGlobal graph: {global_graph.number_of_nodes()} nodes, "
          f"{global_graph.number_of_edges()} edges")
    print(f"Saved to: {out_path}")
    print("Step 3 complete.")


# =============================================================================
# Cleanup intermediate files
# =============================================================================

def cleanup_intermediates(cfg):
    """Remove bulky intermediate files, keeping only final outputs."""
    print("\nCleaning up intermediate files...")
    patterns = [
        os.path.join(cfg.output_dir, "overexpressed_genes_*.csv"),
        os.path.join(cfg.output_dir, "overexpressed_lri_*.csv"),
        os.path.join(cfg.output_dir, "communication_probabilities_*.csv"),
        os.path.join(cfg.output_dir, "E*_*_*.pkl"),   # per-pair network objects
        os.path.join(cfg.output_dir, "E*_*_*.png"),    # per-pair heatmaps
    ]
    removed = 0
    freed = 0
    for pat in patterns:
        for path in glob.glob(pat):
            sz = os.path.getsize(path)
            os.remove(path)
            removed += 1
            freed += sz
    print(f"  Removed {removed} intermediate files ({freed / 1e9:.1f} GB freed)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EcoNet: Build intercellular regulatory network")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--step", nargs="*", type=int, default=[1, 2, 3],
                        help="Steps to run (default: 1 2 3)")
    args = parser.parse_args()

    cfg = Config(args.config)
    steps = sorted(set(args.step))
    cfg.validate(steps)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("EcoNet Network Construction Pipeline")
    print(f"Config: {args.config}")
    print(f"Steps:  {steps}")
    print(f"Output: {cfg.output_dir}")
    cfg.summary()

    # Extract marker genes from EcoTyper model
    print("\nExtracting marker genes from EcoTyper model...")
    cfg.extract_marker_genes()

    # Load scRNA-seq once (shared across all steps that need it)
    adata = None
    if any(s in steps for s in [1, 2, 3]):
        print(f"\nLoading scRNA-seq data: {cfg.scrna_h5ad}")
        adata = sc.read(cfg.scrna_h5ad)
        print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    if 1 in steps:
        run_step1(cfg, adata)
    if 2 in steps:
        run_step2(cfg, adata)
    if 3 in steps:
        run_step3(cfg, adata)

    # Clean up intermediate files after all requested steps complete
    if 3 in steps:
        cleanup_intermediates(cfg)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Merge the 14 Carcinoma-model per-cancer regulatory networks into a single
pan-cancer network, sized for efficient GAT training.

Cancers (14): BLCA, BRCA, CRC, HNSC, LUAD, PDAC, CHOL, LUSC, OV, PRAD, STAD,
THCA, UCEC, ESCA. CCRCC is excluded: it was built from a different EcoTyper
model (614-gene space, ecotypes E1-E11) and a different gene space.

Design:
  - Node set = EcoTyper Carcinoma marker genes (primary, always kept) +
               mediator genes that pass a frequency threshold across cancers.
  - Scaffold removal = *two-gate promiscuity rule* (see scaffold_nodes_merged):
    drop a MEDIATOR node only if it is promiscuous in BOTH
      (1) the NicheNet signaling database (DB-degree > p99.9 of all DB genes), and
      (2) the merged pan-cancer network (degree > p99.7 of network nodes).
    This isolates database-topology artifacts (UBC, APP) — which are
    promiscuously wired in NicheNet *and* become network hubs — while keeping
    legitimate high-degree biology: signaling hubs (MAPK1, SRC, GRB2) fail
    gate 1, and DB-promiscuous-but-not-network-hub TFs (MYC, HSPA8) fail gate 2.
    Markers are always protected (never dropped). The older single-axis
    degree-ratio rule could not separate APP from real TF hubs (TP53, JUN),
    which is why both DB and network promiscuity are required.
  - Edge set = edges present in at least MIN_FREQ cancers. This is a
    pan-cancer network, so cancer-specific (freq<MIN_FREQ) edges are excluded
    regardless of node type.
  - Merged edge weight = (frequency / N_cancers) * mean_weight.

Outputs to ./output_PanCancer/:
  pan_cancer_network.pkl   NetworkX DiGraph with merged edges (14 cancers)
  node_metadata.tsv        per-node type (marker/mediator), degree, # cancers
  edge_metadata.tsv        per-edge frequency, mean_weight, merged_weight
  scaffold_drops.tsv       nodes dropped by the two-gate promiscuity rule
  summary.txt              human-readable counts and parameters
  CE{1..10}_pancancer_consensus.pkl   per-ecotype consensus (14 cancers)
  per_ecotype_consensus_stats.tsv

Run:
  python merge_pan_cancer.py
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================================
# Config
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The 14 Carcinoma-model cancers merged into the pan-cancer network.
# (CCRCC excluded: different EcoTyper model / 614-gene space.)
CANCERS = ["BLCA", "BRCA", "CRC", "HNSC", "LUAD", "PDAC",
           "CHOL", "LUSC", "OV", "PRAD", "STAD", "THCA", "UCEC", "ESCA"]

# Same list is used for the per-ecotype consensus (CE1-CE10).
CARCINOMA = CANCERS

# EcoTyper Carcinoma model directory (marker genes for the merge). EDIT to your
# path, or set the ECONET_ECOTYPER_DIR environment variable.
ECOTYPER_DIR = os.environ.get(
    "ECONET_ECOTYPER_DIR",
    os.path.join(SCRIPT_DIR, "EcotyperModels", "Carcinoma"),
)

# NicheNet signaling network — used to score node "DB promiscuity" for the
# two-gate scaffold rule. Degree here = how connected a gene is in the source
# database (independent of any cancer's expression). EDIT to your path, or set
# the ECONET_NICHENET_DIR environment variable (see README for the download).
NICHENET_SIG_CSV = os.path.join(
    os.environ.get("ECONET_NICHENET_DIR", os.path.join(SCRIPT_DIR, "NicheNet_DB")),
    "weighted_lr_sig.csv",
)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_PanCancer14_minfreq5")

# Uniform edge-frequency threshold (pan-cancer consensus).
# Scaled to ~50% of the 14 cancers (was 3 of 6 in the original 6-cancer merge).
MIN_FREQ = 5  # an edge must be present in >= N of the 14 cancers

# Mediator node filter
MIN_MEDIATOR_FREQ = 5  # non-marker node kept only if present in >= N cancers

# Marker node rule (partner-agnostic). A marker gene is kept if its non-scaffold
# incident edges span >= MIN_INCIDENT cancers, EVEN IF no single edge is shared
# by that many cancers (a gene wired to different partners in different cancers
# is still a conserved network participant). Set equal to MIN_FREQ so the node
# and edge consensus use the same level.
MIN_INCIDENT = 5

# Recovery-edge threshold. A marker kept by MIN_INCIDENT but absent from every
# backbone (freq>=MIN_FREQ) edge is reconnected via its incident edges that
# recur in >= MIN_RECOVERY_FREQ cancers. MIN_INCIDENT>=7 guarantees each such
# marker has a best edge of at least this frequency, so none ends up isolated;
# only edges incident to these markers are added, so the backbone isn't flooded.
MIN_RECOVERY_FREQ = 3

# Two-gate scaffold (promiscuity) rule. A mediator is dropped only if it is an
# upper-tail outlier in BOTH the NicheNet DB and the merged network.
DB_PROMISCUITY_PCTILE = 99.9   # gate 1: NicheNet signaling DB-degree percentile
NET_HUB_PCTILE = 99.7          # gate 2: merged-network degree percentile


# ============================================================================
# Marker-gene extraction from Carcinoma EcoTyper model
# ============================================================================

def canonicalize_gene(name):
    """Canonicalize a gene symbol to the NicheNet/HGNC namespace.

    EcoTyper's gene_info.txt is written by R, which mangles hyphens in HGNC
    symbols to dots (e.g. HLA-DRB1 -> HLA.DRB1, NKX2-1 -> NKX2.1). The
    per-cancer regulatory graphs carry BOTH forms as separate nodes (dash from
    NicheNet, dot from EcoTyper markers), so without normalization a marker like
    'HLA.DRB1' never matches its real network node 'HLA-DRB1' — it gets
    mislabeled as a mediator and its duplicate dot-node is dropped. Converting
    dots to dashes collapses the duplicates and restores marker recognition.
    (Clone/version IDs like RP11.330H6.5 are also converted but are absent from
    the network, so the over-conversion is harmless.)"""
    return name.replace(".", "-")


def find_fraction_root(ecotyper_dir):
    """Locate the Cell_States/discovery/ parent (matches run_pipeline logic)."""
    for candidate in [ecotyper_dir] + [
        os.path.join(ecotyper_dir, e) for e in os.listdir(ecotyper_dir)
    ]:
        if os.path.isdir(os.path.join(candidate, "Cell_States", "discovery")):
            return candidate
    raise FileNotFoundError(f"Cell_States/discovery/ not found under {ecotyper_dir}")


def extract_markers(ecotyper_dir):
    """Return {cell_type: set(marker_genes)} from gene_info.txt in each
    cell-type's NMF-rank subdirectory."""
    root = find_fraction_root(ecotyper_dir)
    disc = os.path.join(root, "Cell_States", "discovery")

    markers = {}
    for cell_type in sorted(os.listdir(disc)):
        ct_path = os.path.join(disc, cell_type)
        if not os.path.isdir(ct_path):
            continue
        for sub in os.listdir(ct_path):
            gi = os.path.join(ct_path, sub, "gene_info.txt")
            if os.path.isfile(gi):
                df = pd.read_csv(gi, sep="\t")
                markers[cell_type] = {canonicalize_gene(g)
                                      for g in df["Gene"].astype(str)}
                break
        else:
            print(f"  Warning: no gene_info.txt for {cell_type}, skipping.")
    return markers


# ============================================================================
# Scaffold detection (two-gate promiscuity rule)
# ============================================================================

def nichenet_db_degree(sig_csv=NICHENET_SIG_CSV, cache=True):
    """Degree of each gene in the NicheNet signaling network (undirected count
    of appearances as `from` or `to`). Cached to a TSV next to the output so
    re-runs don't re-scan the ~180 MB CSV."""
    cache_path = os.path.join(OUTPUT_DIR, "nichenet_sig_degree.tsv")
    if cache and os.path.isfile(cache_path):
        s = pd.read_csv(cache_path, sep="\t", index_col=0)["db_degree"]
        return s.to_dict()

    from collections import Counter
    deg = Counter()
    for chunk in pd.read_csv(sig_csv, usecols=["from", "to"], chunksize=2_000_000):
        deg.update(chunk["from"].values)
        deg.update(chunk["to"].values)
    deg = dict(deg)
    if cache:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pd.Series(deg, name="db_degree").rename_axis("gene").to_csv(
            cache_path, sep="\t")
    return deg


def scaffold_nodes_merged(graph, marker_union, db_degree,
                          db_pctile=DB_PROMISCUITY_PCTILE,
                          net_pctile=NET_HUB_PCTILE):
    """Two-gate promiscuity rule. Drop a MEDIATOR (non-marker) node only if it
    is an upper-tail outlier in BOTH:
      gate 1 (DB promiscuity): NicheNet signaling DB-degree > db_pctile of all
              genes present in the DB, and
      gate 2 (network hub):    degree in the merged graph > net_pctile of the
              merged-graph node degrees.
    Markers are always protected. Returns the set of dropped node names.

    Rationale: artifacts like UBC/APP are promiscuously wired in the database
    AND become network hubs. Real signaling hubs (MAPK1, SRC, GRB2) fail gate 1
    (moderate DB-degree); DB-promiscuous-but-not-hub TFs (MYC, HSPA8) fail
    gate 2; legitimate TF hubs that are borderline on DB-degree (TP53, JUN) sit
    below the network-hub percentile and are kept."""
    db_vals = np.array(list(db_degree.values()), dtype=float)
    db_thr = np.percentile(db_vals, db_pctile)

    net_deg = dict(graph.degree())
    net_thr = np.percentile(np.array(list(net_deg.values()), dtype=float),
                            net_pctile)

    drops = set()
    for n, d in net_deg.items():
        if n in marker_union:
            continue  # markers always protected
        if d > net_thr and db_degree.get(n, 0) > db_thr:
            drops.add(n)
    return drops


# ============================================================================
# Merge
# ============================================================================

def merge_graphs(graphs, marker_union, scaffold_drops):
    """Build the pan-cancer DiGraph.

    Node inclusion:
      - marker  : kept if its non-scaffold incident edges span >= MIN_INCIDENT
                  cancers (partner-agnostic — a conserved participant even if
                  wired to different partners in different cancers).
      - mediator: kept if it lies on a backbone (freq>=MIN_FREQ) edge and is a
                  node in >= MIN_MEDIATOR_FREQ cancers.
    Edge inclusion (both endpoints must be in the node set):
      - backbone: edges with frequency >= MIN_FREQ.
      - recovery: edges with frequency >= MIN_RECOVERY_FREQ incident to a
                  "recovered" marker (kept by MIN_INCIDENT but on no backbone
                  edge), so such markers are connected rather than isolated.
    """
    n_cancers = len(graphs)

    edge_cancers = defaultdict(list)
    edge_weights = defaultdict(list)
    incident_cancers = defaultdict(set)   # gene -> cancers with any non-scaffold edge
    node_cancer_count = defaultdict(set)  # gene -> cancers where it is a node
    for cancer, g in graphs.items():
        for n in g.nodes():
            node_cancer_count[n].add(cancer)
        for u, v, data in g.edges(data=True):
            if u in scaffold_drops or v in scaffold_drops:
                continue
            edge_cancers[(u, v)].append(cancer)
            edge_weights[(u, v)].append(float(data.get("weight", 0)))
            incident_cancers[u].add(cancer)
            incident_cancers[v].add(cancer)

    def attrs_for(u, v):
        cs = edge_cancers[(u, v)]
        freq = len(cs)
        u_m, v_m = u in marker_union, v in marker_union
        pair_type = ("marker_marker" if u_m and v_m
                     else "marker_mediator" if u_m or v_m
                     else "mediator_mediator")
        mean_w = sum(edge_weights[(u, v)]) / freq
        return {
            "frequency": freq,
            "mean_weight": mean_w,
            "merged_weight": (freq / n_cancers) * mean_w,
            "cancers": ",".join(sorted(cs)),
            "pair_type": pair_type,
        }

    # --- backbone edges + their nodes ---
    backbone = {e for e, cs in edge_cancers.items() if len(cs) >= MIN_FREQ}
    backbone_nodes = set()
    for u, v in backbone:
        backbone_nodes.add(u)
        backbone_nodes.add(v)

    # --- node set ---
    final_markers = {m for m in marker_union
                     if len(incident_cancers.get(m, ())) >= MIN_INCIDENT}
    final_mediators = {n for n in backbone_nodes
                       if n not in marker_union
                       and len(node_cancer_count[n]) >= MIN_MEDIATOR_FREQ}
    final_nodes = final_markers | final_mediators

    # markers kept but not on any backbone edge -> need recovery edges
    recovered_markers = final_markers - backbone_nodes

    # --- edge set ---
    kept_edges = {}
    for (u, v), cs in edge_cancers.items():
        if u not in final_nodes or v not in final_nodes:
            continue
        freq = len(cs)
        if freq >= MIN_FREQ:
            kept_edges[(u, v)] = attrs_for(u, v)
        elif (freq >= MIN_RECOVERY_FREQ
              and (u in recovered_markers or v in recovered_markers)):
            kept_edges[(u, v)] = attrs_for(u, v)

    H = nx.DiGraph()
    for (u, v), attrs in kept_edges.items():
        H.add_edge(u, v, weight=attrs["merged_weight"],
                   mean_weight=attrs["mean_weight"],
                   frequency=attrs["frequency"],
                   cancers=attrs["cancers"],
                   pair_type=attrs["pair_type"])

    for n in H.nodes():
        H.nodes[n]["type"] = "marker" if n in marker_union else "mediator"
        H.nodes[n]["n_cancers"] = len(node_cancer_count.get(n, []))

    return H


# ============================================================================
# I/O
# ============================================================================

def write_outputs(H, marker_union, markers_by_cell, scaffold_drops, graphs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "pan_cancer_network.pkl"), "wb") as f:
        pickle.dump(H, f)

    gene_to_cells = defaultdict(list)
    for ct, gs in markers_by_cell.items():
        for gene in gs:
            gene_to_cells[gene].append(ct)

    pd.DataFrame([{
        "gene": n,
        "type": H.nodes[n]["type"],
        "degree": H.degree(n),
        "n_cancers": H.nodes[n]["n_cancers"],
        "marker_cell_types": ";".join(sorted(gene_to_cells.get(n, []))),
    } for n in H.nodes()]).sort_values(
        ["type", "degree"], ascending=[True, False]
    ).to_csv(os.path.join(OUTPUT_DIR, "node_metadata.tsv"),
             sep="\t", index=False)

    pd.DataFrame([{
        "source": u, "target": v,
        "frequency": d["frequency"],
        "mean_weight": round(d["mean_weight"], 6),
        "merged_weight": round(d["weight"], 6),
        "cancers": d["cancers"],
        "pair_type": d["pair_type"],
    } for u, v, d in H.edges(data=True)]).sort_values(
        ["frequency", "merged_weight"], ascending=[False, False]
    ).to_csv(os.path.join(OUTPUT_DIR, "edge_metadata.tsv"),
             sep="\t", index=False)

    drop_rows = []
    for gene in sorted(scaffold_drops):
        row = {"gene": gene}
        for c, g in graphs.items():
            row[f"degree_{c}"] = g.degree(gene) if gene in g else 0
        drop_rows.append(row)
    pd.DataFrame(drop_rows).to_csv(
        os.path.join(OUTPUT_DIR, "scaffold_drops.tsv"),
        sep="\t", index=False,
    )

    n_marker = sum(1 for n in H.nodes() if H.nodes[n]["type"] == "marker")
    pair_counts = defaultdict(int)
    freq_counts = defaultdict(int)
    for _, _, d in H.edges(data=True):
        pair_counts[d["pair_type"]] += 1
        freq_counts[d["frequency"]] += 1

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("Pan-cancer network merge summary\n")
        f.write("================================\n\n")
        f.write(f"Cancers merged : {', '.join(CANCERS)}\n")
        f.write(f"EcoTyper model : {ECOTYPER_DIR}\n\n")
        f.write("Parameters\n----------\n")
        f.write(f"MIN_FREQ           = {MIN_FREQ}   "
                "(edge must appear in >= MIN_FREQ cancers)\n")
        f.write(f"MIN_MEDIATOR_FREQ  = {MIN_MEDIATOR_FREQ}   "
                "(non-marker node must appear in >= MIN_MEDIATOR_FREQ cancers)\n")
        f.write(f"MIN_INCIDENT       = {MIN_INCIDENT}   "
                "(marker kept if its incident edges span >= MIN_INCIDENT cancers, "
                "partner-agnostic)\n")
        f.write(f"MIN_RECOVERY_FREQ  = {MIN_RECOVERY_FREQ}   "
                "(recovery edge for a backbone-orphan marker, freq >= this)\n")
        f.write(f"DB_PROMISCUITY_PCTILE = {DB_PROMISCUITY_PCTILE}   "
                "(scaffold gate 1: NicheNet DB-degree percentile)\n")
        f.write(f"NET_HUB_PCTILE        = {NET_HUB_PCTILE}   "
                "(scaffold gate 2: merged-network degree percentile)\n\n")
        f.write("Marker genes\n------------\n")
        f.write(f"Carcinoma cell types : {len(markers_by_cell)}\n")
        f.write(f"Union of markers     : {len(marker_union)}\n\n")
        f.write("Scaffold-drop rule\n------------------\n")
        f.write(f"Nodes dropped: {len(scaffold_drops)} "
                f"({', '.join(sorted(scaffold_drops)) or 'none'})\n\n")
        f.write("Pan-cancer graph\n----------------\n")
        f.write(f"Nodes    : {H.number_of_nodes()}\n")
        f.write(f"  marker   : {n_marker}\n")
        f.write(f"  mediator : {H.number_of_nodes() - n_marker}\n")
        f.write(f"Edges    : {H.number_of_edges()}\n")
        f.write("  by pair type:\n")
        for r, n in sorted(pair_counts.items(), key=lambda x: -x[1]):
            f.write(f"    {r}: {n}\n")
        f.write("  by frequency:\n")
        for freq in sorted(freq_counts):
            f.write(f"    freq={freq}: {freq_counts[freq]}\n")

    print(f"\nWrote pan-cancer network and metadata to {OUTPUT_DIR}")


# ============================================================================
# Per-ecotype pan-cancer consensus subgraphs
# ============================================================================

ECOTYPES = [f"CE{i}" for i in range(1, 11)]


def build_ecotype_consensus(ecotype, marker_union, scaffold_drops):
    """Merge per-cancer CE{N}_graph.pkl files using the same rules as the
    pan-cancer merge (MIN_FREQ edges, MIN_MEDIATOR_FREQ nodes, scaffold drops).
    Returns an empty DiGraph if fewer than MIN_FREQ cancers have the subgraph."""
    eco_graphs = {}
    for c in CARCINOMA:  # CE1-CE10 only exist for Carcinoma-model cancers
        pkl = os.path.join(SCRIPT_DIR, f"output_{c}", f"{ecotype}_graph.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            g = pickle.load(f)
        eco_graphs[c] = nx.relabel_nodes(
            g, {n: canonicalize_gene(n) for n in g.nodes()}, copy=True)
    return merge_graphs(eco_graphs, marker_union, scaffold_drops)


def write_ecotype_consensus(marker_union, scaffold_drops):
    """Build and save per-ecotype pan-cancer consensus subgraphs into
    OUTPUT_DIR as CE{N}_pancancer_consensus.pkl, plus a stats TSV."""
    rows = []
    for eco in ECOTYPES:
        H = build_ecotype_consensus(eco, marker_union, scaffold_drops)
        out = os.path.join(OUTPUT_DIR, f"{eco}_pancancer_consensus.pkl")
        with open(out, "wb") as f:
            pickle.dump(H, f)
        print(f"  {eco}: {H.number_of_nodes()} nodes, "
              f"{H.number_of_edges()} edges")
        rows.append({
            "ecotype": eco,
            "nodes": H.number_of_nodes(),
            "edges": H.number_of_edges(),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "per_ecotype_consensus_stats.tsv"),
        sep="\t", index=False,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading per-cancer graphs...")
    graphs = {}
    for c in CANCERS:
        pkl = os.path.join(SCRIPT_DIR, f"output_{c}", "global_graph.pkl")
        with open(pkl, "rb") as f:
            g = pickle.load(f)
        # Canonicalize node names (dot->dash) so EcoTyper-derived and
        # NicheNet-derived spellings of the same gene collapse into one node.
        g = nx.relabel_nodes(g, {n: canonicalize_gene(n) for n in g.nodes()},
                             copy=True)
        graphs[c] = g
        print(f"  {c}: {graphs[c].number_of_nodes()} nodes, "
              f"{graphs[c].number_of_edges()} edges")

    print("\nExtracting EcoTyper Carcinoma marker genes...")
    markers_by_cell = extract_markers(ECOTYPER_DIR)
    marker_union = (set.union(*markers_by_cell.values())
                    if markers_by_cell else set())
    print(f"  Marker union: {len(marker_union)}")

    print("\nLoading NicheNet signaling DB-degree (for promiscuity gate)...")
    db_degree = {canonicalize_gene(k): v for k, v in nichenet_db_degree().items()}
    print(f"  DB genes: {len(db_degree)}")

    # First pass: merge with no scaffold drops to get the merged degree profile
    print("\nFirst-pass merge (no scaffold drops)...")
    H0 = merge_graphs(graphs, marker_union, set())

    # Apply two-gate promiscuity rule
    drops = scaffold_nodes_merged(H0, marker_union, db_degree)
    print(f"  Scaffold drops (two-gate promiscuity rule): "
          f"{sorted(drops) or 'none'}")

    # Second pass: final merge excluding scaffold nodes
    print("\nFinal merge...")
    H = merge_graphs(graphs, marker_union, drops)
    isolated = [n for n, d in H.degree() if d == 0]
    assert not isolated, f"isolated (degree-0) nodes in merged graph: {isolated}"
    n_marker = sum(1 for n in H.nodes() if H.nodes[n]["type"] == "marker")
    print(f"  Pan-cancer graph: {H.number_of_nodes()} nodes "
          f"({n_marker} marker), {H.number_of_edges()} edges, "
          f"0 isolated nodes")

    write_outputs(H, marker_union, markers_by_cell, drops, graphs)

    print("\nBuilding per-ecotype pan-cancer consensus subgraphs...")
    write_ecotype_consensus(marker_union, drops)


if __name__ == "__main__":
    main()

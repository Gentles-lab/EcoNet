#!/usr/bin/env python
"""Compare pipeline output global_graph.pkl with reference."""
import pickle
import networkx as nx
import numpy as np
import sys

ref_path = sys.argv[1] if len(sys.argv) > 1 else "../../SourceCode/02.NetworkConstruction/SCP1288RegNetwork/global_graph.pkl"
new_path = sys.argv[2] if len(sys.argv) > 2 else "output/global_graph.pkl"

with open(ref_path, "rb") as f:
    ref = pickle.load(f)
with open(new_path, "rb") as f:
    new = pickle.load(f)

print("=" * 60)
print("GRAPH COMPARISON: Reference vs Pipeline Output")
print("=" * 60)

print(f"\n{'':30s} {'Reference':>12s} {'Pipeline':>12s}")
print(f"{'Nodes':30s} {ref.number_of_nodes():>12d} {new.number_of_nodes():>12d}")
print(f"{'Edges':30s} {ref.number_of_edges():>12d} {new.number_of_edges():>12d}")
print(f"{'Is directed':30s} {str(ref.is_directed()):>12s} {str(new.is_directed()):>12s}")

ref_nodes = set(ref.nodes())
new_nodes = set(new.nodes())
shared_nodes = ref_nodes & new_nodes
only_ref = ref_nodes - new_nodes
only_new = new_nodes - ref_nodes

print(f"\n{'Shared nodes':30s} {len(shared_nodes):>12d}")
print(f"{'Only in reference':30s} {len(only_ref):>12d}")
print(f"{'Only in pipeline':30s} {len(only_new):>12d}")
print(f"{'Node Jaccard similarity':30s} {len(shared_nodes)/len(ref_nodes | new_nodes):>12.4f}")

ref_edges = set(ref.edges())
new_edges = set(new.edges())
shared_edges = ref_edges & new_edges
only_ref_e = ref_edges - new_edges
only_new_e = new_edges - ref_edges

print(f"\n{'Shared edges':30s} {len(shared_edges):>12d}")
print(f"{'Only in reference':30s} {len(only_ref_e):>12d}")
print(f"{'Only in pipeline':30s} {len(only_new_e):>12d}")
if ref_edges | new_edges:
    print(f"{'Edge Jaccard similarity':30s} {len(shared_edges)/len(ref_edges | new_edges):>12.4f}")

# Compare edge weights for shared edges
weight_diffs = []
for u, v in shared_edges:
    rw = ref[u][v].get("weight", 0)
    nw = new[u][v].get("weight", 0)
    weight_diffs.append(abs(rw - nw))

if weight_diffs:
    wd = np.array(weight_diffs)
    print(f"\nEdge weight differences (shared edges):")
    print(f"  Mean absolute diff:  {wd.mean():.6f}")
    print(f"  Max absolute diff:   {wd.max():.6f}")
    print(f"  Exact matches:       {np.sum(wd == 0)}/{len(wd)}")

# Degree distribution comparison
ref_deg = sorted([d for _, d in ref.degree()], reverse=True)
new_deg = sorted([d for _, d in new.degree()], reverse=True)
print(f"\nDegree distribution:")
print(f"  Ref: max={ref_deg[0]}, mean={np.mean(ref_deg):.1f}, median={np.median(ref_deg):.0f}")
print(f"  New: max={new_deg[0]}, mean={np.mean(new_deg):.1f}, median={np.median(new_deg):.0f}")

# Connected components
ref_ug = ref.to_undirected()
new_ug = new.to_undirected()
ref_cc = list(nx.connected_components(ref_ug))
new_cc = list(nx.connected_components(new_ug))
print(f"\nConnected components:")
print(f"  Reference: {len(ref_cc)} (largest: {len(max(ref_cc, key=len))})")
print(f"  Pipeline:  {len(new_cc)} (largest: {len(max(new_cc, key=len))})")

# Sample differences
if only_ref:
    print(f"\nSample nodes only in reference (first 10): {sorted(only_ref)[:10]}")
if only_new:
    print(f"Sample nodes only in pipeline (first 10): {sorted(only_new)[:10]}")
if only_ref_e:
    print(f"Sample edges only in reference (first 5): {sorted(only_ref_e)[:5]}")
if only_new_e:
    print(f"Sample edges only in pipeline (first 5): {sorted(only_new_e)[:5]}")

# Overall verdict
if ref_nodes == new_nodes and ref_edges == new_edges and all(d == 0 for d in weight_diffs):
    print("\n*** PERFECT MATCH ***")
elif len(shared_nodes)/len(ref_nodes | new_nodes) > 0.95 and len(shared_edges)/len(ref_edges | new_edges) > 0.95:
    print("\n*** VERY HIGH ALIGNMENT (>95% overlap) ***")
elif len(shared_nodes)/len(ref_nodes | new_nodes) > 0.8:
    print("\n*** GOOD ALIGNMENT (>80% overlap) ***")
else:
    print("\n*** SIGNIFICANT DIFFERENCES - INVESTIGATE ***")

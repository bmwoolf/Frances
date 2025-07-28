import networkx as nx
import json
import matplotlib.pyplot as plt
import os
import torch

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from networkx.algorithms.traversal.breadth_first_search import bfs_tree

# load BiGG model
with open("iML1515.json", "r") as f:
    model = json.load(f)

G = nx.DiGraph()

# build full metabolic graph
for rxn in model["reactions"]:
    rxn_id = rxn["id"]
    genes = rxn.get("gene_reaction_rule", "")
    G.add_node(rxn_id, type="reaction", genes=genes)

    for met_id, coeff in rxn["metabolites"].items():
        G.add_node(met_id, type="metabolite")

        if coeff < 0:
            G.add_edge(met_id, rxn_id, role="substrate", coeff=coeff)
        else:
            G.add_edge(rxn_id, met_id, role="product", coeff=coeff)

# add engineered limonene pathway
G.add_node("gpp_c", type="metabolite")
G.add_node("limonene_c", type="metabolite")

# add heterologous reaction node
G.add_node("LS", type="reaction", genes="limonene_synthase")

# connect it automatically
G.add_edge("gpp_c", "LS", role="substrate", coeff=1)       # gpp -> LS
G.add_edge("LS", "limonene_c", role="product", coeff=1)    # LS -> limonene

print(f"Total nodes: {len(G.nodes)}")
print(f"Total edges: {len(G.edges)}")
print("Sample nodes:", list(G.nodes(data=True))[:5])
print("Sample edges:", list(G.edges(data=True))[:5])

# extract the 3 hop neighborhood around glucose
neighbors = list(bfs_tree(G, source="glc__D_c", depth_limit=3).nodes)
H = G.subgraph(neighbors)

# remove currency metabolites from the graph
currency_mets = {
    "h2o_c", "atp_c", "adp_c", "pi_c", "h_c", "nadh_c",
    "nad_c", "nadph_c", "nadp_c", "co2_c"
}

H = H.subgraph([n for n in H.nodes if n not in currency_mets])

# add simple node features 
for node in H.nodes:
    ntype = H.nodes[node]["type"]
    deg = H.degree[node]

    # Only check genes if it's a reaction
    if ntype == "reaction":
        gene_str = H.nodes[node].get("genes", "")
        is_target_gene = 1 if any(g in gene_str for g in ["dxs", "idi", "ispG", "gpps", "LS"]) else 0
    else:
        is_target_gene = 0

    H.nodes[node]["x"] = [
        0 if ntype == "metabolite" else 1,  # type encoding
        deg,                                # degree
        is_target_gene                      # relevant for reactions only
    ]


# convert to PyG graph
data = from_networkx(H)
data.x = torch.tensor([H.nodes[n]["x"] for n in H.nodes], dtype=torch.float)
print(f"Graph has {data.num_nodes} nodes and {data.num_edges} edges")
import networkx as nx
import json
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from networkx.algorithms.traversal.breadth_first_search import bfs_tree
from sklearn.manifold import TSNE

# create cobra model
from cobra import io
cobra_model = io.load_json_model("iML1515.json")

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

# ensure all nodes have consistent attributes for PyG conversion
for node in H.nodes:
    if "type" not in H.nodes[node]:
        H.nodes[node]["type"] = "metabolite"
    if "genes" not in H.nodes[node]:
        H.nodes[node]["genes"] = ""

# assign node features consistently
for node in H.nodes:
    ntype = H.nodes[node].get("type", "metabolite")  # default fallback
    deg = H.degree[node]

    if ntype == "reaction":
        gene_str = H.nodes[node].get("genes", "")
        is_target_gene = 1 if any(g in gene_str for g in ["dxs", "idi", "ispG", "gpps", "LS"]) else 0
        type_flag = 1
    else:
        is_target_gene = 0
        type_flag = 0

    H.nodes[node]["x"] = [type_flag, deg, is_target_gene]

# convert to PyG graph
data = from_networkx(H)
data.x = torch.tensor([H.nodes[n]["x"] for n in H.nodes], dtype=torch.float)
print(f"Graph has {data.num_nodes} nodes and {data.num_edges} edges")

# create GNN with the graph as the input
class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

model = GNN(in_dim=3, hidden_dim=16, out_dim=8)  # Changed to 8D embeddings
embeddings = model(data)

print(f"Node embeddings shape: {embeddings.shape}")
print(f"Sample embeddings: {embeddings[:5]}")

# interpret the embeddings
z = embeddings.detach().numpy()
z = TSNE(n_components=2).fit_transform(z)

plt.scatter(z[:, 0], z[:, 1], c=[H.nodes[n]['x'][0] for n in H.nodes])
plt.title("Node Embeddings (colored by node type)")
plt.savefig("network_graphs/node_embeddings.png", dpi=300, bbox_inches='tight')
print("Embeddings visualization saved as 'network_graphs/node_embeddings.png'")
plt.close()

# filter to editable nodes (type == "reaction")
editable_nodes = [i for i, n in enumerate(H.nodes) if H.nodes[n]["type"] == "reaction"]
edit_embeddings = embeddings[editable_nodes]

# score potential edits: shape: [#reactions]
edit_scores = torch.softmax(edit_embeddings.mean(dim=1), dim=0)
# pick top k expressions or knockouts
top_k = torch.topk(edit_scores, k=5)
flat_indices = [i if isinstance(i, int) else i[0] for i in top_k.indices.tolist()]
suggested_edits = [list(H.nodes)[editable_nodes[i]] for i in flat_indices]
print("Suggested edits:", suggested_edits)

# apply edit and simulate yield with cobra
for rxn_id in suggested_edits:
    cobra_model.reactions.get_by_id(rxn_id).knock_out()
cobra_model.optimize()

# loop via RL
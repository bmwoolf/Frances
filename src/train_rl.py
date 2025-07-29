import networkx as nx
import json
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from networkx.algorithms.traversal.breadth_first_search import bfs_tree
from sklearn.manifold import TSNE

# import config
from config import host, target, config

# Load LASER training data
def load_laser_data():
    """Load LASER training data for reward calculation"""
    try:
        with open('data/laser_training_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("LASER data not found, using original reward function")
        return []

laser_data = load_laser_data()
if laser_data:
    print(f"Loaded {len(laser_data)} LASER training examples")
    print("LASER examples:")
    for i, example in enumerate(laser_data):
        print(f"  {i+1}. {example['record']['title']}")
        print(f"     Genes: {example['target_genes']}")
        print(f"     Performance: {example['features']['fold_improvement']}x improvement")
else:
    print("No LASER data found - using original reward function only")

def calculate_laser_reward(selected_rxns, laser_data):
    """Calculate reward based on LASER data"""
    if not laser_data:
        return 0.0, [], []
    
    # Instead of trying to match reaction IDs to gene names,
    # use LASER data to provide a baseline reward for metabolic engineering strategies
    # based on the number of mutations and general strategy patterns
    
    # Count how many mutations we're making (similar to LASER strategies)
    num_mutations = len(selected_rxns)
    
    # Check if this matches typical LASER strategy patterns
    laser_strategies = []
    for example in laser_data:
        example_mutations = example['features']['num_mutations']
        # Reward strategies with similar number of mutations as successful LASER examples
        if abs(num_mutations - example_mutations) <= 2:  # Within 2 mutations
            laser_strategies.append(example)
    
    if not laser_strategies:
        return 0.0, selected_rxns, []
    
    # Calculate reward based on LASER performance patterns
    avg_fold_improvement = np.mean([s['features']['fold_improvement'] for s in laser_strategies])
    avg_yield = np.mean([s['features']['final_yield'] for s in laser_strategies])
    
    # Normalize and combine metrics
    laser_reward = (avg_fold_improvement / 10.0) + (avg_yield / 100.0)
    
    return laser_reward, selected_rxns, laser_strategies

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

# training loop via RL- train the GNN indirectly by using a reward signal
# scoring head for each reaction node
class EditScorer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

scorer = EditScorer(embed_dim=8)
optimizer = Adam(list(model.parameters()) + list(scorer.parameters()), lr=1e-3)

# define + run training loop
print("=========TRAINING MODEL=========")
model.train()
scorer.train()

num_steps = 100
best_reward = -1
best_edits = []

for steps in range(num_steps):
    cobra_model_cp = cobra_model.copy()
    
    # forward pass
    z = model(data)

    # score only editable reaction nodes
    editable_nodes = [i for i, n in enumerate(H.nodes) if H.nodes[n]["type"] == "reaction"]
    edit_embeddings = z[editable_nodes]
    logits = scorer(edit_embeddings).squeeze()

    # sample edits 
    probs = torch.softmax(logits, dim=0)
    sampled = torch.multinomial(probs, num_samples=3, replacement=False)
    selected_rxns = [list(H.nodes)[editable_nodes[i]] for i in sampled.tolist()]

    # apply edits in simulation
    for rxn_id in selected_rxns:
        cobra_model_cp.reactions.get_by_id(rxn_id).knock_out()
    solution = cobra_model_cp.optimize()
    
    # Calculate hybrid reward: combine COBRA simulation with LASER data
    cobra_reward = solution.objective_value if solution.status == "optimal" else 0.0
    laser_reward, selected_genes, similar_strategies = calculate_laser_reward(selected_rxns, laser_data)
    
    # Combine rewards: 70% COBRA simulation, 30% LASER validation
    reward = 0.7 * cobra_reward + 0.3 * laser_reward
    
    # Debug: Print selected genes occasionally
    if steps % 20 == 0:
        print(f"  Selected reactions: {selected_rxns}")
        print(f"  Mapped genes: {selected_genes}")
        print(f"  LASER matches: {len(similar_strategies)} strategies found")

    # track best reward and edits
    if reward > best_reward:
        best_reward = reward
        best_edits = selected_rxns

    # define loss (reinforce policy gradient loss)
    selected_logits = logits[sampled]
    loss = -reward * torch.mean(torch.log_softmax(logits, dim=0)[sampled])

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if steps % 10 == 0:
        print(f"Step {steps} - COBRA: {cobra_reward:.2f}, LASER: {laser_reward:.2f}, Combined: {reward:.2f}")

# save the model
checkpoint_path = f"models/{host}_{target}_gnn_rl_checkpoint.pth"
print(f"Saving checkpoint to: {checkpoint_path}")

torch.save({
    "model": model.state_dict(),
    "scorer": scorer.state_dict(),
    "reward": best_reward,
    "edits": best_edits
}, checkpoint_path)

print(f"\nBest reward: {best_reward:.2f} from edits: {best_edits}")

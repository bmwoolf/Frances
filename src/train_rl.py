import networkx as nx
import json
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from networkx.algorithms.traversal.breadth_first_search import bfs_tree
from sklearn.manifold import TSNE

# import config
from config import host, target, config

# Ensure GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("GPU is required for training. No CUDA device found.")

# Define model classes first
class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)  # Match pretrained model names
        self.conv2 = GATConv(hidden_dim, out_dim)  # Match pretrained model names

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class EditScorer(nn.Module):
    """MLP to score edit sets"""
    def __init__(self, input_dim, hidden_dim=512):
        super(EditScorer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()

# Load pretrained model from supervised training
def load_pretrained_model():
    """Load the pretrained GNN and EditScorer from supervised training"""
    device = torch.device('cuda')
    
    # Initialize models with same architecture as supervised training
    gnn = GNN(in_dim=3, hidden_dim=512, out_dim=256).to(device)
    scorer = EditScorer(input_dim=256, hidden_dim=512).to(device)
    
    try:
        checkpoint_path = "models/E.coli_limonene_supervised_checkpoint.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        gnn.load_state_dict(checkpoint['gnn_state_dict'])
        scorer.load_state_dict(checkpoint['scorer_state_dict'])
        
        print(f"✅ Loaded pretrained model from {checkpoint_path}")
        print(f"   - Best Loss: {checkpoint.get('loss', 'Unknown'):.4f}")
        print(f"   - Epoch: {checkpoint.get('epoch', 'Unknown')}")
        
        return gnn, scorer, device
        
    except Exception as e:
        print(f"❌ Error loading pretrained model: {e}")
        print("Will train from scratch")
        return None, None, device

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

# Load pretrained model
pretrained_gnn, pretrained_scorer, device = load_pretrained_model()

# Move data to device
data = data.to(device)

# Use pretrained models if available, otherwise create new ones
if pretrained_gnn is not None and pretrained_scorer is not None:
    model = pretrained_gnn
    scorer = pretrained_scorer
    print("Using pretrained models for RL training")
else:
    model = GNN(in_dim=3, hidden_dim=512, out_dim=256)
    scorer = EditScorer(input_dim=256, hidden_dim=512)
    model = model.to(device)
    scorer = scorer.to(device)
    print("Training from scratch")

# Move models to device
model = model.to(device)
scorer = scorer.to(device)

optimizer = Adam(list(model.parameters()) + list(scorer.parameters()), lr=1e-3)

# Mixed precision training for GPU acceleration
scaler = GradScaler()

# define + run training loop
print("=========TRAINING MODEL=========")

# Configuration
num_steps = 10000
num_edits_per_step = 3  # Number of reactions to edit per training step
best_reward = -1
best_edits = []

print(f"Training configuration:")
print(f"  - Steps: {num_steps}")
print(f"  - Edits per step: {num_edits_per_step}")
print(f"  - Device: {device}")
print()

model.train()
scorer.train()

for steps in range(num_steps):
    cobra_model_cp = cobra_model.copy()
    
    # forward pass with mixed precision
    with autocast():
        z = model(data)

        # score only editable reaction nodes
        editable_nodes = [i for i, n in enumerate(H.nodes) if H.nodes[n]["type"] == "reaction"]
        edit_embeddings = z[editable_nodes]
        logits = scorer(edit_embeddings).squeeze()

    # sample edits with numerical stability
    # Clip logits to prevent extreme values
    logits = torch.clamp(logits, min=-10.0, max=10.0)
    probs = torch.softmax(logits, dim=0)
    
    # Check for invalid probabilities
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        print(f"Warning: Invalid probabilities detected at step {steps}, using uniform distribution")
        probs = torch.ones_like(probs) / len(probs)
    
    sampled = torch.multinomial(probs, num_samples=num_edits_per_step, replacement=False)
    selected_rxns = [list(H.nodes)[editable_nodes[i]] for i in sampled.tolist()]

    # apply edits in simulation
    for rxn_id in selected_rxns:
        cobra_model_cp.reactions.get_by_id(rxn_id).knock_out()
    solution = cobra_model_cp.optimize()
    
    # Calculate hybrid reward: combine COBRA simulation with pretrained model prediction
    cobra_reward = solution.objective_value if solution.status == "optimal" else 0.0
    
    # Initialize variables for debug output
    selected_genes = []
    similar_strategies = []
    laser_reward = 0.0
    predicted_yield = 0.0
    
    # Use pretrained model to predict yield from edits
    if pretrained_gnn is not None and pretrained_scorer is not None:
        # Get embeddings for the selected edits
        edit_embeddings = z[editable_nodes][sampled]
        
        # Predict yield using pretrained scorer
        with torch.no_grad():
            predictions = pretrained_scorer(edit_embeddings)
            predicted_yield = predictions.mean().item()  # Average the predictions
            
            # Normalize predicted yield to reasonable range [-1, 1]
            predicted_yield = torch.tanh(torch.tensor(predicted_yield)).item()
        
        # Combine rewards: 70% COBRA simulation, 30% normalized predicted yield
        reward = 0.7 * cobra_reward + 0.3 * predicted_yield
        
        if steps % 20 == 0:
            print(f"  Predicted yield: {predicted_yield:.3f}")
    else:
        # Fallback to original LASER reward if pretrained model not available
        laser_reward, selected_genes, similar_strategies = calculate_laser_reward(selected_rxns, laser_data)
        reward = 0.7 * cobra_reward + 0.3 * laser_reward
    
    # Debug: Print selected genes occasionally
    if steps % 20 == 0:
        print(f"  Selected reactions: {selected_rxns}")
        print(f"  Mapped genes: {selected_genes}")
        print(f"  LASER matches: {len(similar_strategies)} strategies found")

    # Ensure reward is reasonable
    if torch.isnan(torch.tensor(reward)) or torch.isinf(torch.tensor(reward)):
        print(f"Warning: Invalid reward detected at step {steps}, using fallback reward")
        reward = 0.0
    
    # track best reward and edits
    if reward > best_reward:
        best_reward = reward
        best_edits = selected_rxns

    # define loss (reinforce policy gradient loss)
    selected_logits = logits[sampled]
    loss = -reward * torch.mean(torch.log_softmax(logits, dim=0)[sampled])

    # backprop with gradient scaling
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if steps % 10 == 0:
        if pretrained_gnn is not None:
            print(f"Step {steps} - COBRA: {cobra_reward:.2f}, Predicted: {predicted_yield:.3f}, Combined: {reward:.2f}")
        else:
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

# Verify limonene-producing reactions exist
limonene_reactions = [r for r in cobra_model.reactions if 'limonene' in r.name.lower()]
print(f"Limonene reactions: {limonene_reactions}")

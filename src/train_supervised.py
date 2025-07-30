import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
from cobra import io
import json
import os

# Import our custom modules
from laser_dataset import LASERDataset
from config import host, target, config

# Ensure GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("GPU is required for training. No CUDA device found.")

# Import GNN components from main
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from networkx.algorithms.traversal.breadth_first_search import bfs_tree
from sklearn.manifold import TSNE

class GNN(nn.Module):
    """Graph Neural Network for node embeddings"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNN, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, out_dim)
        self.out_dim = out_dim
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class EditScorer(nn.Module):
    """MLP to score edit sets"""
    def __init__(self, input_dim, hidden_dim=64):
        super(EditScorer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()

def build_metabolic_network():
    """Build the metabolic network graph"""
    print("Building metabolic network...")
    
    # Load COBRA model
    cobra_model = io.load_json_model("iML1515.json")
    
    # Create graph
    G = nx.DiGraph()
    
    # Add reactions and metabolites
    for reaction in cobra_model.reactions:
        G.add_node(reaction.id, type="reaction", genes=reaction.gene_reaction_rule or "")
        
        for metabolite, coeff in reaction.metabolites.items():
            met_id = metabolite.id
            G.add_node(met_id, type="metabolite", genes="")
            
            if coeff < 0:  # Substrate
                G.add_edge(met_id, reaction.id, role="substrate", coeff=coeff)
            else:  # Product
                G.add_edge(reaction.id, met_id, role="product", coeff=coeff)
    
    # Add engineered pathway for limonene production
    G.add_node("gpp_c", type="metabolite", genes="")
    G.add_node("limonene_c", type="metabolite", genes="")
    
    # Add engineered reactions
    G.add_node("GPPS", type="reaction", genes="gpps")
    G.add_node("LS", type="reaction", genes="limonene_synthase")
    
    G.add_edge("gpp_c", "GPPS", role="substrate", coeff=-1.0)
    G.add_edge("GPPS", "gpp_c", role="product", coeff=1.0)
    G.add_edge("gpp_c", "LS", role="substrate", coeff=-1.0)
    G.add_edge("LS", "limonene_c", role="product", coeff=1.0)
    
    # Ensure all nodes have consistent attributes
    for node in G.nodes():
        if "type" not in G.nodes[node]:
            G.nodes[node]["type"] = "unknown"
        if "genes" not in G.nodes[node]:
            G.nodes[node]["genes"] = ""
    
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    
    return G, cobra_model

def create_node_features(G):
    """Create node features for the graph"""
    features = []
    node_types = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get("type", "unknown")
        node_types.append(node_type)
        
        # Create one-hot encoding for node type
        if node_type == "reaction":
            features.append([1, 0, 0])  # [reaction, metabolite, engineered]
        elif node_type == "metabolite":
            features.append([0, 1, 0])
        else:
            features.append([0, 0, 1])
    
    return torch.tensor(features, dtype=torch.float)

def train_supervised_model():
    """Train the model using LASER data"""
    print("=== SUPERVISED TRAINING WITH LASER DATA ===")
    
    # Set device
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Build network
    G, cobra_model = build_metabolic_network()
    
    # Create node features
    node_features = create_node_features(G)
    
    # Convert to PyTorch Geometric format
    data = from_networkx(G)
    data.x = node_features
    data = data.to(device)
    
    # Store the graph data for LASER dataset
    graph_data = data
    
    # Initialize models
    gnn = GNN(in_dim=3, hidden_dim=512, out_dim=256).to(device)  # Optimal size for this task
    scorer = EditScorer(input_dim=256, hidden_dim=512).to(device)  # Optimal scorer
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        gnn = nn.DataParallel(gnn)
        scorer = nn.DataParallel(scorer)
    
    # Create LASER dataset
    laser_dataset = LASERDataset(
        laser_data_path='data/laser_training_data.json',
        cobra_model=cobra_model,
        network_graph=G,
        device=device
    )
    
    # Update LASER dataset with graph data
    laser_dataset.graph_data = graph_data
    
    # Create data loader with custom collate function
    def custom_collate(batch):
        """Custom collate function to handle variable-length tensors"""
        # Pad node_indices to the same length
        max_length = max(len(item['node_indices']) for item in batch)
        padded_indices = []
        
        for item in batch:
            indices = item['node_indices']
            if len(indices) < max_length:
                # Pad with -1 (invalid index)
                padded = torch.cat([indices, torch.full((max_length - len(indices),), -1, dtype=torch.long, device=indices.device)])
            else:
                padded = indices
            padded_indices.append(padded)
        
        return {
            'node_indices': torch.stack(padded_indices),
            'target_yield': torch.stack([item['target_yield'] for item in batch]),
            'fold_improvement': [item['fold_improvement'] for item in batch],
            'paper_title': [item['paper_title'] for item in batch],
            'reaction_ids': [item['reaction_ids'] for item in batch],
            'lengths': torch.tensor([len(item['node_indices']) for item in batch])
        }
    
    dataloader = DataLoader(laser_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)  # Optimal batch size for learning
    
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=0.1)  # More robust to outliers
    optimizer = optim.AdamW(list(gnn.parameters()) + list(scorer.parameters()), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=1e-6)
    
    # Mixed precision training for better GPU utilization
    scaler = GradScaler()
    
    # Gradient accumulation for effective larger batch sizes
    accumulation_steps = 4  # Effective batch size = 64 * 4 = 256
    
    # Training loop
    num_epochs = 10000 # Scaled up for better learning
    best_loss = float('inf')
    
    print(f"Training on {len(laser_dataset)} LASER examples")
    print(f"Number of epochs: {num_epochs}")
    print(f"Expected training time: ~{num_epochs * 2 / 60:.1f} hours")
    print(f"Starting training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Mixed precision training
            with autocast():
                # Get embeddings for the batch
                batch_embeddings = laser_dataset.get_batch_embeddings(batch_data, gnn)
                
                # Predict yields
                predicted_yields = scorer(batch_embeddings)
                
                # Get target yields
                target_yields = batch_data['target_yield']
                
                # Calculate loss
                loss = criterion(predicted_yields, target_yields)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print batch info occasionally
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
                print(f"  Predicted: {predicted_yields.detach().numpy()}")
                print(f"  Target: {target_yields.detach().numpy()}")
                print(f"  Papers: {[title[:50] + '...' for title in batch_data['paper_title']]}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = f"models/{host}_{target}_supervised_checkpoint.pth"
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': gnn.state_dict(),
                'scorer_state_dict': scorer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    print(f"Training completed! Best loss: {best_loss:.4f}")
    
    # Test the model on a few examples
    print("\n=== TESTING MODEL ===")
    gnn.eval()
    scorer.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches
                break
                
            batch_embeddings = laser_dataset.get_batch_embeddings(batch_data, gnn)
            predicted_yields = scorer(batch_embeddings)
            
            print(f"\nTest Example {i+1}:")
            for j in range(len(batch_data['paper_title'])):
                print(f"  Paper: {batch_data['paper_title'][j][:60]}...")
                print(f"  Reactions: {batch_data['reaction_ids'][j]}")
                print(f"  Predicted Yield: {predicted_yields[j].item():.3f}")
                print(f"  Target Yield: {batch_data['target_yield'][j].item():.3f}")
                print(f"  Fold Improvement: {batch_data['fold_improvement'][j]}x")

if __name__ == "__main__":
    train_supervised_model() 
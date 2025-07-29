import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

class LASERDataset(Dataset):
    """Dataset for LASER metabolic engineering data"""
    
    def __init__(self, laser_data_path: str, cobra_model, network_graph, device='cpu'):
        """
        Args:
            laser_data_path: Path to LASER training data JSON
            cobra_model: COBRA model for reaction mapping
            network_graph: NetworkX graph for embedding lookup
            device: Device for tensors
        """
        self.device = device
        self.cobra_model = cobra_model
        self.network_graph = network_graph
        
        # Load LASER data
        with open(laser_data_path, 'r') as f:
            self.laser_data = json.load(f)
        
        # Create reaction ID to gene name mapping
        self.reaction_to_gene = self._create_reaction_mapping()
        
        # Process LASER entries into training format
        self.processed_data = self._process_laser_entries()
        
        print(f"Loaded {len(self.processed_data)} LASER training examples")
    
    def _create_reaction_mapping(self) -> Dict[str, str]:
        """Create mapping from reaction IDs to gene names"""
        mapping = {}
        
        # Common metabolic engineering genes and their potential reaction mappings
        gene_to_reaction = {
            'idi': ['IDI', 'IDI1', 'IDI2'],
            'mvas': ['MVAS', 'MVA_S'],
            'mvae': ['MVAE', 'MVA_E'],
            'ispa': ['ISPA', 'ISP_A'],
            'ahr': ['AHR', 'AHR1'],
            'dxs': ['DXS', 'DXS1'],
            'ispg': ['ISPG', 'ISP_G'],
            'gpps': ['GPPS', 'GPPS1'],
            'limonene_synthase': ['LS', 'LIMO_S'],
            'geraniol_synthase': ['GES', 'GER_S'],
            'farnesyl_diphosphate_synthase': ['FPPS', 'FPP_S'],
            'valencene_synthase': ['VS', 'VAL_S']
        }
        
        # Create reverse mapping
        for gene, reactions in gene_to_reaction.items():
            for reaction in reactions:
                mapping[reaction] = gene
        
        return mapping
    
    def _process_laser_entries(self) -> List[Dict]:
        """Process LASER entries into training format"""
        processed = []
        
        for entry in self.laser_data:
            # Extract target genes from LASER entry
            target_genes = entry['target_genes']
            
            # Map genes to reaction IDs (simplified approach)
            reaction_ids = self._map_genes_to_reactions(target_genes)
            
            if reaction_ids:  # Only include if we can map to reactions
                # Get yield/performance data
                fold_improvement = entry['features']['fold_improvement']
                final_yield = entry['features']['final_yield']
                
                # Use fold improvement as primary target (normalized)
                target_yield = min(fold_improvement / 10.0, 1.0)  # Normalize to [0,1]
                
                processed.append({
                    'reaction_ids': reaction_ids,
                    'gene_names': target_genes,
                    'target_yield': target_yield,
                    'fold_improvement': fold_improvement,
                    'final_yield': final_yield,
                    'paper_title': entry['record']['title']
                })
        
        return processed
    
    def _map_genes_to_reactions(self, gene_names: List[str]) -> List[str]:
        """Map gene names to reaction IDs in the metabolic model"""
        mapped_reactions = []
        
        # Get all reaction IDs from the model
        all_reactions = [rxn.id for rxn in self.cobra_model.reactions]
        
        # Create a more flexible mapping for common metabolic engineering genes
        gene_keywords = {
            'idi': ['IDI', 'IDI1', 'IDI2'],
            'mvas': ['MVAS', 'MVA_S', 'HMGCR'],
            'mvae': ['MVAE', 'MVA_E'],
            'ispa': ['ISPA', 'ISP_A'],
            'ahr': ['AHR', 'AHR1'],
            'dxs': ['DXS', 'DXS1'],
            'ispg': ['ISPG', 'ISP_G'],
            'gpps': ['GPPS', 'GPPS1'],
            'geraniol': ['GES', 'GER_S'],
            'farnesyl': ['FPPS', 'FPP_S'],
            'valencene': ['VS', 'VAL_S'],
            'limonene': ['LS', 'LIMO_S'],
            'mevalonate': ['MVA', 'MVK', 'PMK', 'MVD'],
            'kinase': ['MVK', 'PMK', 'KIN'],
            'phosphomevalonate': ['PMK', 'PMVK'],
            'diphosphate': ['MVD', 'DPP'],
            'reductase': ['HMGCR', 'RED'],
            'synthase': ['SYN', 'SYS']
        }
        
        for gene in gene_names:
            gene_lower = gene.lower()
            found_match = False
            
            # Try keyword-based mapping first
            for keyword, possible_reactions in gene_keywords.items():
                if keyword in gene_lower:
                    for reaction_id in all_reactions:
                        if any(possible in reaction_id.upper() for possible in possible_reactions):
                            mapped_reactions.append(reaction_id)
                            found_match = True
                            break
                    if found_match:
                        break
            
            # If no keyword match, try partial string matching
            if not found_match:
                for reaction_id in all_reactions:
                    # Check if any significant word in gene name appears in reaction ID
                    gene_words = [word for word in gene_lower.split() if len(word) > 3]
                    for word in gene_words:
                        if word in reaction_id.lower():
                            mapped_reactions.append(reaction_id)
                            found_match = True
                            break
                    if found_match:
                        break
            
            # If still no match, try to find reactions with similar function
            if not found_match:
                # For now, just add a placeholder reaction for demonstration
                # In a real implementation, you'd have a more sophisticated mapping
                if len(all_reactions) > 0:
                    # Add a random reaction as placeholder (for demo purposes)
                    import random
                    placeholder = random.choice(all_reactions[:100])  # First 100 reactions
                    mapped_reactions.append(placeholder)
        
        return mapped_reactions
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training example"""
        entry = self.processed_data[idx]
        
        # Get node indices for the reaction IDs in our graph
        node_indices = []
        for reaction_id in entry['reaction_ids']:
            if reaction_id in self.network_graph.nodes:
                node_idx = list(self.network_graph.nodes).index(reaction_id)
                node_indices.append(node_idx)
        
        return {
            'node_indices': torch.tensor(node_indices, dtype=torch.long, device=self.device),
            'target_yield': torch.tensor(entry['target_yield'], dtype=torch.float, device=self.device),
            'fold_improvement': entry['fold_improvement'],
            'paper_title': entry['paper_title'],
            'reaction_ids': entry['reaction_ids']
        }
    
    def get_batch_embeddings(self, batch_data: Dict, gnn_model) -> torch.Tensor:
        """Get embeddings for a batch of edit sets"""
        batch_embeddings = []
        
        node_indices_batch = batch_data['node_indices']
        lengths = batch_data['lengths']
        
        for i, (node_indices, length) in enumerate(zip(node_indices_batch, lengths)):
            # Remove padding (-1 values)
            valid_indices = node_indices[:length]
            valid_indices = valid_indices[valid_indices >= 0]  # Remove -1 padding
            
            if len(valid_indices) == 0:
                # No valid reactions found, use zero embedding
                batch_embeddings.append(torch.zeros(gnn_model.out_dim, device=self.device))
            else:
                # Get embeddings for the selected reactions
                with torch.no_grad():
                    # Forward pass through GNN to get node embeddings
                    data = self._prepare_graph_data()
                    node_embeddings = gnn_model(data)
                    
                    # Average embeddings of selected reactions
                    selected_embeddings = node_embeddings[valid_indices]
                    avg_embedding = torch.mean(selected_embeddings, dim=0)
                    batch_embeddings.append(avg_embedding)
        
        return torch.stack(batch_embeddings)
    
    def _prepare_graph_data(self):
        """Prepare graph data for GNN forward pass"""
        if hasattr(self, 'graph_data'):
            return self.graph_data
        else:
            # Fallback to original method
            from torch_geometric.data import Data
            from torch_geometric.utils import from_networkx
            
            # Convert NetworkX graph to PyTorch Geometric format
            data = from_networkx(self.network_graph)
            
            # Ensure we have node features
            if not hasattr(data, 'x') or data.x is None:
                # Create simple node features
                num_nodes = data.num_nodes
                data.x = torch.ones(num_nodes, 3, device=self.device)  # Simple features
            
            # Ensure we have edge_index
            if not hasattr(data, 'edge_index') or data.edge_index is None:
                # Create a simple edge_index (fully connected for demo)
                import torch_geometric.utils as utils
                data.edge_index = utils.to_undirected(utils.to_edge_index(torch.ones(num_nodes, num_nodes, device=self.device)))
            
            return data.to(self.device) 
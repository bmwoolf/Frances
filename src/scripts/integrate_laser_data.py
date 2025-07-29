import json
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import GNN, EditScorer

def load_laser_training_data():
    """Load the LASER training data"""
    try:
        with open('data/laser_training_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("LASER training data not found. Run simple_laser_extract.py first.")
        return []

def create_laser_reward_function(training_data):
    """
    Create a reward function based on LASER data
    """
    def calculate_reward(selected_genes, target_product="limonene"):
        """
        Calculate reward based on LASER training data
        """
        # Find similar strategies in LASER data
        similar_strategies = []
        
        for example in training_data:
            # Check if this example has similar genes
            overlap = set(selected_genes) & set(example['target_genes'])
            if len(overlap) > 0:
                similar_strategies.append(example)
        
        if not similar_strategies:
            return 0.0  # No similar strategies found
        
        # Calculate average performance of similar strategies
        avg_fold_improvement = np.mean([s['features']['fold_improvement'] for s in similar_strategies])
        avg_yield = np.mean([s['features']['final_yield'] for s in similar_strategies])
        
        # Normalize and combine metrics
        reward = (avg_fold_improvement / 10.0) + (avg_yield / 100.0)  # Normalize to 0-1 range
        
        return reward
    
    return calculate_reward

def train_with_laser_data(model, scorer, training_data, num_episodes=100):
    """
    Train the GNN using LASER data as guidance
    """
    print("=== Training with LASER Data ===")
    
    # Create reward function
    reward_fn = create_laser_reward_function(training_data)
    
    # Training loop
    optimizer = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=1e-3)
    
    for episode in range(num_episodes):
        # Forward pass (you'll need to adapt this to your actual data)
        # This is a simplified version - you'll need to integrate with your actual graph data
        
        # Simulate gene selection
        selected_genes = ['dxs', 'idi', 'ispG', 'gpps']  # Example genes
        
        # Calculate reward from LASER data
        reward = reward_fn(selected_genes)
        
        # Update model (simplified - you'll need to adapt to your actual training loop)
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {reward:.3f}")
    
    print("Training completed!")

def analyze_laser_patterns(training_data):
    """
    Analyze patterns in LASER data to inform model design
    """
    print("\n=== LASER Data Analysis ===")
    
    if not training_data:
        print("No training data available")
        return
    
    # Analyze mutation patterns
    all_genes = []
    all_fold_improvements = []
    
    for example in training_data:
        all_genes.extend(example['target_genes'])
        all_fold_improvements.append(example['features']['fold_improvement'])
    
    # Find most common genes
    from collections import Counter
    gene_counts = Counter(all_genes)
    
    print("Most commonly modified genes in terpenoid production:")
    for gene, count in gene_counts.most_common(10):
        print(f"  {gene}: {count} times")
    
    print(f"\nPerformance statistics:")
    print(f"  Average fold improvement: {np.mean(all_fold_improvements):.2f}")
    print(f"  Max fold improvement: {max(all_fold_improvements):.2f}")
    print(f"  Min fold improvement: {min(all_fold_improvements):.2f}")
    
    # Analyze mutation types
    mutation_types = []
    for example in training_data:
        features = example['features']
        if features['has_knockout']:
            mutation_types.append('knockout')
        if features['has_overexpression']:
            mutation_types.append('overexpression')
        if features['has_plasmid']:
            mutation_types.append('plasmid')
    
    type_counts = Counter(mutation_types)
    print(f"\nMutation types used:")
    for mut_type, count in type_counts.items():
        print(f"  {mut_type}: {count} times")

def main():
    """Main function to integrate LASER data with your model"""
    print("=== LASER Data Integration ===")
    
    # Load LASER training data
    training_data = load_laser_training_data()
    
    if not training_data:
        print("No LASER data available. Please run simple_laser_extract.py first.")
        return
    
    # Analyze patterns in LASER data
    analyze_laser_patterns(training_data)
    
    # Create model (you'll need to adapt this to your actual model)
    # model = GNN(in_dim=3, hidden_dim=16, out_dim=8)
    # scorer = EditScorer(embed_dim=8)
    
    # Train with LASER data
    # train_with_laser_data(model, scorer, training_data)
    
    print("\nLASER data integration completed!")
    print("You can now use this data to:")
    print("1. Guide your GNN training with real metabolic engineering examples")
    print("2. Validate your model's predictions against known successful strategies")
    print("3. Improve reward functions based on actual performance data")

if __name__ == "__main__":
    main() 
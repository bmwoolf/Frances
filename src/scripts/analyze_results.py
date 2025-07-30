#!/usr/bin/env python3
"""
Analyze RL training results and extract best metabolic engineering strategies
"""

import torch
import json
import numpy as np
from cobra import Model
import networkx as nx
from torch_geometric.utils import from_networkx

def load_training_results(checkpoint_path):
    """Load the final training results"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("=== RL Training Results ===")
    print(f"Best reward achieved: {checkpoint['reward']:.4f}")
    print(f"Best reaction edits: {checkpoint['edits']}")
    print(f"Model saved to: {checkpoint_path}")
    
    return checkpoint

def analyze_metabolic_impact(selected_rxns, cobra_model):
    """Analyze the metabolic impact of the selected reactions"""
    print("\n=== Metabolic Impact Analysis ===")
    
    # Create a copy for simulation
    model_cp = cobra_model.copy()
    
    # Apply the edits
    for rxn_id in selected_rxns:
        if rxn_id in model_cp.reactions:
            rxn = model_cp.reactions.get_by_id(rxn_id)
            print(f"Knocking out: {rxn_id} - {rxn.name}")
            rxn.knock_out()
        else:
            print(f"Warning: Reaction {rxn_id} not found in model")
    
    # Run simulation
    solution = model_cp.optimize()
    
    print(f"\nSimulation Results:")
    print(f"  Status: {solution.status}")
    print(f"  Objective value: {solution.objective_value:.4f}")
    
    # Check limonene production
    limonene_rxns = [r for r in model_cp.reactions if 'limonene' in r.name.lower()]
    if limonene_rxns:
        print(f"\nLimonene production reactions:")
        for rxn in limonene_rxns:
            flux = solution.fluxes.get(rxn.id, 0)
            print(f"  {rxn.id}: {flux:.4f}")
    else:
        print("\nNo limonene production reactions found in model")
    
    return solution

def generate_strategy_report(selected_rxns, cobra_model, laser_data):
    """Generate a comprehensive strategy report"""
    print("\n=== Metabolic Engineering Strategy Report ===")
    
    print(f"\n1. RECOMMENDED GENE EDITS:")
    for i, rxn_id in enumerate(selected_rxns, 1):
        if rxn_id in cobra_model.reactions:
            rxn = cobra_model.reactions.get_by_id(rxn_id)
            print(f"   {i}. {rxn_id} - {rxn.name}")
        else:
            print(f"   {i}. {rxn_id} - (reaction not in model)")
    
    print(f"\n2. IMPLEMENTATION STEPS:")
    print("   a) Clone the target organism")
    print("   b) Design CRISPR guide RNAs for each gene")
    print("   c) Perform gene knockout experiments")
    print("   d) Measure limonene production")
    print("   e) Optimize growth conditions")
    
    print(f"\n3. EXPECTED OUTCOMES:")
    print("   - Reduced competing metabolic pathways")
    print("   - Enhanced carbon flux toward limonene")
    print("   - Improved yield compared to wild type")
    
    # Check if strategy matches LASER data
    if laser_data:
        print(f"\n4. LASER DATABASE COMPARISON:")
        matching_strategies = []
        for entry in laser_data:
            if any(gene in str(entry['target_genes']) for gene in selected_rxns):
                matching_strategies.append(entry)
        
        if matching_strategies:
            print(f"   Found {len(matching_strategies)} similar strategies in LASER:")
            for strategy in matching_strategies[:3]:  # Show top 3
                print(f"   - {strategy['record']['title']}")
                print(f"     Genes: {strategy['target_genes']}")
                print(f"     Improvement: {strategy['features']['fold_improvement']}x")
        else:
            print("   No exact matches found in LASER database")
            print("   This may be a novel strategy combination")

def main():
    """Main analysis function"""
    # Load the training results
    checkpoint_path = "models/E.coli_limonene_gnn_rl_checkpoint.pth"
    
    try:
        checkpoint = load_training_results(checkpoint_path)
        selected_rxns = checkpoint['edits']
        
        # Load COBRA model
        from cobra.io import load_json_model
        cobra_model = load_json_model('iML1515.json')
        
        # Load LASER data
        try:
            with open('data/laser_training_data.json', 'r') as f:
                laser_data = json.load(f)
        except:
            laser_data = []
        
        # Analyze metabolic impact
        solution = analyze_metabolic_impact(selected_rxns, cobra_model)
        
        # Generate strategy report
        generate_strategy_report(selected_rxns, cobra_model, laser_data)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Results saved to: {checkpoint_path}")
        print(f"Best strategy: {selected_rxns}")
        print(f"Expected reward: {checkpoint['reward']:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file {checkpoint_path} not found")
        print("Make sure RL training has completed successfully")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main() 
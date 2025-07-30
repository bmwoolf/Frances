#!/usr/bin/env python3
"""
Analyze data usage per COBRA simulation
"""

import sys
import os
from cobra.io import load_json_model
import json
import numpy as np

def analyze_cobra_model_data():
    """Analyze the data structure and size of COBRA model"""
    
    print("=== COBRA Model Data Analysis ===")
    
    # Load the model
    model = load_json_model('iML1515.json')
    
    # Analyze reactions
    print(f"\n1. REACTIONS:")
    print(f"   Total reactions: {len(model.reactions)}")
    print(f"   Reaction IDs: {len([r.id for r in model.reactions])}")
    print(f"   Reaction names: {len([r.name for r in model.reactions])}")
    
    # Analyze metabolites
    print(f"\n2. METABOLITES:")
    print(f"   Total metabolites: {len(model.metabolites)}")
    print(f"   Metabolite IDs: {len([m.id for m in model.metabolites])}")
    print(f"   Metabolite names: {len([m.name for m in model.metabolites])}")
    
    # Analyze stoichiometric matrix
    print(f"\n3. STOICHIOMETRIC MATRIX:")
    print(f"   Matrix shape: {model.S.shape}")
    print(f"   Non-zero elements: {model.S.nnz}")
    print(f"   Sparsity: {1 - model.S.nnz / (model.S.shape[0] * model.S.shape[1]):.3f}")
    
    # Analyze constraints
    print(f"\n4. CONSTRAINTS:")
    print(f"   Lower bounds: {len([r.lower_bound for r in model.reactions])}")
    print(f"   Upper bounds: {len([r.upper_bound for r in model.reactions])}")
    print(f"   Objective coefficients: {len([r.objective_coefficient for r in model.reactions])}")
    
    # Memory usage estimation
    print(f"\n5. MEMORY USAGE ESTIMATION:")
    
    # Reactions data
    reaction_data = len(model.reactions) * (8 + 8 + 8 + 8)  # id, name, lower_bound, upper_bound
    metabolite_data = len(model.metabolites) * (8 + 8 + 8)   # id, name, compartment
    matrix_data = model.S.nnz * 8  # sparse matrix elements
    constraint_data = len(model.reactions) * 8 * 3  # bounds and objectives
    
    total_bytes = reaction_data + metabolite_data + matrix_data + constraint_data
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"   Reactions: {reaction_data / 1024:.1f} KB")
    print(f"   Metabolites: {metabolite_data / 1024:.1f} KB")
    print(f"   Stoichiometric matrix: {matrix_data / 1024:.1f} KB")
    print(f"   Constraints: {constraint_data / 1024:.1f} KB")
    print(f"   Total estimated: {total_mb:.2f} MB")
    
    return model

def analyze_simulation_data_usage():
    """Analyze data usage for a single simulation"""
    
    print(f"\n=== Single COBRA Simulation Data Usage ===")
    
    model = load_json_model('iML1515.json')
    
    # Data for one simulation
    print(f"\n1. INPUT DATA:")
    print(f"   Model size: ~{len(model.reactions)} reactions")
    print(f"   Knockout data: 3 reaction IDs (minimal)")
    print(f"   Optimization parameters: ~10 parameters")
    
    print(f"\n2. COMPUTATION DATA:")
    print(f"   Linear programming variables: {len(model.reactions)}")
    print(f"   Linear programming constraints: {len(model.metabolites)}")
    print(f"   Matrix operations: {model.S.shape[0]} x {model.S.shape[1]}")
    
    print(f"\n3. OUTPUT DATA:")
    print(f"   Flux values: {len(model.reactions)} values")
    print(f"   Objective value: 1 value")
    print(f"   Solution status: 1 value")
    
    # Estimate computation complexity
    print(f"\n4. COMPUTATION COMPLEXITY:")
    print(f"   Variables: {len(model.reactions)}")
    print(f"   Constraints: {len(model.metabolites)}")
    print(f"   Matrix density: {model.S.nnz / (model.S.shape[0] * model.S.shape[1]):.3f}")
    print(f"   LP solver iterations: ~100-1000 (varies)")
    
    return model

def benchmark_memory_usage():
    """Benchmark actual memory usage"""
    
    import psutil
    import gc
    
    print(f"\n=== Memory Usage Benchmark ===")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Load model
    model = load_json_model('iML1515.json')
    after_load_memory = process.memory_info().rss / 1024 / 1024
    print(f"After model load: {after_load_memory:.2f} MB")
    print(f"Model memory usage: {after_load_memory - initial_memory:.2f} MB")
    
    # Run simulation
    model_cp = model.copy()
    model_cp.reactions.get_by_id('ACALD').knock_out()
    solution = model_cp.optimize()
    
    after_simulation_memory = process.memory_info().rss / 1024 / 1024
    print(f"After simulation: {after_simulation_memory:.2f} MB")
    print(f"Simulation memory overhead: {after_simulation_memory - after_load_memory:.2f} MB")
    
    # Cleanup
    del model, model_cp, solution
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"After cleanup: {final_memory:.2f} MB")

if __name__ == "__main__":
    model = analyze_cobra_model_data()
    analyze_simulation_data_usage()
    benchmark_memory_usage() 
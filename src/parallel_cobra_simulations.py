#!/usr/bin/env python3
"""
Parallel COBRA simulation implementation
"""

import multiprocessing as mp
from functools import partial
import numpy as np
from cobra import Model
import time

def run_cobra_simulation(args):
    """Run a single COBRA simulation"""
    reaction_combination, model_path = args
    
    # Load model (each process needs its own copy)
    from cobra.io import load_json_model
    model = load_json_model(model_path)
    
    # Apply knockouts
    for rxn_id in reaction_combination:
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).knock_out()
    
    # Run simulation
    solution = model.optimize()
    reward = solution.objective_value if solution.status == "optimal" else 0.0
    
    return reaction_combination, reward

def parallel_cobra_batch(reaction_combinations, model_path, num_processes=8):
    """Run multiple COBRA simulations in parallel"""
    
    # Prepare arguments for each simulation
    args_list = [(combo, model_path) for combo in reaction_combinations]
    
    # Run simulations in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_cobra_simulation, args_list)
    
    # Convert to dictionary
    cobra_cache = {}
    for combo, reward in results:
        cobra_cache[tuple(sorted(combo))] = reward
    
    return cobra_cache

def benchmark_parallel_vs_sequential():
    """Benchmark parallel vs sequential performance"""
    
    # Generate test combinations
    all_reactions = ['RXN1', 'RXN2', 'RXN3', 'RXN4', 'RXN5']
    test_combinations = []
    for i in range(50):  # Test 50 combinations
        combo = np.random.choice(all_reactions, size=3, replace=False)
        test_combinations.append(combo.tolist())
    
    model_path = 'iML1515.json'
    
    print("=== Parallel vs Sequential COBRA Benchmark ===")
    
    # Sequential timing
    print("Running sequential simulations...")
    start_time = time.time()
    sequential_results = {}
    for combo in test_combinations:
        result = run_cobra_simulation((combo, model_path))
        sequential_results[tuple(sorted(result[0]))] = result[1]
    sequential_time = time.time() - start_time
    
    # Parallel timing
    print("Running parallel simulations...")
    start_time = time.time()
    parallel_results = parallel_cobra_batch(test_combinations, model_path, num_processes=8)
    parallel_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Sequential time: {sequential_time:.2f} seconds")
    print(f"  Parallel time: {parallel_time:.2f} seconds")
    print(f"  Speedup: {sequential_time/parallel_time:.2f}x")
    print(f"  Simulations per second (sequential): {len(test_combinations)/sequential_time:.2f}")
    print(f"  Simulations per second (parallel): {len(test_combinations)/parallel_time:.2f}")

if __name__ == "__main__":
    benchmark_parallel_vs_sequential() 
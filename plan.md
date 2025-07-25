Frances dev plan

## Phase 1: Close the Loop (connect the target + host all the way to the output gene edits + simulated yield)
  ### 1. Entrypoint Function
  #### --> modeling the inputs --> outputs 
  optimize(target_molecule="limonene", host="E.coli")  
  → { 
      "knockout": [...], 
      "overexpress": [...], 
      "predicted_yield": X g/L 
    }

  ### 2. Build the Simulated Lab
  #### --> Model the underlying biology using the most accurate math and physics formulas we have today (and those formulas are just models themselves)
  Download BiGG GEMs for E. coli: BiGG models
  Run CobraPy FBA:	Simulate gene edits → yield


## Phase 2: Build the Optimizer Model (core engine, proposes the best gene edits to make)
Parse KEGG/MetaCyc pathways to graphs: NetworkX / PyG Data
Create host-specific metabolic graph: Match BiGG to KEGG

  ### 2. Model Components
  GNN + Attention	PyTorch Geometric: GATConv
  RL Policy (actions = gene edits): Ray RLlib (PPO or REINFORCE)
  Simulated environment: CobraPy or toy proxy

  ### 3. Reward Function
  Reward = predicted yield (g/L)
  Penalize over-complex edits (keep interventions minimal)

#### Phase 3: Deploy the API
Wrap optimize() in FastAPI endpoint
Store results in JSON for retraining
Package in Docker for labs to deploy on their local servers

## Extension
Rebuild the same system, but replace CobraPy + RL with deep learning, trained on a high-quality dataset of real gene edits → yields (Ginkgo's dataset, Joint BioEnergy Institute at Berkeley?)
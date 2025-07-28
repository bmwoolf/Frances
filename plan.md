Frances dev plan

Plan for building Frances, an API-first synbio tool that suggests the best gene edits (knockouts + overexpressions) to maximize production of a target molecule in a given microbial host, and predicts the expected yield.

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
  Download BiGG GEMs for E. coli: BiGG model for E Coli iML1515
  Run CobraPy FBA:	Simulate gene edits → yield


## Phase 2: Build the Optimizer Model (core engine, proposes the best gene edits to make)
  ### 1. Import Empirical Data for Pretraining
    - LASER, EDD, ICE, ABF, KBase
    - Clean + unify edit/yield/host/target records
    - Use for supervised warm-up or RL policy initialization

  ### 2. Parse KEGG/MetaCyc pathways to graphs
    - NetworkX / PyG Data

  ### 3. Create host-specific metabolic graph
    - Match BiGG to KEGG

  ### 4. Model Components
  GNN + Attention	PyTorch Geometric: GATConv
  RL Policy (actions = gene edits): Ray RLlib (PPO or REINFORCE)
  Simulated environment: CobraPy or toy proxy

  ### 5. Reward Function
  Reward = predicted yield (g/L)
  Penalize over-complex edits (keep interventions minimal)

#### Phase 3: Deploy the API
Wrap optimize() in FastAPI endpoint
Store results in JSON for retraining
Package in Docker for labs to deploy on their local servers

## Extensions
1. Rebuild the same system, but replace CobraPy + RL with deep learning, trained on a high-quality dataset of real gene edits → yields 

2. BioCAD Integration (TeselaGen, Asimov, SBOL Canvas): Frances fits after target + host selection. It takes design outputs (e.g. host strain, target molecule, constraints) from tools like TeselaGen or SBOL Canvas, computes optimal gene edits, and returns a recommended intervention set with predicted yield—ready for build or simulation.

3. LIMS/ELN Embedding (Benchling, KBase, SciNote): Frances plugs into strain records or assay workflows. It uses existing metadata (host, target, design intent) to generate edits + predicted yield, closing the loop inside tools already used for DBTL tracking.

4. Replace ODEs from older simulation software with Laplace Transforms
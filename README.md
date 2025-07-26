# Frances
A closed-loop metabolic pathway optimizer that optimizes a microbial production loop for a specific target molecule.

## Root function 
`optimize(target_molecule)` --> `strain_design + pathway edits + predicted yield`

Example:
```python
optimize(target_molecule="limonene", host_organism="E.coli") → { "knockout": [geneX], "overexpress": [geneY], "yield": 12.3 g/L }
```

## Steps
1. Ingest a target molecule and host organism (e.g. "limonene", "E.coli")
2. Parse metabolic pathway data and build a host-specific graph
3. Run GNN + attention over the graph to learn pathway structure and bottlenecks
4. Use an RL policy to propose optimal gene knockouts and overexpressions
5. Simulate predicted yield using CobraPy (FBA engine)
6. Return actionable strain design via optimize()
7. Log all input/output pairs for retracing, evaluation, and retraining

## Tools
Data sources: LASER DB, ICE/EDD, MetaCyc, KEGG, BiGG, KBase
Empirical pretraining: LASER, JBEI/ABF EDD datasets
Graph construction: NetworkX, KEGG parser
GNN + Attention: PyTorch Geometric (GATConv)
RL policy (gene edits): Ray RLlib (PPO), optionally SBX for fine control
Flux simulation: CobraPy (BiGG GEMs) or toy FBA proxy
API Deployment: FastAPI + Docker

Extension: SBOL Canvas, Benchling, or CLI input integration

## Running this repo

### Setup
```bash
# Clone the repository
git clone https://github.com/bmwoolf/Frances
cd Frances

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start the API
```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run the Flask API
python app.py
```

The API will start on `http://localhost:5000`

### Usage Examples

#### Basic optimization request
```bash
curl -X POST http://localhost:5000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "target_molecule": "limonene",
    "host": "E.coli",
    "constraints": {
      "max_knockouts": 3,
      "prioritize_growth": true
    }
  }'
```

#### Expected response
```json
{
  "knockout": ["geneA", "geneB"],
  "overexpress": ["geneX"],
  "predicted_yield": 3.17,
  "simulation_status": "optimal",
  "biomass_flux": 0.1
}
```

#### Python client example
```python
import requests

response = requests.post('http://localhost:5000/optimize', json={
    'target_molecule': 'ethanol',
    'host': 'S.cerevisiae',
    'constraints': {
        'max_knockouts': 2,
        'prioritize_growth': False
    }
})

result = response.json()
print(f"Predicted yield: {result['predicted_yield']} g/L")
print(f"Knockout genes: {result['knockout']}")
print(f"Overexpress genes: {result['overexpress']}")
```

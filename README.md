# Frances
A closed-loop metabolic pathway optimizer that optimizes a microbial production loop for a specific target molecule.

## Root function 
`optimize(target_molecule)` --> strain_design + pathway edits + predicted yield

Example:
```python
optimize(target_molecule="limonene", host_organism="E.coli") → { "knockout": [geneX], "overexpress": [geneY], "yield": 12.3 g/L }
```

## Steps
1. Ingests a target molecule + host organism
2. Runs GNN + Attention on the metabolic graph
3. Generates interventions (knockouts/overexpressions)
4. Simulates flux + yield (CobraPy or BiGG proxy)
5. Returns actionable strain design
6. Stores design + result for future reuse or retraining

## Tools
Data sources: MetaCyc, KEGG, BioCyc, FBA outputs  
Graph construction: NetworkX  
GNN + attention: PyTorch Geometric (GATConv)  
RL policy (strain edits): PPO, SBX
Flux simulation: CobraPy (proxy to FBA)

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
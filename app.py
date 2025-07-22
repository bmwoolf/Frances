from flask import Flask, request, jsonify
from flux_simulator import simulator

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    target_molecule = data['target_molecule']
    host = data['host']
    constraints = data.get('constraints', {})
    prior_data = data.get('prior_data', None)

    # Generate gene suggestions (placeholder for now - will be replaced with GNN)
    knockout_genes = ["geneA", "geneB"]
    overexpress_genes = ["geneX"]
    
    # Run flux simulation
    simulation_result = simulator.run_full_simulation(
        host=host,
        knockouts=knockout_genes,
        overexpressions=overexpress_genes
    )
    
    # Format response
    result = {
        "knockout": knockout_genes,
        "overexpress": overexpress_genes,
        "predicted_yield": simulation_result["predicted_yield"],
        "simulation_status": simulation_result["status"],
        "biomass_flux": simulation_result.get("biomass_flux", 0.0)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 
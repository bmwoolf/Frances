from flask import Flask, request, jsonify, render_template_string
from flux_simulator import simulator

app = Flask(__name__)

@app.route('/')
def home():
    """Homepage for the Frances bioinformatics application"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Frances - Bioinformatics Optimization</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .endpoint {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #007bff;
            }
            .method {
                font-weight: bold;
                color: #007bff;
            }
            .url {
                font-family: monospace;
                background: #e9ecef;
                padding: 5px 10px;
                border-radius: 3px;
            }
            .description {
                margin-top: 10px;
                color: #6c757d;
            }
            .example {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
                font-family: monospace;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 Frances - Bioinformatics Optimization API</h1>
            
            <p>Welcome to Frances, a bioinformatics application for metabolic pathway optimization using flux balance analysis.</p>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/optimize</div>
                <div class="description">
                    Optimize metabolic pathways for target molecule production using flux balance analysis.
                </div>
                <div class="example">
                    {
                        "target_molecule": "ethanol",
                        "host": "E. coli",
                        "constraints": {},
                        "prior_data": null
                    }
                </div>
            </div>
            
            <p><strong>Status:</strong> 🟢 Server is running successfully on port 5000</p>
        </div>
    </body>
    </html>
    """
    return html_content

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
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    target_molecule = data['target_molecule']
    host = data['host']
    constraints = data.get('constraints', {})
    prior_data = data.get('prior_data', None)

    # Placeholder output for MVP loop
    result = {
        "knockout": ["geneA", "geneB"],
        "overexpress": ["geneX"],
        "predicted_yield": 10.5
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 
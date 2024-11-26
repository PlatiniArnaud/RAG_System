import subprocess
try:
    subprocess.check_call(['pip', 'install', '-q', 'flask'])
    subprocess.check_call(['pip', 'install', '-q', 'flask_cors'])
except Exception as e:
    print(e)

from flask import Flask, request, jsonify
from Retrieval_Generation import RetrieveContext  # import your RAG class here

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Initialize your retriever class (RAG system)
retriever = RetrieveContext()

@app.route('/')
def home():
    return "Welcome to the Health Information Bot"

@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Retrieve context from RAG system
        context_sources, query = retriever.get_context(query)
        return jsonify({
            'context_sources': context_sources,
            'query': query
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Generate the answer based on RAG system
        response = retriever.Advanced_Generation(query)
        return jsonify({
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
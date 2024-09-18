import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_motor import LLMMotor, initialize_llm_motor
from llm_motor import get_available_models
import os
from dotenv import load_dotenv

# Laad de omgevingsvariabelen uit .env
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Haal de API key uit de omgevingsvariabelen
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Initialize RAG components and LLMMotor
llm_motor = initialize_llm_motor()

app = Flask(__name__)
CORS(app)

@app.route('/api/send-message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message')
    response = llm_motor.get_response(message)
    return jsonify({
        "response": response['answer'],
        "relevant_chunks": response['relevant_chunks'],
        "token_usage": response['token_usage']
    })

@app.route('/api/start-conversation', methods=['POST'])
def start_conversation():
    opening_message = "Hallo! Hoe kan ik u vandaag helpen?"
    return jsonify({"message": opening_message})

@app.route('/api/clear-memory', methods=['POST'])
def clear_memory():
    llm_motor.memory.clear()
    return jsonify({"success": True, "message": "Geheugen gewist"})

@app.route('/api/select-model', methods=['POST'])
def select_model():
    data = request.json
    model = data.get('model')
    try:
        llm_motor.change_model(model)
        return jsonify({"success": True, "message": f"Model {model} geselecteerd"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/available-models', methods=['GET'])
def get_available_models_route():
    models = get_available_models(api_key)
    return jsonify({"models": models})

@app.route('/', methods=['GET'])
def home():
    return "Hello, Flask server is running!"

@app.route('/test', methods=['GET'])
def test():
    logger.info("Test route accessed")
    return "Test route is working!"

if __name__ == '__main__':
    logger.info("Starting Flask server on http://0.0.0.0:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
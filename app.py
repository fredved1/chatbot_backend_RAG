import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_motor import initialize_llm_motor
import os
from dotenv import load_dotenv

# Laad de omgevingsvariabelen uit .env
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Haal de API key uit de omgevingsvariabelen
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY is niet ingesteld in de omgevingsvariabelen")

# Initialize RAG components and LLMMotor
llm_motor = initialize_llm_motor()

app = Flask(__name__)
CORS(app)

@app.route('/api/send-message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Geen bericht ontvangen.'}), 400
    
    try:
        response = llm_motor.get_response(message)
        return jsonify({
            "response": response['answer'],
            "relevant_chunks": response['relevant_chunks'],
            "token_usage": response['token_usage']
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 500
    except Exception as e:
        logger.exception("Onverwachte fout opgetreden.")
        return jsonify({'error': 'Er is een onverwachte fout opgetreden.'}), 500

@app.route('/api/start-conversation', methods=['POST'])
def start_conversation():
    try:
        opening_message = llm_motor.start_new_conversation()
        return jsonify({"message": opening_message})
    except Exception as e:
        logger.exception("Fout bij het starten van een nieuw gesprek.")
        return jsonify({"error": "Fout bij het starten van een nieuw gesprek."}), 500

@app.route('/api/clear-memory', methods=['POST'])
def clear_memory():
    try:
        llm_motor.clear_memory()
        return jsonify({"success": True, "message": "Geheugen gewist"})
    except Exception as e:
        logger.exception("Fout bij het wissen van het geheugen.")
        return jsonify({"success": False, "message": "Fout bij het wissen van het geheugen."}), 500

@app.route('/api/select-model', methods=['POST'])
def select_model():
    data = request.json
    model = data.get('model')
    temperature = data.get('temperature', 0.7)
    if not model:
        return jsonify({"success": False, "message": "Geen model opgegeven"}), 400
    try:
        llm_motor.change_model(model, temperature)
        return jsonify({"success": True, "message": f"Model {model} geselecteerd"})
    except Exception as e:
        logger.exception("Fout bij het selecteren van het model.")
        return jsonify({"success": False, "message": str(e)}), 400

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
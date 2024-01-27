from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = os.getenv('MODEL_DIR', 'model/saved_models')
SERVER_PORT = os.getenv('PORT', '8000')
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

model = None

def load_model():
    """Function to load the latest trained model."""
    global model
    try:
        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')], reverse=True)
        latest_model_file = model_files[0] if model_files else None
        if latest_model_file:
            model = joblib.load(os.path.join(MODEL_DIR, latest_model_file))
            logging.info(f"Model {latest_model_file} loaded successfully.")
        else:
            logging.error("No model file found.")
            raise FileNotFoundError(f"No model file found in '{MODEL_DIR}'.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

@app.before_first_request
def load_model_on_startup():
    """Load the model before the first request."""
    load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make fraud detection predictions."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        df = pd.DataFrame([data])
        prediction = model.predict_proba(df)[0][1]  # Probability of class 1 (fraud)
        return jsonify({'probability': prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    """Home endpoint providing welcome message."""
    return "Welcome to the Fraud Detection API!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=DEBUG_MODE, port=int(SERVER_PORT))

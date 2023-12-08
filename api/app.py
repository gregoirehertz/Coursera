from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_DIR = os.getenv('MODEL_DIR', 'model/saved_models')
SERVER_PORT = os.getenv('PORT', 5001)
DEBUG_MODE = os.getenv('DEBUG', 'False') == 'True'


def load_model():
    """Function to load the latest trained model."""
    try:
        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')])
        latest_model_file = model_files[-1] if model_files else None
        if latest_model_file:
            return joblib.load(f'{MODEL_DIR}/{latest_model_file}')
        else:
            logging.error("No model file found.")
            raise FileNotFoundError(f"No model file found in '{MODEL_DIR}'.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise e


model = load_model()


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
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    """Home endpoint providing welcome message."""
    return "Welcome to the Fraud Detection API!"


if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, port=int(SERVER_PORT))

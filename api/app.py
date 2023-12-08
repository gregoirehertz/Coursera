from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)


def load_model():
    """Function to load the latest trained model."""
    model_files = sorted([f for f in os.listdir('model/saved_models') if f.endswith('.pkl')])
    latest_model_file = model_files[-1] if model_files else None
    if latest_model_file:
        return joblib.load(f'model/saved_models/{latest_model_file}')
    else:
        raise FileNotFoundError("No model file found in 'model/saved_models'.")


model = load_model()


@app.route('/predict', methods=['POST'])
def predict():
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
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"


if __name__ == '__main__':
    app.run(debug=True, port=5001)

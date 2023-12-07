from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the latest trained model
model_files = sorted([f for f in os.listdir('model/saved_models') if f.endswith('.pkl')])
latest_model_file = model_files[-1]
model = joblib.load(f'model/saved_models/{latest_model_file}')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data, index=[0])
        prediction = model.predict_proba(df)[0][1]  # Probability of class 1 (fraud)
        return jsonify({'probability': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

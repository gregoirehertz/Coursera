# Fraud Detection System

## Project Overview
This project implements a machine learning-based fraud detection system designed to automatically detect fraudulent activities in financial transactions. The system is adaptable, allowing for regular retraining to adjust to new data patterns.

## Key Features
- Automated detection of potential fraud in financial transactions.
- Regular retraining capabilities to adapt to data drift.
- RESTful API for accessing model predictions.
- MLflow integration for model performance monitoring.
- Automated workflow using GitHub Actions for model retraining and deployment.

## Repository Structure
- `/data`: Original and simulated datasets.
- `/model`: Training and evaluation scripts for the machine learning model.
- `/api`: Flask application for the RESTful API.
- `/mlflow`: MLflow tracking and monitoring.
- `/.github/workflows`: GitHub Actions workflows for automation.
- `/scripts`: Additional scripts for data drift detection and monitoring.
- `/docs`: Documentation related to the project.
- `/reports`: Generated reports on model evaluation and data drift analysis.
- `config.py`: Centralized configuration settings.
- `requirements.txt`: List of Python package dependencies.

## Setup and Installation
1. **Clone the repository**:
   ```
   git clone https://github.com/mohamadsolouki/MLOps
   ```
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Running the Application
1. **Train the model**:
   ```
   python model/train_model.py
   ```
2. **Start the Flask API**:
   ```
   python api/app.py
   ```
3. **Send prediction requests** (using tools like `curl` or Postman):
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"Time": ..., "V1": ..., ...}' http://127.0.0.1:5000/predict
   ```

## Monitoring and Updates
- **MLflow UI**: Run `mlflow ui` and access at `http://127.0.0.1:5000`.
- **Automated Retraining**: GitHub Actions workflow re-trains and deploys the model regularly or upon data updates.
- **Data Drift Monitoring**: GitHub Actions workflow runs data drift analysis and generates reports regularly or upon data updates.

## Contributing
Contributions to this project are welcome. Please ensure to follow the coding standards and write tests for new features.

## License
This project is licensed under the terms of the [MIT License]

## Contact
Please feel free to contact me if you have any questions or suggestions.
Mohammad Solouki - mohamad.solouki@gmail.com
```
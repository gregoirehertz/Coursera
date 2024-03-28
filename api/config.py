import os

class Config:
    MODEL_DIR = os.getenv('MODEL_DIR', 'model/saved_models')
    SERVER_PORT = os.getenv('PORT', '8000')
    DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
import os
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_file_exists(file_path):
    """
    Check if the given file exists.

    Args:
        file_path: Path to the file to check.

    Returns:
        Boolean indicating whether the file exists.
    """
    return os.path.exists(file_path)

def upload_model_to_server(model_path, server_url):
    """
    Upload the model to the server or cloud storage.

    Args:
        model_path: Path to the model file.
        server_url: URL of the server where the model needs to be uploaded.

    Returns:
        The response object from the server.
    """
    if not check_file_exists(model_path):
        logging.error(f"Model file does not exist at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        with open(model_path, 'rb') as model_file:
            files = {'file': model_file}
            response = requests.post(server_url, files=files)
            response.raise_for_status()
            logging.info("Model successfully uploaded.")
            return response
    except requests.RequestException as e:
        logging.error(f"Error uploading model to server: {e}")
        raise
    except IOError as e:
        logging.error(f"Error opening model file: {e}")
        raise

def notify_server_to_reload_model(notification_url):
    """
    Notify the server to reload the model.

    Args:
        notification_url: URL to notify for model reload.

    Returns:
        The response object from the server.
    """
    try:
        response = requests.post(notification_url)
        response.raise_for_status()
        logging.info("Server successfully notified to reload model.")
        return response
    except requests.RequestException as e:
        logging.error(f"Error notifying server to reload model: {e}")
        raise

if __name__ == "__main__":
    latest_model_path = os.getenv('LATEST_MODEL_PATH', 'model/saved_models/model.pkl')
    server_upload_url = os.getenv('SERVER_UPLOAD_URL', 'http://127.0.0.1:8000/upload_model')
    server_reload_url = os.getenv('SERVER_RELOAD_URL', 'http://127.0.0.1:8000/reload_model')

    try:
        # Upload the model
        upload_response = upload_model_to_server(latest_model_path, server_upload_url)
        logging.info(f"Model upload response: {upload_response.text}")

        # Notify the server to reload the model
        reload_response = notify_server_to_reload_model(server_reload_url)
        logging.info(f"Server reload response: {reload_response.text}")
    except Exception as e:
        logging.error(f"Deployment error: {e}")

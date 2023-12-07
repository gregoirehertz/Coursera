import os
import requests

def upload_model_to_server(model_path, server_url):
    """
    Upload the model to the server or cloud storage.
    Args:
    - model_path: Path to the model file.
    - server_url: URL of the server where the model needs to be uploaded.
    """
    files = {'file': open(model_path, 'rb')}
    response = requests.post(server_url, files=files)
    return response

def notify_server_to_reload_model(notification_url):
    """
    Notify the server to reload the model.
    Args:
    - notification_url: URL to notify for model reload.
    """
    response = requests.post(notification_url)
    return response

# Example usage
if __name__ == "__main__":
    latest_model_path = 'model/saved_models/latest_model.pkl'
    server_upload_url = 'http://yourserver.com/upload_model'
    server_reload_url = 'http://yourserver.com/reload_model'

    # Upload the model
    upload_response = upload_model_to_server(latest_model_path, server_upload_url)
    print(f"Model upload response: {upload_response.text}")

    # Notify the server to reload the model
    reload_response = notify_server_to_reload_model(server_reload_url)
    print(f"Server reload response: {reload_response.text}")

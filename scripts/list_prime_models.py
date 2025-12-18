import os
import requests
import json

def list_prime_models():
    api_key = os.environ.get('PRIME_API_KEY')
    if not api_key:
        print("PRIME_API_KEY not set")
        return

    url = "https://api.pinference.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models = response.json()
            print(f"Found {len(models['data'])} models:")
            for model in models['data']:
                print(f"- {model['id']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Load env vars if needed (similar to eval_runner.py)
    from pathlib import Path
    config_path = Path.home() / ".pandas-rlvr.env"
    if config_path.exists():
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    
    list_prime_models()

import os
import requests
import json
from pathlib import Path

def list_openai_models():
    # Load env vars
    config_path = Path.home() / ".pandas-rlvr.env"
    if config_path.exists():
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models = response.json()
            print(f"Found {len(models['data'])} models:")
            
            # Filter for relevant models to avoid spamming
            relevant = []
            for model in models['data']:
                mid = model['id']
                if 'gpt' in mid or 'o1' in mid or 'codex' in mid:
                    relevant.append(mid)
            
            for mid in sorted(relevant):
                print(f"- {mid}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    list_openai_models()

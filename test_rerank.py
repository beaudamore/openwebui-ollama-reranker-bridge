import requests
import json

url = "http://localhost:8000/v1/rerank"

payload = {
    "model": "bge-m3",  # Make sure you have this model pulled in Ollama: `ollama pull bge-m3`
    "query": "What is the capital of France?",
    "documents": [
        "Paris is the capital of France.",
        "London is the capital of the UK.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is in Paris."
    ],
    "top_n": 3
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

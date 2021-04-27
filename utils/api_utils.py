# import libraries
import json
import requests

# huggingface API key
API_TOKEN = "Replace with your API"
API_URL = "https://api-inference.huggingface.co/models"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# to generate payload for API call
def query(payload, model_name):
    data = json.dumps(payload)
    response = requests.request(
        "POST", f"{API_URL}/{model_name}", headers=headers, data=data
    )
    return json.loads(response.content.decode("utf-8"))


# # to execute huggingface inference API
def translate(text, model_name):
    payload = {"inputs": text}
    translation = query(payload, model_name)[0]["translation_text"]
    return translation

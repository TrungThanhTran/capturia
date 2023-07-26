import json
import requests

def _call_json_api(api_url, json_data):
        response = requests.post(api_url, json=json_data, timeout=500)
        return response.json()

json_data = {
    "file_name":"Busdriver - Imaginary Places.mp3"
    }

api_url = "https://capturia.io/api/v1/transcribe/file"

response = _call_json_api(api_url, json_data)
print(response)
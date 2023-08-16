import json
import requests
from requests.exceptions import ConnectTimeout

import time
def _call_json_api(api_url, json_data):
        start = time.time()
        try:
            response = requests.post(api_url, json=json_data, timeout=10000)
        except ConnectTimeout:
            print('Request has timed out')

        print('time request = ', time.time()-start)
        print(response)
        return response.json()

json_data = {
    "file_name":"QLGvdPvxSemj4UzbJNQs_speed_talking.wav"
    }

api_url = "https://capturia.io/api/v1/transcribe/file"

response = _call_json_api(api_url, json_data)
print(response)

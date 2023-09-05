import json
import requests
from requests.exceptions import ConnectTimeout
from email_sender import Email_Sender

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
    "file_name":"health_check_api.mp3"
    }

api_url = "https://capturia.io/api/v1/transcribe/file"
emailsender = Email_Sender()
status = False
run_time = int(time.time())
while(True):
    try:
        if time.time() - run_time > 36000:
            run_time = time.time()
            emailsender.send_email_text("tranthanhtrung1990@gmail.com", f"STILL RUNNING API!")

        response = _call_json_api(api_url, json_data)
        with open('data/log/api.log', 'r') as afile:
            for line in afile:
                if 'ERROR' in line:
                    print(line)
                    emailsender.send_email_text("tranthanhtrung1990@gmail.com", f"API GOT ERROR: {line}")
                    
        if 'transcript' in list(response.keys()):
            if status == False:
                emailsender.send_email_text("tranthanhtrung1990@gmail.com", "NOTIFY: OK API!")
            print("NOTIFY: OK API!")
            status = True
            time.sleep(800)
        
        

    except Exception as e:
            status = False
            emailsender.send_email_text("tranthanhtrung1990@gmail.com", "NOTIFY: ERROR API!")
            print("NOTIFY: ERROR API!")
            time.sleep(3600)
    
    

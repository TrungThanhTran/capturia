import os
import uuid
import json
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import yaml
from yaml.loader import SafeLoader

from database_aws import S3_Handler, SQS_Handler
from backend import diarize_speaker_whisperX, transcribe_audio_whisperX, sentimet_audio
from functions import save_file_text, save_file_json, clear_gpu
from pydantic import BaseModel

class Item(BaseModel):
    file_name: str

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

S3_BUCKETNAME = os.environ['S3_BUCKETNAME']
s3_handler = S3_Handler(S3_BUCKETNAME)

@app.get('/')
async def add_index():
	return 'This is an API for Takenote.AI'

@app.get('/helloworld')
def hello_world():
	return 'This is an API for Takenote.AI'

@app.post("/api/v1/transcribe/file")
def api_v1_transcribe_file(item: Item):
        print('item = ', item)
        import uuid    
        with open('data/model/model_config.yaml') as file:
            model_config = yaml.load(file, Loader=SafeLoader)
            
        dict_reponse  = {
            "file_name": "",
            "transcript": "",
            "sentiment": [],
            "speaker_diarization": []
        }
        
        random_uuid = uuid.uuid4()
        print("Random UUID (Version 4):", random_uuid)
       
        ### Get data from json
        print('doing this = ', item.file_name)
        audio_path_raw = item.file_name
        dict_reponse["file_name"] = audio_path_raw
        uuid = ""
        ### Download data 
        local_file_path = f'./temp/{random_uuid}_{audio_path_raw}'
        download_flag = s3_handler.download_file_from_s3(audio_path_raw, local_file_path)
        
        assert download_flag == True
        
        ### Transcribe 
        results, _, _, _, audio_path = transcribe_audio_whisperX(model_config,
                local_file_path, 'test', "1234")
        passages = results["text"]
        segments =  results['segments']
        save_file_text(passages, f'./temp/test/passages.txt')
        dict_reponse["transcript"] = passages
        
        ### Sentiment 
        sentiment, sentences = sentimet_audio(passages)
        save_file_json(
            sentiment, f'./temp/test/sentiment.json')
        dict_reponse["sentiment"] = sentiment

        ### Speaker whipser 
        try:
            trans_with_spk = diarize_speaker_whisperX(audio_path, 
                                                        segments, 
                                                        model_config['transcribe']['device'], 
                                                        model_config['transcribe']['hf_token'])
        except Exception as e:
            print(e)
            trans_with_spk = []
            
        save_file_json(
            trans_with_spk, f'./temp/test/matching_speaker.json')
        
        ### Prepare api response
        dict_reponse["speaker_diarization"] = trans_with_spk
        os.remove(local_file_path)
        del segments, passages, results, sentences

        return dict_reponse
        raise HTTPException(status_code=404, detail=f"Could process the file: {file_name}")


if __name__ == '__main__':
    print('Start running fastapi app')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
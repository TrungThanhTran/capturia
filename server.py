import os
import uuid
import json
import torch    
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import gc
import yaml
from yaml.loader import SafeLoader

from database_aws import S3_Handler, SQS_Handler 
from backend import diarize_speaker_whisperX, transcribe_audio_whisperX, sentimet_audio
from functions import save_file_text, save_file_json, download_from_youtube, get_file_size_in_kb
from pydantic import BaseModel

class Item(BaseModel):
    file_name: str
    
class YTLINK(BaseModel):
    link: str
    task: str
    
app = FastAPI()
origins = ['*']
TEMP_AUDIO_FOLDER = "audio"

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

@app.post(f"/api/v1/download/video")
def api_v1_download_video(ytlink: YTLINK):
    try:
        print('api v1 downloading this = ', ytlink)
        link = ytlink.link
        task_id = ytlink.task
        
        audio_file_path, audio_file_path_wav, title = download_from_youtube(link, task_id)
        file_size = get_file_size_in_kb(audio_file_path)
        # Uploading to S3
        audio_file_name = audio_file_path.split('/')[-1]
        upload_flag = s3_handler.upload_file_to_s3(audio_file_path, audio_file_name)
        assert upload_flag == True
        
        response_dict = {}

        response_dict['bucket'] = S3_BUCKETNAME
        response_dict['mimetype'] = 'audio'
        response_dict['audio_path'] = audio_file_name
        response_dict['size'] = file_size
        response_dict['key'] = task_id
        shutil.rmtree(f'temp/youtube_down/{task_id}')
    
        return response_dict
    except Exception as e:
        print(e)
        return JSONResponse(content={'detail': "Could not download the link"}, status_code=404)


@app.post("/api/v1/transcribe/file")
def api_v1_transcribe_file(item: Item):
    try:
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
                local_file_path, TEMP_AUDIO_FOLDER, "1234")
        passages = results["text"]
        segments =  results['segments']
        save_file_text(passages, f'./temp/{TEMP_AUDIO_FOLDER}/passages.txt')
        dict_reponse["transcript"] = passages
        
        ### Sentiment 
        sentiment, sentences = sentimet_audio(passages)
        save_file_json(
            sentiment, f'./temp/{TEMP_AUDIO_FOLDER}/sentiment.json')
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
            trans_with_spk, f'./temp/{TEMP_AUDIO_FOLDER}/matching_speaker.json')
        
        ### Prepare api response
        dict_reponse["speaker_diarization"] = trans_with_spk
        os.remove(local_file_path)
        del segments, passages, results, sentences

        return dict_reponse
    except Exception as e:
        gc.collect()  
        torch.cuda.empty_cache()
        return JSONResponse(content={'detail': "out of memory"}, status_code=404)
      
if __name__ == '__main__':
    print('Start running fastapi app')

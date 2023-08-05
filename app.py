import os
import json

from flask import Flask, request
# from flask_cors import CORS
import yaml
from yaml.loader import SafeLoader

from database_aws import S3_Handler, SQS_Handler
from backend import diarize_speaker_whisperX, transcribe_audio_whisperX, sentimet_audio
from functions import save_file_text, save_file_json, clear_gpu

app = Flask(__name__)
# CORS(app)
S3_BUCKETNAME = os.environ['S3_BUCKETNAME']
s3_handler = S3_Handler(S3_BUCKETNAME)

@app.route('/')
def add_index():
	return 'This is an API for Takenote.AI'

@app.route('/helloworld')
def hello_world():
	return 'This is an API for Takenote.AI'

@app.route(f"/api/v1/transcribe/file", methods=['POST', 'GET'])
def api_v1_transcribe_file():    
    # Init
        with open('data/model/model_config.yaml') as file:
            model_config = yaml.load(file, Loader=SafeLoader)
            
        dict_reponse  = {
            "file_name": "",
            "transcript": "",
            "sentiment": [],
            "speaker_diarization": []
        }
       
    # try:
        ### Get data from json
        print('doing this = ', request.get_json())
        request_data = request.get_json()
        audio_path_raw = request_data['file_name']
        dict_reponse["file_name"] = audio_path_raw

        ### Download data 
        local_file_path = f'./temp/{audio_path_raw}'
        download_flag = s3_handler.download_file_from_s3(audio_path_raw, local_file_path)
        
        assert download_flag == True
        
        ### Transcribe 
        # results, title, language, end_time - start_time, audio_path
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
        # clear_gpu()

    # except Exception as e:
    #     return {
    #         "error": e
    #     }
        return dict_reponse


if __name__ == '__main__':
    print('Start running flask app')
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', debug=False, port=8501, threaded=False)

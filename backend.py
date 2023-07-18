from functions import *
from database import DBHandler
from database_aws import S3_Handler, SQS_Handler
from email_sender import Email_Sender
from datetime import datetime
import json
import gc
from glob import glob


def sentimet_audio(passage):
    sentiment, sentences = sentiment_pipe(passage)
    return sentiment, sentences


def transcribe_audio_whisperX(audio_path, user, task_id):
    start_time = time.time()
    asr_model = load_whisperx_model("medium")
    texts, title, segments, language, audio_path = inference(
        asr_model, audio_path, user, task_id)

    end_time = time.time()
    passages = texts
    gc.collect()
    torch.cuda.empty_cache()
    del asr_model

    return passages, title, segments, language, end_time - start_time, audio_path

def diarize_speaker_whisperX(audio_path, segments):
    colors = ['red', 'green', 'yellow', 'blue',
              'cyan', 'lime', 'magenta', 'pink', 'orange']
    align_result = align_speaker(segments, audio_path)
    result = assign_speaker(align_result, audio_path)
    trans = []

    for seg in result["segments"]:
        dict_spek = {}
        try:
            dict_spek['text'] = seg['text']
            dict_spek['speaker'] = seg['speaker']
            dict_spek['start'] = seg['start']
            dict_spek['end'] = seg['end']
        except:
            continue
        trans.append(dict_spek)
    return trans


def main():
    S3_BUCKETNAME = os.environ['S3_BUCKETNAME']
    TASK_QUEUE = os.environ['TASK_QUEUE']
    ERROR_QUEUE = os.environ['ERROR_QUEUE']
    FINISH_QUEUE = os.environ['FINISH_QUEUE']
    
    emailsender = Email_Sender()
    s3_handler = S3_Handler(S3_BUCKETNAME)
    sqs_handler = SQS_Handler()

    while (True):
        # Get task
        task_id, audio_path_raw, user, rev_email, task_time, status = sqs_handler.get_message(
            TASK_QUEUE)
        
        if (task_id == None) or (audio_path_raw == None):
            continue
        
        print("GET A TASK...")
        # Save file audio into file
        # Create temp path if need
        if not os.path.exists(f'./temp/{user}/'):
           os.mkdir(f'./temp/{user}/') 
        
        # Configure response queue
        json_data = {}
        json_data["task_id"] = task_id
        json_data["file_path"] = audio_path_raw
        json_data["user"] = user
        json_data["email"] = rev_email
        json_data["time"] = task_time

        # Create task_id on S3
        s3_path_task = f'{user}/{task_id}/'

        # Get list user
        registered_user = s3_handler.list_username_in_bucket()
        if user not in registered_user:
            s3_handler.create_s3_folder(f'{user}')

        s3_handler.create_s3_folder(s3_path_task)
        try:
            # Processing task
            # 1. transcribe
            print("TRANSCRIBING...")
            
            file_name = os.path.basename(audio_path_raw)
            local_file_path = f'./temp/{user}/{file_name}'
            audio_path_s3 = audio_path_raw.replace(f"s3://{S3_BUCKETNAME}/", "")
            download_flag = s3_handler.download_file_from_s3(audio_path_s3, local_file_path)
            assert download_flag == True
        
            passages, title, segments, language, running_time, audio_path = transcribe_audio_whisperX(
                local_file_path, user, task_id)
            
            # Save passges into file,
            save_file_text(passages, f'./temp/{user}/passages.txt')
            s3_handler.upload_file_to_s3(
                f'./temp/{user}/passages.txt', s3_path_task)

            save_file_text(title, f'./temp/{user}/title.txt')
            s3_handler.upload_file_to_s3(
                f'./temp/{user}/title.txt', s3_path_task)

            save_file_json(segments, f'./temp/{user}/segments.json')
            s3_handler.upload_file_to_s3(
                f'./temp/{user}/segments.json', s3_path_task)

            # 2. sentiment
            print("SENTIMENT...")
            sentiment, sentences = sentimet_audio(passages)
            # print(sentiment, sentences)
            save_file_json(
                sentiment, f'./temp/{user}/sentiment.json')
            s3_handler.upload_file_to_s3(
                f'./temp/{user}/sentiment.json', s3_path_task)

            save_file_json(
                sentences, f'./temp/{user}/sentences.json')
            s3_handler.upload_file_to_s3(
                f'./temp/{user}/sentences.json', s3_path_task)

            # 3. Speaker Identification
            print("DIARIZE...")
            # trans_with_spk = diarize_speaker(audio_path, segments)
            trans_with_spk = diarize_speaker_whisperX(audio_path, segments)
            save_file_json(
                trans_with_spk, f'./temp/{user}/matching_speaker.json')
            s3_handler.upload_file_to_s3(
                f'./temp/{user}/matching_speaker.json', s3_path_task)

            # Delete task

            now = datetime.now()
            now_ = now.strftime("%b-%d-%Y-%H-%M-%S")

            json_data["status"] = "1"

            message_body = json.dumps(json_data)    
            sqs_handler.send_message(FINISH_QUEUE, message_body)
            
            temp_files = glob(f'./temp/{user}/*')
            for f in temp_files:
                os.remove(f)
            
            task_id_user = task_id + '__' + \
                '-'.join(convert_string_ASCII(user))
            emailsender.send_email_text(rev_email, task_id_user)
            print('Sending email with result!')
        except Exception as e:
            print('ERROR]____:', e)
            now = datetime.now()
            now_ = now.strftime("%b-%d-%Y-%H-%M-%S")
            status = int(status)
            status += 1
            if status > 2:
                url_queue = ERROR_QUEUE
            else:
                url_queue = TASK_QUEUE
            
            json_data["time"] = now_
            json_data["status"] = str(status)
            message_body = json.dumps(json_data)    
            sqs_handler.send_message(url_queue, message_body)


if __name__ == "__main__":
    main()
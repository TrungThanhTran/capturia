from functions import *
from database import DBHandler
from email_sender import Email_Sender
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
from transformers import pipeline as hf_pipe
import json
import gc

def sentimet_audio(passage):
    sentiment, sentences = sentiment_pipe(passage)
    return sentiment, sentences

def transcribe_audio_whisperX(audio_path, user, task_id):
    start_time = time.time()
    asr_model = load_whisperx_model("medium")
    texts, title, segments, language, audio_path = inference(asr_model, audio_path, user, task_id)
    
    end_time = time.time()
    passages = texts
    print(end_time - start_time)
    gc.collect()
    torch.cuda.empty_cache()
    del asr_model

    return passages, title, segments, language, end_time - start_time, audio_path

def summarize_audio(passages, min_len, max_len):
    sum_pipe = hf_pipe("summarization",
                        model="facebook/bart-large-cnn", 
                        tokenizer="facebook/bart-large-cnn",
                        clean_up_tokenization_spaces=True)

    text_to_summarize = chunk_and_preprocess_text(passages)
    
    try:
        if len(passages) < min_len:
            summarized_text = passages
        else:
            summarized_text = summarize_text(sum_pipe, 
                                             text_to_summarize,
                                             max_len=max_len,
                                             min_len=min_len)
            if len(summarized_text) > len(passages):
                summarized_text = passages
            
    except IndexError:
            try:
                text_to_summarize = chunk_and_preprocess_text(passages, 450)
                summarized_text = summarize_text(sum_pipe, 
                                                 text_to_summarize,
                                                 max_len=max_len,
                                                 min_len=min_len)
        
            except IndexError:
                text_to_summarize = chunk_and_preprocess_text(passages, 400)
                summarized_text = summarize_text(sum_pipe, 
                                                 text_to_summarize,
                                                 max_len=max_len,
                                                 min_len=min_len)
    # text_to_summarize = text_to_summarize.replace('+', '\+')
    entity_match_html = highlight_entities(text_to_summarize, summarized_text)

    return entity_match_html

def diarize_speaker_whisperX(audio_path, segments):
    colors = ['red', 'green', 'yellow','blue', 'cyan', 'lime', 'magenta', 'pink', 'orange']
    align_result = align_speaker(segments, audio_path)
    result = assign_speaker(align_result, audio_path)
    trans = []

    for seg in result["segments"]:
        dict_spek  = {}
        try:
            dict_spek['text'] = seg['text']
            dict_spek['speaker'] = seg['speaker']
            dict_spek['start'] = seg['start']
            dict_spek['end'] = seg['end']
        except:
            continue
        trans.append(dict_spek) 
    return trans
    
if __name__ == "__main__":
    dbhandler = DBHandler()
    emailsender = Email_Sender()
    lst_db_tables = dbhandler.list_talbe_db()
    print('list table = ', lst_db_tables)
    
    if 'DONE_QUEUE' not in lst_db_tables:
        print('creating DONE_QUEUE')
        dbhandler.create_table('DONE_QUEUE')
    
    if 'ERROR_QUEUE' not in lst_db_tables:
        print("creating ERROR_QUEUE")
        dbhandler.create_table('ERROR_QUEUE')
    
    
    while(True):
        # Get task
        if dbhandler.get_len_table_db() == 0:
            # print('database is have no task')
            continue
        
        print("GET A TASK...")
        id_process, task_id, audio_path_raw, user, rev_email, task_time, status = dbhandler.query_db_min('TASK_QUEUE')
        print(id_process, task_id, audio_path_raw, user, rev_email, task_time, status)
        try:
            # Processing task
            # 1. transcribe
            print("TRANSCRIBING...")
            passages, title, segments, language, running_time, audio_path  = transcribe_audio_whisperX(audio_path_raw, user, task_id)
                    
            # Save passges into file, 
            save_file_text(passages, f'./temp/{user}/{task_id}/passages.txt')
            save_file_text(title, f'./temp/{user}/{task_id}/title.txt')
            save_file_json(segments, f'./temp/{user}/{task_id}/segments.json')

            # 2. sentiment
            print("SENTIMENT...")
            sentiment, sentences = sentimet_audio(passages)
            # print(sentiment, sentences)
            save_file_json(sentiment, f'./temp/{user}/{task_id}/sentiment.json')
            save_file_json(sentences, f'./temp/{user}/{task_id}/sentences.json')

            # 3. summary
            # Configure
            # print("SUMMARIZE...")
            # config_min_len = 100
            # config_max_len = 400
            # summarized_text = summarize_audio(passages, config_min_len, config_max_len)
            # # Stupid trick
            # summarized_text = summarized_text.replace('border-radius:', ' color:white; border-radius:')
            # with open(f'./temp/{user}/{task_id}/summary.html', 'w') as f:
            #     f.write(summarized_text)
                
            # 4. Speaker Identification
            print("DIARIZE...")
            # trans_with_spk = diarize_speaker(audio_path, segments)
            trans_with_spk = diarize_speaker_whisperX(audio_path, segments)
            save_file_json(trans_with_spk, f'./temp/{user}/{task_id}/matching_speaker.json')
                            
            # Delete task
            del_flag = dbhandler.delete_task_db(id_process)
            print(del_flag)
            now = datetime.now()
            now_ = now.strftime("%b-%d-%Y-%H-%M-%S")

            if not 'success' in del_flag:
                # push to queue again
                status += 1
                if status > 2:
                    table_name = 'ERROR_QUEUE'
                else:
                    table_name = 'TASK_QUEUE'
                dbhandler.writeinfo_db(table_name, task_id, audio_path_raw, user, rev_email, now_, status)
            else: 
                dbhandler.writeinfo_db('DONE_QUEUE', task_id, audio_path_raw, user, rev_email, now_, 1)
            task_id_user = task_id + '__' + '-'.join(convert_string_ASCII(user))
            emailsender.send_email_text(rev_email, task_id_user)
            print('Sending email with result!')
        except Exception as e:
            print(e)
            now = datetime.now()
            now_ = now.strftime("%b-%d-%Y-%H-%M-%S")
            del_flag = dbhandler.delete_task_db(id_process)
            status += 1
            if status > 2:
                table_name = 'ERROR_QUEUE'
            else:
                table_name = 'TASK_QUEUE'
            dbhandler.writeinfo_db(table_name, task_id, audio_path_raw, user, rev_email, now_, status)

            


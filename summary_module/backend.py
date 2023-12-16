from functions import *
from datetime import datetime
import json
import gc
from glob import glob

def transcribe_audio_whisperX(model_config, audio_path, user, task_id):
    start_time = time.time()
    asr_model = load_whisperx_model("medium", 
                                    model_config['transcribe']['device'],
                                    model_config['transcribe']['compute_type'])
    
    # results, "Transcribed Audio", results['segments'], "en", file_path
    results, title, language, audio_path = inference(
        asr_model, audio_path, user, task_id, model_config['transcribe']['batch_size'])

    end_time = time.time()

    gc.collect()
    torch.cuda.empty_cache()
    del asr_model

    return results, title, language, end_time - start_time, audio_path

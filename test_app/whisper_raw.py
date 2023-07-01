
import whisper
from faster_whisper import WhisperModel

asr_model_ori = whisper.load_model("medium")
asr_model_fast = WhisperModel("medium", device="cpu", compute_type="float32")

file_path = "/home/ubuntu/project/offline_Meeting_Analysis/temp/admin/9443026f-8d73-4eca-8c82-7bc03ce1a18e/Facebook (META) Q3 2022 Earnings Call.mp3"
results = {}
# segments, _ = asr_model_fast.transcribe(file_path,  
#                                     language="en",
#                                     beam_size=1, 
#                                     word_timestamps=True)
# whisper_results = []
# for segment in segments:
#     whisper_results.append(segment._asdict())
#     results['text'] = ' '.join([segment['text'] for segment in whisper_results])    

# with open('/home/ubuntu/project/offline_Meeting_Analysis/test_file/fast_data.txt', 'w') as f:
#     f.write(results['text'])

trams = asr_model_ori.transcribe(file_path, task='transcribe', language='en')
with open('/home/ubuntu/project/offline_Meeting_Analysis/test_file/ori_data.txt', 'w') as f:
    f.write(trams)
    
    


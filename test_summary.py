from transformers import pipeline
import time
from functions import *
import gc
from transformers import pipeline
import time
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'
modelxx = SentenceTransformer('all-MiniLM-L6-v2')  
if 'transcription' not in st.session_state:
    st.session_state['transcription'] = ''

def split_into_sentences(text: str):
    """
    Split the text into sentences.
    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.
    :param text: text to be split into sentences
    :type text: str
    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace(",","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    ss = []
    for s in sentences:
        ss.extend(s.split(','))
    sentences = ss
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def on_click_search(st_text, st_query):
    
    sentences1=split_into_sentences(st_text) 
    sentences2=[st_query]
    embeddings1 = modelxx.encode(sentences1, convert_to_tensor=True)
    embeddings2 = modelxx.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    max_score = 0.0
    max_sentence = ""
    

    for i in range(len(sentences1)): 

        if cosine_scores[i][0] > max_score:
            max_score = cosine_scores[i][0] 
            max_sentence = sentences1[i]
    
    if max_score.detach().cpu() > 0.51:
        return max_sentence, None
    
    c = []
    s = []
    for i in range(len(sentences1)): 
        if cosine_scores[i][0].detach().cpu() > 0.3:
            c.append(cosine_scores[i][0])
            s.append(sentences1[i])
            
    c=[x.detach().cpu() for x in c]

    df=pd.DataFrame({'Sentences':s,'Score':c})

    df1=df.sort_values('Score',ascending=False) 
    df1.set_index("Sentences",inplace=True)

    return max_sentence, df1

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

if __name__ == "__main__":
            print("TRANSCRIBING...")
            task_id = 'abcdef123456'
            select_box = st.selectbox('select mode', ['Upload audio', 'Upload text'])
            if select_box == 'Upload audio':
                upload_input = st.empty()
                    # clean_directory([f"./temp/youtube/{username}", f"./temp/vimeo/{username}"])
                upload_input = st.file_uploader("Upload a .wav or .mp3 sound file",
                                                    key="upload", 
                                                    type=['.wav','.mp3','.mp4','.m4a'])
                btn_submit = st.button('Submit')

                if btn_submit:
                        st.session_state['transcription'] = ''
                        if upload_input is not None:
                                    with st.spinner(text="Submitting..."):
                                            with open(f"../temp/test/{upload_input.name}", "wb") as f:
                                                f.write(upload_input.getbuffer())
                                            audio_path_raw = f"../temp/test/{upload_input.name}"
                        else:
                                st.warning("Please upload file!")
                        
                        with st.spinner(text="Transcribing..."):
                            passages, title, segments, language, running_time, audio_path  = transcribe_audio_whisperX(audio_path_raw, 'test', task_id)
                        
                        st.session_state['transcription'] = passages
                        # Save passges into file, 
            
            if select_box == 'Upload text':
                upload_input = st.empty()
                st.session_state['transcription'] = ''

                # clean_directory([f"./temp/youtube/{username}", f"./temp/vimeo/{username}"])
                upload_input = st.file_uploader("Upload a text file",
                                                    key="upload", 
                                                    type=['.txt', '.text'])
                if upload_input is not None:
                    with open(f"../temp/test/{upload_input.name}", "wb") as f:
                        f.write(upload_input.getbuffer())
                    text_file = f"../temp/test/{upload_input.name}"
                    with open(text_file, 'r') as f:
                        passages = f.readlines()
                    st.session_state['transcription'] = passages[0]


            if st.session_state['transcription'] != '':
                st.write(st.session_state['transcription'])
                st_query = st.text_input('Enter your query here')

                s_button = st.button("Search")
                
                if s_button:
                    if st_query != "":
                        s, df1 = on_click_search(st.session_state['transcription'], st_query)
                        if df1 is None:
                            st.write("the info is: ", s)
                        else:
                            st.write(f'the best info is: {s}')
                            st.write('other related result:')
                            st.write(df1)
                    else:
                        st.warning('no text input')

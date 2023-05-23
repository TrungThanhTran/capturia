import whisper
import os
from pytube import YouTube
import pandas as pd
import plotly_express as px
import nltk
import plotly.graph_objects as go
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import streamlit as st
import en_core_web_lg
from PIL import Image
from functions import *
from st_custom_components import st_audiorec
from yaml.loader import SafeLoader
import yaml
import streamlit_authenticator as stauth
# import streamlit.components.v1 as components
# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="HOME", page_icon=image_logo)
# hashed_passwords = stauth.Hasher(['12345', 'equiniti','takenote']).generate()
# print('hashed_passwords = ',hashed_passwords)
print(st.session_state)
nltk.download('punkt')

from nltk import sent_tokenize

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

def infer_audio(state, asr_model, type, is_fast_transcription):
    start_time = time.time()
    texts, title, segments, language = inference(state, asr_model, type, is_fast_transcription)
    end_time = time.time()
    # passages = results
    passages = clean_text(texts, language)
        
    st.session_state['passages'] = passages
    st.session_state['title'] = title
    st.session_state['segments'] = segments
    st.session_state['language'] = language
    st.session_state['running_time_transcript'] = end_time - start_time
    
if __name__ == "__main__":
    with open('data/pass/user_db.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    st.session_state['authenticator'] = authenticator
    name, authentication_status, username = authenticator.login(
        'Login', 'main')
    st.session_state['authentication_status'] = authentication_status
    if authentication_status == False:
        st.error("Username/password is incorrect")

    if authentication_status == None:
        st.warning("Please enter your username and password")
    
    if authentication_status:
        authenticator.logout("Logout", "sidebar")    
        st.sidebar.header("Home")
        asr_model_options = ['medium.en', 'small.en','base.en', 'medium'] #whisper-large-v2
        asr_model_name = st.sidebar.selectbox("Whisper Model Options", options=asr_model_options, key='sbox')
        # is_fast_transcription = st.sidebar.checkbox('Using fast transcription feature')
        is_fast_transcription = True
        col1, col2 = st.columns(2)
        with col1:
            original_title = '<center><p style="font-size: 80px;">TakeNote</p> \n <p>AI MEETING NOTES & SENTIMENT ANALYSIS </p></center>'
            st.markdown(original_title, unsafe_allow_html=True)
            # st.title("TakeNote " + "\n"
            #              "AI MEETING NOTES & SENTIMENT ANALYSIS ",100)
        with col2:
            image = Image.open('data/logo/logo.png')
            st.image(image,width=200)

        st.markdown("<br><br><br>",  unsafe_allow_html=True)

        ### prepare state for 
        if 'sbox' not in st.session_state:
            st.session_state.sbox = asr_model_name
            
        if 'segments' not in st.session_state:
            st.session_state.segments = ''
            
        if 'si_file' not in st.session_state:
            st.session_state.si_file = ''

        if "sen_df" not in st.session_state:
            st.session_state['sen_df'] = ''
            
        if "running_time_transcript" not in st.session_state:
            st.session_state['running_time_transcript'] = 0.0

        def clean_directory(paths):
            if not os.path.exists(f'./temp/si/{username}'):
                os.mkdir(f'./temp/si/{username}')
            if not os.path.exists(f'./temp/{username}'):
                os.mkdir(f'./temp/{username}')
            if not os.path.exists(f'./temp/transcript/{username}'):
                os.mkdir(f'./temp/transcript/{username}')
            for path in paths:
                if not os.path.exists(path):
                    pass
                else:
                    for file in os.listdir(path):
                        if ("." in file):
                            os.remove(os.path.join(path, file))

        ### Preload model
        try:
            if is_fast_transcription:
                ASR_MODEL = load_fast_asr_model(st.session_state.sbox)
            else:    
                ASR_MODEL = load_asr_model(st.session_state.sbox)
        except Exception as e:
            print(e)
            st.session_state.sbox = 'small'
            ASR_MODEL = load_asr_model('small')
        st.markdown("## Please submit your audio or video file",  unsafe_allow_html=True)

        ### UPLOAD AND PROCESS
        choice = st.radio("", ["By uploading a file","By getting from video URL"]) 
        if choice:
            clean_directory([f"./temp/youtube/{username}", f"./temp/vimeo/{username}"])
            if choice == "By uploading a file":
                upload_wav = st.file_uploader("Upload a .wav or .mp3 sound file",key="upload", type=['.wav','.mp3','.mp4','.m4a'])
                if upload_wav:
                    with open(os.path.join(f"./temp/{username}/audio.mp3"),"wb") as f:
                        f.write(upload_wav.getbuffer())
                    st.session_state['url'] = ''
                    st.session_state.si_file = f"./temp/si/{username}/audio.mp3"
                            
            elif choice == "By getting from video URL":
                url_input = st.text_input(
                label="Enter video URL, below is a calling example", value='',
                key="url")
                if 'upload' in st.session_state:
                    st.session_state['upload'] = ''
                if 'vimeo' in url_input:
                    st.session_state.si_file = f"./temp/si/{username}/v_audio.mp3"
                else: 
                    st.session_state.si_file = f"./temp/si/{username}/yt_audio.mp3"
                
            btn_transcribe = st.button('Transcribe')
            if btn_transcribe:
                if 'passage' in st.session_state:
                    st.session_state['passage'] = ""
                    
                with st.spinner(text="Transcribing..."):
                    clean_directory([f'./temp/si/{username}'])
                    try:
                        if ('url' in st.session_state) and (st.session_state['url'] != ''):
                            if len(st.session_state['url']) > 0: 
                                infer_audio(st.session_state['url'], ASR_MODEL, type='url', is_fast_transcription=is_fast_transcription) 

                        if ('upload' in st.session_state) and (st.session_state['upload'] != ''):  
                            import shutil
                            shutil.copy(f"./temp/{username}/audio.mp3", f"./temp/si/{username}/audio.mp3")
                            if st.session_state['upload'] is not None:
                                infer_audio(st.session_state['upload'], ASR_MODEL, type='upload', is_fast_transcription=is_fast_transcription)
                        
                    except Exception as e:
                        print(e)
                        if 'unavailable' in str(e):
                            st.write(f"{e}")
                        else:
                            st.write("No YouTube URL or file upload detected")
                st.success('Transcribing audio done!')
            
        # if "url" not in st.session_state:
        #     st.session_state.url = "https://www.youtube.com/watch?v=agizP0kcPjQ"
            
        # st.markdown(
        #     "<h3 style='text-align: center; color: red;'>OR</h3>",
        #     unsafe_allow_html=True
        # )

        auth_token = os.environ.get("auth_token")


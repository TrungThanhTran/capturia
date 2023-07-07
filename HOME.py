# import whisper
import os
from pytube import YouTube
import pandas as pd
import plotly_express as px
# import nltk
import plotly.graph_objects as go
import streamlit as st
# import en_core_web_lg
from PIL import Image
# from functions import *
from yaml.loader import SafeLoader
import yaml
import streamlit_authenticator as stauth
import uuid
from database import DBHandler
from datetime import datetime
from email_validator import validate_email, EmailNotValidError
from streamlit.runtime.scriptrunner import RerunData, RerunException

image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="HOME", page_icon=image_logo, layout='wide')

# nltk.download('punkt')

# from nltk import sent_tokenize

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

def save_task_to_queue(dbhandler, file_name, user, email, uniq_id, status=0):
    success = True
    now = datetime.now()
    now_ = now.strftime("%b-%d-%Y-%H-%M-%S")
    # Creat an unique id
    file_name = file_name.replace("'", "")
    try:
        dbhandler.writeinfo_db('TASK_QUEUE', uniq_id, file_name, user, email, now_, status)
    except Exception as e:
        return False
    # save to queue
    return success    

def check_email(email):
    try:
      # validate and get info
        v = validate_email(email)
        # replace with normalized form
        email = v["email"] 
        return 'email is valid'
    except EmailNotValidError as e:
        # email is not valid, exception message is human-readable
        return str(e)
 
    
if __name__ == "__main__":
    with open('data/pass/user_db.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
        
    dbhandler = DBHandler()
    if 'TASK_QUEUE' not in dbhandler.list_talbe_db():
        dbhandler.create_table('TASK_QUEUE')
    
    st.experimental_set_query_params()
        
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

        col1, col2 = st.columns(2)
        with col1:
            original_title = '<center><p style="font-size: 80px;">TakeNote</p> \n <p>AI MEETING NOTES & SENTIMENT ANALYSIS </p></center>'
            st.markdown(original_title, unsafe_allow_html=True)

        with col2:
            image = Image.open('data/logo/logo.png')
            st.image(image,width=200)

        st.markdown("<br><br><br>",  unsafe_allow_html=True)
        st.markdown("## Please submit your audio or video file",  unsafe_allow_html=True)

        ### UPLOAD AND PROCESS
        choice = st.radio("", ["By uploading a file","By getting from video URL"]) 
        upload_input = st.empty()
        url_input = st.empty()
        if choice:
            # clean_directory([f"./temp/youtube/{username}", f"./temp/vimeo/{username}"])
            if choice == "By uploading a file":
                upload_input = st.file_uploader("Upload a .wav or .mp3 sound file",
                                            key="upload", 
                                            type=['.wav','.mp3','.mp4','.m4a'])
                            
            elif choice == "By getting from video URL":
                url_input = st.text_input(
                label="Enter video URL, below is a calling example", value='',
                key="url")
            
            email_in = st.text_input(label = 'Email Address')
                
            btn_submit = st.button('Submit')
            if btn_submit:
                uniq_id = uuid.uuid4()
                
                # check existed folder
                if not os.path.exists(f'./temp/'):
                    os.mkdir(f'./temp/')
                    
                if not os.path.exists(f'./temp/{username}'):
                    os.mkdir(f'./temp/{username}')
                    
                if not os.path.exists(f'./temp/{username}/{uniq_id}'):
                    os.mkdir(f'./temp/{username}/{uniq_id}')
                
                if upload_input is not None and url_input != '' and email_in != '':
                    email_status = check_email(email_in)
                    if email_status != 'email is valid':
                        st.error(email_status)
                    else:
                        with st.spinner(text="Submitting..."):
                            if choice == "By uploading a file":
                                with open(f"./temp/{username}/{uniq_id}/{upload_input.name}", "wb") as f:
                                    f.write(upload_input.getbuffer())
                                save_flag = save_task_to_queue(dbhandler,  f"./temp/{username}/{uniq_id}/{upload_input.name}" , username, email_in, uniq_id)
                                print(save_flag)
                            else:
                                save_flag =  save_task_to_queue(dbhandler, url_input, username, email_in, uniq_id)
                                print(save_flag)


                        st.success('Thanks for using our service! The result will be send to your email.', icon="✅")
                else:
                    st.warning("Please upload file and input url or enter email address!")

        auth_token = os.environ.get("auth_token")


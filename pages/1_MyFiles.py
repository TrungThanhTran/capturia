import os
import streamlit as st
# from functions import *
from PIL import Image
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException
import streamlit as st
import streamlit_qs as stqs
import streamlit.components.v1 as components
from yaml.loader import SafeLoader
import yaml
import json
from glob import glob
import pandas as pd
import textwrap
import plotly_express as px
import plotly.graph_objects as go
import streamlit_authenticator as stauth
import base64
from st_clickable_images import clickable_images
from database import DBHandler
from awesome_table import AwesomeTable
from awesome_table.column import (Column, ColumnDType)
                                  
# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
dbhandler = DBHandler()

st.set_page_config(page_title="Myfiles", page_icon=image_logo, layout='wide')

def convert_ASCII_string(ascii_list):
    return ''.join(map(chr, ascii_list))

def convert_string_ASCII(a_string):
    ASCII_values = [str(ord(character)) for character in a_string]
    return '-'.join(ASCII_values)

def convert_url(task_id, user):
    return f'https://capturia.io/MyFiles?task={task_id}__{convert_string_ASCII(user)}'


def extract_queue(queue, status=''):
    task_ids = []
    times = []
    files = []
    titles = []
    statuss = []
    results = []
    
    for each_task in queue:
        task_ids.append(each_task[1])
        if 'temp' in each_task[2]:
            files.append(each_task[2].split('/')[-1])
        else:
            files.append(each_task[2])
        times.append(each_task[5])
        try:
            with open(f'./temp/{each_task[1]}/title.txt', 'r') as f:
                title = f.readline()
        except FileNotFoundError:
            title = ""
        titles.append(title)    
        statuss.append(status)
        if status == 'finished':
            results.append(convert_url(each_task[1], each_task[3]))
        else:
            results.append('')
        
  
    df = pd.DataFrame({'task_id':task_ids, 
                       'file_name':files,
                       'title':titles, 
                       'time':times, 
                       'status':statuss,
                       'result':results})
    return df

def display_sentiment(sentiment, sentences):
    sen_df = pd.DataFrame(sentiment)
    sen_df['text'] = sentences
    
    grouped = pd.DataFrame(sen_df['label'].value_counts()).reset_index()
    grouped.columns = ['sentiment','count']
    st.session_state['sen_df'] = sen_df
    col1, col2 = st.columns(2)        
    # Display number of positive, negative and neutral sentiments
    fig = px.bar(grouped, 
                 x='sentiment', 
                 y='count', 
                 color='sentiment', 
                 color_discrete_map={"Negative":"firebrick","Neutral":\
                                    "navajowhite","Positive":"darkgreen"},\
                                    title='Sentiment Analysis')
    fig.update_layout(
        showlegend=False,
        autosize=True,
        margin=dict(
            l=25,
            r=25,
            b=25,
            t=50,
            pad=2
        )
    )
    with col1:    
        st.plotly_chart(fig)
                
    # Display sentiment score
    sentiment_score = 0.0
    _sc_pos = 0.0
    _sc_neg = 0.0
    if 'Positive' in grouped['sentiment'].to_list():
        _sc_pos = grouped[grouped['sentiment']=='Positive']['count'].iloc[0]
                
    if 'Negative' in grouped['sentiment'].to_list():
        _sc_neg = grouped[grouped['sentiment']=='Negative']['count'].iloc[0]
    sentiment_score = (_sc_pos - _sc_neg) * 100/len(sen_df)
                            
    fig_1 = go.Figure()
                
    fig_1.add_trace(go.Indicator(
        mode = "delta",
        value = sentiment_score,
        domain = {'row': 1, 'column': 1}))
                
    fig_1.update_layout(
        template = {'data' : {'indicator': [{
            'title': {'text': "Sentiment Score"},
            'mode' : "number+delta+gauge",
            'delta' : {'reference': 1}}]
                            }},
        autosize=False,
        width=250,
                    height=250,
                    margin=dict(
                        l=5,
                        r=5,
                        b=5,
                        pad=2
                    )
                )
            
    with col2:    
        st.plotly_chart(fig_1)
        
    hd = sen_df.text.apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=70)))
    # Display negative sentence locations
    fig = px.scatter(sen_df, 
                y='label', 
                color='label', 
                size='score', 
                hover_data=[hd], 
                color_discrete_map={"Negative":"firebrick","Neutral":"navajowhite","Positive":"darkgreen"}, 
                title='Sentiment Score Distribution')
    
    fig.update_layout(
        showlegend=False,
        autosize=True,
        width=800,
        height=350,
        margin=dict(
            b=5,
            t=50,
            pad=4
        )
    )
            
    st.plotly_chart(fig)

def convert_df_json(df):
    out = df.to_json(orient='records')[1:-1].replace('},{', '} {')
    return out
    
footer = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """

st.markdown(footer, unsafe_allow_html=True)

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
    st.sidebar.header("My Files")
    st.markdown("## My Files")

    # Get data in folder results

    # query_string: ip:port/MyFiles?task=xxxx&sign_in=True
    params = st.experimental_get_query_params()
    if len(params) > 0 and 'FormSubmitter:Login-Login' in st.session_state:
        # Blank panels
        # p = open(f"/home/trungtranthanh/avr_c/capturia/offline_Meeting_Analysis/results/user_x/task{params['task'][0]}.html")
        # components.html(p.read())
        # Get task and user name
        task_id, user_ascii = params['task'][0].split('__')
        user_ascii = user_ascii.split('-')
        user_ascii = [int(character) for character in user_ascii]
        user_name = convert_ASCII_string(user_ascii)
        
        folder_task = f'./temp/{user_name}/{task_id}'

        trascription_raw_path = os.path.join(folder_task, 'passages.txt')
        trascription_spk_path = os.path.join(
            folder_task, 'matching_speaker.json')
        segments_path = os.path.join(folder_task, 'segments.json')
        sentiment_path = os.path.join(folder_task, 'sentiment.json')
        sentences_path = os.path.join(folder_task, 'sentences.json')
        summary_path = os.path.join(folder_task, 'summary.html')
        
        files = glob(os.path.join(folder_task, '*'))
        for file in files:
            if 'mp3' in file or 'wav' in file or 'm4a' in file:
                audio_path = file


        #### Step 0: display audio file
        

        #### Step 1: Display Trasncription
        st.markdown(f"## Task: {task_id}")
        st.write("\n")

        # st.markdown("<h1 style='text-align: center; color: white;'>Time to become a comic book character</h1>", unsafe_allow_html=True)
        st.markdown(f"### Audio file: {os.path.basename(audio_path)}")
        audio_file = open(audio_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
        st.write("\n")

        
        st.markdown("### **Transcription:**")
        with open(trascription_raw_path, 'r') as file:
            transcription = file.readlines()
        st.markdown(transcription[0])
        st.write("\n")

        #### Step 2: Display sentiment
        st.markdown("### **Sentiment:**")
        with open(sentiment_path, 'r') as file:
            sentiment = json.load(file)

        with open(sentences_path, 'r') as file:
            sentences = json.load(file)
            
        display_sentiment(sentiment, sentences)
        st.write("\n")

        #### Step 3: Display summary
        # st.markdown("### **Summary:**")
        # with st.expander(label='Summarized  Audio',expanded=True): 
        #     p = open(summary_path)
        #     components.html(p.read())
        # st.write("\n")

        #### Step 4: Display speaker transcription
        st.markdown("### **Diarization:**")
        with open(trascription_spk_path, 'r') as f:
            match_result = json.load(f)
        colors = ['red', 'green', 'yellow','blue', 'cyan', 'lime', 'magenta', 'pink', 'orange']

        for seg in match_result:
                st.markdown("\n" + 'time block: from ' + str(int(seg["start"])) + 's to ' + str(int(seg["end"])) + 's',  unsafe_allow_html=True) 
                st.markdown(f'<span style="color:blue">{seg["speaker"]}</span> :' + ' ' + seg['text'],  unsafe_allow_html=True)
                st.markdown("\n", unsafe_allow_html=True)        

    else:
        st.experimental_set_query_params()

        if 'username' in st.session_state or 'name' in st.session_state:   
            try:
                user_name = st.session_state['username']
            except KeyError:
                user_name = st.session_state['name']
                
        task_queue = dbhandler.get_db_by_user(tb_name="TASK_QUEUE", 
                                              user_name=user_name)
        done_queue = dbhandler.get_db_by_user(tb_name="DONE_QUEUE",
                                               user_name=user_name)
        error_queue = dbhandler.get_db_by_user(tb_name="ERROR_QUEUE",
                                               user_name=user_name)
        df_myfiles = pd.DataFrame(columns=['task_id', 'file_name','title','time','status','result'])
        
        if len(task_queue) != 0:
            df_task_queue = extract_queue(task_queue, 'processing')
            df_myfiles = df_myfiles.append(df_task_queue, ignore_index=True)
            
        if len(done_queue) != 0:
            df_done_queue = extract_queue(done_queue, 'finished')
            df_myfiles = df_myfiles.append(df_done_queue, ignore_index=True)
        
        if len(error_queue) != 0:
            df_error_queue = extract_queue(error_queue, 'error')
            df_myfiles = df_myfiles.append(df_error_queue, ignore_index=True)
        
        df_myfiles.style.set_properties(**{'background-color': 'black',
                           'color': 'white'})
        if len(df_myfiles) > 0:
            AwesomeTable(df_myfiles, columns=[
                Column(name='task_id', label='TASK'),
                Column(name='file_name', label='File Name'),
                Column(name='title', label='Transcript Title'),
                Column(name='time', label='Time Request'),
                Column(name='status', label='Status'),
                Column(name='result', label='Result', dtype=ColumnDType.LINK)
            ], show_order=True, show_search=True)


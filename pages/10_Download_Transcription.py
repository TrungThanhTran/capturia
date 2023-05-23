import os
import base64
import streamlit as st
from glob import glob
from PIL import Image
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException

image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="10_Download_transcription",page_icon=image_logo)
footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)
if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")
    st.sidebar.header("Transcription")
    st.markdown("## Download your transcription with speaker diarization")

    folder_path = '/home/ubuntu/project/capturia/Meeting_Analysis/temp/transcript'
    files = glob(os.path.join(folder_path,st.session_state['username'], '*speaker_diarization*'))
    if len(files) > 0:
        hrefs = []
        col_list = st.columns(3)
        for idx, file in enumerate(files):
            with open(file, 'r') as f:
                content = f.readlines()
            # st.write('\n'.join(content))
            content = ''.join(content)
            content = base64.b64encode(content.encode())
            new_filename = file.split('/')[-1].replace('transcript_', '')
            href = f'<a href="data:file/txt;base64,{content.decode()}" download="{new_filename}">{new_filename}!!</a>'
            with col_list[idx%3]:
                st.markdown(href,unsafe_allow_html=True)
    else:
        st.warning("No transcription found!!!")
else:
    st.subheader("Authentication Required...")
    pages = source_util.get_pages('HOME.py')
    for page_hash, config in pages.items():
        if config["page_name"] == 'HOME':
            page_hash_home = page_hash

    if st.button("Return to Home to login"):
        raise RerunException(
            RerunData(
                page_script_hash=page_hash_home,
                page_name='HOME',
            )
        )
        raise ValueError(f"Could not find page HOME. Must be one of HOME")
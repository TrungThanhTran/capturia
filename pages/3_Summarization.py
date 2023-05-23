import streamlit as st
from functions import *
from PIL import Image
from transformers import pipeline
import gc
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException

# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="Summary",page_icon=image_logo)

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
        
        
st.markdown(footer, unsafe_allow_html=True)

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")

    st.sidebar.header("Summarization")
    st.markdown("## Summarization using AI")

    max_len= st.slider("Maximum length of the summarized text",min_value=100,max_value=200,step=10,value=150)
    min_len= st.slider("Minimum length of the summarized text",min_value=70,max_value=200,step=10)

    st.markdown("####")     
            
    st.subheader("Summarized with matched Entities")

    if "passages" not in st.session_state:
        st.session_state["passages"] = ''

    if st.session_state['passages']:
        
        with st.spinner("Summarizing and matching entities, this takes a few seconds..."):
            sum_pipe = pipeline("summarization",model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn",clean_up_tokenization_spaces=True)
            text_to_summarize = chunk_and_preprocess_text(st.session_state['passages'])
            try:
                if len(st.session_state['passages']) < min_len:
                    summarized_text = st.session_state['passages']
                else:
                    summarized_text = summarize_text(sum_pipe, text_to_summarize,max_len=max_len,min_len=min_len)
                    if len(summarized_text) > len(st.session_state['passages']):
                        summarized_text = st.session_state['passages']
            
            except IndexError:
                try:
                    
                    text_to_summarize = chunk_and_preprocess_text(st.session_state['passages'],450)
                    summarized_text = summarize_text(sum_pipe, text_to_summarize,max_len=max_len,min_len=min_len)

        
                except IndexError:
                    
                    text_to_summarize = chunk_and_preprocess_text(st.session_state['passages'],400)
                    summarized_text = summarize_text(sum_pipe, text_to_summarize,max_len=max_len,min_len=min_len)
                            
            entity_match_html = highlight_entities(text_to_summarize,summarized_text)
            st.markdown("####")
            
            with st.expander(label='Summarized  Audio',expanded=True): 
                st.write(entity_match_html, unsafe_allow_html=True)
            
            st.markdown("####")     
            
            transcript_downloader(summarized_text, "Download Summary",  header="summarization", user_name=st.session_state["username"])
            delete_model(sum_pipe)
            del text_to_summarize
            del summarized_text
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

        
    else:
        st.write("No text to summarize detected, please ensure you have entered the YouTube URL on the Sentiment Analysis page")     
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
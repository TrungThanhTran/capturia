import streamlit as st
from pyvis.network import Network
from functions import *
import streamlit.components.v1 as components
import pickle, math
from PIL import Image
import gc
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException

# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="KnowledgeGraph",page_icon=image_logo)

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)
if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")
    st.sidebar.header("Knowledge Graph")
    st.markdown("## Knowledge Graph")

    filename = "knowledge_network.html"

    if "passages" in st.session_state:

        with st.spinner(text='Loading Babelscape/rebel-large which can take a few minutes to generate the graph..'):
            kg_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
            kg_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
            st.session_state.kb_text = from_text_to_kb(st.session_state['passages'], kg_model, kg_tokenizer, "", verbose=True)
            save_network_html(st.session_state.kb_text, filename=filename)
            st.session_state.kb_chart = filename

        with st.container():
            st.subheader("Generated Knowledge Graph")
            st.markdown("*You can interact with the graph and zoom.*")
            html_source_code = open(st.session_state.kb_chart, 'r', encoding='utf-8').read()
            components.html(html_source_code, width=700, height=700)
            st.markdown(st.session_state.kb_text)
            
        delete_model(kg_model)
        delete_model(kg_tokenizer)
        gc.collect()
        torch.cuda.empty_cache()

    else:

        st.write('No audio text detected, please regenerate from Home page..')
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
import streamlit as st
from faster_whisper import WhisperModel

@st.experimental_singleton(suppress_st_warning=True)
def load_fast_asr_model(asr_model_name):
    asr_model = WhisperModel(asr_model_name, device="cpu", compute_type="float32")
    return asr_model
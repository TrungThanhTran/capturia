from __future__ import annotations

import streamlit as st
from faster_whisper import WhisperModel


@st.cache_resource(show_spinner=False)
def load_fast_asr_model(asr_model_name: str) -> WhisperModel:
    """Load and cache a Faster Whisper model."""
    return WhisperModel(asr_model_name, device="cpu", compute_type="float32")

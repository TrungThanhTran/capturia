from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

from functions import inference, load_whisperx_model

ALPHABETS = r"([A-Za-z])"
PREFIXES = r"(Mr|St|Mrs|Ms|Dr)[.]"
SUFFIXES = r"(Inc|Ltd|Jr|Sr|Co)"
STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
DIGITS = r"([0-9])"
MULTIPLE_DOTS = r"\.{2,}"

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""


def split_into_sentences(text: str) -> list[str]:
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(PREFIXES, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    text = re.sub(DIGITS + "[.]" + DIGITS, "\\1<prd>\\2", text)
    text = re.sub(MULTIPLE_DOTS, lambda m: "<prd>" * len(m.group(0)) + "<stop>", text)
    text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\\s" + ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(ACRONYMS + " " + STARTERS, "\\1<stop> \\2", text)
    text = re.sub(ALPHABETS + "[.]" + ALPHABETS + "[.]" + ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(ALPHABETS + "[.]" + ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + SUFFIXES + "[.] " + STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + ALPHABETS + "[.]", " \\1<prd>", text)
    text = text.replace(".”", "”.").replace('."', '".').replace('!"', '"!').replace('?"', '"?')
    text = text.replace(".", ".<stop>").replace("?", "?<stop>").replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>") if s.strip()]
    return sentences


def on_click_search(st_text: str, st_query: str):
    sentences = split_into_sentences(st_text)
    if not sentences:
        return "", None

    embeddings1 = MODEL.encode(sentences, convert_to_tensor=True)
    embeddings2 = MODEL.encode([st_query], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    max_idx = int(torch.argmax(cosine_scores[:, 0]).item())
    max_sentence = sentences[max_idx]
    if float(cosine_scores[max_idx][0]) > 0.51:
        return max_sentence, None

    scored = [
        (sentences[i], float(cosine_scores[i][0]))
        for i in range(len(sentences))
        if float(cosine_scores[i][0]) > 0.3
    ]
    if not scored:
        return max_sentence, None

    df = pd.DataFrame(scored, columns=["Sentences", "Score"]).sort_values("Score", ascending=False)
    df.set_index("Sentences", inplace=True)
    return max_sentence, df


def transcribe_audio_whisperX(audio_path: str, user: str, task_id: str):
    start_time = time.time()
    asr_model = load_whisperx_model("medium")
    texts, title, segments, language, output_audio_path = inference(asr_model, audio_path, user, task_id)

    running_time = time.time() - start_time
    gc.collect()
    torch.cuda.empty_cache()
    del asr_model
    return texts, title, segments, language, running_time, output_audio_path


def main() -> None:
    st.title("Transcript Search Demo")
    task_id = "abcdef123456"
    temp_dir = Path("../temp/test")
    temp_dir.mkdir(parents=True, exist_ok=True)

    select_box = st.selectbox("select mode", ["Upload audio", "Upload text"])
    if select_box == "Upload audio":
        upload_input = st.file_uploader("Upload a .wav or .mp3 sound file", key="upload_audio", type=[".wav", ".mp3", ".mp4", ".m4a"])
        if st.button("Submit Audio"):
            st.session_state["transcription"] = ""
            if upload_input is None:
                st.warning("Please upload file!")
            else:
                audio_path_raw = str(temp_dir / upload_input.name)
                with open(audio_path_raw, "wb") as fp:
                    fp.write(upload_input.getbuffer())
                with st.spinner(text="Transcribing..."):
                    passages, *_ = transcribe_audio_whisperX(audio_path_raw, "test", task_id)
                st.session_state["transcription"] = passages

    if select_box == "Upload text":
        upload_input = st.file_uploader("Upload a text file", key="upload_text", type=[".txt", ".text"])
        if upload_input is not None:
            text_file = temp_dir / upload_input.name
            with open(text_file, "wb") as fp:
                fp.write(upload_input.getbuffer())
            st.session_state["transcription"] = text_file.read_text(encoding="utf-8")

    if st.session_state["transcription"]:
        st.write(st.session_state["transcription"])
        st_query = st.text_input("Enter your query here")
        if st.button("Search"):
            if not st_query:
                st.warning("no text input")
            else:
                sentence, matches = on_click_search(st.session_state["transcription"], st_query)
                st.write(f"Best match: {sentence}")
                if matches is not None:
                    st.write("Other related results:")
                    st.dataframe(matches)


if __name__ == "__main__":
    main()

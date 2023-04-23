import datetime
import numpy as np
import subprocess
import streamlit as st
import torch
from pyannote.audio import Audio
from pyannote.core import Segment
from pydub import AudioSegment
import pydub
from torchaudio.sox_effects import apply_effects_tensor
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from PIL import Image
from functions import *

import wave
import contextlib
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="TakeNote",page_icon=image_logo)
footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Transcription with speakers")
st.markdown("## Transcribe an audio with speakers")

model, feature_extractor = load_si_model("microsoft/wavlm-large")

audio = Audio()

colors = ['red', 'orange', 'black', 'purple', 'green', 'yellow','blue']

EFFECTS = [
    ["remix", "-"],
    ["channels", "1"],
    ["rate", "16000"],
    ["gain", "-1.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
    ["trim", "0", "10"],
]
# num_speakers = st.slider('How many speakers?', 1, 10, 2)

def load_audio(file_name):
    audio = pydub.AudioSegment.from_file(file_name)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
    arr = arr / (1 << (8 * audio.sample_width - 1))
    return arr.astype(np.float32), audio.frame_rate

def feature_extract(path):
    wav, sr = load_audio(path)
    wav, _ = apply_effects_tensor(torch.tensor(wav).unsqueeze(0), sr, EFFECTS)
    input1 = feature_extractor(wav.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
         emb = model(input1).embeddings
    print(emb)
    emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()[0]
    return emb

# TODO: test
# cosine_sim = torch.nn.CosineSimilarity(dim=-1)
# def similarity_fn(emb1, emb2):
#     similarity = cosine_sim(emb1, emb2).numpy()[0]
#     return similarity

def segment_embedding(idx, segment, duration):
  sound = AudioSegment.from_file(st.session_state['si_file'])

  start = segment["start"]*1000
  end = min(duration, segment["end"]) * 1000
  
  part_sound= sound[start:end]
  part_sound.export(f".{st.session_state['si_file'].split('.')[1]}_{idx}.mp3", format="mp3")
  return feature_extract(f".{st.session_state['si_file'].split('.')[1]}_{idx}.mp3")

def time(secs):
  return datetime.timedelta(seconds=round(secs))

num_spk = None
check_num_spk = st.checkbox("Insert number of speaker ")
if check_num_spk:
    num_spk = st.slider('Number of speakers', 1, 10, 2)
    
auto_num_spk = st.checkbox("Auto detect number of speakers")
do_cluster = st.button('Identify speaker')
if do_cluster:
    if "segments" in  st.session_state and st.session_state['segments'] != "":
        segments = st.session_state["segments"]
        embeddings = []
        duration = segments[-1]["end"]-segments[0]["start"]
        error_idx = []
        with st.spinner("Speaker identifing.."):
            print(len(segments))
            for idx, segment in enumerate(segments):
                try:
                    embeddings.append(segment_embedding(idx, segment, duration))
                except Exception as e:
                    print(e)
                    error_idx.append(idx)
            try:     
                if len(error_idx) > 0:
                    for ei in error_idx:
                        segments.pop(ei)
            except Exception as e:
                print(e)
    
            if auto_num_spk: 
                print('running DBSCAN')
                clustering = DBSCAN(eps=0.42, min_samples=1).fit(embeddings)
            else:
                print('running AgglomerativeClustering')
                clustering = AgglomerativeClustering(n_clusters=num_spk).fit(embeddings)
            print(clustering)
            print('num of clusters = ', len(clustering.labels_))
            print(clustering.labels_)

            st.write(f"number of speakers = {len(set(clustering.labels_))}")

            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    st.write('\n')
                    import random
                    st.markdown(f'<span style="color:{random.choice(colors)}">{segment["speaker"]}</span>',  unsafe_allow_html=True)
                    st.markdown("\n" + 'time block: ' + str(time(segment["start"])),  unsafe_allow_html=True)
                st.write(segment["text"][1:] + ' ')
                st.write("\n")
    else:
        st.write('no audio found!')
        
delete_model(feature_extractor)
delete_model(model)

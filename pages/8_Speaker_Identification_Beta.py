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
from PIL import Image
from functions import *
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException

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
if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")

    st.sidebar.header("Transcription with speakers")
    st.markdown("## Identify speakers within the audio file")

    model, feature_extractor = load_si_model("microsoft/wavlm-base-plus-sv")

    audio = Audio()

    colors = ['red', 'green', 'yellow','blue']

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
        emb = torch.nn.functional.normalize(emb, dim=-1).cpu()
        return emb

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    def similarity_fn(emb1, emb2):
        similarity = cosine_sim(emb1, emb2).numpy()[0]
        return similarity

    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    def compare_with_reg(segment_file, dict_names):
        # upload check get embe
        _emb = feature_extract(segment_file)
        # compare
        max_sim = 0.0
        match_name = ""     
        for _name, emb_reg in zip(dict_names.keys(), dict_names.values()):
            _sim = similarity_fn(_emb, emb_reg)
            if _sim > max_sim:
                match_name = _name
                max_sim = _sim
        return match_name, max_sim

    def segment_embedding(idx, segment, duration, dict_names):
        print('start segment embeddings ...')
        sound = AudioSegment.from_file(st.session_state['si_file'])

        start = segment["start"]*1000
        end = min(duration, segment["end"]) * 1000
        
        part_sound= sound[start:end]
        part_sound.export(f".{st.session_state['si_file'].split('.')[1]}_{idx}.mp3", format="mp3")
        return compare_with_reg(f".{st.session_state['si_file'].split('.')[1]}_{idx}.mp3", dict_names)

    with st.expander("Upload sample voice"):
        upload_registration = st.file_uploader("",key="upload_registration")
        user_name = st.text_input('speaker name')

        confirm_reg = st.button('Register')
        if confirm_reg:
            with st.spinner("Uploading registration..."):
                if upload_registration:
                    if not os.path.exists(f"./registration/{user_name}"):
                        os.mkdir(f"./registration/{user_name}")
                        
                    with open(f"./registration/{user_name}/{user_name}.mp3","wb") as f:
                        f.write(upload_registration.getbuffer())
                    st.success("Upload sample successfully!")

    st.markdown("<br><br>",  unsafe_allow_html=True)

    reg_person = glob('./registration/*')
    reg_person = [p.split('/')[-1] for p in reg_person]
    options = st.multiselect(
        'Select registered speaker',
        reg_person)
    do_identify = st.button('Identify speaker')
    if do_identify:
        folders = glob('./registration/*')
        dict_names = {}
        # Make dict emb of all reg
        with st.spinner("Identifing speaker..."):
            for user_name in folders:
                files = glob(os.path.join(user_name, '*'))
                reg_emb = feature_extract(files[0])
                if user_name.split('/')[-1] in options:
                    dict_names[user_name.split('/')[-1]] = reg_emb       

            if "segments" in  st.session_state and st.session_state['segments'] != "":
                segments = st.session_state["segments"]
                speaker_names = []
                speaker_scores = []
                duration = segments[-1]["end"]-segments[0]["start"]
                error_idx = []
                for idx, segment in enumerate(segments):
                    try:
                            _name, _score = segment_embedding(idx, segment, duration, dict_names)
                            speaker_names.append(_name)
                            speaker_scores.append(_score)
                    except Exception as e:
                            print(e)
                            error_idx.append(idx)
                try:     
                        if len(error_idx) > 0:
                            for ei in error_idx:
                                segments.pop(ei)
                except Exception as e:
                        print(e)
                # st.write(speaker_scores)
                # st.write(speaker_names)
                try:
                    for i in range(len(segments)):
                        segments[i]["speaker"] = 'SPEAKER: ' + str(speaker_names[i]  + ' with confidence of ' + str(int(speaker_scores[i] * 100)) + '%')
                    transcription_down = []
                    with st.expander(label='Transcription with speakers',expanded=False): 
                        for (i, segment) in enumerate(segments):
                                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                                    st.write('\n')
                                    import random
                                    transcription_down.append('\n')
                                    transcription_down.append(segment["speaker"])
                                    transcription_down.append('time block: ' + str(time(segment["start"])))
                                    st.markdown(f'<span style="color:{random.choice(colors)}">{segment["speaker"]}</span>',  unsafe_allow_html=True)
                                    st.markdown("\n" + 'time block: ' + str(time(segment["start"])) + ':' + + str(time(segment["end"])),  unsafe_allow_html=True)
                                transcription_down.append(segment["text"][1:] + ' ')
                                st.write(segment["text"][1:] + ' ')
                                st.write("\n")
                    transcript_downloader("\n".join(transcription_down), "Download transcription with speakers",  header="speaker_identification_beta", user_name=st.session_state["username"])
                except IndexError as ie:
                    print(ie)
                    st.write('Oop.. Sorry, please check the file audio again!')
                    
            else:
                st.write('no audio found!')
                
    delete_model(feature_extractor)
    delete_model(model)
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

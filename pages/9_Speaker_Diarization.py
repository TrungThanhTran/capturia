import streamlit as st
from functions import *
import streamlit.components.v1 as components
import pickle, math
import os
import torch
from pydub import AudioSegment
from PIL import Image
from glob import glob
import torch
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException
from pyannote.audio import Pipeline
import json

# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="Speaker Diarization",page_icon=image_logo)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colors = ['red', 'green', 'yellow','blue', 'cyan', 'lime', 'magenta', 'pink', 'orange']

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

def matching_tran_seg(segs, tran):
    count_tran = 0
    new_segs = []
    for idx, seg in enumerate(segs):
        add_up = False
        if idx == 0:
            seg['speaker'] = tran[0]['speaker']
            new_segs.append(seg)
            continue
            
        # tran[stop] < seg[start]
        if (seg['start'] >= tran[count_tran]['stop']):
            if count_tran < len(tran) - 1:
                count_tran += 1
                add_up = True
            elif count_tran == len(tran):
                seg['speaker'] = tran[count_tran]['speaker']
            else:
                continue
        
        # seg[stop] < tran[start]
        if (seg['end'] < tran[count_tran]['start']):
            if idx > 0 and count_tran > 0:
                if abs(seg['end']  - tran[count_tran]['start']) >= abs(seg['start'] - tran[count_tran - 1]['stop']):
                    seg['speaker'] = tran[count_tran]['speaker']
                else:
                    seg['speaker'] = segs[idx-1]['speaker']
        
        # seg[start] < tran[start] < seg[end]
        if seg['start'] <= tran[count_tran]['start'] and seg['end'] > tran[count_tran]['start']:
            if abs(seg['start'] - tran[count_tran]['start']) >= abs(seg['end'] - tran[count_tran]['start']):
                seg['speaker'] = tran[count_tran]['speaker']
            else: 
                if count_tran < (len(tran) - 1) and add_up == False:
                    seg['speaker'] =  tran[count_tran + 1]['speaker']
                    count_tran += 1
                else:
                    seg['speaker'] = tran[count_tran]['speaker']
        
        # tran[start] < seg[start] < seg[end] < tran[end]
        if (seg['start'] >= tran[count_tran]['start']) and (seg['end'] <= tran[count_tran]['stop']):
            seg['speaker'] = tran[count_tran]['speaker']
        
        # tran[start] < tran[stop] < seg[end]
        if seg['start'] <= tran[count_tran]['stop'] and tran[count_tran]['stop'] < seg['end']:
            if abs(seg['start'] - tran[count_tran]['stop']) >= abs(seg['end'] - tran[count_tran]['stop']):
                seg['speaker'] = tran[count_tran]['speaker']
            else: 
                if count_tran < (len(tran) - 1) and add_up == False:
                    seg['speaker'] =  tran[count_tran + 1]['speaker']
                    count_tran += 1
                else:
                    seg['speaker'] = tran[count_tran]['speaker']
                    
        if 'speaker' not in seg.keys() and idx > 0:
            seg['speaker'] = segs[idx - 1]['speaker']
            
        new_segs.append(seg)
    return new_segs

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")
    st.sidebar.header("Speaker Diarization")
    st.markdown("## Speaker Diarization")
    if os.path.exists(st.session_state['si_file']):
        # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
        #                                     use_auth_token="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")

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                             use_auth_token="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")

        # 4. apply pretrained pipeline
        with st.spinner("Processing file, this may take up 1 hour, please go to Download Transcription later for the file..."):
            process_file = st.session_state['si_file']
            if "wav" not in st.session_state['si_file']:
                # assign files
                output_file = process_file.replace("mp3", "wav")
            
                # convert mp3 file to wav filetr
                sound = AudioSegment.from_file(process_file)
                sound.export(output_file, format="wav")
                
            diarization = pipeline(output_file)

            # 5. print the result
            tran = []
            list_speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                tran.append({'start':turn.start,
                            'stop':turn.end,
                            'speaker':speaker})
                list_speakers.append(speaker)
            with open('../speaker_diarization.json', 'w') as f:
                json.dump(tran, f, indent=6)
            st.write(f"found {len(set(list_speakers))} speakers!")
            # st.write(st.session_state)
            match_result = matching_tran_seg(st.session_state['segments'], tran)
            transcription_down = []
            
            speaker_color_code = {}
            for idx, speaker in enumerate(list(set(list_speakers))):
                if idx > len(colors) - 1:
                    idx = idx % len(colors)
                speaker_color_code[speaker] = colors[idx]
            
            for seg in match_result:
                transcription_down.append("\n" + 'time block: from ' + str(int(seg["start"])) + 's to ' + str(int(seg["end"])))
                st.markdown("\n" + 'time block: from ' + str(int(seg["start"])) + 's to ' + str(int(seg["end"])) + 's',  unsafe_allow_html=True) 
                transcription_down.append("\n" + 'speaker: ' + seg['speaker'] + ': ' + seg['text'])
                st.markdown(f'<span style="color:{speaker_color_code[seg["speaker"]]}">{seg["speaker"]}</span> :' + seg['text'],  unsafe_allow_html=True)
                transcription_down.append("\n")
                st.markdown("\n", unsafe_allow_html=True)
            transcript_downloader("\n".join(transcription_down), "Download transcription with segmentation",  header="speaker_diarization", user_name=st.session_state["username"])
    else:
        st.write("please upload a file or add url link")
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

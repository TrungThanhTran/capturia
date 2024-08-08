from transformers import pipeline
from functions import *
import gc
from transformers import pipeline
import time
import streamlit as st
from transformers import pipeline
from typing import Callable, List, Dict
from backend import transcribe_audio_whisperX
import yaml
from yaml.loader import SafeLoader
import torch
from transformers import pipeline
import pandas as pd
import traceback


alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
st.set_page_config(page_title="Myfiles", layout='wide')
st.title("Summary solution for takenote.ai - ver 2.0 - Large Language Model approach")
import time


from pytube.innertube import _default_clients
from pytube import cipher
import re

_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]



def get_throttling_function_name(js: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    function_patterns = [
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)',
    ]
    #logger.debug('Finding throttling function name')
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            #logger.debug("finished regex search, matched: %s", pattern)
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
                        nfunc=re.escape(function_match.group(1))),
                    js
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    raise RegexMatchError(
        caller="get_throttling_function_name", pattern="multiple"
    )

cipher.get_throttling_function_name = get_throttling_function_name

def download_audio_from_youtube(url: str, video_name: str) -> str:
    """Download the audio from a YouTube video and save it as an MP3 file."""
    video_url= YouTube(url)
    video = video_url.streams.filter(only_audio=True).first()
    filename = video_name + ".mp3"
    video.download(filename=filename)
    return filename

def split_text_into_chunks(document: str, max_tokens: int):
    if not document:
        return []

    chunks, current_chunk, current_length = [], [], 0

    try:
        for sentence in nltk.sent_tokenize(document):
            sentence_length = len(sentence)

            if current_length + sentence_length < max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [sentence], sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        st.write(f"Error splitting text into chunks: {e}")
        return []

def split_text_into_chunks(document: str, max_tokens: int):
    if not document:
        return []

    chunks, current_chunk, current_length = [], [], 0

    try:
        for sentence in nltk.sent_tokenize(document):
            sentence_length = len(sentence)

            if current_length + sentence_length < max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [sentence], sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        st.write(f"Error splitting text into chunks: {e}")
        return []

def get_summary_bart(
    list_chunks: List[str], summarizer: Callable, summarization_params: Dict[str, int]
) -> str:
    # Generate summaries for each text chunk
    try:
        summaries = [
            summarizer(chunk, **summarization_params)[0]["summary_text"]
            for chunk in list_chunks
        ]
        return " ".join(summaries)
    except Exception as e:
        print(f"Error generating summaries: {e}")
        return ""

def save_summary_to_file(summary: str, file_name: str) -> None:
    try:
        # Save the summary to a file
        with open(f"{file_name}.txt", "a") as fp:
            fp.write(summary)
    except Exception as e:
        print(f"Error saving summary to file: {e}")

def file_downloader(raw_text, text_button, header="_", user_name="admin"):
    b64 = base64.b64encode(raw_text.encode())
    new_filename = f"transcript_{header}_{user_name}_{time_str}.txt"
    if 'speaker_diarization' in header:
        try:
            with open(f"./temp/transcript/{user_name}/{new_filename}", "w+") as f:
                f.write(raw_text)
        except Exception as e:
            print(str(e))
    st.markdown(f"#### {text_button} ###")
    href = f'<a href="data:file/txt;base64,{b64.decode()}" download="{new_filename}">Click to Download!!</a>'
    st.markdown(href, unsafe_allow_html=True)

def summary_texts(pipe, text_chunks):
    summary_chuncks = []
    for chunck in text_chunks:
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who give summary concise for a text in plain text not pinpoints.",
            },
            {"role": "user", "content": f"{chunck}"},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        summary_chuncks.append(outputs[0]["generated_text"].split('<|assistant|>')[-1])
    return summary_chuncks

if __name__ == "__main__":
    try:
        print("TRANSCRIBING...")
        select_box = st.selectbox('select mode', ['Past youtube URL', 'Upload audio'])
        if select_box == 'Upload audio':
            upload_input = st.empty()
            upload_input = st.file_uploader("Upload a .wav or .mp3 sound file",
                                                key="upload", 
                                                type=['.wav','.mp3','.mp4','.m4a'])

            if upload_input is not None:
                with st.spinner(text="Submitting..."):
                    with open(f"temp/test/{upload_input.name}", "wb") as f:
                        f.write(upload_input.getbuffer())
                audio_path_raw = f"temp/test/{upload_input.name}"
            else:
                st.warning("Please upload file!")
        elif select_box == 'Past youtube URL':
            url_youtube = st.text_input('Youtube URL')
            if url_youtube != "":
                audio_path_raw = download_audio_from_youtube(url_youtube, "demo")
        
        btn_submit = st.button('Get summary')
        gc.collect()
        # del pipe  
        torch.cuda.empty_cache()
        if btn_submit:
            with st.spinner("Transcribing audio"):
                with open('../data/model/model_config.yaml') as file:
                    model_config = yaml.load(file, Loader=SafeLoader)
                full_text, _, _, _, _ = transcribe_audio_whisperX(model_config, audio_path_raw, "test", "1234567")
                with st.expander("Transcription for the audio:"):
                    st.write(full_text['text'])
                    file_downloader(full_text['text'], "download transcription")
            
            text_chunks = split_text_into_chunks(full_text['text'], max_tokens=7000)

            with st.spinner("Summarizing audio"):
                start = time.time()
                with st.expander("Summary for the audio"):
                    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
                    summary_chuncks = summary_texts(pipe, text_chunks)
                    summary_text =" ".join(summary_chuncks)

                    while len(summary_text) > 4000:
                        print('too long need chuncks.')
                        text_chunks = split_text_into_chunks(summary_text, max_tokens=4000)
                        summary_chuncks = summary_texts(pipe, text_chunks)
                        summary_text =" ".join(summary_chuncks)

                    gc.collect()
                    del pipe  
                    torch.cuda.empty_cache()
                    
                    st.write(summary_text)
                    st.write(f'time to sum = {time.time() - start} seconds' )
                    file_downloader(summary_text, "download summary")
                    try:
                        path_file = url_youtube
                    except: 
                        path_file = upload_input.name
                    import uuid

                    file_name = str(uuid.uuid4())
                    print(path_file)
                    print(full_text['text'])
                    print(summary_text)
                    print(f'time to sum = {time.time() - start} seconds' )

                    df = pd.DataFrame([{'file_path': path_file, 'transcript':full_text['text'], 'summary': summary_text}])
                    df.to_csv(f'temp/result/{file_name}.csv', index=False)

                    gc.collect()
                    torch.cuda.empty_cache()

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        print(e)
        traceback.print_exc()
        st.error(e)
        st.info('please refresh!')

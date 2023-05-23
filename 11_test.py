
from faster_whisper import WhisperModel
import streamlit as st
import time
import whisperx
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import librosa
import soundfile
from helper import *
import os

model_size = "medium.en"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="float32")   

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
vocal_target = f"./temp/{st.session_state['username']}/audio.mp3"
with st.spinner("Transcribing.."):
    results = {}
    start = time.time()
    st.write(start)
    segments, info = model.transcribe(f"./temp/{st.session_state['username']}/audio.mp3", beam_size=5, word_timestamps=True)
    st.write(time.time())
    running_time = time.time() - start
    st.write(int(running_time))
    if running_time / 3600 > 1.0:
        running_time = str(running_time / 3600).format("{:. 2f}")[:4] + ' hours'
    elif running_time / 60 > 1.0:
        running_time = str(running_time / 60).format("{:. 2f}")[:4] + ' minutes'
    else:
        running_time = str(running_time).format("{:. 2f}")[:4] + ' seconds'
    results['segments'] = segments
    results['text'] = ' '.join([segment.text for segment in segments])
    results['info'] = info
    st.write('running time = ', running_time)
    st.write(results['text'])
    
    del WhisperModel
    
    whisper_results = []
    for segment in segments:
        whisper_results.append(segment._asdict())
        # whisper_results.append(segment)
    # clear gpu vram

    if info.language in wav2vec2_langs:
        device = "cpu"
        alignment_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, device
        )
        word_timestamps = result_aligned["word_segments"]
        # clear gpu vram
        del alignment_model
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"text": word[2], "start": word[0], "end": word[1]})


    # convert audio to mono for NeMo combatibility
    signal, sample_rate = librosa.load(vocal_target, sr=None)
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    os.chdir(temp_path)
    soundfile.write("mono_file.wav", signal, sample_rate, "PCM_24")

    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config()).to("cpu")
    msdd_model.diarize()

    del msdd_model

    # Reading timestamps <> Speaker Labels mapping

    output_dir = "nemo_outputs"

    speaker_ts = []
    with open(f"{output_dir}/pred_rttms/mono_file.rttm", "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info['language'] in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    else:
        print(
            f'Punctuation restoration is not available for {whisper_results["language"]} language.'
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    os.chdir(ROOT)  # back to parent dir
    with open(f"test.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"test.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)

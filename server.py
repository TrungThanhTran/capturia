from __future__ import annotations

import gc
import os
import shutil
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import (
    diarize_speaker_whisperX,
    load_model_config,
    sentimet_audio,
    transcribe_audio_whisperX,
)
from database_aws import S3_Handler
from functions import download_from_youtube, get_file_size_in_kb, save_file_json, save_file_text
from logger import log_api_error, log_api_result


class TranscribeItem(BaseModel):
    file_name: str


class YTLink(BaseModel):
    link: str
    task: str


app = FastAPI(title="Takenote API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_AUDIO_FOLDER = "audio"
S3_BUCKETNAME = os.environ["S3_BUCKETNAME"]
s3_handler = S3_Handler(S3_BUCKETNAME)


@app.get("/")
async def add_index() -> str:
    return "This is an API for Takenote.AI"


@app.get("/helloworld")
def hello_world() -> str:
    return "This is an API for Takenote.AI"


@app.post("/api/v1/download/video")
def api_v1_download_video(ytlink: YTLink):
    try:
        log_api_result(f"api v1 downloading this = {ytlink}")
        audio_file_path, _, _, raw_audio_name = download_from_youtube(ytlink.link, ytlink.task)
        file_size = get_file_size_in_kb(audio_file_path)

        audio_file_name = os.path.basename(audio_file_path)
        upload_flag = s3_handler.upload_file_to_s3(audio_file_path, audio_file_name)
        if not upload_flag:
            raise RuntimeError("Could not upload audio to S3")

        response_dict = {
            "bucket": S3_BUCKETNAME,
            "mimetype": "audio/mp3",
            "filename": raw_audio_name,
            "size": file_size,
            "key": ytlink.task,
            "handle": audio_file_name,
        }
        shutil.rmtree(f"temp/youtube_down/{ytlink.task}", ignore_errors=True)
        return response_dict
    except Exception as exc:
        log_api_error(str(exc))
        return JSONResponse(content={"detail": "Could not download the link"}, status_code=404)


@app.post("/api/v1/transcribe/file")
def api_v1_transcribe_file(item: TranscribeItem):
    try:
        log_api_result(str(item))
        model_config = load_model_config()

        response_payload = {
            "file_name": item.file_name,
            "transcript": "",
            "sentiment": [],
            "speaker_diarization": [],
        }

        req_uuid = uuid.uuid4()
        local_file_path = f"./temp/{req_uuid}_{item.file_name}"
        download_flag = s3_handler.download_file_from_s3(item.file_name, local_file_path)
        if not download_flag:
            raise RuntimeError("Could not download file from S3")

        results, _, _, _, audio_path = transcribe_audio_whisperX(
            model_config,
            local_file_path,
            TEMP_AUDIO_FOLDER,
            "1234",
        )
        passages = results["text"]
        segments = results["segments"]

        Path(f"./temp/{TEMP_AUDIO_FOLDER}").mkdir(parents=True, exist_ok=True)
        save_file_text(passages, f"./temp/{TEMP_AUDIO_FOLDER}/passages.txt")
        response_payload["transcript"] = passages

        sentiment, sentences = sentimet_audio(passages)
        save_file_json(sentiment, f"./temp/{TEMP_AUDIO_FOLDER}/sentiment.json")
        response_payload["sentiment"] = sentiment

        try:
            trans_with_spk = diarize_speaker_whisperX(
                audio_path,
                segments,
                model_config["transcribe"]["device"],
                model_config["transcribe"]["hf_token"],
            )
        except Exception as exc:
            log_api_error(str(exc))
            trans_with_spk = []

        save_file_json(trans_with_spk, f"./temp/{TEMP_AUDIO_FOLDER}/matching_speaker.json")
        response_payload["speaker_diarization"] = trans_with_spk

        os.remove(local_file_path)
        del segments, passages, results, sentences
        return response_payload
    except Exception as exc:
        log_api_error(str(exc))
        gc.collect()
        torch.cuda.empty_cache()
        return JSONResponse(content={"detail": "out of memory"}, status_code=404)


if __name__ == "__main__":
    print("Start running fastapi app")

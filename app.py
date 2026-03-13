from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, request
from flask_cors import CORS

from backend import (
    diarize_speaker_whisperX,
    load_model_config,
    sentimet_audio,
    transcribe_audio_whisperX,
)
from database_aws import S3_Handler
from functions import save_file_json, save_file_text
from logger import log_api_error, log_api_result


app = Flask(__name__)
CORS(app)

S3_BUCKETNAME = os.environ["S3_BUCKETNAME"]
s3_handler = S3_Handler(S3_BUCKETNAME)
TEMP_DIR = Path("./temp/test")


@app.route("/")
def add_index() -> str:
    return "This is an API for Takenote.AI"


@app.route("/helloworld")
def hello_world() -> str:
    return "This is an API for Takenote.AI"


@app.route("/api/v1/transcribe/file", methods=["POST", "GET"])
def api_v1_transcribe_file():
    model_config = load_model_config()

    response_payload = {
        "file_name": "",
        "transcript": "",
        "sentiment": [],
        "speaker_diarization": [],
    }

    try:
        request_data = request.get_json(force=True)
        audio_path_raw = request_data["file_name"]
        response_payload["file_name"] = audio_path_raw
        log_api_result(f"transcribe request for {audio_path_raw}")

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        local_file_path = str(TEMP_DIR / os.path.basename(audio_path_raw))

        download_flag = s3_handler.download_file_from_s3(audio_path_raw, local_file_path)
        if not download_flag:
            raise RuntimeError("Download from S3 failed")

        results, _, _, _, audio_path = transcribe_audio_whisperX(
            model_config,
            local_file_path,
            "test",
            "1234",
        )
        passages = results["text"]
        segments = results["segments"]

        save_file_text(passages, str(TEMP_DIR / "passages.txt"))
        response_payload["transcript"] = passages

        sentiment, sentences = sentimet_audio(passages)
        save_file_json(sentiment, str(TEMP_DIR / "sentiment.json"))
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

        save_file_json(trans_with_spk, str(TEMP_DIR / "matching_speaker.json"))
        response_payload["speaker_diarization"] = trans_with_spk
        os.remove(local_file_path)
        del segments, passages, results, sentences
        return response_payload
    except Exception as exc:
        log_api_error(str(exc))
        return {"error": str(exc)}


if __name__ == "__main__":
    print("Start running flask app")
    app.config["JSON_AS_ASCII"] = False
    app.run(host="0.0.0.0", debug=False, port=8000, threaded=False)

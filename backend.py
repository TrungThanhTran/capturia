from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import torch
import yaml
from yaml.loader import SafeLoader

from database_aws import S3_Handler, SQS_Handler
from email_sender import Email_Sender
from functions import (
    align_speaker,
    assign_speaker,
    convert_string_ASCII,
    inference,
    load_sentiment_models,
    load_whisperx_model,
    save_file_json,
    save_file_text,
    sentiment_pipe,
)
from logger import log_api_error, log_api_result


MODEL_CONFIG_PATH = "data/model/model_config.yaml"


def load_model_config(path: str = MODEL_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.load(file, Loader=SafeLoader)


def sentimet_audio(passage: str):
    sent_pipe = load_sentiment_models()
    log_api_result("successfully loaded sentiment model")
    sentiment, sentences = sentiment_pipe(sent_pipe, passage)
    log_api_result("sentiment done")
    gc.collect()
    torch.cuda.empty_cache()
    del sent_pipe
    return sentiment, sentences


def transcribe_audio_whisperX(model_config: dict[str, Any], audio_path: str, user: str, task_id: str):
    log_api_result("Start transcribing audio file")
    start_time = time.time()
    asr_model = load_whisperx_model(
        "medium",
        model_config["transcribe"]["device"],
        model_config["transcribe"]["compute_type"],
    )

    results, title, language, output_audio_path = inference(
        asr_model,
        audio_path,
        user,
        task_id,
        model_config["transcribe"]["batch_size"],
    )

    elapsed = time.time() - start_time
    log_api_result(f"Transcription time = {elapsed}")

    gc.collect()
    torch.cuda.empty_cache()
    del asr_model

    return results, title, language, elapsed, output_audio_path


def diarize_speaker_whisperX(audio_path: str, segments: list[dict[str, Any]], device: str, hf_token: str):
    start = time.time()
    align_result = align_speaker(segments, audio_path, device)
    log_api_result(f"time to align = {time.time() - start} seconds")

    start = time.time()
    result = assign_speaker(align_result, audio_path, hf_token, device)
    log_api_result(f"time to assign = {time.time() - start} seconds")

    transcript_with_speaker = []
    for seg in result.get("segments", []):
        text = seg.get("text")
        speaker = seg.get("speaker")
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        if text is None or speaker is None or seg_start is None or seg_end is None:
            continue
        transcript_with_speaker.append(
            {"text": text, "speaker": speaker, "start": seg_start, "end": seg_end}
        )

    log_api_result("diarization speaker done")
    return transcript_with_speaker


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def _ensure_user_temp_dir(user: str) -> Path:
    temp_dir = Path("temp") / user
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def main() -> None:
    model_config = load_model_config()

    s3_bucket_name = _required_env("S3_BUCKETNAME")
    task_queue = _required_env("TASK_QUEUE")
    error_queue = _required_env("ERROR_QUEUE")
    finish_queue = _required_env("FINISH_QUEUE")

    email_sender = Email_Sender()
    s3_handler = S3_Handler(s3_bucket_name)
    sqs_handler = SQS_Handler()

    while True:
        task_id, audio_path_raw, user, rev_email, task_time, status = sqs_handler.get_message(task_queue)

        if not task_id or not audio_path_raw:
            continue

        print("GET A TASK...")
        user_temp_dir = _ensure_user_temp_dir(user)

        json_data = {
            "task_id": task_id,
            "file_path": audio_path_raw,
            "user": user,
            "email": rev_email,
            "time": task_time,
        }
        s3_path_task = f"{user}/{task_id}/"

        if user not in s3_handler.list_username_in_bucket():
            s3_handler.create_s3_folder(user)
        s3_handler.create_s3_folder(s3_path_task)

        try:
            print("TRANSCRIBING...")
            file_name = os.path.basename(audio_path_raw)
            local_file_path = str(user_temp_dir / file_name)

            if audio_path_raw.startswith(f"s3://{s3_bucket_name}/"):
                audio_path_s3 = audio_path_raw.replace(f"s3://{s3_bucket_name}/", "")
                download_flag = s3_handler.download_file_from_s3(audio_path_s3, local_file_path)
            else:
                local_file_path = audio_path_raw
                download_flag = True

            if not download_flag:
                raise RuntimeError("Failed to download source audio")

            results, title, language, running_time, audio_path = transcribe_audio_whisperX(
                model_config, local_file_path, user, task_id
            )
            passages = results["text"]
            segments = results["segments"]

            save_file_text(passages, str(user_temp_dir / "passages.txt"))
            s3_handler.upload_file_to_s3(str(user_temp_dir / "passages.txt"), s3_path_task)

            save_file_text(title, str(user_temp_dir / "title.txt"))
            s3_handler.upload_file_to_s3(str(user_temp_dir / "title.txt"), s3_path_task)

            save_file_json(segments, str(user_temp_dir / "segments.json"))
            s3_handler.upload_file_to_s3(str(user_temp_dir / "segments.json"), s3_path_task)

            print("SENTIMENT...")
            sentiment, sentences = sentimet_audio(passages)
            save_file_json(sentiment, str(user_temp_dir / "sentiment.json"))
            s3_handler.upload_file_to_s3(str(user_temp_dir / "sentiment.json"), s3_path_task)

            save_file_json(sentences, str(user_temp_dir / "sentences.json"))
            s3_handler.upload_file_to_s3(str(user_temp_dir / "sentences.json"), s3_path_task)

            print("DIARIZE...")
            trans_with_spk = diarize_speaker_whisperX(
                audio_path,
                segments,
                model_config["transcribe"]["device"],
                model_config["transcribe"]["hf_token"],
            )
            save_file_json(trans_with_spk, str(user_temp_dir / "matching_speaker.json"))
            s3_handler.upload_file_to_s3(str(user_temp_dir / "matching_speaker.json"), s3_path_task)

            json_data["status"] = "1"
            sqs_handler.send_message(finish_queue, json.dumps(json_data))

            for path in glob(str(user_temp_dir / "*")):
                os.remove(path)

            task_id_user = task_id + "__" + "-".join(convert_string_ASCII(user))
            email_sender.send_email_text(rev_email, task_id_user)
            print("Sending email with result!")

        except Exception as exc:
            log_api_error(str(exc))
            print("ERROR]____:", exc)

            next_status = int(status) + 1
            json_data["time"] = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
            json_data["status"] = str(next_status)

            retry_queue = error_queue if next_status > 2 else task_queue
            sqs_handler.send_message(retry_queue, json.dumps(json_data))


if __name__ == "__main__":
    main()

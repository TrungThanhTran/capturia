from __future__ import annotations

import logging
import os
import time


LOG_PATH = "data/log/api.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("capturia.api")
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_current_time() -> str:
    current_time = time.localtime()
    return time.strftime("%Y-%m-%d %H:%M:%S", current_time)


def log_api_result(result: str) -> None:
    logger.info(f"RESULT:{get_current_time()}:{result}")


def log_common(text: str) -> None:
    logger.info(text)


def log_api_error(error: str) -> None:
    logger.error(f"ERROR:{get_current_time()}:{error}")

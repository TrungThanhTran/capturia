import logging
import json
import time



# Configure the logging module
logging.basicConfig(filename="data/log/api.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def log_api_result(result):
    time
    logger.info(f"RESULT:{get_current_time()}:{result}")
    
def log_common(text):
    logger.info(text)

def log_api_error(error):
    logger.error(f"ERROR:{get_current_time()}:{error}")
    
def get_current_time():
    # Get the current time
    current_time = time.localtime()

    # Convert the time object to a string
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", current_time)

    return time_string

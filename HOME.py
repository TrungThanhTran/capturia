from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from PIL import Image
from email_validator import EmailNotValidError, validate_email
from yaml.loader import SafeLoader

from database import DBHandler

LOGO_PATH = Path("data/logo/logo.png")
USER_CONFIG_PATH = Path("data/pass/user_db.yaml")


def save_task_to_queue(
    dbhandler: DBHandler,
    file_name: str,
    user: str,
    email: str,
    uniq_id: uuid.UUID,
    status: int = 0,
) -> bool:
    timestamp = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
    sanitized_file_name = file_name.replace("'", "")
    try:
        dbhandler.writeinfo_db("TASK_QUEUE", str(uniq_id), sanitized_file_name, user, email, timestamp, status)
        return True
    except Exception:
        return False


def check_email(email: str) -> str:
    try:
        validate_email(email)
        return "email is valid"
    except EmailNotValidError as exc:
        return str(exc)


def ensure_task_dirs(username: str, task_id: uuid.UUID) -> Path:
    target = Path("temp") / username / str(task_id)
    target.mkdir(parents=True, exist_ok=True)
    return target


def submit_task(
    dbhandler: DBHandler,
    username: str,
    choice: str,
    upload_input,
    url_input: str,
    email_in: str,
) -> tuple[bool, str]:
    if not email_in:
        return False, "Please enter email address!"

    email_status = check_email(email_in)
    if email_status != "email is valid":
        return False, email_status

    task_id = uuid.uuid4()
    task_dir = ensure_task_dirs(username, task_id)

    if choice == "By uploading a file":
        if upload_input is None:
            return False, "Please upload a file first."
        local_file_path = task_dir / upload_input.name
        with open(local_file_path, "wb") as fp:
            fp.write(upload_input.getbuffer())
        save_ok = save_task_to_queue(dbhandler, str(local_file_path), username, email_in, task_id)
    else:
        if not url_input:
            return False, "Please enter a video URL."
        save_ok = save_task_to_queue(dbhandler, url_input, username, email_in, task_id)

    if not save_ok:
        return False, "Could not save task to queue."

    return True, "Thanks for using our service! The result will be sent to your email."


def main() -> None:
    image_logo = Image.open(LOGO_PATH)
    st.set_page_config(page_title="HOME", page_icon=image_logo, layout="wide")

    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with open(USER_CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=SafeLoader)

    dbhandler = DBHandler()
    if "TASK_QUEUE" not in dbhandler.list_talbe_db():
        dbhandler.create_table("TASK_QUEUE")

    st.experimental_set_query_params()

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )
    st.session_state["authenticator"] = authenticator
    _, authentication_status, username = authenticator.login("Login", "main")
    st.session_state["authentication_status"] = authentication_status

    if authentication_status is False:
        st.error("Username/password is incorrect")
        return
    if authentication_status is None:
        st.warning("Please enter your username and password")
        return

    authenticator.logout("Logout", "sidebar")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<center><p style="font-size: 80px;">TakeNote</p>\n<p>AI MEETING NOTES & SENTIMENT ANALYSIS </p></center>',
            unsafe_allow_html=True,
        )
    with col2:
        st.image(image_logo, width=200)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("## Please submit your audio or video file", unsafe_allow_html=True)

    choice = st.radio("", ["By uploading a file", "By getting from video URL"])
    upload_input = None
    url_input = ""

    if choice == "By uploading a file":
        upload_input = st.file_uploader(
            "Upload a .wav or .mp3 sound file",
            key="upload",
            type=[".wav", ".mp3", ".mp4", ".m4a"],
        )
    else:
        url_input = st.text_input("Enter video URL, below is a calling example", value="", key="url")

    email_in = st.text_input("Email Address")

    if st.button("Submit"):
        with st.spinner(text="Submitting..."):
            success, message = submit_task(dbhandler, username, choice, upload_input, url_input, email_in)
        if success:
            st.success(message, icon="✅")
        else:
            st.warning(message)


if __name__ == "__main__":
    main()

from __future__ import annotations

import datetime
import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage

import yaml
from yaml.loader import SafeLoader


class Email_Sender:
    """Small service object for transactional emails."""

    def __init__(self, config_path: str = "data/email/email_config.yaml") -> None:
        with open(config_path, encoding="utf-8") as file:
            email_config = yaml.load(file, Loader=SafeLoader)

        sender_cfg = email_config["sender"]
        self.email_sender = sender_cfg["email"]
        self.email_password = sender_cfg["pass"]
        self.subject = sender_cfg["subject"]
        self.ssl = sender_cfg["ssl"]
        self.host = sender_cfg.get("host", "")

    def get_body(self, time_stamp: str, task_id: str = "") -> str:
        """Return a compact HTML notification body."""
        task_html = f"<p><strong>Task ID:</strong> {task_id}</p>" if task_id else ""
        return f"""
        <html>
          <body>
            <h3>Your transcript request has been processed.</h3>
            {task_html}
            <p><strong>Placed On:</strong> {time_stamp}</p>
            <p>Thank you for using TakeNote AI.</p>
            <p><a href=\"https://www.takenote.ai/\">Visit website</a></p>
          </body>
        </html>
        """

    def setup_email(self, msg: EmailMessage, email_receiver: str, subject: str | None = None) -> EmailMessage:
        selected_subject = subject if subject is not None else self.subject
        msg["Subject"] = selected_subject
        msg["From"] = self.email_sender
        msg["To"] = email_receiver
        return msg

    def _open_smtp(self) -> smtplib.SMTP_SSL:
        context = ssl.create_default_context()
        return smtplib.SMTP_SSL(self.ssl, 465, context=context)

    def check_email(self, rev_email: str) -> str | None:
        if "@" not in rev_email:
            return "Email is in-valid"
        return None

    def send_email_attach(
        self,
        email_receiver: str,
        task_id: str,
        path_to_file: str,
        content: str = "This is a plain text body.",
        subject: str | None = None,
    ) -> None:
        msg = EmailMessage()
        self.setup_email(msg, email_receiver, subject)
        msg.set_content(content)

        mime_type = mimetypes.guess_type(path_to_file)[0] or "application/octet-stream"
        maintype, subtype = mime_type.split("/", 1)

        with open(path_to_file, "rb") as attachment:
            msg.add_attachment(
                attachment.read(),
                maintype=maintype,
                subtype=subtype,
                filename=os.path.basename(path_to_file),
            )

        with self._open_smtp() as smtp:
            smtp.login(self.email_sender, self.email_password)
            smtp.sendmail(self.email_sender, email_receiver, msg.as_string())

    def send_notify(
        self,
        email_receiver: str,
        content: str = "This is a plain text body.",
        subject: str | None = None,
    ) -> None:
        msg = EmailMessage()
        self.setup_email(msg, email_receiver, subject)
        msg.set_content(content)

        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = self.get_body(time_stamp)
        msg.add_alternative(body, subtype="html")

        with self._open_smtp() as smtp:
            smtp.login(self.email_sender, self.email_password)
            smtp.sendmail(self.email_sender, email_receiver, msg.as_string())

    def send_email_text(
        self,
        email_receiver: str,
        notify_text: str,
        content: str = "This is a plain text body.",
        subject: str | None = None,
    ) -> None:
        msg = EmailMessage()
        self.setup_email(msg, email_receiver, subject)
        msg.set_content(content)

        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"{notify_text} {time_stamp}"
        msg.add_alternative(body, subtype="html")

        with self._open_smtp() as smtp:
            smtp.login(self.email_sender, self.email_password)
            smtp.sendmail(self.email_sender, email_receiver, msg.as_string())

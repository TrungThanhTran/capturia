import smtplib
import ssl
import os
from email.message import EmailMessage
from email.utils import make_msgid
import mimetypes

import datetime
import yaml
from yaml.loader import SafeLoader


class Email_Sender():
    def __init__(self):
        with open('data/email/email_config.yaml') as file:
            email_config = yaml.load(file, Loader=SafeLoader)
        self.email_sender = email_config['sender']['email']
        self.email_password = email_config['sender']['pass']
        self.subject = email_config['sender']['subject']
        self.ssl = email_config['sender']['ssl']
        self.host = email_config['sender']['host']

    def get_body(self, time_stamp, task_id):
        body = f"""<td valign="top"
        style="background:white;padding:0cm 0cm 0cm 0cm;border-top:transparent;border-left:transparent;border-bottom:transparent;border-right:transparent">
        <div>
            <div>
                <div>
                    <div align="center">
                        <table border="0" cellspacing="0" cellpadding="0" width="100%"
                            style="width:100.0%;border-collapse:collapse">
                            <tbody>
                                <tr>
                                    <td valign="top" style="padding:0cm 0cm 0cm 0cm">
                                        <div align="center">
                                            <table border="0" cellspacing="0" cellpadding="0" width="500"
                                                style="width:375.0pt;border-collapse:collapse">
                                                <tbody>
                                                    <tr>
                                                        <td width="500" valign="top"
                                                            style="width:375.0pt;padding:0cm 0cm 0cm 0cm">
                                                            <div align="center">
                                                                <table border="0" cellspacing="0" cellpadding="0"
                                                                    width="100%"
                                                                    style="width:100.0%;border-collapse:collapse">
                                                                    <tbody>
                                                                        <tr>
                                                                            <td valign="top"
                                                                                style="background:white;padding:3.75pt 0cm 0cm 0cm">
                                                                                <div>
                                                                                    <div>
                                                                                        <div align="center">
                                                                                            <table border="0"
                                                                                                cellspacing="0"
                                                                                                cellpadding="0" width="100%"
                                                                                                style="width:100.0%;border-collapse:collapse">
                                                                                                <tbody>
                                                                                                    <tr>
                                                                                                        <td valign="top"
                                                                                                            style="padding:0cm 0cm 0cm 0cm">
                                                                                                            <p class="MsoNormal"
                                                                                                                align="center"
                                                                                                                style="text-align:center">
                                                                                                                <a href="https://www.takenote.ai/"
                                                                                                                    target="_blank"
                                                                                                                    data-saferedirecturl="https://www.takenote.ai/"><span
                                                                                                                        style="text-decoration:none"><img
                                                                                                                            width="1000"
                                                                                                                            style="width:2in;"
                                                                                                                            id="takenote"
                                                                                                                            src="{self.host}/media/9ef6936d0816b5ae764ae9ecfafe029e2fde956f9028760baa6f18bc.png"
                                                                                                                            class="CToWUd"
                                                                                                                            data-bit="iit"></span></a><u></u><u></u>
                                                                                                            </p>
                                                                                                        </td>
                                                                                                    </tr>
                                                                                                </tbody>
                                                                                            </table>
                                                                                        </div>
                                                                                        <p class="MsoNormal"
                                                                                            style="background:white;vertical-align:top">
                                                                                            <span
                                                                                                style="display:none"><u></u>&nbsp;<u></u></span>
                                                                                        </p>
                                                                                        <table border="0" cellspacing="0"
                                                                                            cellpadding="0" width="100%"
                                                                                            style="width:100.0%;border-collapse:collapse">
                                                                                            <tbody>
                                                                                                <tr>
                                                                                                    <td valign="top"
                                                                                                        style="padding:11.25pt 7.5pt 11.25pt 7.5pt">
                                                                                                        <div>
                                                                                                            <div>
                                                                                                                <p align="center"
                                                                                                                    style="margin:0cm;text-align:center;line-height:33.75pt;word-break:break-word">
                                                                                                                    <strong><span
                                                                                                                            style="font-size:21.0pt;font-family:&quot;Open Sans&quot;,sans-serif;color:#333333">Your
                                                                                                                            Automated
                                                                                                                            Transcript
                                                                                                                            is
                                                                                                                            ready!
                                                                                                                        </span></strong><span
                                                                                                                        style="font-size:22.5pt;font-family:&quot;Open Sans&quot;,sans-serif;color:#4a4a4a"><u></u><u></u></span>
                                                                                                                </p>
                                                                                                            </div>
                                                                                                        </div>
                                                                                                    </td>
                                                                                                </tr>
                                                                                            </tbody>
                                                                                        </table>
                                                                                    </div>
                                                                                </div>
                                                                            </td>
                                                                        </tr>
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        <div align="center">
            <table border="0" cellspacing="0" cellpadding="0" width="100%" style="width:100.0%;border-collapse:collapse">
                <tbody>
                    <tr>
                        <td valign="top" style="padding:0cm 0cm 0cm 0cm">
                            <div align="center">
                                <table border="0" cellspacing="0" cellpadding="0" width="500"
                                    style="width:375.0pt;border-collapse:collapse">
                                    <tbody>
                                        <tr>
                                            <td width="250" valign="top" style="width:187.5pt;padding:0cm 0cm 0cm 0cm">
                                                <div align="center">
                                                    <table border="0" cellspacing="0" cellpadding="0" width="100%"
                                                        style="width:100.0%;border-collapse:collapse">
                                                        <tbody>
                                                            <tr>
                                                                <td valign="top" style="padding:3.75pt 0cm 3.75pt 0cm">
                                                                    <div>
                                                                        <div>
                                                                            <table border="0" cellspacing="0"
                                                                                cellpadding="0" width="100%"
                                                                                style="width:100.0%;border-collapse:collapse">
                                                                                <tbody>
                                                                                    <tr>
                                                                                        <td valign="top"
                                                                                            style="padding:7.5pt 7.5pt 7.5pt 7.5pt">
                                                                                            <div>
                                                                                                <div>
                                                                                                    <p
                                                                                                        style="margin:0cm;line-height:12.75pt">
                                                                                                        <strong><span
                                                                                                                style="font-size:10.5pt;font-family:&quot;Open Sans&quot;,sans-serif;color:#4a4a4a">Order
                                                                                                                #:</span></strong><span
                                                                                                            style="font-size:10.5pt;font-family:&quot;Open Sans&quot;,sans-serif;color:#4a4a4a">
                                                                                                            <a href="{self.host}/MyFiles?task={task_id}"
                                                                                                                target="_blank"
                                                                                                                data-saferedirecturl="{self.host}/MyFiles?task={task_id}"><span
                                                                                                                    style="color:#4a90e2;text-decoration:none">{task_id}</span></a><u></u><u></u></span>
                                                                                                    </p>
                                                                                                </div>
                                                                                            </div>
                                                                                        </td>
                                                                                    </tr>
                                                                                </tbody>
                                                                            </table>
                                                                        </div>
                                                                    </div>
                                                                </td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </td>
                                            <td width="250" valign="top"
                                                style="width:187.5pt;padding:0cm 0cm 0cm 0cm;border-top:transparent;border-left:transparent;border-bottom:transparent;border-right:transparent">
                                                <div align="center">
                                                    <table border="0" cellspacing="0" cellpadding="0" width="100%"
                                                        style="width:100.0%;border-collapse:collapse">
                                                        <tbody>
                                                            <tr>
                                                                <td valign="top"
                                                                    style="padding:3.75pt 0cm 3.75pt 0cm;word-break:break-word">
                                                                    <div>
                                                                        <div>
                                                                            <table border="0" cellspacing="0"
                                                                                cellpadding="0" width="100%"
                                                                                style="width:100.0%;border-collapse:collapse">
                                                                                <tbody>
                                                                                    <tr>
                                                                                        <td valign="top"
                                                                                            style="padding:7.5pt 7.5pt 7.5pt 7.5pt">
                                                                                            <div>
                                                                                                <div>
                                                                                                    <p align="right"
                                                                                                        style="margin:0cm;text-align:right;line-height:12.0pt">
                                                                                                        <strong><span
                                                                                                                style="font-size:10.5pt;font-family:&quot;Open Sans&quot;,sans-serif;color:#4a4a4a">&nbsp;
                                                                                                                Placed
                                                                                                                On:</span></strong><span
                                                                                                            style="font-size:10.5pt;font-family:&quot;Open Sans&quot;,sans-serif;color:#4a4a4a"><u></u><u></u></span>
                                                                                                    </p>
                                                                                                    <p align="right"
                                                                                                        style="margin:0cm;text-align:right;line-height:12.0pt;word-break:break-word">
                                                                                                        <span
                                                                                                            style="font-size:10.5pt;font-family:&quot;Open Sans&quot;,sans-serif;color:#4a4a4a">{time_stamp}"""
        return body

    def setup_email(self, msg, email_receiver, task_id, subject=None):
        if subject != None:
            _subject = _subject
        else:
            _subject = self.subject

        msg['Subject'] = _subject
        msg['From'] = self.email_sender
        msg['To'] = email_receiver
        return msg

    def check_email(self, rev_email):
        if "@" not in rev_email:
            return "Email is in-valid"

    def send_email_attach(self,
                          email_receiver,
                          task_id,
                          path_to_file,
                          content='This is a plain text body.',
                          subject=None):
        
        msg = EmailMessage()
        msg = self.setup_email(msg, email_receiver, task_id, subject)
        mime_type, _ = mimetypes.guess_type(path_to_file)
        
        mime_type, mime_subtype = mime_type.split('/', 1)
        # msg.set_content(content)
        with open(path_to_file, 'r') as ap:
            msg.add_attachment(ap.read(), maintype=mime_type, subtype=mime_subtype,
                            filename=os.path.basename(path_to_file))
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.ssl, 465, context=context) as smtp:
            smtp.login(self.email_sender, self.email_password)
            smtp.sendmail(self.email_sender, email_receiver, msg)

    def send_email_text(self,
                        email_receiver,
                        task_id,
                        content='This is a plain text body.',
                        subject=None):
        msg = EmailMessage()
        msg = self.setup_email(msg, email_receiver, task_id, subject)
        msg.set_content(content)

        x = datetime.datetime.now()
        time_stamp = x.strftime("%Y-%m-%d %H:%M:%S")
        body = self.get_body(time_stamp, task_id)

        msg.add_alternative(body, subtype='html')
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.ssl, 465, context=context) as smtp:
            smtp.login(self.email_sender, self.email_password)
            smtp.sendmail(self.email_sender, email_receiver, msg.as_string())

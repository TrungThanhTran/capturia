from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.environ.get("AWS_REGION", "eu-west-2")


class S3_Handler:
    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self.s3_connector = boto3.resource(
            service_name="s3",
            region_name=AWS_REGION,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        self.bucket = self.s3_connector.Bucket(bucket_name)
        self.s3_client = boto3.client(
            service_name="s3",
            region_name=AWS_REGION,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

    def list_username_in_bucket(self) -> list[str]:
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Delimiter="/")
        folder_names: list[str] = []
        for folder in response.get("CommonPrefixes", []):
            folder_names.append(folder["Prefix"].rstrip("/"))
        return folder_names

    def download_file_from_s3(self, object_key: str, local_file_path: str) -> bool:
        try:
            self.bucket.download_file(object_key, local_file_path)
            return True
        except Exception as exc:
            print(f"[ERROR] downloading file '{object_key}' from S3 bucket: {exc}")
            return False

    def upload_file_to_s3(self, file_path: str, destination_key: str) -> bool:
        try:
            self.bucket.upload_file(file_path, destination_key)
            print(
                f"File '{file_path}' uploaded to bucket '{self.bucket_name}' with key '{destination_key}'"
            )
            return True
        except Exception as exc:
            print(f"[ERROR] uploading file '{file_path}' to S3 bucket '{self.bucket_name}': {exc}")
            return False

    def create_s3_folder(self, folder_name: str) -> dict[str, Any]:
        normalized = folder_name if folder_name.endswith("/") else f"{folder_name}/"
        return self.bucket.put_object(Key=normalized)


class SQS_Handler:
    def __init__(self) -> None:
        self.sqs_client = boto3.client(
            service_name="sqs",
            region_name=AWS_REGION,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

    def create_queue(self, queue_name: str, delay_seconds: str, visibility_timeout: str):
        try:
            return self.sqs_client.create_queue(
                QueueName=queue_name,
                Attributes={
                    "DelaySeconds": delay_seconds,
                    "VisibilityTimeout": visibility_timeout,
                },
            )
        except ClientError:
            print(f"Could not create SQS queue - {queue_name}.")
            raise

    def list_queues(self) -> list[str]:
        try:
            response = self.sqs_client.list_queues()
            return response.get("QueueUrls", [])
        except ClientError:
            print("[ERROR]: Could not list queues.")
            raise

    def get_message(self, queue_url: str):
        response = self.sqs_client.receive_message(
            QueueUrl=queue_url,
            AttributeNames=["SentTimestamp"],
            MaxNumberOfMessages=1,
            MessageAttributeNames=["All"],
            VisibilityTimeout=0,
            WaitTimeSeconds=0,
        )

        messages = response.get("Messages")
        if not messages:
            time.sleep(5)
            return None, None, None, None, None, None

        message = messages[0]
        self.sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])

        message_task = json.loads(message["Body"])
        print(f"Received and deleted message: {message_task}")
        return (
            message_task.get("task_id"),
            message_task.get("file_path"),
            message_task.get("user"),
            message_task.get("email"),
            message_task.get("time"),
            message_task.get("status"),
        )

    def send_message(self, queue_url: str, message_body: str) -> bool:
        try:
            response = self.sqs_client.send_message(QueueUrl=queue_url, MessageBody=message_body)
            print(f"Message sent successfully with MessageId: {response['MessageId']}")
            return True
        except Exception as exc:
            print(f"[ERROR]: {exc}")
            return False

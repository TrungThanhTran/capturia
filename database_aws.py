import os
import boto3
import json
import time
from botocore.exceptions import ClientError

class S3_Handler():
    def __init__(self, bucket_name) -> None:
        self.s3_connector = boto3.resource(
            service_name='s3',
            region_name='eu-west-2',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
        self.bucket_name = bucket_name
        self.bucket = self.s3_connector.Bucket(bucket_name)

        self.s3_client = boto3.client(
            service_name='s3',
            region_name='eu-west-2',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )

    def list_username_in_bucket(self):
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Delimiter='/')
        folder_namese = []
        if 'CommonPrefixes' in response:
            for folder in response['CommonPrefixes']:
                folder_name = folder['Prefix'].rstrip('/')
                folder_namese.append(folder_name)
        return folder_namese

    def download_file_from_s3(self, object_key, local_file_path):
        '''
        # Usage example
        local_file_path = '/path/to/local/file.txt'
        s3_bucket_name = 'your-s3-bucket'
        s3_key = 'desired/s3/directory/'  # Include the desired directory structure here
        '''
        # Upload the file to S3 preserving the directory structure
        download_flag = False
        try:
            self.bucket.download_file(object_key, local_file_path)
            download_flag = True
        except Exception as e:
            print(
                f"[ERROR]____ downloading file '{object_key}' from S3 bucket: {e}")
            download_flag = False
        return download_flag

    def upload_file_to_s3(self,
                          file_path,
                          destination_directory):
        # Use the 'destination_directory' to construct the object key (file path within the bucket)
        object_key = destination_directory.strip(
            "/") + "/" + file_path.split("/")[-1]
        upload_flag = False
        try:
            self.bucket.upload_file(file_path, object_key)
            print(
                f"File '{file_path}' uploaded to S3 bucket '{self.bucket_name}' in directory '{destination_directory}'")
            upload_flag = True
        except Exception as e:
            print(
                f"[ERROR]____:Error uploading file '{file_path}' to S3 bucket '{self.bucket_name}': {e}")
            upload_flag = False
        return upload_flag

    def create_s3_folder(self,
                         folder_name):
        # If the folder name doesn't end with '/', add it for consistency
        if not folder_name.endswith('/'):
            folder_name += '/'

        # Create an empty object (0 bytes) with the specified key (folder name)
        response = self.bucket.put_object(Key=folder_name)
        return response


class SQS_Handler():
    def __init__(self) -> None:
        self.sqs_client = boto3.client(
            service_name='sqs',
            region_name='eu-west-2',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )

    def create_queue(self, queue_name, delay_seconds, visiblity_timeout):
        """
        Create a standard SQS queue
        """
        try:
            response = self.sqs_client.create_queue(QueueName=queue_name,
                                                    Attributes={
                                                        'DelaySeconds': delay_seconds,
                                                        'VisibilityTimeout': visiblity_timeout
                                                    })
        except ClientError:
            print(f'Could not create SQS queue - {queue_name}.')
            raise
        else:
            return response

    def list_queues(self):
        """
        Creates an iterable of all Queue resources in the collection.
        """
        try:
            sqs_queues = []
            for queue in self.sqs_client.queues.all():
                sqs_queues.append(queue)
        except ClientError:
            print('[ERROR]____: Could not list queues.')
            raise
        else:
            return sqs_queues

    def get_message(self, queue_url):
        # Create an SQS client
        # Receive message from SQS queue
        response = self.sqs_client.receive_message(
            QueueUrl=queue_url,
            AttributeNames=[
                'SentTimestamp'
            ],
            MaxNumberOfMessages=1,
            MessageAttributeNames=[
                'All'
            ],
            VisibilityTimeout=0,
            WaitTimeSeconds=0
        )
        if  'Messages' not in response:
            time.sleep(5)  # Add some delay before checking again
            return None, None, None, None, None, None
        else:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']

            # Delete received message from queue
            self.sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            print('Received and deleted message: %s' % json.loads(message['Body']))
            message_task = json.loads(message['Body'])
            return message_task['task_id'], message_task['file_path'], message_task['user'], message_task['email'], message_task['time'], message_task['status']

    def send_message(self, queue_url, message_body):
        # Create an SQS client
        try:
            # Send the message to the SQS queue
            response = self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=message_body
            )

            print(
                f"Message sent successfully with MessageId: {response['MessageId']}")
        except Exception as e:
            print(f"[ERROR]____: {str(e)}")
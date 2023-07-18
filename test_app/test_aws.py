import os
import boto3
import json
from botocore.exceptions import ClientError

class S3_Handler():
    def __init__(self, bucket_name) -> None:
        self.s3_connector = boto3.resource(
                service_name='s3',
                region_name='eu-west-2',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
            )
        
        self.s3_client = boto3.client(
                service_name='s3',
                region_name='eu-west-2',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
            )
        
        self.bucket_name = bucket_name
        self.bucket = self.s3_connector.Bucket(bucket_name)
        
    def list_folders_in_bucket(self):
        
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Delimiter='/')
        folder_namese = []
        if 'CommonPrefixes' in response:
            for folder in response['CommonPrefixes']:
                folder_name = folder['Prefix'].rstrip('/')
                folder_namese.append(folder_name)
        return folder_namese
        
    def download_file(self, local_file_path, object_key):
        '''
        # Usage example
        local_file_path = '/path/to/local/file.txt'
        s3_bucket_name = 'your-s3-bucket'
        s3_key = 'desired/s3/directory/'  # Include the desired directory structure here
        '''
        # Upload the file to S3 preserving the directory structure
        self.bucket.download_file(object_key, local_file_path)

        
    def upload_file_to_s3(self, 
                          file_path, 
                          destination_directory):
        # Use the 'destination_directory' to construct the object key (file path within the bucket)
        object_key = destination_directory.strip("/") + "/" + file_path.split("/")[-1]

        try:
            self.bucket.upload_file(file_path, object_key)
            print(f"File '{file_path}' uploaded to S3 bucket '{self.bucket_name}' in directory '{destination_directory}'")
        except Exception as e:
            print(f"Error uploading file '{file_path}' to S3 bucket '{self.bucket_name}': {e}")

    def create_s3_folder(self,
                         folder_name):        
        # If the folder name doesn't end with '/', add it for consistency
        if not folder_name.endswith('/'):
            folder_name += '/'

        # Create an empty object (0 bytes) with the specified key (folder name)
        response = self.bucket.put_object(Key=folder_name)
              
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
            print('Could not list queues.')
            raise
        else:
            return sqs_queues
        
    def get_message(self, queue_url):
        # Create an SQS client
        # Receive message from SQS queue
        response =  self.sqs_client.receive_message(
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
        
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']

        # Delete received message from queue
        self.sqs_client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
        print('Received and deleted message: %s' % json.loads(message['Body']))
            
    def send_message(self, queue_url, message_body):
        # Create an SQS client
        try:
            # Send the message to the SQS queue
            response = self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=message_body
            )

            print(f"Message sent successfully with MessageId: {response['MessageId']}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    bucket_name = 'takenoteai'
    folder_name = 'user_name_y'
    sub_name = 'user_x'
    # s3_handler = S3_Handler(bucket_name)
    # local_path = '/home/ubuntu/storage/capturia/temp/test/y2mate.is - Facebook META Q3 2022 Earnings Call-iWobmXvCM0c-192k-1687330838.mp3'
    # for obj in s3_handler.bucket.objects.all():
    #     print(obj.key)
        
    # print('list of folder in bucket = ', s3_handler.list_folders_in_bucket())
        
    # s3_handler.create_s3_folder(folder_name)
    
    # local_file_path = '/home/ubuntu/storage/capturia/temp/trungtt/1f4cb8c8-51ef-4877-8609-b764b57c89ee/Busdriver - Imaginary Places.mp3'
    # s3_handler.upload_file_to_s3(local_file_path, f"{folder_name}/{sub_name}")
    # s3_handler.download_file(local_path, f'user_namex/user_x/Busdriver - Imaginary Places.mp3')
    sqs_url = 'Takenote_TaskQueue'
    
    
    #### SQS $%%
    sqs_handler = SQS_Handler()
    #                 # TASKID           TEXT      NOT NULL,
    #                 # FILE_PATH        TEXT     NOT NULL,
    #                 # USER             TEXT    NOT NULL,
    #                 # EMAIL            TEXT    NOT NULL,
    #                 # TIME             TEXT,
    #                 # STATUS           INT);
    json_data = {
        "task_id": "b1aa1942-432f-43cc-bdc5-6136a2307462",
        "file_path": "s3://takenoteai/trungtt/b1aa1942-432f-43cc-bdc5-6136a2307462/Busdriver - Imaginary Places.mp3",
        "user": "trungtt",
        "email": "trung.tranthanh@pixta.co.jp",
        "time": "Jul-18-2023-15-01-32",
        "status": "0"
    }   
        
    message_body = json.dumps(json_data)
    sqs_handler.send_message(sqs_url, message_body)
    # # sqs_handler.list_queues()
    
    # # sqs_handler.get_message(sqs_url)
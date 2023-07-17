import os
import boto3

class AWS_Handler():
    def __init__(self, bucket_name) -> None:
        self.s3_connector = boto3.resource(
                service_name='s3',
                region_name='eu-west-2',
                aws_access_key_id='AKIAZAOFBE65XYMS3YEO',
                aws_secret_access_key='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'
            )
        self.bucket_name = bucket_name
        self.bucket = self.s3_connector.Bucket(bucket_name)

        
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


if __name__ == "__main__":
    bucket_name = 'takenoteai'
    folder_name = 'user_namex'
    sub_name = 'user_x'
    aws_handler = AWS_Handler(bucket_name)
    local_path = '/home/ubuntu/storage/capturia/temp/test/y2mate.is - Facebook META Q3 2022 Earnings Call-iWobmXvCM0c-192k-1687330838.mp3'
    for obj in aws_handler.bucket.objects.all():
        print(obj.key)
        
    # aws_handler.create_s3_folder(folder_name)
    
    local_file_path = '/home/ubuntu/storage/capturia/temp/trungtt/1f4cb8c8-51ef-4877-8609-b764b57c89ee/Busdriver - Imaginary Places.mp3'
    aws_handler.upload_file_to_s3(local_file_path, f"{folder_name}/{sub_name}")
    aws_handler.download_file(local_path, f'user_namex/user_x/Busdriver - Imaginary Places.mp3')
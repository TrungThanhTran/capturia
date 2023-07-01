import time
from glob import glob
from moviepy.editor import *
from pytube import YouTube

def MP4ToMP3(path_to_mp4, path_to_mp3):
    FILETOCONVERT = AudioFileClip(path_to_mp4)
    FILETOCONVERT.write_audiofile(path_to_mp3)
    FILETOCONVERT.close()

url = 'https://www.youtube.com/live/iWobmXvCM0c?feature=share'
start = time.time()
yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
yt.streams.filter(progressive=True, file_extension='mp4') \
    .order_by('resolution') \
    .desc() \
    .first() \
    .download(f"/home/ubuntu/project/offline_Meeting_Analysis/test_file/")

print(f"download time = {time.time() - start}")
for file in glob('/home/ubuntu/project/offline_Meeting_Analysis/test_file/*.mp4'):
    path_to_mp4 = file
    break

path_to_mp3 = "/home/ubuntu/project/offline_Meeting_Analysis/test_file/"

start = time.time()
MP4ToMP3(path_to_mp4, path_to_mp3)
print(f"convert time = {time.time() - start}")

import cv2
import os
import time
from datetime import datetime
import pandas as pd

"""
ffprobe -v quiet timestamp_1.avi -print_format json -show_entries stream=index,codec_type:stream_tags=creation_time:format_tags=creation_time
ffprobe -v quiet timestamp_1.avi -print_format json -show_entries stream_tags:format_tags
"""

path_to_movie = "Z:\\DNF\\GROUPS\\RESEARCH\\NEUROBAU\\PUBLIC\\LN\\Maria\\09 Timestamp Bonsai\\timestamp_1.avi"

file_creation_timestamp = os.path.getctime(path_to_movie)
creation_date = pd.Timestamp(file_creation_timestamp, unit='s', tz='Europe/Berlin')

# get file duration
cap = cv2.VideoCapture(path_to_movie)
cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
file_duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
cap.release()

# file_duration = 125

start_timestamp = file_creation_timestamp - file_duration
file_start_date = pd.Timestamp(start_timestamp, unit='s', tz='Europe/Berlin')
print('start date: {}'.format(file_start_date))
print('file end: {}'.format(creation_date))
print('file duration: {}'.format(file_duration))

a = pd.Timestamp('2021-06-28T17:58:55.7284864+02:00', tz='Europe/Berlin')

# pd.Timestamp(start_timestamp, unit = 's', tz='Europe/Berlin')

# generate time step from framerate and creation date
file_start_date
pd.date_range(start=file_start_date, end=creation_date, freq='ms')
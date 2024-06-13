import os
import numpy as np
import pandas as pd
import cv2


"""
2022-09-27

> INPUT: a raw movie (.avi) and the post-processed output from the networks (.csv)
> OUTPUT: a PIP (Picture In Picture) video with visual label for grooming

CSV file:
- used to find the head's coordinates and to label the grooming episodes.
- is processed from the raw output of networks 1 and 2
- contains at least:
    - X, Y coordinates
    - Automated grooming label
"""

##################################################################################################
ROOT_DIR = r"Z:\DNF\GROUPS\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\15 Build_composite_movie"
filename = "25_degrees_6mW_9_3"
##################################################################################################

movie_file = os.path.join(ROOT_DIR, filename + ".avi")
data_file = os.path.join(ROOT_DIR, filename + ".csv")
out_movie_file_name = os.path.join(ROOT_DIR, filename + "_PIP.avi")

data_in = pd.read_csv(data_file, sep=',')

focus_size = 64 # size of head window

cap = cv2.VideoCapture(movie_file) # capturing the video from the given path
frameRate = cap.get(cv2.CAP_PROP_FPS) # int(np.ceil(frameRate))
number_of_frames_in_movie = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # how many frames in movie

# actual coordinates cannot be used since network 1 has been trained on a 256 x 210 frame set
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame width
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame height

data_in.X = data_in.X * w / 320
data_in.Y = data_in.Y * h / 256

codec = cv2.VideoWriter_fourcc(*"MJPG")
out_movie = cv2.VideoWriter(out_movie_file_name, codec , frameRate , (1280, 1024))

while cap.isOpened():

    for i in range(number_of_frames_in_movie):

        x = int(data_in.iloc[i].X)
        y = int(data_in.iloc[i].Y)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, tmp_frame = cap.read()

        # crop frame to focus
        crop_frame = tmp_frame[x - focus_size : x + focus_size, y - focus_size : y + focus_size] / 255
        scale_percent = 300 # percent of original size
        width = int(crop_frame.shape[1] * scale_percent / 100)
        height = int(crop_frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        crop_frame = cv2.resize(crop_frame, dim, interpolation = cv2.INTER_AREA) 
        
        print('frame {} '.format(i))

        #------------------
        g = np.uint8(crop_frame / np.max(crop_frame) * 255)
        tmp_frame[:384,:384,:] = g


        #------------------
        if data_in.iloc[i]["auto_grooming_0.5"] == 1:
            image = cv2.circle(tmp_frame, (30, 30), 25, (255, 255, 255), -1) # Circle for grooming
            image = cv2.rectangle(image,  (y-32,x-32), (y+32,x+32), (255, 255, 255), 1) # Rectangle for head
            out_movie.write(image)
        else:
            image = cv2.rectangle(tmp_frame,  (y-32,x-32), (y+32,x+32), (255, 255, 255), 1)
            out_movie.write(image)

        ###################### DEBUG <<<
        # if i == 120:
        #     cap.release()
        #     break
        ######################

    cap.release()

out_movie.release()

print("DONE")
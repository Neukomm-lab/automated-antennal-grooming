from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torch

#--- used by custom dataset loader
from datetime import datetime, date
import pandas as pd
import time
import sys
import os

import cv2
import glob
import json

'''
INPUT: movie files in `source_videos` folder + corresponding (i.e. same filename) coordinates files from '"..\\..\\results\\head_coordinates\\"' 
OUTPUT: Grooming timeline (.csv file)

# 2023-02-14
- The parts that save the video output have been commented out (to speed up the execution time)
- The lines that save the metadata have been commented out (to avoid collision during running of the analysis script)
'''

################################################################
flag = str(sys.argv[1])

if flag.lower() == "before":
    METADATA_FILE_PATH = '../lib/metadata/metadata_fly_grooming_inference_network_2_BEFORE.json'


elif flag.lower() == "after":
    METADATA_FILE_PATH = '../lib/metadata/metadata_fly_grooming_inference_network_2_AFTER.json'
else:
    print(">>> ERROR: missing required argument (BEFORE | AFTER)")
    sys.exit()
################################################################

print("\nProcessing files with checkpoint: {}".format(flag.upper()))

with open(METADATA_FILE_PATH) as f:
  metadata = json.load(f)
  print(metadata)

#\\\\\\\\\\\\\\\\\\\\\\ SEED \\\\\\\\\\\\\\\\\\\\\
torch.manual_seed(0)
np.random.seed(0)
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# use GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load ResNet
model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# resize the output layer
# Here the size of each output sample is set to 2.
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

# load model to GPU, if available
model_ft = model_ft.to(device)

#################################################
checkpoint_name =  metadata['CHECKPOINT']  # load the weights of the trained network
#################################################

checkpoint = torch.load('../lib/checkpoints/' + checkpoint_name + '.pt')
model_ft.load_state_dict(checkpoint['model_state_dict'])
model_ft.eval()
focus_size = 64
frames_to_add = 3 # how many frames to use for the standard projection
font = cv2.FONT_HERSHEY_SIMPLEX # Font to be used for writing the label on frames (output movie)

execution_time = []

for movie_file in glob.iglob(os.path.join(metadata['MOVIE_DIR'], '*.' + metadata['MOVIE_FILE_EXTENSION'])):
    
    since = time.time() # star timer

    # >>> 2021-06-30: used to get the time stamp (to be used for aligning LED onset)
    file_creation_timestamp = os.path.getctime(movie_file) 
    creation_date = pd.Timestamp(file_creation_timestamp, unit='s', tz='Europe/Berlin')
    
    coordinates = pd.read_csv(os.path.join(metadata['COORDINATES_DIR'], os.path.splitext(os.path.basename(movie_file))[0] + '.' + metadata['COORD_FILE_EXTENSION']), sep=',')

    cap = cv2.VideoCapture(movie_file) # capturing the video from the given path
    frameRate = cap.get(cv2.CAP_PROP_FPS) # int(np.ceil(frameRate))
    number_of_frames_in_movie = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # how many frames in movie
    total_frames = list(range(number_of_frames_in_movie))

    print('Total number of frames: {}'.format(len(total_frames)))
    print('\nProcessing ...')
    
    frames_sequences = [total_frames[i:i + frames_to_add] for i in range(number_of_frames_in_movie)] # index of frames grouped in groups with length equal the numbe rof frames to be added

    # actual coordinates cannot be used since network 1 has been trained on a 256 x 210 frame set
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame height

    coordinates.X = coordinates.X * w / metadata['RESIZE_FRAME'][0] # 320
    coordinates.Y = coordinates.Y * h / metadata['RESIZE_FRAME'][1] # 256

    # pred_labels = torch.zeros(number_of_frames_in_movie)
    output_soft_max = []

    # UNCOMMENT TO SAVE VIDEO OUTPUT
    # codec = cv2.VideoWriter_fourcc(*"MJPG")
    # out_movie = cv2.VideoWriter(os.path.join(metadata['OUTPUT_MOVIE_DIR'], os.path.splitext(os.path.basename(movie_file))[0] + '.avi'), codec , frameRate , (w, h))

    with torch.no_grad():
        #  movie streaming buffer is on
        while cap.isOpened():

            # generate single batch for prediction
            empty_batch = np.zeros([128, 128, 3, 1])

            # `sequence` holds the N of `frames_to_add` (i.e. Z-Projection)
            for i, sequence in enumerate(frames_sequences): # from now on every operation is at frame-level

                # print('{} | sequence: {}'.format(i, sequence[0]))
                x = int(coordinates.iloc[sequence[0]].X)
                y = int(coordinates.iloc[sequence[0]].Y)
                
                # generate sequence of frames to add up (i.e. Z-projection)
                empty_array_of_frames = np.zeros([h, w, 3, frames_to_add]) 

                for frame_counter, frame_number in enumerate(sequence):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, tmp_frame = cap.read()
                    empty_array_of_frames[:, :, :, frame_counter] = tmp_frame 

                crop_frame = empty_array_of_frames[x - focus_size : x + focus_size, y - focus_size : y + focus_size] / 255
                crop_frame = np.std(crop_frame, 3)
                
                # this can be improved by arranging dimensions without the need to transpose
                empty_batch[:crop_frame.shape[0], :crop_frame.shape[1], :, 0] = crop_frame # w h c b
                batch_for_prediction = torch.FloatTensor(empty_batch.transpose((3, 2, 1, 0))) # needs b c w h, then converts to tensor
                prediction = model_ft(batch_for_prediction.to(device))

                # UNCOMMENT TO HAVE PRINT
                # _, preds = torch.max(prediction, 1)

                soft_max_pred = torch.nn.functional.softmax(prediction, dim=1)[0]

                # UNCOMMENT TO PRINT FRAME-WISE FEEDBACK
                # print('frame {} | predict class: {} | classes (softmax): {}'.format(i, preds.item(), soft_max_pred ))

                output_soft_max.append(soft_max_pred)

                # UNCOMMENT TO SAVE VIDEO OUTPUT
                # write probability of grooming (0) and probability of no_grooming (1) on each frame
                # cv2.putText(tmp_frame, str(soft_max_pred), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                # out_movie.write(tmp_frame * 255)

                ## Draw red spot on frames if prediction is GROOMING (0)
                # if preds < 1:
                #     image = cv2.circle(tmp_frame * 255, (50, 50), 14, (0, 0, 255), -1)
                #     out_movie.write(image)
                # else:
                #     cv2.putText(tmp_frame, str(soft_max_pred), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                #     out_movie.write(tmp_frame * 255)


                # # ///////////////////////////// DEBUGGING ////////////////////////////////////
                # if frame_number == 100: # use for debugging (i.e. limit inference to N frames)
                #     break
                # # ////////////////////////////////////////////////////////////////////////////


            # ////////////////////////////////////////////////////////////////////////////
            # set to last frames and get frame count
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            file_duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            cap.release()

    # UNCOMMENT TO SAVE VIDEO OUTPUT
    # out_movie.release()


    # >>> 2021-06-30: extract info about movie's timestamp
    start_timestamp = file_creation_timestamp - file_duration
    file_start_date = pd.Timestamp(start_timestamp, unit='s', tz='Europe/Berlin')
    print('start date: {}'.format(file_start_date))
    print('file end: {}'.format(creation_date))
    print('file duration: {}'.format(file_duration))

    grooming_behavior_timeline = pd.DataFrame()
    softmax_probabilities = [row.detach().cpu().numpy()[0] for row in output_soft_max]
    grooming_behavior_timeline['timestamp'] = np.arange(0, frame_number, 1 / frameRate)[:frame_number + 1]
    grooming_behavior_timeline['date_timestamp'] = pd.date_range(start=file_start_date, end=creation_date, periods=number_of_frames_in_movie)[:frame_number + 1]
    
    ## >>> the following two lines can be used when debugging on a short frame sequence (i.e. trimmed by lines 148-149)
    # grooming_behavior_timeline['timestamp'] = np.arange(0, frame_number, 1 / frameRate)[:frame_number-1]
    # grooming_behavior_timeline['date_timestamp'] = pd.date_range(start=file_start_date, end=creation_date, periods=number_of_frames_in_movie)[:frame_number-1]
    
    grooming_behavior_timeline['P(grooming)'] = softmax_probabilities
    grooming_behavior_timeline['checkpoint'] = flag
    grooming_behavior_timeline.to_csv(os.path.join(metadata['TARGET_DIR'], os.path.splitext(os.path.basename(movie_file))[0] + '.' + metadata['COORD_FILE_EXTENSION']))

    time_elapsed = time.time() - since # stop timer
    print('Done in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    now = datetime.now()
    today = date.today()
    execution_time.append([today.strftime("%B %d, %Y"), now.strftime("%H:%M:%S"), os.path.basename(movie_file) , time_elapsed])

# UNCOMMENT TO SAVE METADATA LOG
with open(os.path.join(metadata['TARGET_DIR'], 'metadata.' + metadata['COORD_FILE_EXTENSION']),'a') as metadata_output_file:
    metadata_output_file.writelines(str(row) + '\n' for row in execution_time)

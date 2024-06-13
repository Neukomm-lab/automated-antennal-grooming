import torch
import sys
sys.path.insert(0, "../lib/")
import models

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import importlib
import json
import time
import glob
import cv2
import os


#####################################################################################
METADATA_FILE_PATH = '../lib/metadata/metadata_fly_grooming_inference_network_1.json'
#####################################################################################

with open(METADATA_FILE_PATH) as f:
  metadata = json.load(f)
  print(metadata)

# Seed
torch.manual_seed(0)
np.random.seed(0)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn = torch.backends.cudnn.version()
print('Pytorch version: {}'.format(torch.__version__))
print('Using: {}'.format(device))
print('Using cuDNN: {}'.format(cudnn))

# Instantiate the model
MyClass = getattr(importlib.import_module("models"), metadata['MODEL'])
model = MyClass()
model = model.to(device)

# Use CPU on Mac
if os.name == 'posix':
    checkpoint = torch.load('../lib/checkpoints/' + metadata['CHECKPOINT_NAME'] + '.pt', map_location='cpu')
else:
    checkpoint = torch.load('../lib/checkpoints/' + metadata['CHECKPOINT_NAME'] + '.pt')

model.load_state_dict(checkpoint['model_state_dict']) # Load model
codec = cv2.VideoWriter_fourcc(*"MJPG") # Codec for reading video files
counter = 0 # keep track of frames (to display)

for movie_file in glob.iglob(os.path.join(metadata['ROOT_DIR'], '*.' + metadata['FILE_EXTENSION'])):
    
    file_name = os.path.basename(movie_file).split('.')[0]

    print('\n>>> processing: {}'.format(movie_file))

    since = time.time() # star timer

    cap = cv2.VideoCapture(movie_file) # capturing the video from the given path
    frameRate = cap.get(cv2.CAP_PROP_FPS) # int(np.ceil(frameRate))
    
    # >>> UNCOMMENT TO SAVE VIDEO OUTPUT <<<
    # out_movie = cv2.VideoWriter('../../results/videos/head_tracking/' + file_name + '_head.avi', codec , frameRate , (metadata['RESIZE_FRAME'][0],metadata['RESIZE_FRAME'][1]))
    
    counter +=1

    out = [] # network output
    path = [] # coordinates of the path
    size = [] # number of pixels above THRESHOLD (to be improved!)
    speed = None # speed (frame by frame)
    light_frames = [] # frame crop containing the visible LED

    with torch.no_grad():

        model.eval()
        counter = 0

        while cap.isOpened():
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()

            if (ret != True):
                break

            # convert to grayscale (1 channel)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
            gray = gray[:metadata['CROP_Y'],:metadata['CROP_X']]
            gray = cv2.resize(gray, (metadata['RESIZE_FRAME'][0], metadata['RESIZE_FRAME'][1])) 

            # add extra empty dimension
            frame = gray[..., np.newaxis, np.newaxis]

            # rearrange channels and size
            frame = frame.transpose((2, 3, 0, 1))

            # convert to tensors
            frame = torch.FloatTensor(frame)

            prediction = model(frame.to(device))

            if counter % 300 == 0:
                print('frame # : {}'.format(frameId))

            tmp={}

            tmp['input_frame'] = gray
            tmp['score_map'] = prediction[0][0].detach().cpu().numpy()
            tmp['frame_number'] = frameId
            tmp['frame_rate'] = frameRate
            tmp['coordinates'] = np.array(np.unravel_index(tmp['score_map'].argmax(), tmp['score_map'].shape))
            px_above_threshold = np.where(tmp['score_map'] >= metadata['THRESHOLD'])

            tmp['size'] = px_above_threshold[0].shape[0]

            # >>> UNCOMMENT TO SAVE VIDEO OUTPUT <<< !
            # Hack to get the coordinates right for OpenCV rectangle
            # correct_arrangement = tmp['coordinates'][::-1]
            # c_x = correct_arrangement[0] - 8
            # c_y = correct_arrangement[1] - 8
            # c_x_y = (c_x,c_y)
            # c_x_e = correct_arrangement[0] + 8
            # c_y_e = correct_arrangement[1] + 8
            # c_x_y_e = (c_x_e,c_y_e)

            # >>> UNCOMMENT TO SAVE VIDEO OUTPUT <<< !
            # write movie # coordinates must be flipped
            # image = cv2.rectangle((gray*255).astype(np.uint8), c_x_y, c_x_y_e, (255, 255, 255), 1)
            
            # >>> UNCOMMENT TO SAVE VIDEO OUTPUT <<< !
            # im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            # out_movie.write(im)

            light_frames.append(tmp['input_frame'][200:250, 150:250])
            path.append(tmp['coordinates'])
            size.append(tmp['size'])
            out.append(tmp)
            counter += 1

            ########################### DEBUG ############################
            # if frameId ==30: # used to debug
            #     break
            ##############################################################

        cap.release() # stop video buffer

        # >>> UNCOMMENT TO SAVE VIDEO OUTPUT  <<< !
        # out_movie.release()
        
        time_elapsed = time.time() - since # stop timer
        
        path = np.array(path)
        
        speed = np.sqrt(np.sum(np.square(np.diff(path, axis=0)), axis=1)) # speed
        speed = np.append(0, speed) # first frame has null speed (or np.nan?)

        print('Done in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #/////////////////////// HACK to detect LED onset ////////////////////////////
    print('>>> computing standard deviation ...')
    avg = np.mean(np.array(light_frames[:200]), axis=0) # computes average of first 200 frames
    std_dev = np.array([np.std(i) for i in light_frames - avg]) # standard deviation on raw , subtracting initial average
    z = (std_dev - np.mean(std_dev)) / np.std(std_dev) # convert to z-scores
    onset = np.where(z >= 2)[0] # 2 standard deviations above mean
    light_ix = np.where(np.diff(onset)>100)[0]
    light = np.zeros(len(light_frames))
    light_1 = np.arange(onset[0], onset[light_ix[0]] - 1, 1)
    light_2 = np.arange(onset[light_ix[0] + 1], onset[light_ix[1] - 1], 1)
    light_3 = np.arange(onset[light_ix[1] + 1], onset[-1], 1)
    np.add.at(light, light_1, 1)
    np.add.at(light, light_2, 1)
    np.add.at(light, light_3, 1)
    # ////////////////////////////////////////////////////////////////////////////

    # Save head coordinates to csv file
    data = pd.DataFrame({'X':path[:, 0], 'Y':path[:, 1], 'size':size, 'speed':speed, 'light': light})
    data.to_csv(os.path.join(metadata['TARGET_DIR'], file_name + '.csv'), sep=',', index_label='frame')

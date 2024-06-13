from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torch
from PIL import Image

#--- used by custom dataset loader
from datetime import datetime, date
import pandas as pd
import time
import os

import cv2
import glob
import json

'''
'''

################################################################
METADATA_FILE_PATH = '../../lib/metadata/metadata_fly_grooming_test_network_2.json'
################################################################


with open(METADATA_FILE_PATH) as f:
  metadata = json.load(f)
  print(metadata)


#\\\\\\\\\\\\\\\\\\\\\\ SEED \\\\\\\\\\\\\\\\\\\\\
torch.manual_seed(0)
np.random.seed(0)
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# use GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load resnet
model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# resize the output layer
# Here the size of each output sample is set to 2.
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

# load model to GPU, if available
model_ft = model_ft.to(device)

checkpoint_name =  metadata['CHECKPOINT']  # load the weights of the trained network
test_frames_directory = metadata['FRAMES_DIR']
checkpoint = torch.load('../../lib/checkpoints/' + checkpoint_name + '.pt')
model_ft.load_state_dict(checkpoint['model_state_dict'])
model_ft.eval()


convert_tensor = transforms.ToTensor()

for label in ['grooming','no_grooming']:
    target = 0
    auto = 0
    for frame in glob.iglob(os.path.join(test_frames_directory, label, "*.png")):
        
        empty_batch = np.zeros([128, 128, 3, 1])
        target += 1

        with torch.no_grad():
            img = Image.open(frame)
            tensor_frame = convert_tensor(img)
            tensor_frame = tensor_frame[None,:]
            prediction = model_ft(tensor_frame.to(device))
            _, preds = torch.max(prediction, 1)
            soft_max_pred = torch.nn.functional.softmax(prediction, dim=1)[0]
            
            if preds < 1:
                auto_label = 'GROOMING'
                if label == 'grooming':
                    auto += 1
            else:
                auto_label = 'NO GROOMING'
                if label == 'no_grooming':
                    auto += 1

            # print('label:{} | {} ; p = {}'.format(label, auto_label, soft_max_pred.data[0]))
            print('{}\t{}'.format(label, soft_max_pred.data[0]))
    
    print('proportion correct: {}'.format(auto / target))        

        

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import random
import cv2
import re
import os


class CustomDataset(Dataset):
    """
    Define the dataset for subject body or individual body parts
    read image, mask and coordinates
    """

    def __init__(self, root_data_dir, transform=False, resize=None):
        self.transform = transform
        self.root_data_dir = root_data_dir
        self.resize = resize
      
    def __len__(self):
        return len(os.listdir(self.root_data_dir + 'frames/'))

    def __getitem__(self, idx):
        img_name = sorted(os.listdir(self.root_data_dir +'frames/'))[idx]
        coordinates_name = sorted(os.listdir(self.root_data_dir +'coordinates/'))[idx]
      
        input_image = cv2.imread(self.root_data_dir +'frames/' + img_name, cv2.IMREAD_GRAYSCALE)
        input_mask = cv2.imread(self.root_data_dir +'masks/' + img_name, cv2.IMREAD_GRAYSCALE)
        input_coordinates = pd.read_csv( self.root_data_dir +'coordinates/' + coordinates_name, sep='\t')
        
        if self.resize is not None:
            input_mask = cv2.resize(input_mask, (self.resize[0], self.resize[1]))
            input_mask = np.clip(input_mask, 0, 1) * 255 # the rescaled mask has been interpolated. convert to 0/255 values
            input_image = cv2.resize(input_image, (self.resize[0], self.resize[1]))

        if self.transform:
            
            # FLIP - H
            if random.random() < 0.5:
                input_image = np.flip(input_image, axis=1).copy()
                input_mask = np.flip(input_mask, axis=1).copy()
                input_coordinates['X'] = input_mask.shape[1] - input_coordinates['X']

            # BLUR
            if random.random() < 0.5:
                input_image = cv2.GaussianBlur(input_image,(3, 3), 0)

            # BRIGHTNESS (+ 2021-03-19: to be tested)
            if random.random() < 0.5:
                input_image += 35
                input_image = np.clip(input_image, a_min = 0, a_max = 255)

            # NOISE    
            if random.random() < 0.5:
                input_image = input_image * np.random.uniform(low=0.8, high=1, size=input_image.shape)

            # FLIP - V
            if random.random() < 0.5:
                input_image = np.flip(input_image, axis=0).copy()
                input_mask = np.flip(input_mask, axis=0).copy()
                # input_coordinates['X'] = input_mask.shape[1] - input_coordinates['X']


        # add extra empty dimension
        input_image = input_image[..., np.newaxis] 
        input_mask = input_mask[..., np.newaxis]
        
        # rearrange channels and size
        input_image = input_image.transpose((2, 0, 1)) 
        input_mask = input_mask.transpose((2, 0, 1))
        
        # convert to tensors
        coordinates = torch.FloatTensor(np.array([input_coordinates['X'], input_coordinates['Y']]))
        input_image = torch.FloatTensor(input_image)
        input_mask = torch.FloatTensor(input_mask)
        
        # at this point the coordinates are inferred from the mask
        # this allows the flexible resizing of the image and mask
        # the original coordinates may be store for reference to the original frame size
        tmp_x = input_mask[0].sum(0).nonzero()
        tmp_y = input_mask[0].sum(1).nonzero()
        coordinates[0] = tmp_x[tmp_x.size()[0] // 2]
        coordinates[1] = tmp_y[tmp_y.size()[0] // 2]
        
        # return structured output
        return {'input_image':input_image, 'frame_id':img_name[0], 'input_mask':input_mask, 'input_coordinates': coordinates}    
    
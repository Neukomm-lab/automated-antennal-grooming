import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torch.optim as optim
#--- used by custom dataset loader
import pandas as pd
import random
import cv2
import re
import os

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch.optim as optim
from torch.optim import lr_scheduler

import copy
from datetime import datetime
import time
import json

#\\\\\\\\\\\\\\\\\\\\\\ SEED \\\\\\\\\\\\\\\\\\\\\
torch.manual_seed(0)
np.random.seed(0)
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#######################################################################################
METADATA_FILE_PATH = '../../lib/metadata/metadata_fly_grooming_train_network_2.json'
#######################################################################################

with open(METADATA_FILE_PATH) as f:
  metadata = json.load(f)
  print(metadata)

data_transforms = {
    'train': transforms.Compose([
    # transforms.transforms.Grayscale(num_output_channels=1),
    # transforms.RandomAffine(180, translate=(0.1, 0.1), scale=None, shear=None, resample=False, fillcolor=0),
    transforms.RandomAffine(180, translate=(0.1, 0.1), scale=None, shear=None, fill=0),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()]),
    'validation': transforms.Compose([
    # transforms.transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(metadata["ROOT_DIR"], x), data_transforms[x]) for x in ['train', 'validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=0) for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=45): # 45
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(outputs)
                    

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print(outputs)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc

### MODEL INSTANCE
model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

### OPTIMIZATION
optimizer_ft = optim.Adam(model_ft.parameters())

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # stepsize=7

since = time.time()
model_ft, val_accuracy = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs= 25) #41 or 25
time_elapsed = time.time() - since

# TRAIN stats and parameters
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# print('Best val distance: {:.2f} px (average)'.format(best_acc))

torch.save({'model_state_dict': model_ft.state_dict()}, '../../lib/checkpoints/' + metadata["CHECKPOINT"] + '.pt')

# output stats + info for VALIDATION
now = datetime.now()
metadata['training on'] = now.strftime("%d/%m/%Y %H:%M:%S")
metadata['# Validation samples'] = dataset_sizes['validation']
metadata['# Train samples'] = dataset_sizes['train']
metadata['Validation Accuracy (%)'] = "{0:.2f}".format(val_accuracy.detach().cpu().item()*100)
metadata['execution time (s)'] = "{0:.1f}".format(time_elapsed)

# store VALIDATION stats in the metadata JSON file
with open(METADATA_FILE_PATH, 'w') as outfile:
    json.dump(metadata, outfile, separators=(',', ': '), indent=4)


#######################################################################################
# the following lines probe the model with samples used during train/validation
#######################################################################################

# model_ft.eval()
# image_datasets['train']
# # batch arrangement: [B C H W]
# test_image = image_datasets['validation'][0][0]
# batch_for_prediction = np.zeros([1, 3, 128, 128])
# batch_for_prediction[0, :, :, :] = test_image
# batch_for_prediction = torch.FloatTensor(batch_for_prediction)

# prediction = model_ft(batch_for_prediction.to(device))
# _, preds = torch.max(prediction, 1)
# print(prediction)
# print(torch.nn.functional.softmax(prediction, dim =1))
# print(preds)

# # to plot
# test_image = image_datasets['validation'][0][0]
# a = test_image.permute(2, 1, 0)
# plt.imshow(a.numpy())
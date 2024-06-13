from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
# import torchvision
import torch

import sys
sys.path.insert(0, "../../lib/")
from custom_dataset import CustomDataset

from datetime import datetime
import numpy as np
import importlib
import json
import copy
import time


##############################################################################
METADATA_FILE_PATH = '../../lib/metadata/metadata_fly_grooming_train_val.json'
##############################################################################

# load the metadata file (JSON) containing parameters to be used 
with open(METADATA_FILE_PATH) as f:
  metadata = json.load(f)

# These are selected via the JSON metadata file
hyper_parameters = {'ADAM': torch.optim.Adam,
                    'nn.MSELoss': nn.MSELoss,
                    'Adamax':torch.optim.Adamax}

# instantiate the model
MyClass = getattr(importlib.import_module("models"), metadata['MODEL'])
MODEL = MyClass()

#\\\\\\\\\\\\\\\\\\\\\\ SEED \\\\\\\\\\\\\\\\\\\\\
torch.manual_seed(0)
np.random.seed(0)
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# Detect if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn = torch.backends.cudnn.version()
print('Pytorch version: {}'.format(torch.__version__))
print('Using: {}'.format(device))
print('Using cuDNN: {}'.format(cudnn))

# Model setup + hyperparameters
tmp_optimizer = hyper_parameters[metadata['OPTIMIZER']]
OPTIMIZER = tmp_optimizer(MODEL.parameters(), lr=metadata['LEARNING_RATE'])
tmp_criterion =  hyper_parameters[metadata['CRITERION']]
CRITERION = tmp_criterion()
MODEL = MODEL.to(device)
EXP_LR_SCHEDULER = lr_scheduler.StepLR(OPTIMIZER, step_size=7, gamma=0.1)

# Data loaders
train_dataset = CustomDataset(root_data_dir = metadata['PATH_TO_TRAIN_SET'], transform=metadata['TRANSFORM'], resize=metadata['RESIZE_FRAME'])
validation_dataset = CustomDataset(root_data_dir = metadata['PATH_TO_VAL_SET'], transform=metadata['TRANSFORM'], resize=metadata['RESIZE_FRAME'])
train_size = len(train_dataset)
validation_size = len(validation_dataset)
image_datasets = {'train':train_dataset, 'val':validation_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=metadata['BATCH_SIZE'], shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('Train set: {}'.format(train_size))
print('Validation set: {}'.format(validation_size))

# variables to store training's parameters
train_loss_history = []
val_loss_history = []
best_model_wts = copy.deepcopy(MODEL.state_dict())
best_acc = 10000 # This is the distance in pixels between predicted and label image. Starts with an unreasonably high number
best_classification = 100

since = time.time()
for epoch in range(metadata['NUM_EPOCHS']):
    print('Epoch {}/{}'.format(epoch, metadata['NUM_EPOCHS'] - 1))
    print('-' * 10)

    # Each epoch has a TRAIN and VALIDATION phase
    for phase in ['train', 'val']:
        if phase == 'train':
            MODEL.train()
        else:
            MODEL.eval()

        running_loss = 0.0
        running_dist = 0.0
        running_acc = 0.0

        # Iterate over data.
        for i, datapoint in enumerate(dataloaders[phase]):
            input_image = datapoint['input_image'].to(device)
            label_image = datapoint['input_mask'].to(device)
            input_coordinates = datapoint['input_coordinates'].to(device)
            
            input_image /= 255

            # zero the parameter gradients
            OPTIMIZER.zero_grad()

            # forward
            # track history if only in TRAIN mode
            with torch.set_grad_enabled(phase == 'train'):
                outputs = MODEL(input_image)
                loss = CRITERION(outputs, label_image)

                tmp = {}
                dist=[]

                for ix, d in enumerate(outputs):
                    tmp['score_map'] = d[0].detach().cpu().numpy()
                    tmp['input_coordinates'] = input_coordinates[ix].detach().cpu().numpy()

                    # coordinates of max point in score-map
                    max_y, max_x = np.unravel_index(tmp['score_map'].argmax(), tmp['score_map'].shape)
                    dist.append(np.sqrt( (tmp['input_coordinates'][0].data - max_x) ** 2 + (tmp['input_coordinates'][1].data - max_y) ** 2))
                
                # backward + optimize only if in TRAIN phase
                if phase == 'train':
                    loss.backward()
                    OPTIMIZER.step()

            # statistics
            batch_dist = np.sum(dist) / len(dist)
            running_loss += loss.item() * input_image.size(0)
            running_dist += np.sum(dist)
            running_acc += np.sum(np.array(dist) < metadata['PIXEL_ACCURACY'])
            # print('Batch distance ({}) {:.2f} px'.format(phase, batch_dist))
            # print(np.sum(np.array(dist) < metadata['PIXEL_ACCURACY']))
            # print('len dist: {}, len dataset_sizes[phase]: {}, running_dist: {} '.format(len(dist), dataset_sizes[phase], running_dist))

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_dist = running_dist / dataset_sizes[phase]
        epoch_accuracy = (running_acc / dataset_sizes[phase]) * 100

        print('> > > {} Loss: {:.4f}  |  Distance: {:.4f} | Accuracy {:.2f}'.format(phase, epoch_loss, epoch_dist, epoch_accuracy))

        if phase == 'train':
            EXP_LR_SCHEDULER.step()
            train_loss_history.append(epoch_loss)
        else:
            val_loss_history.append(epoch_loss)

        # Copy best model so far
        if phase == 'val' and epoch_dist < best_acc:
            print("*")
            best_acc = epoch_dist
            best_model_wts = copy.deepcopy(MODEL.state_dict())
    print()

# TRAIN stats and parameters
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val distance: {:.2f} px (average)'.format(best_acc))

# Load best model weights (for final validation + visualization + Plotting | Optional)
MODEL.load_state_dict(best_model_wts)
torch.save({'model_state_dict': MODEL.state_dict()},'../../lib/checkpoints/' + metadata['CHECKPOINT_NAME'] + '.pt')

#############################################################################
# VALIDATION ON BEST MODEL
#############################################################################

validation_loader = torch.utils.data.DataLoader(validation_dataset)

# Store parameter and stats for VALIDATION
validation = [] # images for visualizations
dist = [] # distance between detected object and target
size = [] # size of detected object
coord = [] # coordinates of the detected object

# VALIDATION starts using weights from best model from TRAIN phase
with torch.no_grad():
    MODEL.eval()
    for datapoint in validation_loader:
        tmp={}
        input_image = datapoint['input_image'].to(device)/255
        prediction = MODEL(input_image)
        tmp['input_image'] = datapoint['input_image'][0][0].detach().cpu().numpy()
        tmp['score_map'] = prediction[0][0].detach().cpu().numpy()
        tmp['input_coordinates'] = datapoint['input_coordinates'][0].detach().cpu().numpy()

        max_value_in_scoremap = tmp['score_map'].max()
        max_value_in_scoremap_coordinates = np.where(tmp['score_map'] == max_value_in_scoremap)
        x = max_value_in_scoremap_coordinates[1][0]
        y = max_value_in_scoremap_coordinates[0][0]

        distance_predicted_target = np.sqrt( (tmp['input_coordinates'][0].data - x) ** 2 + (tmp['input_coordinates'][1].data - y) ** 2)
        dist.append(distance_predicted_target)
        tmp['scoremap_coordinates'] = [x, y]
        tmp['distance'] = distance_predicted_target
        coord.append([x, y])
        validation.append(tmp)
            
    dist = np.array(dist)

    print("\nVALIDATION W/ BEST MODEL\n")
    print('Number of samples in VALIDATION set: {}'.format(len(validation_dataset)))
    print('Average distance (+/- STD): {:.2f} +/- {:.2f} px'.format(np.mean(dist), np.std(dist)))
    print('Median distance (+/- MAD): {:.2f} +/- {:.2f} px'.format(np.median(dist), np.median(dist - np.median(dist))))
    print("Pixel accuracy threshold: {} px".format(metadata['PIXEL_ACCURACY']))
    print("Accuracy:  {0:.2f} %".format((np.sum(dist < metadata['PIXEL_ACCURACY']) / len(dist)) * 100))


# output stats + info for VALIDATION
now = datetime.now()
metadata['training on'] = now.strftime("%d/%m/%Y %H:%M:%S")
metadata['# Validation samples'] = validation_size
metadata['# Train samples'] = train_size
metadata['Validation Accuracy'] = "{0:.2f} %".format((np.sum(dist < metadata['PIXEL_ACCURACY']) / len(dist)) * 100)
metadata['Validation MEDIAN distance (+/- MAD)'] = '{:.2f} +/- {:.2f} px'.format(np.median(dist), np.median(dist - np.median(dist)))
metadata['Validation AVERAGE distance (+/- STD)'] = '{:.2f} +/- {:.2f} px'.format(np.mean(dist), np.std(dist))
metadata['execution time (s)'] = time_elapsed


# store VALIDATION stats in the metadata JSON file
with open(METADATA_FILE_PATH, 'w') as outfile:
    json.dump(metadata, outfile, separators=(',', ': '), indent=4)
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import importlib
import torch
import time
import json
import os

import sys
from datetime import datetime
sys.path.insert(0, "../../lib/")
from custom_dataset import CustomDataset


########################################################################
METADATA_FILE_PATH = '../../lib/metadata/metadata_fly_grooming_test.json'
########################################################################

with open(METADATA_FILE_PATH) as f:
  metadata = json.load(f)
  print('\n', metadata,'\n')

# Seed
torch.manual_seed(0)
np.random.seed(0)

# Detect GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn = torch.backends.cudnn.version()
print('\nPytorch version: {}'.format(torch.__version__))
print('Using: {}'.format(device))
print('Using cuDNN: {}'.format(cudnn))

# Model instantiation (from metadata JSON)
MyClass = getattr(importlib.import_module("models"), metadata['MODEL'])
MODEL = MyClass()
MODEL = MODEL.to(device)

# use CPU if running on Mac
if os.name == 'posix':
    checkpoint = torch.load('../../lib/checkpoints/' + metadata['CHECKPOINT_NAME'] + '.pt', map_location='cpu')
else:
    checkpoint = torch.load('../../lib/checkpoints/' + metadata['CHECKPOINT_NAME'] + '.pt')

MODEL.load_state_dict(checkpoint['model_state_dict'])

# load test dataset
test_dataset = CustomDataset(root_data_dir = metadata['PATH_TO_TEST_SET'], transform=metadata['TRANSFORM'], resize=metadata['RESIZE_FRAME'])
test_size = len(test_dataset)
print('\nSamples in test set: {}\n'.format(test_size))
test_loader = torch.utils.data.DataLoader(test_dataset)

# STORE parameters for visualization and inspection
test = [] # images for visualizations
dist = [] # distance between detected object and target
size = [] # size of detected object
coord = [] # coordinates of the detected object

# TEST starts here
since = time.time()
with torch.no_grad():
    MODEL.eval()
    for datapoint in test_loader:
        tmp={}
        input_image = datapoint['input_image'].to(device) / 255
        prediction = MODEL(input_image)
        tmp['frame_id'] = datapoint['frame_id']
        tmp['input_image'] = datapoint['input_image'][0][0].detach().cpu().numpy()
        tmp['score_map'] = prediction[0][0].detach().cpu().numpy()
        tmp['input_coordinates'] = datapoint['input_coordinates'][0].detach().cpu().numpy()

        # use coordinates of max value in scoremap
        max_value_in_scoremap = tmp['score_map'].max()
        max_value_in_scoremap_coordinates = np.where(tmp['score_map'] == max_value_in_scoremap)
        x = max_value_in_scoremap_coordinates[1][0]
        y = max_value_in_scoremap_coordinates[0][0]

        # Compute distances (detected object, target)
        distance_predicted_target = np.sqrt( (tmp['input_coordinates'][0].data - x) ** 2 + (tmp['input_coordinates'][1].data - y) ** 2)
        dist.append(distance_predicted_target)
        tmp['scoremap_coordinates'] = [x, y]
        tmp['distance'] = distance_predicted_target
        coord.append([x, y])
        test.append(tmp)
            
    dist = np.array(dist)
    time_elapsed = time.time() - since

    # Output statistics
    print('\nNumber of samples in TEST set: {}'.format(len(test_dataset)))
    print('Average distance (+/- STD): {:.2f} +/- {:.2f} px'.format(np.mean(dist), np.std(dist)))
    print('Median distance (+/- MAD): {:.2f} +/- {:.2f} px'.format(np.median(dist), np.median(dist - np.median(dist))))
    print("Pixel accuracy threshold: {} px".format(metadata['PIXEL_ACCURACY']))
    print("Accuracy:  {0:.2f} %".format((np.sum(dist < metadata['PIXEL_ACCURACY']) / len(dist)) * 100))

# Update metadata
now = datetime.now()
metadata['Test on'] = now.strftime("%d/%m/%Y %H:%M:%S")
metadata['# Test samples'] = test_size
metadata['Test Accuracy'] = "{0:.2f} %".format((np.sum(dist < metadata['PIXEL_ACCURACY']) / len(dist)) * 100)
metadata['Test MEDIAN distance (+/- MAD)'] = '{:.2f} +/- {:.2f} px'.format(np.median(dist), np.median(dist - np.median(dist)))
metadata['Test AVERAGE distance (+/- STD)'] = '{:.2f} +/- {:.2f} px'.format(np.mean(dist), np.std(dist))
metadata['Execution time (s)'] = time_elapsed

with open(METADATA_FILE_PATH, 'w') as outfile:
    json.dump(metadata, outfile, separators=(',', ': '), indent=4)

# Plot the distribution of errors (distance in pixels of predicted vs target)
sns.displot(data=dist, kde=True);

# plot a 
frame_number = np.random.randint(0, len(test))
plt.imshow(test[frame_number]['input_image']);
plt.scatter(test[frame_number]['scoremap_coordinates'][0], test[frame_number]['scoremap_coordinates'][1],color='red',s =1.5)
plt.title(test[frame_number]['frame_id'][0]);
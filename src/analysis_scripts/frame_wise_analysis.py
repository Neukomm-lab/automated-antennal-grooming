from fileinput import filename
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, matthews_corrcoef

'''
2022-12-16

script for frame-wise analysis of grooming
'''

#--- root directory
ROOT_DIR = r"..\..\results"

#--- read file in 
data = pd.read_csv(os.path.join(ROOT_DIR, '2022-12-20_frame_wise_grooming.csv'))

#--- select subject
subjects = np.unique(data.filename) # list of available subjects


for sbj in subjects:
    tmp = data[data.filename == sbj]
    print(tmp['auto_grooming_0.55'].corr(tmp['target_grooming']),'\n')



tn, fp, fn, tp = confusion_matrix(data['target_grooming'], data['auto_grooming_0.55']).ravel()
(tn, fp, fn, tp)
print('true positive: {:.3f}'.format(tp / np.sum((tn, fp, fn, tp))))
print('true negative: {:.3f}'.format(tn / np.sum((tn, fp, fn, tp))))
print('false positive: {:.3f}'.format(fp / np.sum((tn, fp, fn, tp))))
print('false negative: {:.3f}'.format(fn / np.sum((tn, fp, fn, tp))))

light_only = data[data['light'] == 1]

tn, fp, fn, tp = confusion_matrix(light_only['target_grooming'], light_only['auto_grooming_0.55']).ravel()
(tn, fp, fn, tp)
print('true positive: {:.3f}'.format(tp / np.sum((tn, fp, fn, tp))))
print('true negative: {:.3f}'.format(tn / np.sum((tn, fp, fn, tp))))
print('false positive: {:.3f}'.format(fp / np.sum((tn, fp, fn, tp))))
print('false negative: {:.3f}'.format(fn / np.sum((tn, fp, fn, tp))))

light_only['auto_grooming_0.55'].sum() - light_only['target_grooming'].sum()
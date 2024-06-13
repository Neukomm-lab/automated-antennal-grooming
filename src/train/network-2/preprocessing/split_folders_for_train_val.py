import os
import glob
from pathlib import Path
import shutil

import random as rnd
import numpy as np

rnd.seed(123)

'''
2022-08-25
distribute labeled frames to train-validation folders
'''

DIR = "no_grooming"

# ROOT_DIR = r"Z:\DNF\GROUPS\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\TRAIN-VALIDATION-TEST-SETS\2022-08-24"
ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\16 After injury test\AFTER_INJURY_TRAIN_VAL_TEST_SETS"

SOURCE_DIR = os.path.join(ROOT_DIR, "training")
OUTPUT_DIR = os.path.join(ROOT_DIR, "training")

list_of_files = [f for f in glob.glob(os.path.join(SOURCE_DIR, DIR, "*.png"))]

train_set_size = int(len(list_of_files) * 0.8)
val_set_size = len(list_of_files) - train_set_size

ixs = np.arange(0, len(list_of_files))
val_set_ixs =np.sort(rnd.sample(list(ixs), val_set_size))
train_set_ixs= np.delete(ixs, val_set_ixs)

train_set_files = [list_of_files[i] for i in train_set_ixs]
val_set_files = [list_of_files[i] for i in val_set_ixs]

for src in train_set_files:
    print(Path(src).stem)
    dst = os.path.join(ROOT_DIR,'training', 'train_val_set', 'train', DIR, Path(src).stem + ".png")
    shutil.copyfile(src, dst)

for src in val_set_files:
    print(Path(src).stem)
    dst = os.path.join(ROOT_DIR,'training', 'train_val_set', 'validation', DIR, Path(src).stem + ".png")
    shutil.copyfile(src, dst)

print(" >>>>>>>>>>> DONE")
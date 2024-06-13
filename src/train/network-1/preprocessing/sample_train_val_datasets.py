import sys
import os
import glob
import random
from shutil import copyfile
import shutil

'''
Randomly samples labeled frames and assign to TRAIN, VALIDATION sets
'''

ROOT_DIR = "D:\\data-sets\\neukomm\\train-val-test-sets\\2022-06-17\\"

list_of_files = []

for f in glob.glob(ROOT_DIR + "pool\\frames\*.png"):
    f_name = os.path.splitext(f)[0].split('\\')[-1]
    list_of_files.append(f_name)

random.shuffle(list_of_files)

# to be sent to train (80%)
train_size = int(len(list_of_files) * 0.8)
validation_size = len(list_of_files) - train_size

train_ixs = range(0, train_size)
validation_ixs = range(train_size, len(list_of_files))

_b = ['frames','masks','coordinates']
_a = ['train_set','validation_set']

for i in _a:
    for j in _b:
        TARGET_DIR = ROOT_DIR + i +"\\"+ j
        if os.path.exists(TARGET_DIR):
            shutil.rmtree(TARGET_DIR)
        os.makedirs(TARGET_DIR)

for i in train_ixs:

    TARGET_DIR = ROOT_DIR + "train_set\\frames\\"
    SRC_DIR = ROOT_DIR + "pool\\frames\\"
    dst = TARGET_DIR + list_of_files[i] + '.png'
    src = SRC_DIR + list_of_files[i] + '.png'
    copyfile(src, dst)

    TARGET_DIR = ROOT_DIR + "train_set\\masks\\"
    SRC_DIR = ROOT_DIR + "pool\\masks\\"
    dst = TARGET_DIR + list_of_files[i] + '.png'
    src = SRC_DIR + list_of_files[i] + '.png'
    copyfile(src, dst)

    TARGET_DIR = ROOT_DIR + "train_set\\coordinates\\"
    SRC_DIR = ROOT_DIR + "pool\\coordinates\\"
    dst = TARGET_DIR + list_of_files[i] + '.txt'
    src = SRC_DIR + list_of_files[i] + '.txt'
    copyfile(src, dst)


for i in validation_ixs:
    TARGET_DIR = ROOT_DIR + "validation_set\\frames\\"
    SRC_DIR = ROOT_DIR + "pool\\frames\\"
    dst = TARGET_DIR + list_of_files[i] + '.png'
    src = SRC_DIR + list_of_files[i] + '.png'
    copyfile(src, dst)

    TARGET_DIR = ROOT_DIR + "validation_set\\masks\\"
    SRC_DIR = ROOT_DIR + "pool\\masks\\"
    dst = TARGET_DIR + list_of_files[i] + '.png'
    src = SRC_DIR + list_of_files[i] + '.png'
    copyfile(src, dst)

    TARGET_DIR = ROOT_DIR + "validation_set\\coordinates\\"
    SRC_DIR = ROOT_DIR + "pool\\coordinates\\"
    dst = TARGET_DIR + list_of_files[i] + '.txt'
    src = SRC_DIR + list_of_files[i] + '.txt'
    copyfile(src, dst)
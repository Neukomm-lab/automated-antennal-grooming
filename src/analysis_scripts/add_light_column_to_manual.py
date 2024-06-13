from fileinput import filename
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
2023-11-21
INPUT: folder containing files with manually scored data (Ethovision files) and automated data
OUTPUT: manual data are saved with the light_column from the auto data
"""


# root directory
ROOT_DIR = "..\\..\\results\\"


def get_file_name(filename_in):
    '''
    returns the filename without extension
    
    >>> get_file_name("z:/some/path/to/file.csv")
    'file'
    '''
    return os.path.splitext(os.path.basename(filename_in))[0]


def get_AUTO_data(filename_in):
    '''
    get automated scoring data (.csv format)
    '''
    return pd.read_csv(filename_in)


def get_EV_data(filename_in):
    '''
    Get Ethovision file
    >>> Number of header rows must be 37! --> see skiprows
    '''
    ethovision_data_out = pd.read_excel(filename_in, skiprows=37)
    ethovision_data_out.columns = ['Trial time','Recording time','X center','Y center','Area','Areachange','Elongation','grooming','light','Result 1']
    return(ethovision_data_out.drop(['Result 1'], axis=1))

manual_files = glob.glob(os.path.join(ROOT_DIR, "manual_data", "*.xlsx"))
automated_files = glob.glob(os.path.join(ROOT_DIR, "raw_data_output", "*.csv"))

n_of_files =  len(manual_files)

OUTPUT_DIR = os.path.join(ROOT_DIR, "manual_data", "output_w_light")

for i in range(n_of_files):
    auto_tmp = get_AUTO_data(automated_files[i])
    manual_tmp = get_EV_data(manual_files[i])
    manual_tmp.light = auto_tmp.light_frames
    manual_tmp.to_excel(os.path.join(OUTPUT_DIR , manual_files[i].split("\\")[-1]))

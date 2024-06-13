from fileinput import filename
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
INPUT: folder containing files with manually scored data (Ethovision files)
OUTPUT: table with statistics on manual scoring
"""


# root directory
ROOT_DIR = r"..\..\results\manual_data"


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


def assign_filename_to_column(dataframe_in, filename_in):
    '''
    insert filename as column
    '''
    dataframe_in['filename'] = filename_in
    return (dataframe_in)


dump = []
for f in glob.glob(os.path.join(ROOT_DIR, "*.xlsx")):
    print(f)
    tmp = get_EV_data(f)
    assign_filename_to_column(tmp,get_file_name(f))
    dump.append(tmp)

print("\nDone READING FILES\n")

data = pd.concat(dump)

light_only = data[data["light"] > 0]
light_only.groupby(by=["filename"]).sum()
table_out = light_only.groupby(by=["filename"]).sum()
table_out["%time_grooming(MANUAL)"] = table_out["grooming"] / table_out["light"] * 100
table_out.to_csv(os.path.join(ROOT_DIR,"2023-11-28_time_course_5-7dpe.csv"))

print("\nDone SAVING FILE\n")
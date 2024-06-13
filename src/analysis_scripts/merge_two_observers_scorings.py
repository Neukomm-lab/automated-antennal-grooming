import os
import glob
import numpy as np
import pandas as pd

'''
merges two scorings from independent observers.
'''


def get_EV_data(filename_in):
    '''
    Get Ethovision file
    >>> Number of header rows must be 37! --> see skiprows
    '''
    ethovision_data_out = pd.read_excel(filename_in, skiprows=37)
    ethovision_data_out.columns = ['Trial time','Recording time','X center','Y center','Area','Areachange','Elongation','grooming','light','Result 1']
    return(ethovision_data_out.drop(['Result 1'], axis=1))


def get_file_name(filename_in):
    '''
    + return the filename without extension
    >>> get_file_name("z:/some/path/to/file.csv")
    'file'
    '''
    return os.path.splitext(os.path.basename(filename_in))[0]


def assign_filename_to_column(dataframe_in, filename_in):
    '''
    Add one extra column containing the filename
    '''
    dataframe_in['filename'] = filename_in
    return (dataframe_in)


# directories and file paths *IMMUTABLE*
# ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\20 Arnau manual scoring\Maria_manual_scoring"
ROOT_DIR = r"D:\switchdrive\Neuro-BAU\Code\GIT\deep_behavior\CNN\build\Fly-grooming-Neukomm-lab\results\raw_data_output"

db = []
# get list of file paths & check that the same number of files is present in both directories
files_list = glob.glob(os.path.join(ROOT_DIR, '*.xlsx'))
for f in sorted(files_list):
    f_name = get_file_name(f)
    current_file = get_EV_data(f)
    current_file = assign_filename_to_column(current_file, f_name)
    db.append(current_file)

out = pd.concat(db)
out.to_csv(os.path.join(ROOT_DIR,"GPU_merged.csv"))
print('DONE')
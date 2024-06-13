import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# figure(figsize=(18, 6))

'''
Takes output of network1 (`head coordinates`) and network2 (`grooming timeline`)
and merges them into a single file that is saved in folder `raw_data_output`
the output of this script is a single file for each animal
the grooming variable (0/1) is computed for a range of thresholds (from 0.1 to 0.9)
'''

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


def compute_auto_grooming(auto_df, threshold=0.5):
    '''
    transform P(grooming) into 0/1 column based on input threshold value
    after transform, the 0/1 column is smoothed (see smooth_function_grooming)
    >>> add new column to input dataframe
    '''
    new_grooming_column = 'auto_grooming_{}'.format(threshold)
    auto_df[new_grooming_column] = np.where(auto_df['P(grooming)'] >= threshold, 1, 0)
    smooth_function_grooming(auto_df, new_grooming_column)


def smooth_function_grooming(auto_df, grooming_column):
    '''
    Every grooming event (n) is expanded into a 3n series
    i.e. for every grooming frame , the two consecutive frames are labeled as grooming
    The network is trained on stacks of 10 frames (standard deviation projection).
    However, using a 10X expansion of grooming frames leads to too many false positive. 
    '''
    df_1 = np.concatenate((np.array([0]), auto_df[grooming_column][:-1]))
    df_2 = np.concatenate((np.array([0,0]), auto_df[grooming_column][:-2]))
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_1)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_2)

#..............................................................................................

print("\nRunning <{}> script ...\n".format(os.path.basename(__file__)))

# directories and file paths *IMMUTABLE*
ROOT_DIR = r"../../results"

# get list of file paths & check that the same number of files is present in both directories
head_coordinates_files_list = glob.glob(os.path.join(ROOT_DIR, 'head_coordinates', '*.csv'))
grooming_files_list = glob.glob(os.path.join(ROOT_DIR, 'grooming_timeline', '*.csv'))
assert len(head_coordinates_files_list) == len(grooming_files_list), '>>>>> NUMBER of files do not match in folders: `grooming_timeline` and `head_coordinates`'

# get list of filenames & check that the file names do match
head_coordinates_filenames = np.array([get_file_name(i) for i in head_coordinates_files_list])
grooming_filenames = np.array([get_file_name(i) for i in grooming_files_list])
assert (head_coordinates_filenames == grooming_filenames).all(), 'FILE NAMES in `grooming_timeline` and `head_coordinates` do not match '

for i in range(len(head_coordinates_filenames)):

    print(head_coordinates_files_list[i])

    # open head coordinates file
    current_head_coordinates_file = pd.read_csv(head_coordinates_files_list[i])
    
    # add column with filename
    assign_filename_to_column(current_head_coordinates_file, head_coordinates_filenames[i])
    
    # open grooming file / force convert P(grooming) data into float
    current_grooming_file = pd.read_csv(grooming_files_list[i], dtype = {'P(grooming)':np.float64})

    # generate merged dataframe
    new_df = pd.DataFrame({
        'filename': current_head_coordinates_file.filename,
        'frame': current_head_coordinates_file.frame,
        'timestamp': current_grooming_file.timestamp,
        'date_timestamp': current_grooming_file.date_timestamp,
        'X': current_head_coordinates_file.X,
        'Y': current_head_coordinates_file.Y,
        'speed': current_head_coordinates_file.speed,
        'P(grooming)': current_grooming_file['P(grooming)'],
        'light_frames': current_head_coordinates_file.light,
        'checkpoint': current_grooming_file.checkpoint
    })
    compute_auto_grooming(new_df, 0.1)
    compute_auto_grooming(new_df, 0.2)
    compute_auto_grooming(new_df, 0.3)
    compute_auto_grooming(new_df, 0.4)
    compute_auto_grooming(new_df, 0.5)
    compute_auto_grooming(new_df, 0.6)
    compute_auto_grooming(new_df, 0.7)
    compute_auto_grooming(new_df, 0.8)
    compute_auto_grooming(new_df, 0.9)

    new_df.to_csv(os.path.join(ROOT_DIR, 'raw_data_output', head_coordinates_filenames[i] + '.csv'), index = False)
    
print("\n+++ DONE +++\n")
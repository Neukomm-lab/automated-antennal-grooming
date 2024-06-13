from datetime import date
import pandas as pd
import numpy as np
import glob
import os


'''
2023-01-17

Read files from folders `output_raw_data` and `manual_data`.
merge files and generates a summary table for LIGHT-ONSET period.
The individual files are saved to the `evaluation_auto_manual` folder
The summary table is saved to the `output_analysis` folder

the resulting file can be used to compare manual and automated scoring files
'''


#--- root directory
ROOT_DIR = r"../../results"


today = date.today()
print("Today's date:", today, "\n")

def get_file_name(filename_in):
    '''
    returns the filename without extension
    
    >>> get_file_name("z:/some/path/to/file.csv")
    'file'
    '''
    return os.path.splitext(os.path.basename(filename_in))[0]


# get list of file paths & check that the same number of files is present in both directories
AUTO_files_list = glob.glob(os.path.join(ROOT_DIR, 'raw_data_output', '*.csv'))
MANUAL_files_list = glob.glob(os.path.join(ROOT_DIR, 'manual_data', '*.xlsx'))
assert len(AUTO_files_list) == len(MANUAL_files_list), '>>>>> NUMBER of files do not match in folders: `manual_data` and `raw_data_output`'

# get list of filenames & check that the file names do match
AUTO_filenames = np.array([get_file_name(i) for i in AUTO_files_list])
MANUAL_filenames = np.array([get_file_name(i) for i in MANUAL_files_list])
assert (AUTO_filenames == MANUAL_filenames).all(), 'FILE NAMES in `manual_data` and `raw_data_output` do not match '

n_of_files = len(AUTO_files_list)


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


db = []

for i in range(n_of_files):

    # screen output
    print("{}, Processing : {}".format(i, AUTO_filenames[i]))

    # get auto data
    auto = get_AUTO_data(AUTO_files_list[i])
    
    # get manual data
    manual = get_EV_data(MANUAL_files_list[i])

    # add column to auto data
    auto['target_grooming'] = manual['grooming']
    
    # place all processed files in a list (to be concatenated as a single dataframe)
    db.append(auto)

    # save file to folder
    auto.to_csv(os.path.join(ROOT_DIR, 'evaluation_auto_manual', AUTO_filenames[i] + '.csv'), index = False)

merged_df = pd.concat(db)

# find LIGHT ONSET periods
# get total number of frames grooming (manual or automated) during LIGHT ON
# save table to csv file
light_only = merged_df[merged_df["light"] > 0]
light_only = light_only.drop(['frame', 'timestamp', 'date_timestamp', 'X', 'Y', 'speed','P(grooming)'], axis=1)
table_out = light_only.groupby(by=["filename"]).sum()
table_out["%time_grooming(MANUAL)"] = table_out["target_grooming"] / table_out["light"] * 100
table_out["%time_grooming(AUTO)"] = table_out["auto_grooming_0.5"] / table_out["light"] * 100
table_out.to_csv(os.path.join(ROOT_DIR, "output_analysis", str(today) + "_LIGHT-ONLY_quality_assessment.csv"))
print('\n>>> DONE\n')
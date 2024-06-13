from datetime import date
import pandas as pd
import numpy as np
import glob
import os


'''
2023-02-13

this file is called by the main script: `data analysis`
it generates a result table 
'''

#------------------------------------------------
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

#------------------------------------------------

print("\nRunning <{}> script ...\n".format(os.path.basename(__file__)))


#--- root directory
ROOT_DIR = r"../../results"

today = date.today()
print("Today's date:", today, "\n")

# get list of file paths & check that the same number of files is present in both directories
AUTO_files_list = glob.glob(os.path.join(ROOT_DIR, 'raw_data_output', '*.csv'))

# get list of filenames & check that the file names do match
AUTO_filenames = np.array([get_file_name(i) for i in AUTO_files_list])

n_of_files = len(AUTO_files_list)


db = []

for i in range(n_of_files):

    # screen output
    print("{}, Processing : {}".format(i, AUTO_filenames[i]))

    # get auto data
    auto = get_AUTO_data(AUTO_files_list[i])
    
    # place all processed files in a list (to be concatenated as a single dataframe)
    db.append(auto)

merged_df = pd.concat(db)

# find LIGHT ONSET periods
# get total number of frames grooming (manual or automated) during LIGHT ON
# save table to csv file
light_ON = merged_df[merged_df["light_frames"] > 0]
light_ON = light_ON.drop(['frame', 'timestamp', 'date_timestamp', 'X', 'Y', 'speed','P(grooming)'], axis=1)
table_ON = light_ON.groupby(by=["filename","checkpoint"]).sum()
table_ON.insert(1,"light", 1)


light_OFF = merged_df[merged_df["light_frames"] < 1]
light_OFF = light_OFF.drop(['frame', 'timestamp', 'date_timestamp', 'X', 'Y', 'speed','P(grooming)'], axis=1)
table_OFF = light_OFF.groupby(by=["filename","checkpoint"]).sum()
table_OFF.light_frames = light_OFF.groupby(by=["filename"]).count().light_frames.iloc[0]
table_OFF.insert(1,"light", 0)


table_out = pd.concat([table_OFF, table_ON])
# table_out["%time_grooming(AUTO)"] = table_out["auto_grooming_0.5"] / table_out["light"] * 100
table_out.rename({"light_frames":"number_of_frames"}, inplace=True, axis=1)
table_out.to_csv(os.path.join(ROOT_DIR, "output_analysis", str(today) + "_RESULTS_table_out.csv"))

print("\n+++ DONE +++\n")





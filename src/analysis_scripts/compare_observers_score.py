import os
import numpy as np
import pandas as pd
from datetime import datetime

"""
2023-06-23
compare performances of two human observers
input: csv file with merged trials (i.e. individual flies) scored by 
OB1, OB2, GPU
"""

# Root directory path
ROOT = r"D:\switchdrive\Neuro-BAU\Research-projects\Neukomm\data"
ROOT = "G:\\RESEARCH\\NEUROBAU\\PUBLIC\\LN\\Maria\\23 Humans vs Machine after injury"

# define the column name where the auto grooming is stored
auto_grooming_column_name = "auto_grooming_0.4"

# Get a list of CSV and XLSX files in the directory
# data = pd.read_excel(os.path.join(ROOT, 'multiple_observers_scoring_raw.xlsx'))
data = pd.read_csv(os.path.join(ROOT, 'muliple_obesrvers_AFTER.csv'))

# total number of observations
number_of_frames = len(data)

# total number of grooming episodes estimated (take max from two observers)
data['total_estimate'] = np.where((data.maria_grooming + data.arnau_grooming) > 0, 1, 0)

# OB1 deviation from total estimate (both false positive and false negative)
data['OB1|total_estimate'] = np.where(data.maria_grooming != data.total_estimate, 1, 0)

# OB2 deviation from total estimate (both false positive and false negative)
data['OB2|total_estimate'] = np.where(data.arnau_grooming != data.total_estimate, 1, 0)

# GPU deviation from total estimate (both false positive and false negative)
data['GPU|total_estimate'] = np.where(data[auto_grooming_column_name] != data.total_estimate, 1, 0)

data['OB1|false_positive'] = np.where(data["maria_grooming"] - data['arnau_grooming'] == 1, 1, 0)
data['OB2|false_positive'] = np.where(data["arnau_grooming"] - data['maria_grooming'] == 1, 1, 0)
data['GPU|false_positive'] = np.where(data[auto_grooming_column_name] - data['total_estimate'] == 1, 1, 0)
data['OB1_OB2|concordance'] = np.where(data["maria_grooming"] + data['arnau_grooming'] == 2, 1, 0)
data['OB1_GPU|concordance'] = np.where(data["maria_grooming"] + data[auto_grooming_column_name] == 2, 1, 0)
data['OB2_GPU|concordance'] = np.where(data["arnau_grooming"] + data[auto_grooming_column_name] == 2, 1, 0)
data['OB1_OB2|median'] = np.median([data["maria_grooming"],data['arnau_grooming']])

data["disagreement_OBS1|OBS2"] = np.where(data["maria_grooming"] != data['arnau_grooming'], 1, 0)
data["disagreement_OBS1|GPU"] = np.where(data["maria_grooming"] != data[auto_grooming_column_name], 1, 0)
data["disagreement_OBS2|GPU"] = np.where(data["arnau_grooming"] != data[auto_grooming_column_name], 1, 0)

total_frame_count = data.light.groupby(by=data.Filename).count() # choose any column present in the dataframe

# filter by light ON only
data = data[data["light"]==1]

out = data.groupby(by=data.Filename).sum()
count = data.light.groupby(by=data.Filename).count()
out["light_frame_count"] = count
out["total_frame_count"] = total_frame_count

out["StdDev_OBS1_OBS2"] = np.sqrt(((out.maria_grooming - out.arnau_grooming)/2)**2)
out["StdDev_OBS1_GPU"] = np.sqrt(((out.maria_grooming - out[auto_grooming_column_name])/2)**2)
out["StdDev_OBS2_GPU"] = np.sqrt(((out.arnau_grooming - out[auto_grooming_column_name])/2)**2)

out["OB1_accuracy"] = 100 - out["OB1|total_estimate"] / out["light_frame_count"] * 100
out["OB2_accuracy"] = 100 - out["OB2|total_estimate"] / out["light_frame_count"] * 100
out["GPU_accuracy"] = 100 - out["GPU|total_estimate"] / out["light_frame_count"] * 100

out["OB1_FP_rate"] = 100 - out["OB1|false_positive"] / out["light_frame_count"] * 100
out["OB2_FP_rate"] = 100 - out["OB2|total_estimate"] / out["light_frame_count"] * 100
out["GPU_FP_rate"] = 100 - out["GPU|total_estimate"] / out["light_frame_count"] * 100

out["OB1_%"] = out["maria_grooming"] / out["light"] * 100
out["OB2_%"] = out["arnau_grooming"] / out["light"] * 100
out["GPU_%"] = out[auto_grooming_column_name] / out["light"] * 100


out.to_csv("inter_observer_accuracy_table_AFTER.csv")

print('DONE!')

import os
import glob
from pathlib import Path
import shutil


'''
Generates unique IDs using the file path.
folders are named according to exepriment type
subjects may have non-unique IDs (e.g. 1_2, 3_3, 6_4 ...)
the resulting file is named (example):
20_degrees_6mW_6_4

'''

ROOT_DIR = r'Z:\DNF\GROUPS\RESEARCH\NEUROBAU\PUBLIC\LN\Maria'
OUTPUT_DIR = r'Z:\DNF\GROUPS\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\TRAIN-VALIDATION-TEST-SETS\2022-08-24'

experiment = ["20_degrees_6mW", "20_degrees_10mW","25_degrees_6mW","25_degrees_10mW"]
file_type = ["Rec","Eth"]
data_set = ["Training","Test"]

for ex in experiment:
    for ft in file_type:
        for ds in data_set:

            if ft == "Rec":
                extension = ".avi"
            elif ft == "Eth":
                extension = ".xlsx"
            
            for f in glob.iglob(os.path.join(ROOT_DIR, "13 Manual_vs_Automated_update", ex, ft, ds, "*" + extension)):
                
                file_name_out = "_".join([ex, Path(f).stem])
                dst = os.path.join(OUTPUT_DIR, ds, ft, file_name_out + extension)
                
                if not os.path.exists(dst):
                    # print("{} will be copied".format(Path(f).stem))
                    shutil.copyfile(f, dst)

print(">>>>>>> DONE <<<<<<<")
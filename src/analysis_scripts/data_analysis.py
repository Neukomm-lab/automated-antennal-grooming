import os
import subprocess

# from: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, 'analysis_scripts')

# import consolidate_raw_data
# import generate_result_table

"""
2023-02-13

*Main script for running data analysis on the network's output.*
it can be run directly after the processing of Network2 is completed

Imports sequentially two script for analysis

+++ How to run +++
- cd into the `analysis scripts` folder and type:
>>> python data_analysis.py
"""

# from: https://stackoverflow.com/questions/517970/how-to-clear-the-interpreter-console
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

cls()

subprocess.call(['python', 'consolidate_raw_data.py'])
subprocess.call(['python', 'generate_result_table.py'])
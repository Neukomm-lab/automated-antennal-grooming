import os
import sys
import subprocess

'''
Main script to sequentially run the two networks.
'''

# from: https://stackoverflow.com/questions/517970/how-to-clear-the-interpreter-console
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
    
cls()

# from: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'src\inference')

flag = str(sys.argv[1]).lower()

if flag == "before":
    pass
elif flag == "after":
    pass
else:
    print(">>> ERROR: missing required argument (BEFORE | AFTER)")
    sys.exit()

print("\nProcessing networks with checkpoint: {}".format(flag.upper()))

subprocess.call(['python', 'inference_network_1.py'])
subprocess.call(['python', 'inference_network_2.py', flag])

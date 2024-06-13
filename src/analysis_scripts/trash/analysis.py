from fileinput import filename
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, matthews_corrcoef

# root directory
ROOT_DIR = r"D:\switchdrive\Neuro-BAU\Code\GIT\deep_behavior\CNN\build\results"


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


def merge_auto_EV_dataframes(auto_df, EV_df):
    '''
    takes two dataframes (df) AUTO or EV
    extract light and grooming from EV files and attach them to AUTO df
    load extra columns onto auto_df
    >>> Does not return:the input dataframe is changed in_place!
    '''
    # make sure that the two dataframes have the same length
    if auto_df.shape[0] == EV_df.shape[0]:
        # make sure that it is same animal in both df
        if auto_df.filename[0] == EV_df.filename[0]:
            # merge the dataframes along the rows
            auto_df['ev_light'] = EV_df.light
            auto_df['target_grooming'] = EV_df.grooming
        else:
            print('the two dataframes are from different animals')
    else:
        print('the two dataframes have different number of rows')


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
    this is done because, the network is trained on stacks of 3 frames to detect grooming
    '''
    df_1 = np.concatenate((np.array([0]), auto_df[grooming_column][:-1]))
    df_2 = np.concatenate((np.array([0,0]), auto_df[grooming_column][:-2]))
    df_3 = np.concatenate((np.array([0,0,0]), auto_df[grooming_column][:-3]))
    df_4 = np.concatenate((np.array([0,0,0,0]), auto_df[grooming_column][:-4]))
    df_5 = np.concatenate((np.array([0,0,0,0,0]), auto_df[grooming_column][:-5]))
    df_6 = np.concatenate((np.array([0,0,0,0,0,0]), auto_df[grooming_column][:-6]))
    df_7 = np.concatenate((np.array([0,0,0,0,0,0,0]), auto_df[grooming_column][:-7]))
    df_8 = np.concatenate((np.array([0,0,0,0,0,0,0,0]), auto_df[grooming_column][:-8]))
    df_9 = np.concatenate((np.array([0,0,0,0,0,0,0,0,0]), auto_df[grooming_column][:-9]))
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_1)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_2)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_3)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_4)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_5)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_6)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_7)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_8)
    auto_df[grooming_column] = np.maximum(auto_df[grooming_column], df_9)


#----------------------------------------------------------------
def compute_frame_wise_accuracy(df_in):
    '''
    Compute the accuracy frame-wise for the selected dataframe
    over a range of probabilities
    '''
    for p in probability:
        threshold = "auto_grooming_" + str(p)
        classification_deviation = np.sum(np.abs(df_in[threshold] - df_in.target_grooming))
        total_grooming = np.sum(df_in.target_grooming)
        accuracy = (1 - (classification_deviation / total_grooming)) * 100
        print("{} \t {}".format(p , accuracy))
#----------------------------------------------------------------

#----------------------------------------------------------------
# def compute_average_accuracy():
#     '''
#     find unique data set of IDs
    
#     for every ID in set:
#         compute accuracy
#         append value to list

#     compute average and std.dev of list
#     '''
    # dump_out = []
    # for id in list(set(data.filename)):
    #     # print(matthews_corrcoef(data[data.filename == id]["target_grooming"],data[data.filename == id]["auto_grooming_0.5"]))
    #     dump_out.append(matthews_corrcoef(data[data.filename == id]["target_grooming"],data[data.filename == id]["auto_grooming_0.5"]))
    # print(np.average(dump_out))
    #  np.average(np.abs(table_out_all["auto_grooming_0.5"] - table_out_all.target_grooming))
#----------------------------------------------------------------

#----------------------------------------------------------------
def plot_n_save(id_number, save_file=0):
    '''
    plot individual subjects, using the index of IDS list
    '''
    f,ax = plt.subplots(figsize=(10, 8))
    ax.plot(data[data.filename == ids[id_number]]["P(grooming)"])
    ax.plot(data[data.filename == ids[id_number]]["light"],'r')
    ax.plot(data[data.filename == ids[id_number]]["target_grooming"] + 1, 'g')
    plt.title(ids[id_number])
    if save_file != 0:
        plt.savefig(ids[id_number] + ".svg")
#----------------------------------------------------------------

#----------------------------------------------------------------
def concatenate_auto_data(auto_data_folder=None):
    '''
    retrieve the data from the `raw_data_output` folder
    - can be added an extra folder inside the `raw_data_output`
    '''
    auto_data = []
    # the path can be either the TRAINING output data folder or the TEST output data folders
    for f in glob.glob(os.path.join(ROOT_DIR, 'raw_data_output', auto_data_folder, '*.csv')):
        print(f)
        # read file
        current_dataframe = get_AUTO_data(f)
        # get file name
        current_filename = get_file_name(f)
        # add filename & dump
        auto_data.append(assign_filename_to_column(current_dataframe, current_filename))
    
    return auto_data
#----------------------------------------------------------------

#----------------------------------------------------------------
def concatenate_ethovision_files():
    '''
    requires .xlsx files
    '''
    # list of manually scored data files
    EV_dump_out = []

    for g in glob.glob(os.path.join(ROOT_DIR, 'manual_data', '*.xlsx')):
        print(g)
        # read file
        current_dataframe = get_EV_data(g)
        # get file name
        current_filename = get_file_name(g)
        # add filename & dump
        EV_dump_out.append(assign_filename_to_column(current_dataframe, current_filename))
    
    return EV_dump_out
#----------------------------------------------------------------

#----------------------------------------------------------------
def merge_AUTO_MANUAL_data(auto_df_in, manual_df_in):
    '''
    '''
    tmp = []

    for i in range(len(auto_df_in)):
        merge_auto_EV_dataframes(auto_df_in[i], manual_df_in[i])
        tmp.append(auto_df_in[i])
        for p in probability:
            compute_auto_grooming(auto_df_in[i], p)

    return pd.concat(tmp)
#----------------------------------------------------------------

#----------------------------------------------------------------
def find_bouts(data_in):
    '''
    >>> find_bouts(np.array([0,0,0,0,1,1,1,1,1,1,0,0,0]))
    [np.array(4,9)]
    '''
    pass
    

#----------------------------------------------------------------
def true_positive(auto_df):
    return np.sum(auto_df.target_grooming * auto_df.auto_grooming)

def true_negative(auto_df):
     return sum(np.where(auto_df.target_grooming - auto_df.auto_grooming == 0, 1 ,0))

def false_positive(auto_df):
     return sum(np.where(auto_df.auto_grooming - auto_df.target_grooming == 1, 1 ,0))

def false_negative(auto_df):
     return sum(np.where(auto_df.auto_grooming - auto_df.target_grooming == -1, 1 ,0))

def sensitivity(auto_df):
    return true_positive(auto_df) / (true_positive(auto_df) + false_negative(auto_df))

def specificity(auto_df):
    return true_negative(auto_df) / (false_positive(auto_df) + true_negative(auto_df))

def g_mean(auto_df):
    return np.sqrt(sensitivity(auto_df) * specificity (auto_df))

def precision(auto_df):
    return true_positive(auto_df) / (true_positive(auto_df) + false_positive(auto_df))

def recall(auto_df):
    return true_positive(auto_df) / (true_positive(auto_df) + false_negative(auto_df))

def F_measure(auto_df):
    return (2 * precision(auto_df) * recall(auto_df)) / (precision(auto_df) + recall(auto_df))
#..............................................................................................
#..............................................................................................


AUTO_dump = concatenate_auto_data('2022-12-20')

EV_dump = concatenate_ethovision_files()

probability = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

data = merge_AUTO_MANUAL_data(AUTO_dump, EV_dump)

# get list of IDs
ids =  list(set(data.filename))


#----------------------------------------------------------------
# Compute scores of classification quality for a range of probabilities
# for fn in [matthews_corrcoef,f1_score, fbeta_score]:
#     print(fn.__name__)
#     for p in probability:
#         p_column = "auto_grooming_" + str(p)
#         if fn.__name__ == 'fbeta_score':
#             print(p, '\t', fn(data.target_grooming, data[p_column], 0.4))
#         else:
#             print(p, '\t', fn(data.target_grooming, data[p_column]))


#----------------------------------------------------------------
# save RAW DATA with range of grooming classification according to probabilities and manual grooming
data.to_csv(os.path.join(ROOT_DIR,"2022-12-20_frame_wise_grooming.csv"))

#----------------------------------------------------------------
# find LIGHT ONSET periods
# get total number of frames grooming (manual or automated) during LIGHT ON
# save table to csv file
light_only = data[data["light"] > 0]
light_only = light_only.drop(['frame', 'timestamp', 'date_timestamp', 'X', 'Y', 'speed','P(grooming)'], axis=1)
light_only.groupby(by=["filename"]).sum()
table_out = light_only.groupby(by=["filename"]).sum()
table_out["%time_grooming(MANUAL)"] = table_out["target_grooming"] / table_out["light"] * 100
table_out["%time_grooming(AUTO)"] = table_out["auto_grooming_0.5"] / table_out["light"] * 100
table_out.to_csv(os.path.join(ROOT_DIR,"2022-12-20_quality_assessment.csv"))


#----------------------------------------------------------------
# get total number of frames grooming (manual or automated) throughout ENTIRE RECORDING
# save table to csv file
data.groupby(by=["filename"]).sum()
table_out_all = data.groupby(by=["filename"]).sum()
table_out_all["%time_grooming(MANUAL)"] = table_out_all["target_grooming"] / table_out_all["light"] * 100
table_out_all["%time_grooming(AUTO)"] = table_out_all["auto_grooming_0.5"] / table_out_all["light"] * 100
table_out_all.to_csv(os.path.join(ROOT_DIR,"2022-12-20_quality_assessment_all.csv"))




#----------------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(data.target_grooming, data["auto_grooming_0.5"]).ravel()
compute_frame_wise_accuracy(table_out)
compute_frame_wise_accuracy(table_out_all)

#----------------------------------------------------------------
# scatterplot for automated and manual grooming
f,ax = plt.subplots(figsize=(8, 8))
ax.scatter(table_out.target_grooming, table_out["auto_grooming_0.5"])
ax.plot([0, 200],[0, 200])








############################################################################




# SELECT FILE
file_number = 0

a_df = AUTO_dump[file_number]
e_df = EV_dump[file_number]


# merge dataframes
merge_auto_EV_dataframes(a_df, e_df)


# SELECT TRHRESHOLD
# threshold = 0.81

sensitivity_curve = []
G_mean_curve = []
precision_curve = []
recall_curve = []
f1_curve = []


for threshold in np.arange(0.5, 0.925, 0.025):

    # find grooming given p(grooming)
    compute_auto_grooming(a_df, threshold)
    # smooth grooming (max,3)
    smooth_function_grooming(a_df)
    sensitivity_curve.append(sensitivity(a_df))
    G_mean_curve.append(g_mean(a_df))
    precision_curve.append(precision(a_df))
    recall_curve.append(recall(a_df))
    f1_curve.append(F_measure(a_df))

#------------------------------------------
#>>> TODO : Use F_beta (or F_0.5 to give different weights to the imbalanced class labels)
#>>> TODO: if F_0.5 does not work, try to optimize G-Mean (looks good so far!)
#------------------------------------------
# hyperparam = np.max(precision_curve)
# hyperparam = np.max(f1_curve)
hyperparam = 0.5

figure(figsize=(12, 6))
compute_auto_grooming(a_df, hyperparam)
smooth_function_grooming(a_df)
plt.plot(a_df.auto_grooming)
plt.plot(a_df['P(grooming)'] - 2)
plt.plot(a_df.target_grooming - 1)
plt.plot(a_df.ev_light - 2)
plt.title("Threshold: p={}".format(hyperparam));
print('Sensitivity: {}'.format(sensitivity(a_df)))
print('G-Mean: {}'.format(g_mean(a_df)))
print('Precision: {}'.format(precision(a_df)))
print('Recall: {}'.format(recall(a_df)))
print('F1-Score: {}'.format(F_measure(a_df)))

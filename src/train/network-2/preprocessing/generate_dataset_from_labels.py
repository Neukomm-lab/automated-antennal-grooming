import pandas as pd
import numpy as np
import os

import cv2
import pandas as pd
import matplotlib.pyplot as plt

import unittest

"""
2022-08-24

Requires first the head coordinates retrieved from network1
It generates sequences of frames and store them in two folders:
- grooming
- no_grooming

Processes files individually
"""


def detect_state_change(array_in):
    '''convert one-hot encoded var to 1|0|-1'''
    _state_change = [0]
    tmp = list(np.diff(array_in))
    _state_change += tmp
    return(_state_change)


def find_frame_sequence_greater_than_N(array2D_IN, N):
    is_greater_than_N = array2D_IN[:, 1] - array2D_IN[:, 0] >= N
    return array2D_IN[is_greater_than_N]


def find_boundaries(array_in, flag):
    '''
    offset is always in between the onset boundaries defined by the array items
    '''
    
    if flag == 'onset':
        loop_over_this = range(0, len(array_in), 2)
    
    elif flag == 'offset':
        loop_over_this = range(1, len(array_in) - 1, 2)
    
    out = np.zeros([int(len(array_in) / 2), 2])
    
    for i in loop_over_this:
        ix = int(i/2)
        out[ix][0] = array_in[i]
        out[ix][1] = array_in[i + 1] -1
    
    return out


def sample_frames(frames_list_IN):
    '''
    # define sampling interval (0 <= index range <= N-10)
    '''
    out = []
    for frames_list in frames_list_IN:
        where_to_sample = np.arange(frames_list[0], frames_list[1] - 3)
        number_of_samples = int(np.floor(where_to_sample.shape[0] / 3))
        if number_of_samples > 0 :
            # print(np.random.choice(where_to_sample, number_of_samples, False))
            out.append(list(np.random.choice(where_to_sample, number_of_samples, False)))
        else:
            # print(np.random.choice(where_to_sample, 1, False))
            out.append(list(np.random.choice(where_to_sample, 1, False)))
    return [item for sublist in out for item in sublist]


rows_to_skip = np.arange(0, 36, 1).tolist()
rows_to_skip.append(37)


################################
# temp = "25_degrees"
# power = "10mW"
# subject_number = "11_1"
# subject_id = temp + "_" + power + "_" + subject_number

condition = "nmnat_after_injury"
subject_number = "13_6_6"
subject_id = condition + "_" + subject_number
type_of_behavior= 'no_grooming' # grooming / no_grooming

# ROOT_DIR = r'D:/data-sets/neukomm/2022-08-09'
# OUTPUT_DIR = r'D:/data-sets/neukomm/2022-08-09/train-validation-set/train'
# ROOT_DIR = r"Z:\DNF\GROUPS\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\TRAIN-VALIDATION-TEST-SETS\2022-08-24\training"
# OUTPUT_DIR = r"Z:\DNF\GROUPS\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\TRAIN-VALIDATION-TEST-SETS\2022-08-24\training\frames"

# ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\16 After injury test\Ctrl after injury"
# OUTPUT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\16 After injury test\Ctrl after injury\TRAIN-VAL-TEST-SETS\frames"

ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\16 After injury test\Nmnat after injury"
OUTPUT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\16 After injury test\Nmnat after injury\TRAIN-VAL-TEST-SETS\frames"

'''
Make sure that Ethovision files in folder `labels` have the header on rows 33 or 37 !!!
'''
################################


# read ethovision containing the  manual scoring
# d = pd.read_excel('D://data-sets//neukomm//2020-11-10//labels//' + subject_id + '.xlsx', sheet_name=0, skiprows=rows_to_skip, na_values = '-')
d = pd.read_excel(os.path.join(ROOT_DIR, 'Eth', subject_id + '.xlsx'), sheet_name=0, skiprows=rows_to_skip, na_values = '-')

# movie to sample
# movie_file = 'D://data-sets//neukomm//2020-11-10//01_uncropped_movies//all_together//' + subject_id + '_uncropped.avi'
movie_file = os.path.join(ROOT_DIR, 'Rec', subject_id + '.avi')

# head coordinates
# coordinates_file = 'D://switchdrive//Neuro-BAU//code//GIT//deep_behavior//CNN//build//results//head_coordinates//' + subject_id + '_uncropped.csv'
coordinates_file = os.path.join(ROOT_DIR, 'head_coordinates', subject_id + '.csv')

# define ONSET/OFFSET
state_change = detect_state_change(d['grooming'])

# find start of grooming sequences
is_onset = np.array(state_change) > 0

# get end of grooming epochs
is_offset = np.array(state_change) < 0

# get epochs onset (both grooming and no grooming)
epochs_ONSET = np.bitwise_or(is_onset, is_offset)
ON_epochs = d[epochs_ONSET]

epochs_ixs = ON_epochs.index

on = find_boundaries(epochs_ixs, 'onset')
off = find_boundaries(epochs_ixs, 'offset')

ON_selection = find_frame_sequence_greater_than_N(on, 5)
OFF_selection = find_frame_sequence_greater_than_N(off, 5)

# list of frames to extract and process
ON_frames = sample_frames(ON_selection)
OFF_frames = sample_frames(OFF_selection)

cap = cv2.VideoCapture(movie_file, cv2.CAP_ANY) # capturing the video from the given path
frameRate = cap.get(cv2.CAP_PROP_FPS) # int(np.ceil(frameRate))
number_of_frames_in_movie = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames in movie
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame width
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame height
frames_to_add = 10 # how many frames to use for the standard deviation projection
db = []

# coordinates = pd.read_csv(coordinates_file, sep='\t') # tab separated
coordinates = pd.read_csv(coordinates_file, sep=',') #comma separated

focus_size = 64 # !!!! half window

coordinates.X = coordinates.X * h / 256
coordinates.Y = coordinates.Y * w / 320

# debug ---
# cap.release()

if type_of_behavior == 'grooming':
    frame_sequence = ON_frames
elif type_of_behavior == "no_grooming":
    frame_sequence = OFF_frames
else:
    raise "type of behavior not specified correctly!!!"

while cap.isOpened():
    for start_frame in frame_sequence:
        start_frame = int(start_frame)
        empty_stack_of_frames = np.zeros([h, w, 3, frames_to_add])
        print('subject:{} >> stack | {} <-> {}'.format(subject_number,start_frame, start_frame + 10))
        x = int(coordinates.iloc[start_frame].X)
        y = int(coordinates.iloc[start_frame].Y)
        for i in range(10):
            cap.set(1, start_frame + i)
            # print(start_frame + i)
            ret, tmp_frame = cap.read()
            empty_stack_of_frames[:, :, :, i] = tmp_frame

        # Z projection along the 3rd dimension
        # frame = np.amax(empty_stack_of_frames, 3) # the max projection does not work for capturing grooming
        frame = np.std(empty_stack_of_frames, 3)
        crop_frame = frame[x - focus_size : x + focus_size, y - focus_size : y + focus_size]
        cv2.imwrite(os.path.join(OUTPUT_DIR, type_of_behavior, subject_id + '_' + str(start_frame) + '.png'), crop_frame)
        db.append(frame)
    
    cap.release()


########################### ||| TEST UNITS ||| ###########################

# class MainTest(unittest.TestCase):

#     def test_detect_state_change(self):
#         array_in = np.array([0, 0, 0, 1, 1, 1, 0])
#         self.assertEqual(detect_state_change(array_in), [0, 0, 0, 1, 0, 0, -1])
    
#     def test_find_frame_sequence_greater_than_N(self):
#         ''
#         array_in = np.array([[0, 10],[30, 50], [60, 69]])
#         expected_result = np.array([[0, 10],[30, 50]])
#         np.testing.assert_almost_equal(find_frame_sequence_greater_than_N(array_in, 10), expected_result)

#     def test_find_boundaries(self):
#         array_in = np.array([10, 20, 30, 50])
#         expected_result_onset = np.array([[10, 19], [30, 49]])
#         expected_result_offset = np.array([[20, 29], [0, 0]])
#         np.testing.assert_almost_equal(find_boundaries(array_in, 'onset'), expected_result_onset)
#         np.testing.assert_almost_equal(find_boundaries(array_in, 'offset'), expected_result_offset)

#     # def test_sample_frame(self):
#     #     np.random.seed(999)
#     #     array_in = np.array([[10, 30], [40, 60]])
#     #     expected_result = np.array([16, 12, 18, 48, 44, 41])
#     #     np.testing.assert_almost_equal(sample_frames(array_in), expected_result)


# if __name__ == '__main__':
#     unittest.main()


import os
import numpy as np
import pandas as pd
import cv2

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


"""
2022-09-27

> INPUT:    1. raw movie (.avi)
			2. head coordinates (.csv)*
			3. p(grooming) (.csv)

> OUTPUT:   1. original video with square centered on head (.avi)
			2. cropped video of the head (contains p value on top-right corner) (.avi)
			3. sequence of p values for plotting (.csv)

*The coordinates are smoothed so that the resulting crop is less jittering

CSV file:
- used to find the head's coordinates and to label the grooming episodes.
- is processed from the raw output of networks 1 and 2
- contains at least:
	- X, Y coordinates
	- Automated grooming label
"""


##################################################################################################
# ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\19 video production"
# ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\21 video paper"
ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\22 video without antennae"
filename = "ctrl_after_20_12_4_1"
##################################################################################################

font = cv2.FONT_HERSHEY_SIMPLEX # Font to be used for writing on frames (output movie)

movie_file = os.path.join(ROOT_DIR, 'videos', filename + ".avi")
head_coordinates_file = os.path.join(ROOT_DIR, 'head_coordinates', filename + ".csv")
p_grooming_file = os.path.join(ROOT_DIR, 'grooming_timeline', filename + ".csv")

head_coordinates = pd.read_csv(head_coordinates_file, sep=',')
p_grooming = pd.read_csv(p_grooming_file, sep=',')
p_grooming['light'] = head_coordinates['light']
p_grooming['frame'] = head_coordinates['frame']

# Create a function for interpolation
timestamps = head_coordinates["frame"]
x_coord = head_coordinates["X"]
y_coord = head_coordinates["Y"]

window_size = 15 # originally = 10

# Apply moving average to x and y coordinates separately
smoothed_x = np.convolve(x_coord, np.ones(window_size), 'same') / window_size
smoothed_y = np.convolve(y_coord, np.ones(window_size), 'same') / window_size

head_coordinates["x_smooth"] = smoothed_x
head_coordinates["y_smooth"] = smoothed_y

focus_size = 64 # half-size of head window

cap = cv2.VideoCapture(movie_file) # capturing the video from the given path
frameRate = cap.get(cv2.CAP_PROP_FPS) # int(np.ceil(frameRate))
number_of_frames_in_movie = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # how many frames in movie

# actual coordinates cannot be used since network 1 has been trained on a 256 x 210 frame set
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame width
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame height

# head_coordinates.X = head_coordinates.X * w / 320
# head_coordinates.Y = head_coordinates.Y * h / 256

head_coordinates.x_smooth = head_coordinates.x_smooth * w / 320
head_coordinates.y_smooth = head_coordinates.y_smooth * h / 256


codec = cv2.VideoWriter_fourcc(*"MJPG")

# 5 seconds before and after the red light
peri_seconds = int(frameRate * 5)

light_on_ixs = np.where(p_grooming.light.diff() == 1)[0] - peri_seconds
light_off_ixs = np.where(p_grooming.light.diff() < 0)[0] - 1 + peri_seconds

scale_percent = 300 # percent of original size (used to downscale coordinates for crop frame)

def extract_peristimulus_movie_data(start_frame, stop_frame):

	out_movie_file_name = os.path.join(ROOT_DIR, filename + "_" + str(start_frame) + "_" + str(stop_frame)+ "_out.avi")
	out_movie_file_crop_name = os.path.join(ROOT_DIR, filename + "_" + str(start_frame) + "_" + str(stop_frame)+ "_out_crop.avi")
	out_movie = cv2.VideoWriter(out_movie_file_name, codec , frameRate , (1280, 1024))
	out_movie_crop = cv2.VideoWriter(out_movie_file_crop_name, codec , frameRate , (384, 384))
	
	for i in range(start_frame, stop_frame):
		# x = int(head_coordinates.iloc[i].X)
		# y = int(head_coordinates.iloc[i].Y)

		x = int(head_coordinates.iloc[i].x_smooth)
		y = int(head_coordinates.iloc[i].y_smooth)
		
		cap.set(cv2.CAP_PROP_POS_FRAMES, i)
		ret, tmp_frame = cap.read()

		# crop frame to focus
		crop_frame = tmp_frame[x - focus_size : x + focus_size, y - focus_size : y + focus_size] / 255
		
		width = int(crop_frame.shape[1] * scale_percent / 100)
		height = int(crop_frame.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		crop_frame = cv2.resize(crop_frame, dim, interpolation = cv2.INTER_AREA) 
		
		print('frame {} '.format(i))

		g = np.uint8(crop_frame / np.max(crop_frame) * 255)
		
		# hack to prevent scientific notation to mess up with the string formatting
		
		if p_grooming["P(grooming)"][i] < 0.0001:
			print(p_grooming["P(grooming)"][i], '*************')
			p_grooming["P(grooming)"][i] = 0.0001


		cv2.putText(g, str(p_grooming["P(grooming)"][i])[:4], (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_4)
		out_movie_crop.write(g)

		# write bounding box on original movie
		image = cv2.rectangle(tmp_frame,  (y - focus_size,x - focus_size), (y + focus_size,x + focus_size), (255, 255, 255), 1)
		out_movie.write(image)

		###################### DEBUG <<<
		# if i == 100:
		# 	cap.release()
		# 	break
		######################

		p_grooming[["frame", "light","P(grooming)"]][start_frame:stop_frame].to_csv(os.path.join(ROOT_DIR, filename + "_" + str(start_frame) + "_" + str(stop_frame) + '_p_grooming_crop.csv'))

	out_movie.release()
	out_movie_crop.release()
	return crop_frame


for i in range(3):
	print(light_on_ixs[i], light_off_ixs[i])
	crop = extract_peristimulus_movie_data(light_on_ixs[i], light_off_ixs[i])

cap.release()

print("DONE")
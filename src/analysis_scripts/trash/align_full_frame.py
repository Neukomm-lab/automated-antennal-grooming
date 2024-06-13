import os
import numpy as np
import pandas as pd
import cv2

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


"""
2023-06-09
"""


##################################################################################################
ROOT_DIR = r"G:\RESEARCH\NEUROBAU\PUBLIC\LN\Maria\19 video production"
filename = "ctrl 10 mW 25 degrees_training_8_3_374_657_out_crop"
##################################################################################################

movie_file = os.path.join(ROOT_DIR, filename + ".avi")

cap = cv2.VideoCapture(movie_file) # capturing the video from the given path
frameRate = cap.get(cv2.CAP_PROP_FPS) # int(np.ceil(frameRate))
number_of_frames_in_movie = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # how many frames in movie

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame width
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame height

codec = cv2.VideoWriter_fourcc(*"MJPG")

out_movie_file_name = os.path.join(ROOT_DIR, filename + "_aligned.avi")
out_movie = cv2.VideoWriter(out_movie_file_name, codec , frameRate , (384, 384))


for i in range(200):
	# set template frame
	cap.set(cv2.CAP_PROP_POS_FRAMES, i)
	ret, template_frame_original = cap.read()

	# set to-be-aligned frame
	cap.set(cv2.CAP_PROP_POS_FRAMES, i+1)
	ret, align_frame = cap.read()

	# Convert both frames to grayscale
	template_frame = cv2.cvtColor(template_frame_original, cv2.COLOR_BGR2GRAY)
	align_frame = cv2.cvtColor(align_frame, cv2.COLOR_BGR2GRAY)
	

	height, width = align_frame.shape
	
	# Create ORB detector with 5000 features.
	orb_detector = cv2.ORB_create(5000)
	
	# Find keypoints and descriptors.
	# The first arg is the image, second arg is the mask
	#  (which is not required in this case).
	kp1, d1 = orb_detector.detectAndCompute(template_frame, None)
	kp2, d2 = orb_detector.detectAndCompute(align_frame, None)
	
	# Match features between the two images.
	# We create a Brute Force matcher with 
	# Hamming distance as measurement mode.
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	
	# Match the two sets of descriptors.
	matches = matcher.match(d1, d2)
	
	# Sort matches on the basis of their Hamming distance.
	matches.sort(key = lambda x: x.distance)
	
	# Take the top 90 % matches forward.
	matches = matches[:int(len(matches)*0.9)]
	no_of_matches = len(matches)
	
	# Define empty matrices of shape no_of_matches * 2.
	p1 = np.zeros((no_of_matches, 2))
	p2 = np.zeros((no_of_matches, 2))
	
	for i in range(len(matches)):
		p1[i, :] = kp1[matches[i].queryIdx].pt
		p2[i, :] = kp2[matches[i].trainIdx].pt
	
	# Find the homography matrix.
	homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 15.0)
	
	# Use this matrix to transform the
	# colored image wrt the reference image.
	transformed_img = cv2.warpPerspective(template_frame_original, homography, (width, height))

	out_movie.write(transformed_img)
	

out_movie.release()

cap.release()

print("DONE")
#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#### LIST OF COLOR CONVERSION METHODS IN OPENCV
#import cv2
#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#for i in flags:
#    print i

############ Intensity Filtering on video ##################

import cv2
import six
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

##################################
def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.uint16)
    out[mask] = np.concatenate(data)
    return out
    
##### get the list of colors from matplotlib
colors_ = list(six.iteritems(colors.cnames))

# Add the single letter colors.
for name, rgb in six.iteritems(colors.ColorConverter.colors):
    hex_ = colors.rgb2hex(rgb)
    colors_.append((name, hex_))

# Transform to hex color values.
hex_ = [color[1] for color in colors_]
# Get the rgb equivalent.
rgb = [colors.hex2color(color) for color in hex_]
rgb = np.asarray(rgb)*255
rgb = rgb.astype(int)
bgr = rgb[:,::-1]
##################################

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/Users/nbastian/Desktop/Drones_2016_inNatesMac/Videos/rhino1.avi')
framecount = 0
intlist = []

while True:

	_, frame = cap.read()
	frame = cv2.flip(frame,0)
	framegray = frame.copy()
	framegray = cv2.cvtColor(framegray, cv2.COLOR_RGB2GRAY)
	framegray[:,580:]=0
	framecount += 1

	# masking
	thresh = 155
	mask = np.where(framegray<=thresh)
	resim = framegray.copy()
	resim[mask] = 0
	
	#finding and drawing contours
	contours, hierarchy = cv2.findContours(resim,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	# Initialize empty list
	intlist_frame = []

	# For each list of contour points...
	for i in range(len(contours)):
		# Create a mask image that contains the contour filled in
		cimg = np.zeros_like(resim)
		cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
		
		# Access the image pixels and create a 1D numpy array then add to list
		pts = np.where(cimg == 255)
		intlist_frame.append(np.sum(framegray[pts]))
		cv2.drawContours(frame, contours, i, bgr[i], 1)		
    	
	intlist.append(intlist_frame)
	#cv2.drawContours(frame, contours, -1, (0,0,255), 1)
	
	#plotting
	cv2.imshow('frame', frame)
	cv2.imshow('framegray', framegray)
	cv2.imshow('resim', resim)

	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
	    
cap.release()
cv2.destroyAllWindows()

#Making intlist into array to plot
intlist = np.asarray(intlist)
array = numpy_fillna(intlist)

makeplot = raw_input('do you want plot individual contours?: (Y/N)')
if makeplot=='Y':
	#plot
	for i in np.arange(array.shape[1]):
	
		plt.plot(np.arange(array.shape[0]),array[:,i], label='contour'+str(i))
		plt.legend()
		plt.show()
		
makeplot = raw_input('do you want plot?: (Y/N)')
if makeplot=='Y':
	#plot
	for i in np.arange(array.shape[1]):
	
		plt.plot(np.arange(array.shape[0]),array[:,i], label='contour'+str(i))
		
	plt.legend()	
	plt.show()

	



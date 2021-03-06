#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-



#### LIST OF COLOR CONVERSION METHODS IN OPENCV
#import cv2
#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#for i in flags:
#    print i

#Last modified 10/Mar/2017 MR
#Script modifyed to work with tif files instead of avi video
#GOAL: work with absolute temperatures only available as
#.tif or .csv outputs from the camera software


############ Intensity Filtering on video ##################
import cv2
import six
import PIL
import os
import time
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
    out = np.zeros(mask.shape, dtype=np.uint64)
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
# alternate list of colors
colors_py = [(255,0,0), (0,0,255), (0,255,0), (0,0,0), (255,165,0),(255,255,255),(0,165,255), (255,0,255),(128,0,128)] #['red','blue','green','black','orange','white','cyan','magenta','purple']# for matplotlib this is RGB
colors_cv2 = [(0,0,255), (255,0,0), (0,255,0), (0,0,0),(0,165,255),(255,255,255), (255,165,0), (255,0,255), (128,0,128)] #for cv2 thi is BGR


####################################
####### START IMAGE ANALYSIS #######
####################################

# ONLINE/VIDEO READING

#cap = cv2.VideoCapture(0)
#cap=cv2.VideoCapture('/Users/nbastian/Desktop/Drones_2016_inNatesMac/Videos/ApolloandOsi/Apolloandosi1.avi') #/Users/nbastian/Desktop/Drones_2016_inNatesMac/Videos/rhino1.avi


# LIST TIF FILES FOR READING
directory = '/Volumes/TheBananaStand/Osiframes' 
dirlist=os.listdir(directory)
filelist = sorted([s for s in dirlist if ".tif" in s])

# OUTPUT for saving video
#fourcc = cv2.cv.CV_FOURCC(*'MJPG')#(*'avc1')
#output = cv2.VideoWriter(directory+'/'+'Osi_autoplot.avi',-1, 15,(640,512), True)


framecount = 0
intlist = []
arealist = []
meancont = []
framelist = []
cimglist = []
plt.ion()

#Frame loop
##################

#while True:
#	_, frame = cap.read()
## 	frame = cv2.flip(frame,0)
#	framegray = frame.copy()
#	framegray = cv2.cvtColor(framegray, cv2.COLOR_RGB2GRAY)

# Initialize empty list
intlist_frame = []
arealist_frame = []
meancont_frame = []
arealistno_frame = []
output = [] 
avgheight = []

# with open ('/Volumes/TheBananaStand/Flight_2_GPS.csv', 'r') as f:
#     GPS = []                              # create an empty list
#     for line in f:
#         GPS.append([float(i) for i in line.strip('\n').split(',')])
#     GPS = np.transpose(GPS)
#     #print(GPS[:][:])                     # [column][row]
# 
# with open ('/Volumes/TheBananaStand/Flight_2_Vid_Data.csv', 'r') as f:
#     Data = [] #create an empty list
#     for line in f:
#         Data.append([float(i) for i in line.strip('\n').split(',')])
#     Data = np.transpose(Data)
#     #print(Data[:][:])                    # [column][row]

for ifile,file in enumerate(filelist):
	frame = cv2.imread(directory+'/'+file, -1)

	if ifile==0:

		maxframe = frame.max()

		minframe = frame.min()
# 	frame = cv2.flip(frame,0)
	framegray = frame.copy()
	framegray = framegray*(1/256.) #resim0.convertTo(resim0,CV_8U,1/256.)
	framegray = np.uint8(framegray)	
#	framegray = cv2.cvtColor(framegray, cv2.COLOR_RGB2GRAY)
#	framegray[:,630:]=0 #block the colorbar
	framegray[:350,:] = 0 #block top 60 in y
#	framegray[300:, :] = 0 #block bottom in y
	framecount = ifile #+= 1
# 	height = []
# 	x = np.where((ifile+1) == Data) #finding the framecount in Data
# 	y = Data[1,x[1]] #accessing the timestamp for that frame
# 
# 	for j in range(len(GPS[0]) - 1):
# 		z = np.where(GPS[1,:] == y) #finding the instances where the timestamp in Data matches those in GPS
# 		z = np.asarray(z) #converting to a numpy array for easier transformations
# 	
# 	for i in np.nditer(z):
# 		h = GPS[0,i] #finding heights corresponding to timestamps
# 		height.append(h)
# 		s = sum(height)
# 		avg = s/len(height) #calculating average height from multiple height values
		
    		

	# adaptive threshold and masking
 	thresh = np.percentile(framegray.ravel(),99.) #real adaptive threshold, which didn't work very well
	mask = np.where(framegray<=thresh)
	resim0 = framegray.copy()
	resim0[mask] = 0
# 	resim0 = resim0*(1/256.) #resim0.convertTo(resim0,CV_8U,1/256.)
# 	resim0 = np.uint8(resim0)


	#finding and drawing contours	
	#adding dilation to get rid of noise before finding contours
	kernel = np.ones((10,10),np.uint8)
	resim = cv2.dilate(resim0,kernel,iterations = 1) # resim0 after dilation 
	image, contours, hierarchy = cv2.findContours(resim,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	#Clean contour list from noisy contours and append	
	contours = [contour for contour in contours if cv2.contourArea(contour)>=110.]		
#	cimglist_frame = []
#	contposlist = []
#	#Countour loop 1
#	for i in range(len(contours)):
#		# Create a mask image that contains the contour filled in
#		cimg = np.zeros_like(resim)
#		cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
#		cimglist_frame.append(cimg)			
#
#		# Comparing pixels in contour with ones in prev frame
#		if framecount>1:
#			crosslist = []
#			for cimg2 in cimglist[framecount-2]:
#				cross = len(np.where((cimg==255)&(cimg2==255))[0])
#				crosslist.append(cross)	
#			contpos = np.argmax(crosslist)
#			contposlist.append(contpos)	
#			
#	cimglist.append(cimglist_frame)


	# For each list of contour points...
	for i in range(len(contours)):
		# Create a mask image that contains the contour filled in
		cimg = np.zeros_like(resim)
		cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
		# Access the image pixels, count them, and create a 1D numpy array then add to list
		pts = np.where(cimg == 255)		
		area = len(pts[0])
		# Removing small noisy contours
# 		if area <= 100:
# 			del contours[i]	
		arealist_frame.append(area)
		intlist_frame.append(np.sum(framegray[pts]))

		plt.figure('Plot')
		plt.grid()
		plt.plot(intlist_frame, color='k')
		plt.xlabel('Frame No.')
		plt.ylabel('Integrated Intensity')
		plt.show()
		plt.pause(0.001) 

		   
		
	#ordering everything with high int -> low int
	#arealist_frame = np.asarray(arealist_frame)
	#intlist_frame = np.asarray(intlist_frame)
	#contours = np.asarray(contours)
	
	#order = intlist_frame.argsort()[::-1]
	#contours = contours[order]		
	#intlist_frame = intlist_frame[order]
	#arealist_frame = arealist_frame[order]
	
	#contours = contours.tolist()
	
	#drawing contours
	for i in range(len(contours)): 
		#cv2.drawContours(frame, contours, i, bgr[i], 1)#with full list of colors
		cv2.drawContours(frame, contours, i,255, 1)#short&defined list of colours colors_cv2[i]

	intlist.append(intlist_frame)
	arealist.append(arealist_frame)
	framelist.append(frame)
# 	avgheight.append(avg)

	#cv2.drawContours(frame, contours, -1, (0,0,255), 1)


	#saving video
	#frameforvid = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
	#frameforvid = np.uint8(frameforvid)

 	#output.write(frameforvid)


	#plotting
	plt.figure('Track')
	plt.imshow(frame, vmin=minframe,vmax=maxframe)
	plt.title('Frame '+str(framecount))
	plt.axis('off')
	plt.show()
	#image.set_data(frame)
	#fig.suptitle('Frame '+str(framecount))
	plt.draw()
# 	plt.pause()
#	time.sleep(0.2)
# 	cv2.imshow('frame', frame)
# 	cv2.imshow('framegray', framegray)
# 	cv2.imshow('resim', resim)



	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


#cap.release()
#output.release()
#cv2.destroyAllWindows()


#Making intlist into array to plot
intlist = np.asarray(intlist)
intarray = numpy_fillna(intlist)


#same with areas list
arealist = np.asarray(arealist)
arearray = numpy_fillna(arealist)


# makeplot = raw_input('do you want plot individual contours?: (Y/N)')
# if makeplot=='Y':
# 	#plot
#	plt.figure()
# 	for i in np.arange(intarray.shape[1]):
# 	    plt.plot(np.arange(intarray.shape[0]),intarray[:,i], label='contour'+str(i))
# 	    plt.legend()
# 	    plt.show()

# 
# makeplot = raw_input('do you want plot?: (Y/N)')
# if makeplot=='Y':
# 	#plot
# 	sns.set(color_codes=True)
# 	sns.set(style='ticks')
# 	sns.color_palette("colorblind")
# 	sns.despine()
# 	plt.figure(2)
# 	for i in np.arange(intarray.shape[1]):
# 	    plt.plot(np.arange(intarray.shape[0]),intarray[:,i], label='contour'+str(i),color=bgr[i])
# 	    plt.plot(np.arange(intarray.shape[0]),intarray[:,i], label='contour'+str(i),color=colors_py[i])
# 	    plt.grid()


#		
#	plt.legend(loc=2)	
#	plt.show()


# makeplot = raw_input('do you want plot surface brightness of countours?: (Y/N)')
# if makeplot=='Y':
# 	#plot
#	plt.figure()
# 	for i in np.arange(arearray.shape[1]):	
# 		plt.plot(np.arange(arearray.shape[0]),intarray[:,i]/arearray[:,i], label='Surf. Bright. count. '+str(i)) 		
# 	plt.legend()
# 	plt.show()

"""______________________________________________________________________________________
| Author   : Md Akib Hosen Khan								|
|	     4th year student, dept of Information & Communication Engineering  	|
|	     Islamic University, Bangladesh						|
| Date     : August 25, 2018								|
| OS       : Windows 10 x64 machine							|
| Language : Python (V-2.7)								|							|
| Library  : OpenCV (V-3.0)								|							|
|_______________________________________________________________________________________|"""


import cv2
import numpy as np
import os, sys, time

# initialize the camera
camera = cv2.VideoCapture(0)
camera.set(3,320)
camera.set(4,240)
time.sleep(0.5)

# define master frame as none
master = None

while True:
	# for grab a frame
	(grabbed, frame0) = camera.read()

	# end of the feed
	if not grabbed:
		break

	# To convert BGR to Gray frame
	frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

	# To blur the frame1 that is gray frame
	frame2 = cv2.GaussianBlur(frame1, (21, 21), 0)

	# initialize master
	if master is None:
		master = frame2
		continue

	# calculate absolute difference between master and frame2 
	frame3 = cv2.absdiff(master, frame2)

	# apply threshold on frame3
	frame4 = cv2.threshold(frame3, 15, 255, cv2.THRESH_BINARY)[1]

	# dilate the threshold image 
	kernel = np.ones((5,5), np.uint8)
	frame5 = cv2.dilate(frame4, kernel, iterations=4)

	# find contours on thershold image
	_, contours, _ =cv2.findContours(frame5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# make contour frame
	frame6 = frame0.copy()

	# loop over the contours
	for c in contours:
		
		if cv2.contourArea(c) < 400:   
			continue


		# contour data 
		M = cv2.moments(c) 
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		x, y, w, h =cv2.boundingRect(c)
		rx = x + int(w/2)
		ry = y + int(h/2)
		ca = cv2.contourArea(c)

		# plot contours
		cv2.drawContours(frame6, [c], 0, (1,1,1), 2)
		cv2.rectangle(frame6, (x,y), (x+w, y+h), (255,255,255), 2)



	# update master
	master = frame2

	# display
	cv2.imshow('Frame0 : Original ', frame0)
	cv2.imshow('Frame1 : Gray ', frame1)
	cv2.imshow('Frame2 : Blur ', frame2)
	cv2.imshow('Frame3 : Delta ', frame3)
	cv2.imshow('Frame4 : Threshold ', frame4)
	cv2.imshow('Frame5 : Dialated ', frame5)
	cv2.imshow('Frame6 : Contours ', frame6)


	key = cv2.waitKey(1)
	if key == 27:
		break

# release camera
camera.release()

# close all windows
cv2.destroyAllWindows()

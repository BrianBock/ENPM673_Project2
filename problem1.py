
# Import required packages
import numpy as np
import cv2
#import imutils
import time

write_to_video=False


# Define the codec and initialize the output file
if write_to_video:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	today = time.strftime("%m-%d__%H.%M.%S")
	videoname=str(today)
	fps_out = 29
	out = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))
	print("Writing to Video, Please Wait")

# open the video specified by video_src
video = cv2.VideoCapture('media/Night Drive - 2689.mp4') 
start_frame=0
# move the video to the start frame and adjust the counter
video.set(1,start_frame)
count = start_frame

while(video.isOpened()):
	ret, frame = video.read() # ret is false if the video cannot be read
	if ret:

		# cv2.imshow("Original",frame)
		# equ = cv2.equalizeHist(frame)

		# cv2.imshow("Hist",equ)
		# https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
		img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# equalize the histogram of the Y channel
		img_hsv[:,:,0] = cv2.equalizeHist(img_hsv[:,:,0])

		# convert the YUV image back to RGB format
		img_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

		cv2.imshow('Color input image', frame)
		cv2.imshow('Histogram equalized', img_output)












	else:
		# if ret is False release the video which will exit the loop
		video.release()

	# if the user presses 'q' release the video which will exit the loop
	if cv2.waitKey(1) == ord('q'):
		video.release()

	if write_to_video:
		out.release()




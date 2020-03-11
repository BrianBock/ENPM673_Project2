
# Import required packages
import numpy as np
import cv2
import imutils
import time
# import argparse	
		

write_to_video=True
show_output=False


# Define the codec and initialize the output file
if write_to_video:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	today = time.strftime("%m-%d__%H.%M.%S")
	videoname=str(today)
	fps_out = 29
	out = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))
	print("Writing to Video, please Wait")


# open the video specified by video_src
video = cv2.VideoCapture('../media/Problem1/Night Drive - 2689.mp4') 
start_frame=0
# move the video to the start frame and adjust the counter
video.set(1,start_frame)
count = start_frame



#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
		# equ = cv2.equalizeHist(highcontrast)



def adjust_brightness(image,brightness, show_output,write_to_video):
	frame32=np.asarray(frame,dtype="int32")
	bright_image=frame32+brightness
	bright_image=np.clip(bright_image,0,255)
	bright_image=np.asarray(bright_image,dtype="uint8")
	if show_output:
		cv2.imshow("Original",imutils.resize(frame,600))
		cv2.imshow("Brighter (+"+str(brightness)+")",imutils.resize(bright_image,600))
	if write_to_video:
		out.write(bright_image)
	
	return bright_image



def adjust_contrast(image,contrast,show_output,write_to_video):
	frame32=np.asarray(frame,dtype="int32")
	high_contrast=frame32*contrast
	high_contrast=np.clip(high_contrast,0,255)
	high_contrast=np.asarray(high_contrast,dtype="uint8")
	if show_output:
		cv2.imshow("Original",imutils.resize(frame,600))
		cv2.imshow("Contrast (*"+str(contrast)+")",imutils.resize(high_contrast,600))
	if write_to_video:
		out.write(high_contrast)
	
	return high_contrast





while(video.isOpened()):
	ret, frame = video.read() # ret is false if the video cannot be read
	if ret:

		# Increase Brightness
		brightness=50
		# bright_image=adjust_brightness(frame,brightness,show_output,write_to_video)
		# out.write(bright_image)
		

		# Increase Contrast
		high_contrast=adjust_contrast(frame,4,show_output,write_to_video)
		# out.write(high_contrast)



		#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
		# construct the argument parse and parse the arguments
		# ap = argparse.ArgumentParser()
		# ap.add_argument("-i", "--image", required=True,
		# 	help="path to input image")
		# args = vars(ap.parse_args())
		# # # load the original image
		# original = frame#cv2.imread(args["image"])

		# # loop over various values of gamma
		# for gamma in np.arange(0.0, 3.5, 0.5):
		# 	# ignore when gamma is 1 (there will be no change to the image)
		# 	if gamma == 1:
		# 		continue
		# 	# apply gamma correction and show the images
		# 	gamma = 2.5
		# 	adjusted = adjust_gamma(original, gamma=gamma)
		# 	# cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
		# 	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
		# 	cv2.imshow("Images", adjusted)
			# cv2.waitKey(0)



		# https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
		# img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# # equalize the histogram of the Y channel
		# img_hsv[:,:,0] = cv2.equalizeHist(img_hsv[:,:,0])

		# # convert the YUV image back to RGB format
		# img_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

		# cv2.imshow('Color input image', frame)
		# cv2.imshow('Histogram equalized', img_output)












	else:
		# if ret is False release the video which will exit the loop
		video.release()

	# if the user presses 'q' release the video which will exit the loop
	if cv2.waitKey(1) == ord('q'):
		video.release()

if write_to_video:
	out.release()




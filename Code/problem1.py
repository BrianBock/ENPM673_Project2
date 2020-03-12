
# Import required packages
import numpy as np
import cv2
import imutils
import time
# import argparse	
		

write_to_video=False
show_output=True
change_brightness=True
change_contrast=False
change_histYUV=False
change_histHSV=False
change_gamma=False

brightness=100 # int between 0-255
contrast=9 # int between 0-15
gamma = 2


# open the video specified by video_src
video = cv2.VideoCapture('../media/Problem1/Night Drive - 2689.mp4')
start_frame=0#365+24
# move the video to the start frame and adjust the counter
video.set(1,start_frame)
count = start_frame


# Define the codec and initialize the output file
if write_to_video:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	today = time.strftime("%m-%d__%H.%M.%S")
	videoname=str(today)
	fps_out = 29
	out = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))
	print("Writing to video, please wait...")





#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(image, gamma, show_output, write_to_video):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	new_img=cv2.LUT(image, table)
	if show_output:
		cv2.imshow("Original",imutils.resize(frame,900))
		cv2.imshow("Gamma (+"+str(gamma)+")",imutils.resize(new_img,900))
	if write_to_video:
		out.write(new_img)

	return new_img
		# equ = cv2.equalizeHist(highcontrast)



def adjust_brightness(frame,brightness, show_output,write_to_video):
	frame32=np.asarray(frame,dtype="int32")
	bright_image=frame32+brightness
	bright_image=np.clip(bright_image,0,255)
	bright_image=np.asarray(bright_image,dtype="uint8")
	if show_output:
		cv2.imshow("Original",imutils.resize(frame,900))
		cv2.imshow("Brighter (+"+str(brightness)+")",imutils.resize(bright_image,900))
	if write_to_video:
		out.write(bright_image)
	
	return bright_image



def adjust_contrast(frame,contrast,show_output,write_to_video):
	frame32=np.asarray(frame,dtype="int32")
	high_contrast=frame32*contrast
	high_contrast=np.clip(high_contrast,0,255)
	high_contrast=np.asarray(high_contrast,dtype="uint8")
	if show_output:
		cv2.imshow("Original",imutils.resize(frame,900))
		cv2.imshow("Contrast (*"+str(contrast)+")",imutils.resize(high_contrast,900))
	if write_to_video:
		out.write(high_contrast)
	
	return high_contrast


def equalize_Hist_YUV(frame,show_output,write_to_video):
	# https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
	img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	if show_output:
		cv2.imshow('Color input image', frame)
		cv2.imshow('Histogram equalized', img_output)

	if write_to_video:
		out.write(img_output)

	return img_output


def equalize_Hist_HSV(frame,show_output,write_to_video):
	# https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
	img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# equalize the histogram of the V channel
	img_HSV[:,:,0] = cv2.equalizeHist(img_HSV[:,:,0])

	# convert the HSV image back to RGB format
	img_output = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)

	if show_output:
		cv2.imshow('Color input image', frame)
		cv2.imshow('Histogram equalized', img_output)

	if write_to_video:
		out.write(img_output)

	return img_output




while(video.isOpened()):
	ret, frame = video.read() # ret is false if the video cannot be read
	if ret:

	# Increase Brightness
		if change_brightness:
			bright_image=adjust_brightness(frame,brightness,show_output,write_to_video)
			# cv2.imwrite("brightness"+str(brightness)+".jpg",bright_image)
			# cv2.waitKey(0)


		# out.write(bright_image)
		

	# Increase Contrast
		if change_contrast:
			high_contrast=adjust_contrast(frame,contrast,show_output,write_to_video)
			# cv2.imwrite("contrast"+str(contrast)+".jpg",high_contrast)
			# cv2.imwrite("original.jpg",frame)
			# out.write(high_contrast)
			# cv2.waitKey(0)

	# Equalize Histogram in YUV Space
		if change_histYUV:
			hist_eq=equalize_Hist_YUV(frame,show_output,write_to_video)

	# Equalize Histogram in HSV Space
		if change_histHSV:
			hist_eq=equalize_Hist_HSV(frame,show_output,write_to_video)

		# cv2.imwrite("HisteqHSV.jpg",hist_eq)
		# cv2.waitKey(0)

	# Adjust Gamma
		if change_gamma:
			new_gamma = adjust_gamma(frame, gamma,show_output,write_to_video)
			# cv2.imshow("Gamma "+str(gamma), new_gamma)
		# cv2.imwrite("Gamma"+str(gamma)+".jpg", new_gamma)
		# cv2.waitKey(0)















	else:
		# if ret is False release the video which will exit the loop
		video.release()
		print("End of video")

	# if the user presses 'q' release the video which will exit the loop
	if cv2.waitKey(1) == ord('q'):
		video.release()

if write_to_video:
	out.release()




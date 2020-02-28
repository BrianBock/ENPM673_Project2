
# Import required packages
import numpy as np
import cv2
import time




write_to_video=False
data_set=2


# Define the codec and initialize the output file
if write_to_video:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	today = time.strftime("%m-%d__%H.%M.%S")
	videoname="problem2_data"+str(data_set)+"_"+str(today)
	fps_out = 29
	out = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))
	print("Writing to Video, Please Wait")




def getFrame(data_set,imagenum):
	if data_set==1:
		filepath="media/Problem2/data_1/data/"
		imagepath=filepath+('0000000000'+str(imagenum))[-10:]+'.png'
		image=cv2.imread(imagepath)
		return image

	elif data_set==2:
		videopath="media/Problem2/data_2/challenge_video.mp4"
		video = cv2.VideoCapture(videopath) 
		# move the video to the start frame and adjust the counter
		video.set(1,imagenum)
		ret, frame = video.read() # ret is false if the video cannot be read
		if ret:
			return frame
		else:
			print("Frame "+str(imagenum)+" exceeds video length or you've reached the end of video. Quitting...")
			exit()



image=getFrame(data_set,500)
cv2.imshow("Image",image)
cv2.waitKey(0)






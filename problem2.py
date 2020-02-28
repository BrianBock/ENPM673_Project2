
# Import required packages
import numpy as np
import cv2
import time
from p2functions import*



write_to_video=False
data_set=1


# Define the codec and initialize the output file
if write_to_video:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	today = time.strftime("%m-%d__%H.%M.%S")
	videoname="problem2_data"+str(data_set)+"_"+str(today)
	fps_out = 29
	out = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))
	print("Writing to Video, Please Wait")








image=getFrame(data_set,50)

pts=(514,318),(195,512),(939,512), (767,320) #region of road

# H=homography(pts,500)
# square_road=fastwarp(image,H,500,500)

cv2.imshow("Image",image)
cv2.waitKey(0)






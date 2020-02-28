
# Import required packages
import numpy as np
import cv2
import time
import imutils
from p2functions import*



write_to_video=False
data_set=1

if data_set != 1 and data_set!= 2:
	print("Invalid data_set selected. Please use data_set=1 or data_set=2. Quitting")
	exit()


# Define the codec and initialize the output file
if write_to_video:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	today = time.strftime("%m-%d__%H.%M.%S")
	videoname="problem2_data"+str(data_set)+"_"+str(today)
	fps_out = 29
	out = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))
	print("Writing to Video, Please Wait")








image=getFrame(data_set,50)




# Define the HSV bounds that make the just road lines really clear (determined experimentally)
if data_set==1:
	colorLower = (0, 0, 201)
	colorUpper = (255, 49, 255)
	pts=[(470,318),(770,320),(950,512),(140,512)] #region of road

elif data_set==2:
	colorLower = (0, 123, 183)
	colorUpper = (255, 255, 255)


H=homography(pts,500)
square_road=fastwarp(np.linalg.inv(H),image,500,500)
cv2.imshow("Road",square_road)
cv2.waitKey(0)


# # Mask the top half of the video so only the road half remains
# mask=np.zeros((image.shape[0],image.shape[1]),dtype="uint8")
# mask_points=np.array([[0,0],[1392,0],[1392,225],[0,255]],dtype=np.int32)
# cv2.fillConvexPoly(mask,pts,-1,255,-1)
# masked=cv2.bitwise_and(image,image,mask=mask)
# cv2.show(masked)
# cv2.waitKey(0)

# Convert the image to HSV space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Thresh the image based on the HSV max/min values
hsv_binary_image=cv2.inRange(hsv_image, colorLower, colorUpper)

# Blur the image a little bit
img_blur=cv2.GaussianBlur(hsv_binary_image,(15,15),0)

# Find the edges
all_cnts, hierarchy = cv2.findContours(img_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw the edges
edges=cv2.drawContours(image.copy(),all_cnts,-1,(0,0,255), 5)
cv2.imshow("Edges", imutils.resize(edges,width=640))

cv2.imshow("Image",img_blur)
cv2.waitKey(0)







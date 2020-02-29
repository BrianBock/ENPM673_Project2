
# Import required packages
import numpy as np
import cv2
import time
import imutils
from p2functions import*
import matplotlib
import matplotlib.pyplot as plt


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








image=getFrame(data_set,60)


dst_height = 500
dst_width = 1000

# Define the HSV bounds that make the just road lines really clear (determined experimentally)
if data_set==1:
	colorLower = (0, 0, 201)
	colorUpper = (255, 49, 255)
	src_pts=np.float32([(570,275),(715,275),(950,512),(140,512)]).reshape(-1,1,2) #region of road #(0,225),(1392,225)
	dst_pts=np.float32([(.25*dst_width,0),(.75*dst_width,0),(.75*dst_width,dst_height),(.25*dst_width,dst_height)]).reshape(-1,1,2)#,(0,0),(dst_width,0)

elif data_set==2:
	colorLower = (0, 123, 183)
	colorUpper = (255, 255, 255)



#H=homography(pts,height,width)
H=cv2.findHomography(src_pts,dst_pts)[0]
print(H[0])
cv2.imshow("Road-before",image)

square_road=fastwarp(np.linalg.inv(H),image,dst_height,dst_width)
cv2.imshow("Road-after",square_road)
cv2.waitKey(0)


# # Mask the top half of the video so only the road half remains
# mask=np.zeros((image.shape[0],image.shape[1]),dtype="uint8")
# mask_points=np.array([[0,0],[1392,0],[1392,225],[0,255]],dtype=np.int32)
# cv2.fillConvexPoly(mask,pts,-1,255,-1)
# masked=cv2.bitwise_and(image,image,mask=mask)
# cv2.show(masked)
# cv2.waitKey(0)

# Convert the image to HSV space
hsv_image = cv2.cvtColor(square_road, cv2.COLOR_BGR2HSV)

# Thresh the image based on the HSV max/min values
hsv_binary_image=cv2.inRange(hsv_image, colorLower, colorUpper)
#cv2.imshow("HSV",hsv_binary_image)

# Blur the image a little bit
img_blur=cv2.GaussianBlur(hsv_binary_image,(15,15),0)

# Find the edges
#edges = cv2.Canny(img_blur, 5, 10)
all_cnts, hierarchy = cv2.findContours(img_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw the edges
blank=np.zeros((square_road.shape[0],square_road.shape[1]),np.uint8)
filledContours=cv2.drawContours(blank,all_cnts,-1,(255), -1)
cv2.imshow("filledContours", imutils.resize(filledContours,width=1000))

inds=np.nonzero(filledContours)
# print(inds)
# for x in inds[1]:
# 	if len(count_array)%10:
# 		count=0
whitebin=[]
stepsize=200
steps=dst_width//stepsize
for i in range (steps):
	whitebin.append(0)

for x in inds[1]:
	i=0
	for x_c in range (stepsize,dst_width,stepsize):
		if(x<x_c):
			whitebin[i]+=1
			break
		i+=1

# print(whitebin)
# stepsize=10
# for x in range(0,dst_width,stepsize):
# 	count=0
# 	for y in range(0,dst_height):
# 		if filledContours[y][x]==255:
# 			count[x]+=1


# plt.plot(whitebin,'.')
# #plt.plot(x,y,label='Original Data')
# plt.xlabel('X Bins')
# plt.ylabel('White count')
# plt.title("My Histogram")
# plt.show()

maxbinstep=whitebin.index(max(whitebin))
print(maxbinstep)

leftlanex=[]
leftlaney=[]
for i,x in enumerate(inds[1]):
	y=inds[0][i]
	if(x<(maxbinstep*stepsize)):
		leftlanex.append(x)
		leftlaney.append(y)

line=np.polyfit(leftlanex,leftlaney,2)
xpts=np.linspace(0,dst_width)
pts=[]
for x in xpts:
	y=int(line[0]*x**2+line[1]*x+line[2])
	# print(x,y)
	pts.append((x,y))
	# print(pts)
print(pts)

pts=np.array(pts,np.int32)
pts=pts.reshape((-1,1,2))

mylines=cv2.polylines(square_road,[pts],False,(0,0,255),15)


# line=cv2.line(square_road,)
# cv2.imshow("Line",line)

cv2.imshow("Lines",mylines)
cv2.waitKey(0)







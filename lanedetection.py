import numpy as np
import cv2
import matplotlib.pyplot as plt
from p2functions import*

image=getFrame(1,204)
# cv2.imshow('Original Frame',image)

dst_h = 1000
dst_w = 1000

#region of road - determined experimentally
src_corners = [(585,275),(715,275),(950,512),(140,512)] 
src_pts=np.float32(src_corners).reshape(-1,1,2) #change form for homography computation

# How the source corners will match up in the destination image
# x1 = x3, x2 = x4 so the lanes will be parallel
top_offset = 370 # can be modified to adjust position of car in front
dst_corners = [(.25*dst_w,-top_offset),(.75*dst_w,-top_offset),(.75*dst_w,dst_h),(.25*dst_w,dst_h)]
dst_pts=np.float32(dst_corners).reshape(-1,1,2) #change form for homography computation 

# Find the homography matrix for the conversion
H=cv2.findHomography(dst_pts,src_pts)[0]

# Warp the image
square_road=fastwarp(H,image,dst_h,dst_w)
# cv2.imshow('Square Road',square_road)

# Convert the image to HSV space
hsv_image = cv2.cvtColor(square_road, cv2.COLOR_BGR2HSV)

# Thresh the image based on the HSV max/min values
colorLower = (0, 0, 201)
colorUpper = (255, 49, 255)
hsv_binary_image=cv2.inRange(hsv_image, colorLower, colorUpper)

# Blur the image a little bit
img_blur=cv2.GaussianBlur(hsv_binary_image,(15,15),0)

# Find the edges
edges = cv2.Canny(img_blur, 5, 10)
cnts, hierarchy = cv2.findContours(img_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Fill in edges on blank image
blank=np.zeros((square_road.shape[0],square_road.shape[1]),np.uint8)
filledContours=cv2.drawContours(blank,cnts,-1,255, -1)
# cv2.imshow('Filled Contours',filledContours)

# Find all points that correspond to a white pixel
inds=np.nonzero(filledContours)

# Create a histogram of the number of nonzero pixels
num_pixels,bins = np.histogram(inds[1],bins=dst_w,range=(0,dst_w))

# Find the first peak
lane_width = 100
peak1 = np.argmax(num_pixels)

# Set the region around the first peak to zero
for i in range(-lane_width//2,lane_width//2):
    num_pixels[peak1+i] = 0

# Find the second peak
peak2 = np.argmax(num_pixels)

# Collect all the points that are within a lane-width of the two peaks
left_pts,right_pts = [], []
for i,x in enumerate(inds[1]):
    y = inds[0][i]
    if peak1-lane_width//2 <= x <= peak1+lane_width//2:
        right_pts.append([x,y])
    elif peak2-lane_width//2 <= x <= peak2+lane_width//2:
        left_pts.append([x,y])

left_pts = np.asarray(left_pts)
right_pts = np.asarray(right_pts)

lane1=np.polyfit(left_pts[:,1],left_pts[:,0],1)
lane2=np.polyfit(right_pts[:,1],right_pts[:,0],1)

x = [lane1[1],dst_h*lane1[0]+lane1[1],dst_h*lane2[0]+lane2[1],lane2[1]]
y = [0,dst_h,dst_h,0]
X_s = np.array([x, y, np.ones_like(x)])

sX_c = H.dot(X_s)
X_c = sX_c/sX_c[-1]

corners = []
for i in range(4):
    corners.append((X_c[0][i],X_c[1][i]))

contour = np.array(corners, dtype=np.int32)

lane_img = image.copy()
cv2.drawContours(lane_img,[contour],-1,(0,255,0),-1)

alpha = 0.3
overlay = cv2.addWeighted(image, 1-alpha, lane_img, alpha, 0) 

cv2.imshow('Lane',overlay)
cv2.waitKey(0)

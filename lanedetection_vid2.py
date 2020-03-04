import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from p2functions import*

write_to_video = False
show_output = True

for k in range(1):

    if write_to_video and k == 0:
        print('Writing to Video, Please Wait')
        print('Frame ' + str(k+1) + ' of ?')
        image=getFrame(2,k)
        k += 1
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        frame_size = (image.shape[1], image.shape[0])
        videoname='lane_detection'
        fps_out = 15
        out = cv2.VideoWriter(str(videoname)+".mp4", fourcc, fps_out, frame_size)
    else:
        print('Frame ' + str(k+1) + ' of ?')
        image=getFrame(2,k)
        k += 1

    dst_h = 500
    dst_w = 500

    #region of road - determined experimentally
    src_corners = [(615,460),(745,460),(1175,720),(250,720)] 
    src_pts=np.float32(src_corners).reshape(-1,1,2) #change form for homography computation

    # How the source corners will match up in the destination image
    # x1 = x3, x2 = x4 so the lanes will be parallel
    top_offset = 370 # can be modified to adjust position of car in front
    dst_corners = [(.1*dst_w,-top_offset),(.9*dst_w,-top_offset),(.9*dst_w,dst_h),(.1*dst_w,dst_h)]
    dst_pts=np.float32(dst_corners).reshape(-1,1,2) #change form for homography computation 

    # Find the homography matrix for the conversion
    H=cv2.findHomography(dst_pts,src_pts)[0]

    # Warp the image
    square_road=fastwarp(H,image,dst_h,dst_w)
    # cv2.imshow('Square Road',square_road)

    # Convert the image to HSV space
    hsv_image = cv2.cvtColor(square_road, cv2.COLOR_BGR2HSV)

    # Thresh the image based on the HSV max/min values
    colorLower = (0, 0, 171)
    colorUpper = (91, 255, 216)
    hsv_binary_image=cv2.inRange(hsv_image, colorLower, colorUpper)
    cv2.imshow('Filled Contours',hsv_binary_image)
    cv2.waitKey(0)

    # Blur the image a little bit
    img_blur=cv2.GaussianBlur(hsv_binary_image,(15,15),0)

    # Find the edges
    edges = cv2.Canny(img_blur, 5, 10)
    cnts, hierarchy = cv2.findContours(img_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Fill in edges on blank image
    blank=np.zeros((square_road.shape[0],square_road.shape[1]),np.uint8)
    filledContours=cv2.drawContours(blank,cnts,-1,255, -1)
    # cv2.imshow('Filled Contours',filledContours)
    # cv2.waitKey(0)

    # Find all points that correspond to a white pixel
    inds=np.nonzero(filledContours)

    # Create a histogram of the number of nonzero pixels
    num_pixels,bins = np.histogram(inds[1],bins=dst_w,range=(0,dst_w))

    peaks = []
    peaks = signal.find_peaks_cwt(num_pixels, np.arange(1,25))
    
    peak_vals = []
    for peak in peaks:
        peak_vals.append(num_pixels[peak])
        
    max1_ind = peak_vals.index(max(peak_vals))
    temp = peak_vals.copy()
    temp[max1_ind] = 0
    max2_ind = temp.index(max(temp))

    peak1 = peaks[max1_ind]
    peak2 = peaks[max2_ind]

    # plt.hist(inds[1],bins=dst_w,range=(0,dst_w))
    
    # for peak in peaks:
    #     plt.vlines(peak,0,dst_h)
    # plt.vlines(peak1,0,dst_h,'r')
    # plt.vlines(peak2,0,dst_h,'r')
    # plt.show()

    # Collect all the points that are within a lane-width of the two peaks
    lane_width = 100
    left_pts,right_pts = [], []
    for i,x in enumerate(inds[1]):
        y = inds[0][i]
        if peak1-lane_width//2 <= x <= peak1+lane_width//2:
            right_pts.append([x,y])
        elif peak2-lane_width//2 <= x <= peak2+lane_width//2:
            left_pts.append([x,y])

    left_pts = np.asarray(left_pts)
    right_pts = np.asarray(right_pts)

    # Find coefficients for the best fit lines for the points
    lane1=np.polyfit(left_pts[:,1],left_pts[:,0],1)
    lane2=np.polyfit(right_pts[:,1],right_pts[:,0],1)

    # Find the corners of the polygon that bounds the lane in the squared image
    x = [lane1[1],dst_h*lane1[0]+lane1[1],dst_h*lane2[0]+lane2[1],lane2[1]]
    y = [0,dst_h,dst_h,0]

    line_color = (0,0,0)
    arrow_color = (0,255,255)
    lane_color = (0,255,0)
    lane=np.zeros((square_road.shape[0],square_road.shape[1],3),np.uint8)
    corners = []
    for i in range(4):
        corners.append((int(x[i]),int(y[i])))
    contour = np.array(corners, dtype=np.int32)
    cv2.drawContours(lane,[contour],-1,lane_color,-1)
    cv2.line(lane,corners[0],corners[1],line_color, 20)
    cv2.line(lane,corners[2],corners[3],line_color, 20)

    mid = dst_h//2
    mid_x = int(((mid*lane1[0]+lane1[1])+(mid*lane2[0]+lane2[1]))/2)

    arrow_length = dst_h//10
    for i in range(5):
        start_y = i*(2*arrow_length)+20
        end_y = start_y+arrow_length
        cv2.arrowedLine(lane, (mid_x,end_y), (mid_x,start_y), arrow_color, 10,tipLength = 0.5) 
    # cv2.imshow('Lane',lane)
    # cv2.waitKey(0)
    
    H_inv = np.linalg.inv(H)
    warped_lane = fastwarp(H_inv,lane,image.shape[0],image.shape[1])
    # cv2.imshow('Warped Lane',warped_lane)
    # cv2.waitKey(0)

    # Find the location of those corners in the camera frame
    X_s = np.array([x, y, np.ones_like(x)])
    sX_c = H.dot(X_s)
    X_c = sX_c/sX_c[-1]

    corners = []
    for i in range(4):
        corners.append((X_c[0][i],X_c[1][i]))

    # Overlay a green polygon that represents the lane
    contour = np.array(corners, dtype=np.int32)
    lane_img = image.copy()
    cv2.drawContours(lane_img,[contour],-1,(0,0,0),-1)

    lane_img = cv2.bitwise_or(lane_img,warped_lane)

    alpha = 0.5 # determine the transparency of the polygon
    overlay = cv2.addWeighted(image, 1-alpha, lane_img, alpha, 0) 

    if show_output:
        cv2.imshow('Lane',overlay)
        if cv2.waitKey(1) == ord('q'):
            break

    if write_to_video:
        out.write(overlay)

    # cv2.waitKey(0)

if write_to_video:
    out.release()
import numpy as np
import cv2
from scipy import signal

class Road:
    def __init__ (self,dst_size,data_set,initial_frame_num):
        self.dst_w=dst_size[0]
        self.dst_h=dst_size[1]
        self.frame_num=initial_frame_num
        if data_set==1:
            self.HSVLower=(0, 0, 220)
            self.HSVUpper=(255, 49, 255)
        elif data_set==2:
            self.HSVLower = (0, 0, 171)
            self.HSVUpper = (91, 255, 216)
        else:
            print("Invalid data_set entered. Quitting...")
            exit()
        self.get_frame()

    def compute_homography(self):
        road_points=self.road_points
        h=self.h
        w=self.w
        #Define the eight points to compute the homography matrix
        x,y = [],[]
        for point in road_points:
            x.append(point[0])
            y.append(point[1])

        xp = [0,w,w,0]
        yp = [0,0,h,h]

        n = 9
        m = 8
        A = np.empty([m, n])

        val = 0
        for row in range(0,m):
            if (row%2) == 0:
                A[row,0] = -x[val]
                A[row,1] = -y[val]
                A[row,2] = -1
                A[row,3] = 0
                A[row,4] = 0
                A[row,5] = 0
                A[row,6] = x[val]*xp[val]
                A[row,7] = y[val]*xp[val]
                A[row,8] = xp[val]

            else:
                A[row,0] = 0
                A[row,1] = 0
                A[row,2] = 0
                A[row,3] = -x[val]
                A[row,4] = -y[val]
                A[row,5] = -1
                A[row,6] = x[val]*yp[val]
                A[row,7] = y[val]*yp[val]
                A[row,8] = yp[val]
                val += 1

        U,S,V = np.linalg.svd(A)
        # x is equivalent to the eigenvector column of V that corresponds to the 
        # smallest singular value. A*x ~ 0
        x = V[-1]

        # reshape x into H
        H = np.reshape(x,[3,3])
        return H


    def get_frame(self):
        if self.data_set==1:
            filepath="media/Problem2/data_1/data/"
            imagepath=filepath+('0000000000'+str(self.frame_num))[-10:]+'.png'
            frame=cv2.imread(imagepath)
            if frame is None:
                print("Unable to import '"+str(imagepath)+"'. Quitting...")
                exit()
            self.frame=frame

        elif self.data_set==2:
            videopath="media/Problem2/data_2/challenge_video.mp4"
            video = cv2.VideoCapture(videopath)
            # move the video to the start frame and adjust the counter
            video.set(1,self.frame_num)
            ret, frame = video.read() # ret is false if the video cannot be read
            if ret:
                # cv2.imwrite('frame.jpg',frame)
                self.frame=frame
            else:
                print("Frame "+str(self.frame_num)+" exceeds video length or you've reached the end of video. Quitting...")
                exit()


    def warp(self):
        H=self.H
        src=self.src
        h=self.h
        w=self.w
         # create indices of the destination image and linearize them
        indy, indx = np.indices((h, w), dtype=np.float32)
        lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

        # warp the coordinates of src to those of true_dst
        map_ind = H.dot(lin_homg_ind)
        map_x, map_y = map_ind[:-1]/map_ind[-1] 
        map_x = map_x.reshape(h,w).astype(np.float32)
        map_y = map_y.reshape(h,w).astype(np.float32)

        new_img = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
        
        return new_img

    
    def fill_contours(self):
        # Convert the image to HSV space
        hsv_image = cv2.cvtColor(self.top_down_image, cv2.COLOR_BGR2HSV)

        # Thresh the image based on the HSV max/min values
        hsv_binary_image=cv2.inRange(hsv_image, self.HSVLower, self.HSVUpper)
        # cv2.imshow('Filled Contours',hsv_binary_image)
        # cv2.waitKey(0)

        # Blur the image a little bit
        img_blur=cv2.GaussianBlur(hsv_binary_image,(15,15),0)

        # Find the edges
        edges = cv2.Canny(img_blur, 5, 10)
        cnts, hierarchy = cv2.findContours(img_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Fill in edges on blank image
        blank=np.zeros((square_road.shape[0],square_road.shape[1]),np.uint8)
        self.filled_image=cv2.drawContours(blank,cnts,-1,255, -1)


    def find_peaks(self):
        errorbound=50
        # Find all points that correspond to a white pixel
        inds=np.nonzero(self.filled_image)

        # Create a histogram of the number of nonzero pixels
        num_pixels,bins = np.histogram(inds[1],bins=self.dst_w,range=(0,self.dst_w))

        peaks = signal.find_peaks_cwt(num_pixels, np.arange(1,50))
        
        peak_vals = []
        for peak in peaks:
            peak_vals.append(num_pixels[peak])
            # plt.vlines(peak,0,dst_h)

        max1_ind = peak_vals.index(max(peak_vals))
        temp = peak_vals.copy()
        temp[max1_ind] = 0
        max2_ind = peak_vals.index(max(temp))

        if len(peaks)==0: # No peaks detected
            right_peak=self.right_peak
            left_peak=self.left_peak

        elif len(peaks)==1: #only one peak detected
        #Determine if the peak is closer to the right or left
            if peak[max1_ind]>=self.dst_w/2:
                right_peak=peak[max1_ind]
                if abs(right_peak-self.right_peak)<errorbound: #lane is within our margin of error
                    pass
                else:
                    right_peak=self.right_peak
                left_peak=self.left_peak
            else:
                left_peak=peak[max1_ind]
                if abs(left_peak-self.left_peak)<errorbound: #lane is within our margin of error
                    pass
                else:
                    left_peak=self.left_peak
                left_peak=self.left_peak

        else: # At least 2 peaks found
            big_peaks=[peaks[max1_ind],peaks[max2_ind]]
            big_peaks.sort()

            left_peak=big_peaks[0]
            right_peak=big_peaks[1]

        self.right_peak=right_peak
        self.left_peak=left_peak


    def find_lane_line(self):
         # Collect all the points that are within a lane-width of the two peaks
        line_width = 100
        left_pts,right_pts = [], []
        for i,x in enumerate(inds[1]):
            y = inds[0][i]
            if self.left_peak-line_width//2 <= x <= self.left_peak+line_width//2:
                left_pts.append([x,y])
            elif self.right_peak-line_width//2 <= x <= self.right_peak+line_width//2:
                right_pts.append([x,y])

        left_pts = np.asarray(left_pts)
        right_pts = np.asarray(right_pts)

        # Find coefficients for the best fit lines for the points
        self.left_lane_coeffs=np.polyfit(left_pts[:,1],left_pts[:,0],1)
        self.right_lane_coeffs=np.polyfit(right_pts[:,1],right_pts[:,0],1)



    def overlay(self):

        # Find the corners of the polygon that bounds the lane in the squared image
        x = [self.left_lane_coeffs[1],self.dst_h*self.left_lane_coeffs[0]+self.left_lane_coeffs[1],self.dst_h*self.right_lane_coeffs[0]+self.right_lane_coeffs[1],self.right_lane_coeffs[1]]
        y = [0,self.dst_h,self.dst_h,0]

        line_color = (0,0,0)
        arrow_color = (0,255,255)
        lane_color = (0,255,0)
        lane_image=np.zeros((self.dst_h,self.dst_w,3),np.uint8)
        corners = []
        for i in range(4):
            corners.append((int(x[i]),int(y[i])))
        contour = np.array(corners, dtype=np.int32)
        cv2.drawContours(lane_image,[contour],-1,lane_color,-1)
        cv2.line(lane_image,corners[0],corners[1],line_color, 20)
        cv2.line(lane_image,corners[2],corners[3],line_color, 20)

        mid = self.dst_h//2
        mid_x = int(((mid*self.left_lane_coeffs[0]+self.left_lane_coeffs[1])+(mid*self.right_lane_coeffs[0]+self.right_lane_coeffs[1]))/2)

        arrow_length = self.dst_h//10
        for i in range(5):
            start_y = i*(2*arrow_length)+20
            end_y = start_y+arrow_length
            cv2.arrowedLine(lane_image, (mid_x,end_y), (mid_x,start_y), arrow_color, 10,tipLength = 0.5) 
        # cv2.imshow('Lane',lane)
        # cv2.waitKey(0)
        
        H_inv = np.linalg.inv(self.H)
        warped_lane = self.warp(H_inv,lane_image,self.frame.shape[0],self.frame.shape[1])
        # cv2.imshow('Warped Lane',warped_lane)
        # cv2.waitKey(0)

        # Find the location of those corners in the camera frame
        X_s = np.array([x, y, np.ones_like(x)])
        sX_c = self.H.dot(X_s)
        X_c = sX_c/sX_c[-1]

        corners = []
        for i in range(4):
            corners.append((X_c[0][i],X_c[1][i]))

        # Overlay a green polygon that represents the lane
        contour = np.array(corners, dtype=np.int32)
        lane_overlay_img = self.frame.copy()
        cv2.drawContours(lane_overlay_img,[contour],-1,(0,0,0),-1)

        lane_overlay_img = cv2.bitwise_or(lane_overlay_img,warped_lane)

        alpha = 0.5 # determine the transparency of the polygon
        overlay = cv2.addWeighted(self.frame, 1-alpha, lane_overlay_img, alpha, 0) 


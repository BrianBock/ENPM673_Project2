import numpy as np
import cv2
from scipy import signal
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Road:
    def __init__ (self,dst_size,data_set,initial_frame_num):
        self.dst_w=dst_size[0]
        self.dst_h=dst_size[1]
        self.video=True
        self.data_set=data_set
        self.count = 0
        if data_set==1:
            self.HSVLower=(0, 0, 190)
            self.HSVUpper=(255, 255, 255)
            #region of road - determined experimentally
            src_corners = [(585,275),(715,275),(950,512),(140,512)] 
            self.errorbound=50

        elif data_set==2:
            self.HSVLower=(0, 8, 170)
            self.HSVUpper=(255, 255, 255)
            #region of road - determined experimentally
            src_corners = [(610,480),(730,480),(1020,680),(240,680)] 
            self.errorbound=25
        else:
            print("Invalid data_set entered. Quitting...")
            exit()
        self.get_frame(initial_frame_num)

        src_pts=np.float32(src_corners).reshape(-1,1,2) #change form for homography computation

        # How the source corners will match up in the destination image
        # x1 = x3, x2 = x4 so the lanes will be parallel
        dst_corners = [(.2*self.dst_w,0),(.8*self.dst_w,0),(.8*self.dst_w,self.dst_h),(.2*self.dst_w,self.dst_h)]
        dst_pts=np.float32(dst_corners).reshape(-1,1,2) #change form for homography computation 
        self.H=cv2.findHomography(dst_pts,src_pts)[0]




    def get_frame(self,frame_num):
        if self.data_set==1:
            filepath="media/Problem2/data_1/data/"
            imagepath=filepath+('0000000000'+str(frame_num))[-10:]+'.png'
            frame=cv2.imread(imagepath)
            if frame is None:
                # print("Unable to import '"+str(imagepath)+"'. Quitting...")
                self.video=False
            self.frame=frame

        elif self.data_set==2:
            videopath="media/Problem2/data_2/challenge_video.mp4"
            video = cv2.VideoCapture(videopath)
            # move the video to the start frame and adjust the counter
            video.set(1,frame_num)
            ret, frame = video.read() # ret is false if the video cannot be read
            if ret:
                # cv2.imwrite('frame.jpg',frame)
                self.frame=frame
            else:
                # print("Frame "+str(self.frame_num)+" exceeds video length or you've reached the end of video. Quitting...")
                self.video=False


    def warp(self,H,src,h,w):
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
        # cv2.imwrite('top_down_image.jpg',self.top_down_image)
        # cv2.imshow("Top Down",self.top_down_image)
        # cv2.imwrite('top_down.jpg',self.top_down_image)
        # cv2.waitKey(0)

        # Convert the image to HSV space
        hsv_image = cv2.cvtColor(self.top_down_image, cv2.COLOR_BGR2HSV)
        # cv2.imwrite("hsv.jpg",hsv_image)
        # cv2.imshow("hsv",hsv_image)
        # cv2.waitKey(0)

        # Thresh the image based on the HSV max/min values
        hsv_binary_image=cv2.inRange(hsv_image, self.HSVLower, self.HSVUpper)
        # cv2.imshow('HSV Binary',hsv_binary_image)
        # cv2.imwrite('hsv_binary_image.jpg',hsv_binary_image)
        # cv2.waitKey(0)

        # Blur the image a little bit
        img_blur=cv2.GaussianBlur(hsv_binary_image,(15,15),0)
        # cv2.imshow('GaussianBlur',img_blur)
        # cv2.imwrite('GaussianBlur.jpg',img_blur)
        # cv2.waitKey(0)

        # Find the edges
        edges = cv2.Canny(img_blur, 5, 10)
        cnts, hierarchy = cv2.findContours(img_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('Edges',edges)
        # cv2.waitKey(0)

        # Fill in edges on blank image
        blank=np.zeros((self.top_down_image.shape[0],self.top_down_image.shape[1]),np.uint8)
        self.filled_image=cv2.drawContours(blank,cnts,-1,255, -1)
        # cv2.imshow("Contours",self.filled_image)
        # cv2.imwrite("filled_contours.jpg",self.filled_image)
        # cv2.waitKey(0)

        plt.sca(self.axs[0,0])
        plt.axis('off')
        plt.title('Original Top Down')
        plt.imshow(cv2.cvtColor(self.top_down_image,cv2.COLOR_BGR2RGB))

        plt.sca(self.axs[0,1])
        plt.axis('off')
        plt.imshow(cv2.cvtColor(hsv_binary_image,cv2.COLOR_GRAY2RGB))
        plt.title('After HSV Thresh')
        
        plt.sca(self.axs[1,0])
        plt.axis('off')
        plt.title('Edge detection')
        plt.imshow(cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))

        plt.sca(self.axs[1,1])
        plt.title('Filled Contours')
        plt.axis('off')
        plt.imshow(cv2.cvtColor(self.filled_image,cv2.COLOR_GRAY2RGB))


    def find_peaks(self):
        # Find all points that correspond to a white pixel
        self.inds=np.nonzero(self.filled_image)
        num_pixels,bins = np.histogram(self.inds[1],bins=self.dst_w,range=(0,self.dst_w))

        plt.sca(self.axs[2,0])
        plt.axis('off')
        plt.title('Histogram Overlay')
        # Create a histogram of the number of nonzero pixels
        plt.imshow(cv2.cvtColor(self.filled_image,cv2.COLOR_GRAY2RGB),extent=[0,500,0,500])
        plt.hist(self.inds[1],bins=self.dst_w,range=(0,self.dst_w),color='yellow',histtype='step',lw=2)
        # plt.xlabel('X Column')
        # plt.ylabel('Count of White Pixels')
        # plt.title("Histogram Lane Detection")
        # fig, ax = plt.subplots()

        # try:
        peaks = signal.find_peaks_cwt(num_pixels, np.arange(1,25))
        # except RuntimeWarning:
        #     peaks = []

        if len(peaks)==0: # No peaks detected
            if self.count == 0:
                print('Could not find both lanes on first frame')
                exit()
            else:
                right_peak = self.right_peak
                left_peak = self.left_peak
                self.found_right_lane = False
                self.found_left_lane = False

        elif len(peaks)==1: #only one peak detected
            if self.count == 0:
                print('Could not find both lanes on first frame')
                exit()
            else:
                if peaks[0]>=self.dst_w/2 and abs(peaks[0]-self.right_peak)<self.errorbound:
                    right_peak = peaks[0]
                    left_peak = self.left_peak
                    self.found_right_lane = True
                    self.found_left_lane = False
                elif peaks[0]<=self.dst_w/2 and abs(peaks[0]-self.left_peak)<self.errorbound:
                    left_peak = peaks[0]
                    right_peak = self.right_peak
                    self.found_right_lane = False
                    self.found_left_lane = True
                else:
                    right_peak = self.right_peak
                    left_peak = self.left_peak
                    self.found_right_lane = False
                    self.found_left_lane = False
            
        else: # At least 2 peaks found
            # Find the value associated with the peak 
            peak_vals = []
            for peak in peaks:
                peak_vals.append(num_pixels[peak])

            # Find the two highest peaks
            max1_ind = peak_vals.index(max(peak_vals))
            temp = peak_vals.copy()
            temp[max1_ind] = 0
            max2_ind = peak_vals.index(max(temp))
            big_peaks=[peaks[max1_ind],peaks[max2_ind]]
            big_peaks.sort()

            if self.count == 0: #Assume first peaks found are correct
                left_peak = big_peaks[0]
                right_peak = big_peaks[1]
                self.found_left_lane = True
                self.found_right_lane = True
            else:
                found_left_peak = False
                found_right_peak = False
                for peak in peaks:
                    if abs(peak-self.left_peak) <= self.errorbound:
                        found_left_peak = True
                        left_peak = peak
                    if abs(peak-self.right_peak) <= self.errorbound: 
                        found_right_peak = True
                        right_peak = peak
                
                if found_left_peak and found_right_peak:
                    self.found_left_lane = True
                    self.found_right_lane = True
                elif found_right_peak:
                    self.found_right_lane = True
                    self.found_left_lane = False
                    left_peak = self.left_peak
                    self.found_left_lane = False  
                elif found_left_peak:
                    self.found_left_lane = True
                    self.found_right_lane = False
                    right_peak = self.right_peak
                    self.found_right_lane = False
                else:
                    right_peak = self.right_peak
                    left_peak = self.left_peak
                    self.found_right_lane = False
                    self.found_left_lane = False                

        self.right_peak=right_peak
        self.left_peak=left_peak

        plt.sca(self.axs[2,1])
        plt.axis('off')
        plt.title('Peaks')
        plt.imshow(cv2.cvtColor(self.filled_image,cv2.COLOR_GRAY2RGB))
        if self.found_left_lane:
            plt.vlines(self.left_peak,0,self.dst_h,'r')

        if self.found_right_lane:
            plt.vlines(self.right_peak,0,self.dst_h,'b')

    def find_lane_lines(self):
         # Collect all the points that are within a lane-width of the two peaks
        line_width = 100
        left_pts,right_pts = [], []
        for i,x in enumerate(self.inds[1]):
            y = self.inds[0][i]
            if self.left_peak-line_width//2 <= x <= self.left_peak+line_width//2:
                left_pts.append([x,y])
            elif self.right_peak-line_width//2 <= x <= self.right_peak+line_width//2:
                right_pts.append([x,y])

        left_pts = np.asarray(left_pts)
        right_pts = np.asarray(right_pts)

        # Find coefficients for the best fit lines for the points
        if not self.found_right_lane and not self.found_left_lane:
            pass
        elif not self.found_left_lane:
            self.right_lane_coeffs=np.polyfit(right_pts[:,1],right_pts[:,0],1)
        elif not self.found_right_lane:
            self.left_lane_coeffs=np.polyfit(left_pts[:,1],left_pts[:,0],1)
        else:
            self.left_lane_coeffs=np.polyfit(left_pts[:,1],left_pts[:,0],1)
            self.right_lane_coeffs=np.polyfit(right_pts[:,1],right_pts[:,0],1)

        x1 = self.left_lane_coeffs[1],self.dst_h*self.left_lane_coeffs[0]+self.left_lane_coeffs[1]
        x2 = self.dst_h*self.right_lane_coeffs[0]+self.right_lane_coeffs[1],self.right_lane_coeffs[1]
        y1 = 0,self.dst_h
        y2 = self.dst_h,0

        plt.sca(self.axs[3,0])
        plt.axis('off')
        plt.title('Lane Lines Overlay')
        plt.imshow(cv2.cvtColor(self.top_down_image,cv2.COLOR_BGR2RGB))
        plt.plot(x1,y1,linewidth=2,c='red')
        plt.plot(x2,y2,linewidth=2,c='red')




    def make_overlay(self):
        # Find the corners of the polygon that bounds the lane in the squared image
        x = [self.left_lane_coeffs[1],self.dst_h*self.left_lane_coeffs[0]+self.left_lane_coeffs[1],self.dst_h*self.right_lane_coeffs[0]+self.right_lane_coeffs[1],self.right_lane_coeffs[1]]
        y = [0,self.dst_h,self.dst_h,0]

        line_color = (0,0,255)
        arrow_color = (0,255,255)
        lane_color = (0,255,0)
        line_thick = 10
        lane_image=np.zeros((self.dst_h,self.dst_w,3),np.uint8)
        corners = []
        for i in range(4):
            corners.append((int(x[i]),int(y[i])))
        contour = np.array(corners, dtype=np.int32)
        cv2.drawContours(lane_image,[contour],-1,lane_color,-1)
        cv2.line(lane_image,corners[0],corners[1],line_color, line_thick)
        cv2.line(lane_image,corners[2],corners[3],line_color, line_thick)
        # cv2.imshow("Lane lines",lane_image)


        mid = self.dst_h//2
        mid_x = int(((mid*self.left_lane_coeffs[0]+self.left_lane_coeffs[1])+(mid*self.right_lane_coeffs[0]+self.right_lane_coeffs[1]))/2)

        arrow_length = self.dst_h//10
        for i in range(5):
            start_y = i*(2*arrow_length)+20
            end_y = start_y+arrow_length
            cv2.arrowedLine(lane_image, (mid_x,end_y), (mid_x,start_y), arrow_color, 10,tipLength = 0.5) 
        # cv2.imshow('Arrows',lane_image)
        # cv2.imwrite('arrows.jpg',lane_image)
        # cv2.waitKey(0)
        
        H_inv = np.linalg.inv(self.H)
        warped_lane = self.warp(H_inv,lane_image,self.frame.shape[0],self.frame.shape[1])
        # cv2.imshow('Warped Lane',warped_lane)
        # cv2.waitKey(0)

        # Find the location of those corners in the camera frame
        x[0]-=line_thick/2
        x[1]-=line_thick/2
        x[2]+=line_thick/2
        x[3]+=line_thick/2

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
        # cv2.imshow("Lane Overlay",lane_overlay_img)
        # cv2.waitKey(0)

        lane_overlay_img = cv2.bitwise_or(lane_overlay_img,warped_lane)
        # cv2.imshow("Lane Overlay",lane_overlay_img)
        # cv2.waitKey(0)

        alpha = 0.5 # determine the transparency of the polygon
        self.overlay = cv2.addWeighted(self.frame, 1-alpha, lane_overlay_img, alpha, 0) 


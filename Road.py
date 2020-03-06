import numpy as np
import cv2
from scipy import signal

class Road:
    def __init__ (self,dst_size,data_set,initial_frame_num):
        self.dst_w=dst_size[0]
        self.dst_h=dst_size[1]
        self.video=True
        self.data_set=data_set
        if data_set==1:
            self.HSVLower=(0, 0, 220)
            self.HSVUpper=(255, 49, 255)
            #region of road - determined experimentally
            src_corners = [(585,275),(715,275),(950,512),(140,512)] 

        elif data_set==2:
            self.HSVLower = (0, 0, 171)
            self.HSVUpper = (91, 255, 216)
            #region of road - determined experimentally
            src_corners = [(615,460),(745,460),(1175,720),(250,720)] 
        else:
            print("Invalid data_set entered. Quitting...")
            exit()
        self.get_frame(initial_frame_num)

        src_pts=np.float32(src_corners).reshape(-1,1,2) #change form for homography computation

        # How the source corners will match up in the destination image
        # x1 = x3, x2 = x4 so the lanes will be parallel
        dst_corners = [(.1*self.dst_w,0),(.9*self.dst_w,0),(.9*self.dst_w,self.dst_h),(.1*self.dst_w,self.dst_h)]
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
        blank=np.zeros((self.top_down_image.shape[0],self.top_down_image.shape[1]),np.uint8)
        self.filled_image=cv2.drawContours(blank,cnts,-1,255, -1)


    def find_peaks(self):
        errorbound=50
        # Find all points that correspond to a white pixel
        self.inds=np.nonzero(self.filled_image)

        # Create a histogram of the number of nonzero pixels
        num_pixels,bins = np.histogram(self.inds[1],bins=self.dst_w,range=(0,self.dst_w))

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
        self.left_lane_coeffs=np.polyfit(left_pts[:,1],left_pts[:,0],1)
        self.right_lane_coeffs=np.polyfit(right_pts[:,1],right_pts[:,0],1)



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


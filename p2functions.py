import cv2
import numpy as np


def getFrame(data_set,imagenum):
    if data_set==1:
        filepath="media/Problem2/data_1/data/"
        imagepath=filepath+('0000000000'+str(imagenum))[-10:]+'.png'
        image=cv2.imread(imagepath)
        if image is None:
            print("Unable to import '"+str(imagepath)+"'. Quitting...")
            exit()
        return image

    elif data_set==2:
        videopath="media/Problem2/data_2/challenge_video.mp4"
        video = cv2.VideoCapture(videopath)
        # move the video to the start frame and adjust the counter
        video.set(1,imagenum)
        ret, frame = video.read() # ret is false if the video cannot be read
        if ret:
            # cv2.imwrite('frame.jpg',frame)
            return frame
        else:
            print("Frame "+str(imagenum)+" exceeds video length or you've reached the end of video. Quitting...")
            exit()





def homography(road_points,h,w):
    #Define the eight points to compute the homography matrix
    x,y = [],[]
    for point in road_points:
        x.append(point[0])
        y.append(point[1])

    xp = [0,w,w,0]
    yp = [0,0,h,h]

    # dr = x[2]-x[3]

    # w1 = w - x[3]
    # xp=[w1,w1+dr,w1+dr,w1]
    # yp=[0,0,h,h]

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


def warp(H,src,h,w):
    # create indices of the destination image and linearize them
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1] 
    map_x = map_x.reshape(h,w).astype(np.float32)
    map_y = map_y.reshape(h,w).astype(np.float32)

    # generate new image
    new_img = np.zeros((h,w,3),dtype="uint8")

    # map_x[map_x>=src.shape[1]] = -1
    # map_x[map_x<0] = -1
    # map_y[map_y>=src.shape[0]] = -1
    # map_x[map_y<0] = -1

    for new_x in range(w):
        for new_y in range(h):
            x = int(map_x[new_y,new_x])
            y = int(map_y[new_y,new_x])

            if x == -1 or y == -1:
                pass
            else:
                new_img[new_y,new_x] = src[y,x]

    return new_img


def fastwarp(H,src,h,w):
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
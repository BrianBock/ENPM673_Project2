
data_set=1
dst_size=(500,500)
write_to_video=False

# Construct object Road
road=Road(dst_size,data_set,0)

# Read in the frame


# Prepare video output
if write_to_video and k == 0:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frame_size = (road.frame.shape[1], road.frame.shape[0])
    videoname='lane_detection'
    fps_out = 15
    out = cv2.VideoWriter(str(videoname)+".mp4", fourcc, fps_out, frame_size)
    print('Writing to Video, Please Wait')

# Road Region

# How the source corners will match up in the destination image

# Find the homography matrix for the conversion

# Warp the image

# Prepare the Image

    # Convert the image to HSV space

    # Thresh the image to HSV space

    # Blur the image

# Edge detection

# Create new image to fill with contours

# Find peaks

# Find line of best fit for points in peaks

# Draw the lane overlay


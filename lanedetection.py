
import cv2
from Road import Road

data_set=2
dst_size=(500,500)
write_to_video=False
initial_frame_num=0
cur_frame=initial_frame_num

# Construct object road
road=Road(dst_size,data_set,initial_frame_num)

# Prepare video output
if write_to_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frame_size = (road.frame.shape[1], road.frame.shape[0])
    videoname='lane_detection'
    fps_out = 15
    out = cv2.VideoWriter(str(videoname)+".mp4", fourcc, fps_out, frame_size)
    print('Writing to Video, Please Wait')


while road.video:

    # Warp the image
    road.top_down_image=road.warp(road.H,road.frame,road.dst_h,road.dst_w)

    # Prepare the Image, Edge detection, create new image to fill with contours
    road.fill_contours()

    # Find peaks
    road.find_peaks()

    # Find line of best fit for points in peaks
    road.find_lane_lines()

    # Draw the lane overlay
    road.make_overlay()

    cv2.imshow("Fancy Lanes",road.overlay)

    if cv2.waitKey(1) == ord('q'):
        break

    road.count += 1
    cur_frame += 1

    road.get_frame(cur_frame)


    # road.video=False
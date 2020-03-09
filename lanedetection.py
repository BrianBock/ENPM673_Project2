from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
import os
from Road import Road

data_set= 2
dst_size=(500,500)
write_to_video = True
show_plot = True
initial_frame_num=0
cur_frame=initial_frame_num

# Construct object road
road=Road(dst_size,data_set,initial_frame_num)

# Prepare video output
if write_to_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frame_size = (road.frame.shape[1], road.frame.shape[0])
    filename = 'output/lane_detection'+str(data_set)+'.mp4'
    fps_out = 15
    
    if os.path.exists(filename):
        os.remove(filename)
    
    out = cv2.VideoWriter(filename, fourcc, fps_out, frame_size)

    if data_set == 2:
        filename = 'output/plots.mp4'
        if os.path.exists(filename):
            os.remove(filename)

        out_plt = cv2.VideoWriter(filename, fourcc, fps_out, (700,900))
    
    print('Writing to Video, Please Wait')


while road.video:

    if road.count == 0 and data_set==2:
        plt.ion()
        fig = plt.figure(figsize=(7, 9))
        road.axs = fig.subplots(4,2)
        plt.subplots_adjust(wspace=.15, hspace=.15)
        plt.tight_layout()

    # Warp the image
    road.top_down_image=road.warp(road.H,road.frame,road.dst_h,road.dst_w)

    # Prepare the Image, Edge detection, create new image to fill with contours
    road.HSV_thresh()

    # Find peaks
    road.find_peaks()

    # Find line of best fit for points in peaks
    road.find_lane_lines()

    # Draw the lane overlay
    road.make_overlay()

    if data_set == 2 and show_plot:
        for i in range(4):
            for j in range(2):
                road.axs[i,j].cla()

        road.make_plot()

    print("Frame: "+str(cur_frame))
    if write_to_video:
        out.write(road.overlay)
        if data_set == 2 and show_plot:
            canvas = FigureCanvas(fig)
            canvas.draw() 
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image_from_plot = cv2.cvtColor(image_from_plot,cv2.COLOR_RGB2BGR)
            out_plt.write(image_from_plot)
    else:
        cv2.imshow("Lane Overlay",road.overlay)
    
    if cv2.waitKey(1) == ord('q'):
        break

    road.count += 1
    cur_frame += 1

    road.get_frame(cur_frame)

    # Use this for viewing only one frame at a time
    # road.video=False

if write_to_video:
    out.release()
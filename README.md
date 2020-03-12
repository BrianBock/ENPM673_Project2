# ENPM673_Project2

Justin Albrecht and Brian Bock


This project has two parts. Part 1 focuses on enhancing a dark video shot at night. Part 2 is lane detection on two videos. 

## Packages Required

This entire project is written in Python 3.7 and requires the following packages:

`numpy`, `cv2`, `matplotlib`, `scipy`, `os`, `imutils`, `time`

## How to Run
Download the entire directory. All files are required. 

### Part 1
There are several program parameters that are easily adjustable via a boolean toggles at the top of `problem1.py`:

`write_to_video` toggles the video output. If set to `True`, an AVI file will be saved in the same directory. If set to `False`, no video will be exported. 

`show_output` toggles the visualization (`True`=ON, `False`=OFF). This is very useful for seeing the program as it runs in real time. If you only want to export a video, we suggest turning this to `False` for speed. 

There are several different modes you have available for improving the video. If `write_to_video=True`, you should only have one of these set to `True`. If more than one mode is set to true, each frame of the video will alternate between modes, producing an undesirable flashing effect. 

`change_brightnes` toggles a brightness adjustment on the video

`change_contrast` toggles a contrast adjustment on the video

`change_histYUV` toggles a histogram equalization in the Y channel of the video converted to YUV space

`change_histHSV` toggles a histogram equalization in the V channel of the video converted to HSV space

`change_gamma` toggles a gamma adjustment on the video

You have the option to change the values for the brightness, contrast, and gamma. The preset values produced favorable results. 

To run Part 1, run `python3 problem1.py` in Terminal. 


### Part 2
There are several program parameters that are easily adjustable via a series of toggles at the top of `lanedetection.py`:

`data_set` chooses the data set to work with (either 1 or 2)

`write_to_video` toggles the video output. If set to `True`, an MP4 file will be saved in the same directory. The video name includes the data_set. If set to `False`, no video will be exported. 

`show_plot` toggles the pipeline visualization (`True`=ON, `False`=OFF). This is very useful for understanding what the program is doing, but it does slow the program down. If you run the program with `write_to_video=True` and `show_plot=True`, the program will output both videos. 

`initial_frame_num` is the first frame that will be read into the program. If you want to start at an arbitrary frame, you can change this number to whatever frame you want. 


To run Part 2, run `python3 lanedetection.py` in Terminal.  



# Part 1 - Enhancing a Dark Video


# Part 2 - Lane Detection on 2 Videos
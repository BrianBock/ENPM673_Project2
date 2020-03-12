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

There are several different modes you have available for improving the video. If `write_to_video=True`, you should only have one of these set to `True`. If more than one mode is set to true, each frame of the video will alternate between modes, producing an undesirable flashing effect. If `write_to_video=False`, you can run any combination of these parameters and view their outputs in separate windows. 

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

The task for this section was to improve the lightning and visibility in the provided video. The video is shot at night, which makes it dark. You can view the original video here: \url{https://youtu.be/s23jnVK4rHs}. Unlike the videos in Part \ref{sec:lanedetection} which use a rigidly mounted camera, the video in this part is shot with a hand controlled camera. Throughout the sequence, the camera moves and changes the view. To compound the difficulty,  most of the video is both compressed and very out of focus. The lack of focus makes it nearly impossible to know if a smoothing kernel is over-applied - the video is blurry regardless. The compression shows up in interesting ways later; since most of the video is dark, large clusters of pixels are stored as the same color, and these clusters become readily apparent with a variety of image manipulations. 



## Brightness
Brightness (or pixel intensity) is simply a piece-wise addition to the image matrix. Since the original image is 8 bit, pixel values cannot exceed 255. Any value that does wraps around and is interpreted as a near 0 number. This produces a result that is visually interesting (if not useful) where the bright spots of the image show up black, but the bordering light rays appear as normal:

![Over bright frame without 8 bit correction](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/bright_burn2.jpg)

To combat this issue, we use convert the image to 32 bit, add brightness, use `np.clip` to cap any values above 255 at 255, and then convert the image back to 8 bit for saving and displaying. On its own, brightness is not a great fix for this video. The sky and road both tend toward grey (and then white) as the brightness increases, without much improvement in visibility or image quality:


![original image](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/original.jpg)
*Original Image*

![Brightness +25](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/brightness25.jpg)
*Brightness +25*

![Brightness +50](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/brightness50.jpg)
*Brightness +50*


![Brightness +100](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/brightness100.jpg)
*Brightness +100*




Here are two versions of the video with boosted brightness:

+25 - https://youtu.be/VFmRoxNZOys

+50 - https://youtu.be/LHDYsPBT0Z8






## Contrast
Contrast is a scalar piece-wise multiplication with the image matrix. It is susceptible to the same 8 bit wrap around issue as the brightness, and is fixed with the same approach. Increased contrast offers some small incremental improvements in the video, which you can see here:

![Contrast 3](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/contrast3.jpg)
*Contrast \*3*

![Contrast 5](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/contrast5.jpg)
*Contrast \*5*

![Contrast 9](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/contrast9.jpg)
*Contrast \*9*



\*2 - https://youtu.be/Ob4xFdAFl68

\*3 - https://youtu.be/I_JWFZJkPkM

\*4 - https://youtu.be/2xkwpyO-ajc

\*5 - https://youtu.be/IyASQW2F_0s



At higher contrast levels, we start getting a noisier, redder sky, and washed out road signs (Figure \ref{fig:contrast9}). Car headlights also become excessively bright (Figure \ref{fig:contrast9headlights}). This would be detrimental to the performance of an autonomous reliant on this video for navigation, as this headlight sun obstructs a significant portion of the field of view and misrepresents the size of the offending car.

![overcontrasted headlights](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/contrast9headlights.jpg)
*Over-contrasted (\*9) frame with overbright car headlights}*







## Gamma Correction
We next explore gamma correction based on the work done by Adrian Rosebrock (https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)

Gamma=2 - https://youtu.be/zg0S08nHYJg

![Gamma 1.5](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/gamma1.5.jpg)
*Gamma 1.5*


\begin{figure}[H]
\begin{subfigure}{.5\textwidth}
  \centering
  % include first image
  \includegraphics[width=.95\linewidth]{images/original.jpg}
  \caption{Original frame}
  \label{fig:ogframe_gamma}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  % include 2nd image
  \includegraphics[width=.95\linewidth]{images/Gamma1.5.jpg}
  \caption{Gamma=1.5}
  \label{fig:gamma1.5}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  % include first image
  \includegraphics[width=.95\linewidth]{images/Gamma2.5.jpg}
  \caption{Gamma=2.5}
  \label{fig:gamma2.5}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  % include 2nd image
  \includegraphics[width=.95\linewidth]{images/Gamma5.jpg}
  \caption{Gamma=5}
  \label{fig:gamma5}
\end{subfigure}
\caption{Frames with different gamma correction}
\label{fig:gamma}
\end{figure}










## Histogram Equalization
We attempted histogram equalization on the image. Histogram equalization on an RGB/BGR image would distort the colors, which is problematic for an autonomous car (which needs to know colors for traffic lights, brake lights, road signs, etc.). We convert the image to YUV color space, which has a dedicated channel for illumination (Y). We equalize the histogram of the Y channel and then convert the image back to BGR space. \cite{yuvhist}\cite{hsvhist} The result looks like an old TV (Figure \ref{fig:histeqYUV}): \url{https://youtu.be/53uUhmN6sZA}


\begin{figure}[H]
  \centering
  % include first image
  \includegraphics[width=.8\linewidth]{images/Histeq.jpg}
  \caption{Frame with histogram equalization in the Y (YUV) channel}
  \label{fig:histeqYUV}
\end{figure}

We can also try converting the image to the HSV color space, which similarly has a dedicated channel for intensity (V). The result has largely purple skewed color and is still a noisy mess (Figure \ref{fig:histeqHSV}): \url{https://youtu.be/FYoVCjUv1X0}

\begin{figure}[H]
  \centering
  % include first image
  \includegraphics[width=.8\linewidth]{images/HisteqHSV.jpg}
  \caption{Frame with histogram equalization in the V (HSV) channel}
  \label{fig:histeqHSV}
\end{figure}



# Part 2 - Lane Detection on 2 Videos
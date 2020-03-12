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



At higher contrast levels, we start getting a noisier, redder sky, and washed out road signs. Car headlights also become excessively bright:

![overcontrasted headlights](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/contrast9_headlight.jpg)
*Over-contrasted (\*9) frame with overbright car headlights}*

This would be detrimental to the performance of an autonomous reliant on this video for navigation, as this headlight sun obstructs a significant portion of the field of view and misrepresents the size of the offending car.





## Gamma Correction
We next explore gamma correction based on the work done by Adrian Rosebrock (https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)

Gamma=2 - https://youtu.be/zg0S08nHYJg

![Gamma 1.5](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/Gamma1.5.jpg)
*Gamma 1.5*

![Gamma 2.5](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/Gamma2.5.jpg)
*Gamma 2.5*

![Gamma 5](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/Gamma5.jpg)
*Gamma 5*








## Histogram Equalization
We attempted histogram equalization on the image. Histogram equalization on an RGB/BGR image would distort the colors, which is problematic for an autonomous car (which needs to know colors for traffic lights, brake lights, road signs, etc.). We convert the image to YUV color space, which has a dedicated channel for illumination (Y). We equalize the histogram of the Y channel and then convert the image back to BGR space. (https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image)(https://stackoverflow.com/questions/17185151/how-to-obtain-a-single-channel-value-image-from-hsv-image-in-opencv-2-1) The result looks like an old TV:


![YUV equalization](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/HistYUV.jpg)
*Histogram Equalization in the Y channel of the YUV image*

https://youtu.be/53uUhmN6sZA


We can also try converting the image to the HSV color space, which similarly has a dedicated channel for intensity (V). The result has largely purple skewed color and is still a noisy mess: 

![HSV equalization](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part%201/HisteqHSV.jpg)
*Frame with histogram equalization in the V (HSV) channel*

https://youtu.be/FYoVCjUv1X0
































# Part 2 - Lane Detection on 2 Videos

The goal of this assignment is to detect the lane lines from a dash cam of a car driving on a highway. We need to detect where the lane lines are in the image and then overlay a virtual lane over top the original video. There are two sets of videos, the first of which is mostly on a straight road, while the second video the car makes gradual turns.  

In the first data set, the video to analyze is comprised of 302 individual frames saved as images. The second data set is a 16 second video. With the exception of reading the video in, the steps to detect the lane lines in both of these data sets are very similar. In both cases, we treat the video as a series of independent frames.

## Overview of image processing pipeline}

1. Import the next frame
1. Use homography to warp the image to create a top-down view of the road
1. Apply HSV threshold to create a binary image of the lane lines
1. Generate a histogram of the number of white pixels in each column of the binary hsv threshed image
1. Find the peaks of the histogram that represent the lane lines
1. Gather the points around each peak and create a line of best fit through those points 
1. Create a new blank image, the same dimensions as the top down image and draw the lane lines, the lane, and arrows 
1. Warp the image back into the original camera coordinates and overlay it with transparency onto the original frame
1. Determine turning direction



## Top Down Image}

In order to generate a homography matrix that will allow us to view the road as if we were looking from above we need to pick eight points. Four from the original source image and four where those points will map onto the destination image which in this case is the top down image. The first four points we pick by selecting two points just in front of the car on the lane and another two far ahead of the car on the lane. Each set of points share the same y-position in the image. In order to ensure that the top down image contains both lanes even if the car shifts position in the lane, we map the points not to the edge of the destination but instead 20\% in on each side. Figure \ref{fig:point-mapping} shows how the points map between the two images.

![point mapping](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/mapping.png)
*Point mapping between original and top down view*


We have written a function that computes this homography but for speed considerations we used the built-in opencv function `cv2.findhomography()`. 

After we determine a homography matrix we can then warp the image from the normal camera coordinate system into the top down view. The resulting top down image is:

![data1](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/top_down_ds1.png)

*Data set 1*


![data2](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/top_down_ds2.png)

*Data set 2*


## HSV Threshing
We converted each image from BGR space to HSV space:

![top down image](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/top_down_image.jpg)

*Original Warped Road Frame*

![HSV image](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/HSV_1.jpg)

*Frame in HSV color space*

The HSV color space is more robust and consistent under varied lighting conditions (https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/). We experimentally determined the ideal HSV max and min threshold values that made just the lane lines clearly visible in the frame and eliminated most undesired image elements. We used these values to convert the HSV image into a binary image. 

![HSV binary image](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/hsv_binary_image.jpg)

*Binary frame threshed from HSV*



We then apply a Gaussian blur with a square kernel size of 15 to the binary image. This is necessary to reduce noise that would otherwise make edge detection difficult. 

![Blurred image](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/GaussianBlur.jpg)

*Frame with Gaussian blur*


Using a single threshold worked very well for the first data-set where both lane lines were white, but it started to fail for the second video where the left-lane was yellow. We tested with a variety of thresholds but we were never able to find a threshold which worked for the entirety of the video. In order to mitigate this issue we used two thresholds, one for white and one for yellow. Each of the threshold values were used to create a binary image. These binary images were then added together with a `cv2.bitwise_or`. This way even if one of the threshold catches part of the other lane it is not counted twice.

![Yellow Threshold](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/yellow_threshold.png)

*Yellow Threshold*

![White Threshold](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/white_threshold.png)

*White Threshold*

![Combo Threshold](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/added_threshold.png)

*Combination of the two thresholds*


## Histogram Peak Finding
Now that we have a threshed binary image, we need to determine what pixels are associated with which lane. To do this we found the peaks for a histogram of white pixels in the binary image for each pixel column in the image. The reasoning is that the lane lines from the top down should be fairly straight, therefore there should be peaks in the histogram where vertical lines are. 


To find the peaks we considered multiple strategies but eventually decided to use a built in package within scipy named `find_peaks_cwt`. This function smooths the histogram then finds anywhere where there are peaks. This function also allows the user to specify the minimum distance between peaks, which allows for us to compensate for when there would be multiple peaks within the region of a lane line.

![Histogram](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/histogram.png)

*Histogram for white pixels in threshed image*

![Histogram with peaks](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/histogram_with_peaks.png)

*Vertical Lines where peaks were detected*



Once we have found all the peaks in the histogram we need to determine which peaks is associated with the lane lines. For the first frame we just use the peaks with the highest value. On subsequent frames we test to see if the peaks are within a reasonable threshold of the last frame. Doing this we are able to initialize two booleans, `found_left_lane` and `found_right_lane`. These booleans determine whether we try to generate a new lane line for the current image or just reuse the line from the previous frame. This is helpful for regions like under the bridge where we are unable to see the lane lines for a few frames. This also ensures that the lane lines are not too jittery. This approach does however require the first frame to be correct.

![Histogram with correct peaks](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/histogram_correct_peaks.png)

*Red peak denotes left lane center, Blue peak denotes right lane center*


## Lane Pixel Candidate
After the histogram peak detection, we have found a point close to the center of each lane. We then want to gather all the points within some region around the peak and fit a line to the points. We gather points within $\pm$25 pixels of each peak location. These pixels are the lane candidates that we use for line fitting.


For simplicity we decided to use only fit a straight line to the points but the method would allow for a more complex curve to be fitted to the data without too much effort. We then use `polyfit` to find the lane line coefficients. Our peak finding already distinguishes which peak is which so we now have an equation for both the left and right lane that fits the points of the HSV threshed image. As mentioned previously if either of the booleans for peak detection are `False`, we just use the line coefficients from the previous frame.  



## Lane Overlay
From the left and right lane coefficients, we generate the points that bound the lane in the top down squared image. We create a new all black image with the same dimensions as the top-down squared image. Within this image, we use the defined corners to draw a polygon which becomes our green lane overlay. Thick red lines are drawn from the top to bottom corners to bound the lane, becoming our lane lines. The next major step is to add the arrows to the center of our lane. We take the two top points and average them to get a midpoint between them. We repeat this to get a midpoint at the bottom, and then draw a line between these midpoints. This is the line that our arrows will be drawn on. We define the length and spacing of our arrows, and then use `cv2.arrowedLine` to generate the arrows. We now have a generated lane overlay with lane lines, a green road, and yellow arrows in the world frame:

![overlay](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/arrows.jpg)

*Overlay generated*

Now we use the inverse Homgraphy to warp this overlay into the camera frame. The shape of this overlay is blacked out in a copy of the original frame, and then a bitwise OR combines the lane overlay with the road blacked frame. 

![warped overlay](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/warped_overlay.png)

*Frame with road region blacked out*

![blacked out road](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/black_road.jpg)

*Frame with road region blacked out*


This new frame with the image and lane overlay is then recombined with the original frame (using `cv2.addWeighted` and alpha=0.5) to make the final image with a semi-transparent lane overlay:


![final1](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/Fancylanes1.1.jpg)

*1st Frame from Data Set 1 with the lane overlay shown*


![final2](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/Fancylanes1.1.jpg)

*1st Frame from Data Set 2 with the lane overlay shown*






## Turn Prediction
We take the slope of our lane midpoint line (used to draw the arrows) and compute it's slope. 

    m=(y_2-y_1)/(x_2-x_1

This slope has a bit of noise and changes a small amount between frames. We experimentally determined a reliable range of $m$ which denote the car is turning right, driving straight, or turning left. Based on this information, we print the slope and car action on the frame, and show it to the screen:

![left](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/left.jpg)

*Car turning left*

![straight](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/straight.jpg)

*Car driving straight*

![right](https://github.com/BrianBock/ENPM673_Project2/blob/master/output/Part2/right.jpg)

*Car turning right*

There is no turning in the first data set, so this method is not implemented with that video. 



## Discussion on Hough Lines

Another possible way to solve the problem of lane detection would be to use Hough lines. Hough lines work by using a voting system. The process is as follows:

1. Determine the edges of the image
1. Initialize a Hough Matrix $H$ with all values set to zero
1. For each edge point in the image find many equations of a line, that passes through the point, in the form d = xcos(theta)+ysin(theta).
  1. To do this you should find a new d for each theta between 0 degrees and 180 degree
  1. Add 1 vote to that index (H[d,theta])
1. Find the maximum in the $H$ matrix. These indicies of the max are the d and theta$that represents a line that fit the most edges in the image.
1. For lane detection you should find the equation that fits the each of the lane lines as maximums within H, you will probably need to add some filtering to remove extraneous lines in the image.  



## Applicability to Other Videos
We were able to apply our solution from the first data set of this project to the second with minimal modification (only adjusted the HSV threshold and road bounding points). This increases our confidence in the applicability of our solution to other videos. The techniques used in this project should be work with minimal modification on any fixed camera driving on a road with the following conditions:

The road must be reasonably well lit. This approach would probably not work as well at night, unless the lane lines were illuminated or highly reflective. In any condition where the lane lines are clearly visible, this technique should work. The HSV thresholds might need to be adjusted to compensate for drastically different lighting conditions.

Using multiple thresholds, lane lines can be yellow or white. These are the most common colors for lane lines, so this should have wide applicability. In order to capture other lane line colors, you could add another HSV threshold, which is an easy task.

The road points would need to be reconfigured for any new camera configuration or vehicle, but if chosen well they should be applicable for any driving that car does (given the above conditions). Since the program analyzes histogram peaks and not position, it does not care about the lane width. The program could therefore not restricted to roads of a comparable lane width, and should handle roads with a variety of lane widths. Lanes are often wider on highways and parkways than they are on local roads. It is important to define the road region points on a wide lane, and to allow for some buffer width. If the points were defined on a narrow road, regions of a wider road (and the lane lines) would be lost. 

Since this project is founded on lane line detection, it would be useless on roads without any lane lines. 


## Videos
**Data Set 1**

Lane Overlay - \url{https://youtu.be/__XLG9Cgs0I}


**Data Set 2**

Lane Overlay - \url{https://youtu.be/NvaUFfDh_DU}

Lane Overlay with turning - \url{https://youtu.be/QNkY0j8-9CY}

Pipeline Visualization - \url{https://youtu.be/twkbwrYAsu4}





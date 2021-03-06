**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/detect1.png
[image41]: ./examples/detect2.png
[image42]: ./examples/detect3.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/example1.png
[image7]: ./examples/example2.png
[image8]: ./examples/example3.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2o code cell of the IPython notebook "Vehicle Detection and Tracking" on the function `single_img_features()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and, after many tests, I decided to set the parameters as follws:

- color_space = 'YCrCb' 
- HOG orientations = 5 
- HOG pixels per cell = 8 
- HOG cells per block = 2 
- HOG Channels = 'ALL' 
- Spatial binning dimensions = (32, 32) 
- Number of histogram bins = 32  

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial fesatures, histogram features, and hog features.
This processes can be seen on the seccond cell of the notebook "Vehicle Detection and Tracking" on the function `single_img_features`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding windows search is implemented on the file `Vehicle_Detection_and_Tracking.py` on the lines 67 to 100 of the function `find_cars`.
The search area is determinated by the variables `ystart, ystop`, and `scale` defines the scale of the image to use.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image41]
![alt text][image42]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_output3.mp4)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/7YSzeImvwI8/0.jpg)](http://www.youtube.com/watch?v=7YSzeImvwI8)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

It was need to hold the last 60 frames from the heat map, and calculate the average of the values to stabilize the detection box over the entire video.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

![alt text][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The are some problems with the detection, were some times, it canot distinguish the car in some light situations. So it´s need a much bigger image data-base to train the detector.

The other problem I've found was the estabilization of the bounding boxes, that try to wobling arround. I work arround it holding the last seccond of frames (60 frames) and calculate the average.



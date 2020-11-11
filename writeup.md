## Advance Lane Finding
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: output_images/undistorted.jpg "Undistortion Example"
[image2]: output_images/distorted.jpg "Original Frame"
[image3]: output_images/warped_and_thresholded.jpg "Thresholding Example"
[image4]: output_images/warped_lanes.jpg "Warp Example"
[image5]: output_images/roi.jpg "Src Poly"
[image6]: output_images/marked.jpg "Output"
[image7]: output_images/vehicle_location.jpg "Vehicle Indicator Example"
[video1]: output_images/output.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in script file called `calibrate.py`. This script should be run with `path_to_calibration_folder` as command line argument.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_p` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

I save the output of the calibration in file `calibration.json` in current working directory. The formal of json is as follows;
```json
{
"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
"dist": [[k_1, k_2, p_1, p_2, k_3]]
}
```

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I used OpenCV `undistort` method by giving camera matrix and distortion coefficients as arguments. The signature of
OpenCV `undistord` method  is `undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) -> dst` as per OpenCV
documentation.

After distortion-correction the above image turns into following image:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 63 through 
72 in `lane.py`). I look for pixels with saturation value above `200` or pixels with sobel_x edge strength above `20`.

Here's an example of my output for this step.  (note: this is not actually from one of the test 
images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is divided into three methods of `Lane` class in `lane.py` namely:
 * `estimate_perspective_transform()` line `379` through `391`.
 * `perspective_wrap()` line `393` through `400`.
 * `perspective_wrap()` line `402` through `409`.

The `estimate_perspective_transform()` method uses image size width offset and `roi_source` to estimate `roi_dest` 
(the polygon where the image will be projected) and then uses `roi_source` and `roi_dest` to estimate perspective 
transform matrices using OpenCV `getPerspectiveTransform` method and store the matrices in `M_` and `M_inv_` class
variables of `Lane` class to use later for perspective wrapping and unwraping respectively.

I chose the hardcode the source points and calculated destination points from them in the following manner:

```python
poly = np.array([
    [681, 446],
    [1118, 720],
    [192, 720],
    [600, 446]
])
poly = np.float32(poly)

roi_dest = np.float32([[w - offset, 0], [w - offset, h], [offset, h], [offset, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 681, 446      | 930, 0        | 
| 1118, 720     | 930, 720      |
| 192, 720      | 350, 720      |
| 600, 446      | 350, 0        |

I verified that my perspective transform was working as expected by drawing the `src` points onto a test image to verify that the lines appear parallel.

![alt text][image5]

The following was output of perspective transform:

![alt text][image4]
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used sliding window method to initialize the lane-line pixels with code in `lane.py` from line `74` through `183`.
I start with identifying left and right lane-line pixel at bottom of image using histograms as done in method 
`find_hist_based_anchors` and the use sliding windows to extract all the possible candidate pixels for both left and
right lane-lines. And finally I fit the polynomials through extracted candidate pixels if number of votes are greater
than a threshold in my case `100` as done in method `extract_poly`. I also extract metric polynomials to use later for
curvature and vehicle position detection.

Once the lane-line polynomials are initialized I used `process()` method of `Lane` class to use previously identified
polynomials and offset to look for candidate pixels. The code for this part is location from line 273 through 335 in 
`lane.py`.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 185 through 203 under method `estimate_curvature_and_position` in `lane.py` using help function 
`curvature_of_poly` implemented in `lane.py` in lines 15 through 23.

I estimate vehicle position by taking mean of metric x positions of left and right lane-lines close to car and
subtracting it from `3.7/2` which should be actual mid-point of lane with lane width being `3.7` with reference to 
left lane.

I also implemented a visual of vehicle location relative to lane, and results look as follow:
![alt text][image7]
The highlight color also changes from green to yellow when vehicle moves from lane center to sides of lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 205 through 271 in my code in `lane.py` in the function `mask_frame()`. 
 Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](output_images/output.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
  
* I used a fixed polynomial for perspective transform as it was simplest solution but I don't think it is scalable for
different cameras even lane widths. I think the width of polygon we draw from bottom toward center of image is dependent
of the field of view of camera and it should be used to estimate possible Region of Interest.

* The current method assumes and highly rely on well marked lanes, lanes with old marking will not work with this method.
* The method assumes a flat road which means if it is exposed to a road from a mountain area with sudden rise and falls,
the method will struggle with finding lanes.
Github: 
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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

a. convert image format to grayscale
b. cv2.findChessboardCorners() to find the corners in images
c. cv.2.calibrateCamera to calibrate the image
d. cv2.undistort() to undistort the image

The code for this step is contained in the `Cell 293` code cell of the IPython notebook located in `Advanced Lane Line Detection.ipynb` .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
```python
images = glob.glob('camera_cal/calibration*.jpg')
count = 1
fig = plt.figure(figsize=(20, 15))
mtx = []
dist = []
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

for img_dir in images:
img = cv2.imread(img_dir)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (cx,cy), None)
if ret == True:
objpoints.append(objp)
imgpoints.append(corners)
img = cv2.drawChessboardCorners(img, (cx,cy), corners, ret)
fig.add_subplot(4,5,count)
count += 1
plt.imshow(img)
```

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


![Calibrate Camera & Undistort ](http://upload-images.jianshu.io/upload_images/2528310-07ff348436dad22f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/860)

The codes are as follows:
```python
images = glob.glob('camera_cal/calibration*.jpg')
fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(24, 9))
fig.tight_layout()
img = cv2.imread(images[0])
fig.add_subplot(1,2,1)
ax1.set_title('Before', fontsize=50)
plt.imshow(img)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
fig.add_subplot(1,2,2)
ax2.set_title('After', fontsize=50)
plt.imshow(dst)
```
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![distortion-corrected image](http://upload-images.jianshu.io/upload_images/2528310-f2acbe8a31d95998.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/860)

The codes are as follows:
```
images = glob.glob('test_images/test3.jpg')
fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(24, 9))
fig.tight_layout()
img = plt.imread(images[0])
fig.add_subplot(1,2,1)
ax1.set_title('Before', fontsize=50)
plt.imshow(img)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
fig.add_subplot(1,2,2)
ax2.set_title('After', fontsize=50)
plt.imshow(dst)
```
#### 2. Describe how you used color transforms `Cell 622, 621, 620`, gradients `Cell 297, 622` to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (HLS, HSV, Lab) and gradient thresholds to generate a binary image (thresholding steps at (Cell 622, 621, 620),).  Here's an example of my output for this step. 
a. absolute sobel operator in both x and y dimension
b. gradient sobel operator for x and y dimension
c. use the specific channels in HLS, HSV, Lab format to remove noises (such as shaddow) and detect white and yellow lane lines
d. Combine all the above results together and generate the binary image

![A binary image](http://upload-images.jianshu.io/upload_images/2528310-24508277b2fdd892.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/860)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in Cell 626 of the IPython notebook).  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
leftupperpoint  = [568,470]
rightupperpoint = [717,470]
leftlowerpoint  = [260,680]
rightlowerpoint = [1043,680]

src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
dst = np.float32([[200,0], [200,680], [1000,0], [1000,680]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 568,470      | 200, 0        | 
| 260,680     | 200, 680      |
| 1043,680    | 1000, 680      |
| 717,470      | 1000, 0        |

Then, I use the cv2.getPerspectiveTransform(source, destination) to generate the perspective transform matrix.  Finally, the cv2.warpPerspective is employed to warp the image.


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


![Perspective transform](http://upload-images.jianshu.io/upload_images/2528310-601ad2c2e23bc018.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/860)



#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?
As shown in `Cell 629` in  `Advanced Lane Line Detection.ipynb`,  I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

How to identify the lane line correctly in the above image? We first draw the histogram of the lane line. The two peaks in the histogram corresponds to the left and right lines. Then we use 10 sliding windows to go through the image to identify the right position of the lane line. Finally, a second order polynomial function is used to fit the points that we detect. 

```python
histogram = np.sum(dst[int(img.shape[0]/2):,:], axis=0)
plt.plot(histogram)
```
![Histogram of the lane line](http://upload-images.jianshu.io/upload_images/2528310-9ade7ee11543098b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/860)
![Fit 2nd order curve](http://upload-images.jianshu.io/upload_images/2528310-93077b33a80e8c2e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/860)


In addition, we use a Queue (size = 10) to smooth the curves that we detect. In some cases, the camera may not detect lane lines. With a queue, the car may keep on the track according to the information in previous frames.


![Windows for lane line detection](http://upload-images.jianshu.io/upload_images/2528310-46e73c9170f07039.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/2480)

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `Cell 591` in my code in `Advanced Lane Line Detection.ipynb`.
a. Use the Radius of Curvature to get the curvature for left and right lane lines, respectively
b. Calculate the relation between pixes and meters. As the US high way specifications. 700 pixes vs 3.7 meters.
c. According to the points, we can calculate the offset for both left and right lane lines. The average of the both lane line position minus the center of the lane line is used to evaluate whether the detected lane line is corrected. We mark the radius of curvature and position of the vehicle with respect to center in the video.

```python
def calculate_curvature(leftx, rightx, lefty, righty):
'''Calculate the radius of curvature in meters'''

# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
#y_eval = np.max(ploty)
y_eval = 719
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters

return left_curverad, right_curverad

def calculate_offset(undist, left_fit, right_fit):
'''Calculate the offset of the lane center from the center of the image'''

xm_per_pix = 3.7/700 # meters per pixel in x dimension
ploty = undist.shape[0]-1 # height
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

offset = (left_fitx+right_fitx)/2 - undist.shape[1]/2 # width
offset = xm_per_pix*offset

return offset
```

![Test image on the road](http://upload-images.jianshu.io/upload_images/2528310-e8fc6f466322d28c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `Cell  636` in my code in `Advanced Lane Line Detection.ipynb`.  Here is an example of my result on a test image:


![Lane Line detection](http://upload-images.jianshu.io/upload_images/2528310-b31e2ebf02bf512f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
from moviepy.editor import VideoFileClip
from IPython.display import HTML
global left 
left = Queue()
global right 
right = Queue()

global left_pre_avg 
left_pre_avg = 0
global right_pre_avg
right_pre_avg = 0
def process_frame(img):

undist = cv2.undistort(img, mtx, dist, None, mtx)

combined_binary = combined(undist)
warped, minv = warp(combined_binary)

warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

left_fitx, right_fitx, plot_y, curve_pickle = fit_line(combined_binary)

# Smooth the curve line
global left_pre_avg
if left_pre_avg == 0:
left_pre_avg = np.mean(left_fitx)
right_pre_avg = np.mean(right_fitx)
if left.size()<=10:
left.put(left_fitx)
right.put(right_fitx)
elif abs(np.mean(left_fitx)-left_pre_avg)<10:
left.get()
right.get()
left.put(left_fitx)
right.put(right_fitx)

left_pre_avg = np.mean(left_fitx)
right_pre_avg = np.mean(right_fitx)

left_x = left.avg()
right_x = right.avg()

pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])
pts = np.hstack((pts_left, pts_right))

cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
newwarp = cv2.warpPerspective(color_warp, minv, (image.shape[1], image.shape[0])) 

# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)
return result

image = plt.imread('test_images/test1.jpg')
process_frame(image)
```
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  
Here's a [https://www.youtube.com/watch?v=ivDAtCu-XzY](project_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The present method works well on the project video. However, it doesn't work on the challenge and hard_challenge video. The shadow on the road, the lightness and the sharp bend are also challenging situations that my method will fail. Preprocessing the data is a way to make the method robust. However, the parameters in my code are fixed. It's a promising direction for deep learning model to identify the lane line automatically (without fixing parameters). However, data collection of the self-driving car is the limitation.

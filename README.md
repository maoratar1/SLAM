# SLAM project
This project were taken under the course 
"Computer vision based navigation" - 67604 that was given at the 
Hebrew University of Jerusalem in 2023 by Mr David Arnon and Dr 
Refael Vivanti.
This is my [Hebrew summary](https://drive.google.com/file/d/19_4wjf477zzoSyrLiXZ66g6sY3s-dHds/view?usp=sharing)
for the course.


## tl;dr
SLAM, shortcut of **S**imultaneous **L**ocalization **A**nd **M**apping, 
is a computational problem of constructing a map of an unknown
environment while keeping track of an agent's location within it.
We accomplish this mission using the `Bundle Adjustment` algorithm which is 
an algorithm for solving SLAM. 
This project is an implementation, and in some meaning, a refined version,
of the algorithm presented at the
[Frame SLAM](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=hczHVxEAAAAJ&citation_for_view=hczHVxEAAAAJ:WF5omc3nYNoC&tzom=-120) 
article.

The project is built from, basically, 5 main steps:
1. Creating first trajectory's estimation using deterministic approach.
2. Building Features' tracking database.
3. Performing Bundle Adjustment optimization.
4. Building a Pose graph.
5. Performing Loop Closure.

In this project we are using the following concepts and many others:
- Bundle Adjustment.
- Loop Closure.
- Feature's detecting and description.
- Triangulation.
- PnP.
- RANSAC.
- Consensus match.
- SVD.
- Least Square.
- Camera matrices: Extrinsic and Intrinsic.
- Outliers rejection policies.
- Factor graph.


The data we are using is of KITTI's Benchmark Suite.
In this project we are trying to estimate a trajectory of a moving car
that has a stereo camera on its roof, and it is filming its travel route.

At the end of the project we will get a pretty food estimation for the car's 
trajectory while going through 3 stations: Deterministic estimation, Bundle Adjustment
optimization and Loop Closure strengthening.

### Runtime 
Computing the first trajectory's estimation takes time of ~ 8 min.
Building the database is similar. After that, Bundle Adjustment, with 
multiprocessing, takes 3.5 min and the loop closure is about 3 min.
Practically, the whole process can be done in a parallel so the total rum time
is 8 min that is almost the real time of ~ 6 min.




## KITTI's Benchmark

KITTI uses a car, includes some sensors, travelling around several streets
in Germany. The sensors are stereo camera (color and black-white), GPS and
Lidar.
[KITTI's benchmark](http://www.cvlibs.net/datasets/kitti/) 
supplies a lot of ground truth which allows us to compare 
are results of our algorithms.
In our project we use only the black and white stereo cameras.

KITTI's supplies for us:
- Cameras' intrinsic matrices between right and left.
- Cameras' extrinsic matrices between left and right cameras 
(Where the left camera is at the origin)
- 3450 Frames.
- Vehicle location during the ride - a ground truth to compare to.

### Stereo camera
KITTI uses stereo camera with two or more lenses with a separate image sensor or
film frame for each lens. This allows the camera to simulate human binocular vision, 
and therefore gives it the ability to capture three-dimensional images. So In our project 
when we say `Frame` when mean two images, for the left and the right cameras.


### Cameras' matrices
Every camera has two matrices, extrinsic and intrinsic.

#### Extrinsic matrix
<img src=README_Images/KITTI/ExCam.png width="230" height="180" align="right">

The extrinsic matrix used for 
converting between two coordinates system, the world coordinates and the camera coordinate
systems. By saying "world coordinates" we mean to the first left camera coordinates system 
and by saying "camera coordinates" we mean a system where the camera is 
at the origin, x-axis point to right, y_axis points down and the z axis points to the front
(see image on the right).
The extrinsic matrix is a 4X3 matrix which consists from two matrices, 
a Rotation and translation. and can be written as [R | t] where R is the rotation 3X3 matrix
and t is the 3x1 translation matrix. 
> Image from the courtesy of David Arnon and Refael Vivanti

#### Intrinsic matrix
The intrinsic matrix is used for projecting a 3d point, 
at the **camera coordinate**, to the camera's image plane. 
It's includes the inner parameters of the camera such as 
principal point at the image coordinates system (top left or centered)
focal length at pixel units, Skew and Distortion.

> We will denote the camera's intrinsic matrix as K and the extrinsic's as M1
> for the left camera and M2 for the right camera.

#### KITTI's cameras' matrices
Those are KITTI's left and right camera extrinsic matrices and intrinsic matrix:

<img src=README_Images/KITTI/KITTIcameras.png width="" height="">

> You can see that there's that the left and right cameras are differ by their
> x-axis's location where the right camera is 0.54 to the right. The way we conclude
> that will be explained later under the sub subject of finding cameras' location
> by its extrinsic camera.


### Projecting matrix
From those 2 matrices we can build a matrix which maps a 3d point at the **world coordinates**
to the image plane. This matrix is simply, K * [R | t]. The extrinsic matrix, [R | t], 
maps the 3d point from the world coordinates to camera's coordinates
and K project the point to the image plane.

> This idea of mapping between two Euclidean coordinates system is stems from the
> the Euler's Theorem saying that "Two Euclidean coordinate systems differ by rotation and translation."

## Bundle Adjustment - First glimpse
As mentioned above, we are doing localization and mapping, localization
is finding object's location in **world coordinates**
and mapping is creating some mapping of object's in the world.

Every camera in the whole trajectory sees several objects (we call them landmarks)
in the world and every object in the world is seeing by several cameras. 
By saying "seeing by a camera" we mean that it can be projected to the camera plane.
So we can look at this structure as "Bundles" that suppose to fit perfectly. As 
we can see at the following image:


<img src=README_Images/BundleAdjustmentPart/BundleStructure.png width="300" height="200">

> Courtesy of Ackermann


So we want to fit the bundles, means that we want that all cameras and landmarks
poses would be in a place that the landmarks projections would fit to 
our measures. "measure" means pixels in some camera that some landmark 
appears at. Practically, bundle adjustment algorithm get as input, 
Landmarks' projections at the Cameras' image plane and outputs cameras and landmarks poses that
minimize the projection error.

Due to the fact that Bundle Adjustment algorithm is abased on a the `Levenberg Marquadt` algorithm
which 
finds a local minimum, there is a greate importance in choosing the algorithm's
starting point, or as we call it in our code `initial_estimation`. So before
diving in to the Bundle Adjustment algorithm we will start with a deterministic
approach to the problem that will obtain some, and I might say not bad, 
trajectory estimation that will be used as our initial estimation.

## Estimate trajectory - Deterministic approach 
So, we want to create a world mapping and to find our 
location simultaneously. Let's start with mapping.

### Mapping
"Mapping" means that, for a given camera's pose, we want to create some
map of the world surrounds us. There are several ways to define a map on the world like
geometric map, topological map and others. In our project we will use the 
`Point cloud` map.

#### Point cloud

A point cloud is a map that tells us about every coordinate in the world 
if there is some object in that place, but it does not tell us if there is **no**
object there.

In order to create a point cloud we will use 2 concepts: `Images Matching` and `Triangulation`

### Images matching
Images matching is the process of finding matching key points between two images.
The process of matching include 3 steps:
1. Features detection.
2. Feature description.
3. Feature matching.

There are several algorithms for the first two steps as AKAZE, SIFT, ORB, and others.
In our project we made comparisons between the first three, and we finally chose `AKAZE`.

For the 3rd step we had tried Brute force and Knn using significance test. Finally,
we use the brute force method due to better results and with a run time payment of
40 sec only. At the Brute force matching we use the `Hamming norm` which is 
more suitable for AKAZE that its feature vectors are binary vectors.

This process can be done by the following code:
```python
from utils.utills import detect_and_match
detect_and_match(left_img, right_img)
```
> This function returns kpts, dsc anf matches list for those images.
> 
> There is some similar functions that differ by parameters and other functionality
> for other purposes.

### Outliers rejections policies
Of course, those matching algorithm are not accurate so there are some outliers that we get
in the matching process. In general, outliers rejection is a very important process
and especially in our project which uses the `Least Square` method and the assumption that our
measures are `Normally distributed` that are very sensitive to outliers. 
In our project we will meet the following outliers' rejection policies:
1. Blurring.
2. Rectification test.
3. Triangulation.
4. Significance test.
5. Consensus match.

Currently, we can explain the `Rectification test` and the `Blurring`
rejection policies, the other policies we will meet later. 

#### Blurring
Blurring the image may help with removing noises. Noise's environment tends to be unique, 
therefore it's a good candidate for being a feature since detectors, at least some of them, 
characterize a feature by its environment - the more unique its environment is, the more likely
it is to be a feature.

In our code we use the `Gaussian blurr` with kernel size of 11.
```python
import cv2
img1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), 0)
```

> We had examined several kernels as 3, 5, 7, 9 and 11. Finally, we chose the one
> with a good mean track and the total number of tracks. The term `track` will be explained
> later, at the Database part.


#### Rectification test
Each frame consists of two stereo camera means that matching key points would be with the same 
y-axis value. Thus, we can produce a nice outliers' rejection policy - a match that
it's left_y and right_y is not equal (up to some threshold) is considered as an 
outlier.

```python
from utils.utills import rectified_stereo_pattern_rej
rectified_stereo_pattern_rej(img1_kpts, img2_kpts, matches)
```

> This function returns the inliers and outlines indexes at the original `matches` list

----
Now, that we have the ability to match between two key points at the frame, we 
can perform the `Triangulation` operation. But, before that we should mention the `Least Square`

#### Least square
Least square is an algorithm that is used for fitting a model to a given data 
when the data's amount is larger than the model's parameters and with the assumption
that the data is noisy, this algorithm obtains the "closest", in meaning of Residual sum
of squares, model to the desired one.

There are 2 cases of Least square, the linear case, where there is some linear
relations between the data and the value we want to predict, and the non linear one.
Formally, we can present the linear case as:

<img src=README_Images/DeterministicApproach/LeastSquareDesc.png width="380" height="100">

if b is not equal to zero the solutions go through the pseudo-inverse and if b = 0, 
the homogeneous case, We calculate A's SVD and the desired vector is **the last column
of V**

#### Linear equations
There are 3 options for linear system equations, Ax = b:
1. if A is cubic matrix - need to check if A is invertible 
2. A is fat, more columns than rows - there are infinite solutions in that case we will search
for the minimum norm solution
3. A is thin, more rows than columns, there are zero solutions, so we search for the closest solution,
means the projection of x on the A dimension


### Triangulation
In order to create a world mapping we can use the operation called `Triangulation`.
It is called that way due to a geometric consideration. The Triangulation operation
get as input, a pair of match cameras' pixels and a projections' metrics for each camera
and return the 3d point in the world they represent.

<img src=README_Images/DeterministicApproach/Triangulation.png width="250" height="150">

> Courtesy of David Arnon and Refael Vivanti 

Triangulation is none other than solving a linear system of 4 equations.
If we denote by P and Q the left and right cameras' projection matrices respectively, 
the desired 3d point by X and by p and q the left and right pixels of X's projection
we get that the multiplication of P with X, at homogeneous coordinates, yields p_hat
(at homogeneous coordinates) that equals to p and the same for q. This leaves us
with a linear equation system that can be represented as a matrix of 4X3. As 
we can see for P:

<img src=README_Images/DeterministicApproach/PX.png width="" height="">

Doing the same for Q and concatenating those equations we get:

<img src=README_Images/DeterministicApproach/TriangulationEquations.png width="200" height="130">

This linear system does not necessarily have a solution, which in geometrically meanings 
it represents the case where the two rays does not intersect but crossed. 
So, for solving those
equations we will use `SVD` which is very helpful tool in the `Linear Least Square` problems.

The bottom line is that we are 'SVDing' that matrix and returns the last vector at V 
from the SVD decomposition.


```python
from utils.utills import triangulate
# Performs triangulation for a list key points and returns a list of a 3d points.
p3d = triangulate(left_cam_proj_mat, right_cam_proj_mat, left_kp_lst, right_kpt_lst)
```
#### Triangulation rejection policy
Practically, the triangulation provides us with another outliers'
rejection policy. We can reject two kind of 3d points :
1. Very far points
2. Points with negative z value. 

The reason for rejecting very far points is that although those point can be a good match, they
are not providing us much information about their location because far points tends to 
have a high
inaccuracy of their triangulation estimation so for our future purposes we would prefer to 
ignore them.
The reason that far points error tend to get high values can be explained by the following:
The key thing that causes the triangulation's error is the Feature detector inaccuracy and 
by simply assuming that the detector inaccuracy over the image is even, means there is no 
reason that far away points will have a higher inaccuracy than closer ones
because after all they both are pixels in the image and the detectors are looking, basically, 
at the pixel's environment which is not influenced by the real location at the world. 
Now, for simplicity lets assume that the triangulation's triangle is an 
isosceles triangle with base angles of 'a' and baseline '2m', so we have that the
object's distance from the camera equals to X = m / tan(a) so the distance depends on tg(a) 
and due to the tangent function behavior in the range [0, pi / 2] 
it holds that for error of 'e' and two angles 'a' > 'b' : 
: tg(a + e) / tg(a) > tg(b + e) / tg(b). Therefore inaccuracy of e is more dramatic at a larger
angles. 

Rejecting points with negative values is obviously but let's explain how such points are created.
Because the left camera is from the left of the right camera we suppose to
have that a 3d point in the world would project to the left image's and the right image's 
pixels such that x_left > x_right. So as x_right get closer to x_left
we get that the triangulation result
will be far as well until we have the situation where x_r > x_l and in that case, the 
triangulation triangle is changing direction and now the ray's intersection will be **behind**
the cameras. So when we are collecting our data we can add this rejection policy:
```python
if x_r > x_l or x_l - x_r < TRIANGULATION_THRESH:
    return
```

Actually we do this with numpy conditioning in the following way:
```python
diff_matches = left0_matches_coor[:, 0] - right0_matches_coor[:, 0] 
closer_pts = diff_matches > TRIANGULATION_THRESH
positive_pts = diff_matches > 0
return closer_pts * positive_pts
```

We have done this policy rejection at the following function:

```python
from utils.utills import far_or_neg_pts_rej

far_or_neg_pts_rej(left_img_pixles, right_img_pixles)
```
>This functions returns the lists indexes that their corresponding values passed the test.

---
So now we have the ability to create world mapping with a point cloud: 

<img src=README_Images/DeterministicApproach/pointcloud.png width="300" height="250">

> Camera is represented by the red point.

Now, we are moving for the localization part


### Localization
The localization is done inductively by the following pseudocode:
1. Initialize frame 0 to the origin
2. For the ith frame, i in range()
   1. Perform _tracking_ between ith and (i - 1)th frames
   2. Create a point cloud of the (i - 1)th frame  - `Triangulation`.
   3. Compute the transformation between those frames using steps 1 - 2 - `PnP`.
   4. Compute the ith frame 3d location.

> Setting the first frame to the origin means setting its extrinsic matrix to be 
> the [Identity | 0]

Let's explain each step, but because `PnP` is the main algorithm here we will explain
it first.

#### PnP algorithm
`PnP`, shortcut of Projective n Points, is an algorithm that Given n landmarks at the
world coordinates, their projection on the camera plane and the 
`Intrinsic camera matrix`, it returns the
transformation between world and camera coordinates system.

In the following image, the Xs represents the points at the world coordinates, 
the zs are their projection at the camera plane and X_p is the camera pose that we
are interested to find.

<img src=README_Images/DeterministicApproach/PnP.png width="230" height="150">

> Courtesy of David Arnon and Refael Vivanti 

The following code will perform the PnP algorithm:
```python
from utils.utills import pnp 
ex_cam_matrix = pnp(world_p3d_pts, img_proj_coor, calib_mat, pnp_method)
```
> This function wraps the `cv2.pnp()` method. The cv2's PnP returns a `Rodriguez` vector
so at our function we call cv2's PnP and, if it succeeds, our function will convert it
to a transformation matrix.
`world_p3d_pts` are the X's, `img_proj_coor` are the z's, `calib_mat` is the 
calibration or intrinsic matrix of the camera and `flag` is the cv2's `SOLVEPNP` method 
that the `cv2.pnp()` is expected to get.
>
> `Rodirguez` vector is a 6dim vector that represent a camera pose by its 
> euler angles and its 3d location with the order (azimut, pitch, roll, x, y, z)

So, for using PnP method we need a point cloud that, as mentioned at the pseudocode,
is done from the (i - 1)th frame and match each 3d point to a pixel at the ith frame
image plane, for doing that we will need to use the `Tracking` process.

#### Tracking
`Tracking` is the process of finding key points in consecutive frames.
In this process we wish to find a key point which appears in all 4 images. Practically speaking,
we perform a keypoint matching at the first frame, second frame and between the 
lefts images of these consecutive frames. For convenience, we will call the first frame
, frame 0, the second frame, frame 1 and their left and right images left_i and right_i
correspondingly.

<img src=README_Images/DeterministicApproach/tracking.png width="250" height="200">

> Courtesy of David Arnon and Refael Vivanti 

As mentioned above, matching is done with the rectified rejection policy, so
the tracking process is done as follows:
1. Find matches between frame 0 with the rectified rejection policy
2. Find matches between frame 1 with the rectified rejection policy
3. Find matches between left0 and left1 images
4. Return matches that are in all 3 lists.

Example of tracking:

<img src=README_Images/DeterministicApproach/4PointTracking.png width="650" height="420">

###### Implementation
The tracking process is done by the following function:
```python
from utils.utills import tracking_4images
tracking_4images(left0_dsc, left1_dsc, 
                 pair0_matches, pair0_rec_matches_idx,
                 pair1_matches, pair1_rec_matches_idx)
```
>`pair0_matches` is a list of matches (`DMatch` cv2 object) at pair 0
This function returns 2 lists, one for pair 0 matches that were tracked in all 4 images and
the other for the pair 1 matches. We did not write them due to row space limit.
Notice that this function receives a matches at pair 0 and pair 1 and their indexes where
there is a match that passed the rectified policy

Let's dive in to this function a little and explain how we have done this tracking:

At first, we perform matching between left 0 and left 1:
```python
from utils.utills import match
# Find matches between left0 and left1
left0_left1_matches = match(left0_dsc, left1_dsc)
```
Then, for each pair, assume pair 0 for convenience, we create a dictionary
whose keys are the `left0_dsc`'s indexes (where each descriptor index corresponds to 
one key point) and the values for each key is the `pair0_rec_matches_idx` values, means the
index at the pair 0 matches list which passed the rectified rejection policy.
This done by:
```python
from utils.utills import create_rec_dic
# dict of {left kpt idx: pair rec id}
rec0_dic = create_rec_dic(pair0_matches, pair0_rec_matches_idx)
```
 We do the same for pair1. Now, we pass over the left0 and left1 matches.
For each match, which is a tuple (left0 key point, left1 key point) of a matching key points 
we check if the left0 key point is in its rec0_dic - means it passed
the rectified test, and the same for the left1 key point. If they both passed the test, it
means this feature is in the all 4 images, Therefore we choose its index at the
pair 0 rectified indexes and at the pair 1 rectified indexes. That way we create a new 
 **3** lists where each of them contains the indexes of the matches lists that were tracked
in all 4 images. It's important to note that those lists are "sharing indexes", That is the
ith value in each list points to an index at the matches list and those matches match to each 
other in the left0 and left1 list.

All this is done by:
```python
from utils.utills import find_kpts_in_all_4_rec
find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)
```
Let's denote the 2 lists (actually it's returns 3, but we use just two of them) with 
`tracked_pair0_idx` and `tracked_pair1_idx`, so finally the function returns :

```python
return pair0_matches[tracked_pair0_idx],  pair1_matches[tracked_pair1_idx]
```

#### Creating point cloud
As mentioned above, at the PnP algorithm we need tuples of (pixel, 3d point) such that the
3d point is projected to the pixel at the camera plane. So, we need triangulate
at the tracked features only.

This can be done simply by:
```python
from utils.utills import get_feature_obj, get_features_left_coor, \
                            get_features_right_coor, triangulate
# Get frame 0 Feature objects (which passed the rec test)
frame0_features = get_feature_obj(tracked_pair_0_matches, left0_kpts, right0_kpts)

# Here we take only their d2_points
left0_matches_coor = get_features_left_coor(frame0_features)
right0_matches_coor = get_features_right_coor(frame0_features)

# Frame 0 triangulation
pair0_p3d_pts = triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)
```

At the function ,at the first row, `get_feature_obj` we create a list of `Feature`
objects that represents a feature in a frame. This object can be found at the
`DataBaseDirectory`

For testing our PnP transformation result quality, We plotted two points cloud, one for first frame
and another for the second frame, each in its own coordinates system, and then We mapped the
first point cloud to the coordinates' system of the second frame, and we've the following result:

<img src=README_Images/DeterministicApproach/2PointsCloud.png width="350" height="280">


#### Finding location
Now that we have in hand the transformation between frame (i - 1) to frame i we 
can compute the ith camera location simply by:

<img src=README_Images/DeterministicApproach/findloc.png width="390" height="110">

Where the first equation stems from that the Transformation we have got from the PnP
transforms between world (Here it's the previous camera world) coordinates to the camera 
coordinates means that the camera location at the previous camera coordinate, denoted by
[x, y, z], is mapped by this transformation to [0, 0, 0] as the camera placed in the 
origin in its own coordinates system.

-------
Before moving along to the trajectory estimation lets strength our PnP model with the `RANSAC` 
method.

#### RANSAC
RANSAC, shortcut of Random Sample consensus is an algorithm that wish to estimate the 
parameters of a mathematical model from a data containing outliers by using iterative method. 
This method is based on the idea that by repeated points drawing we except to draw a set of point
which contains inliers only. The way we choose the best set of points is that after 
computing our model we "check" (depends on the model) how many points supports this model,
and the one who have the bigger set is the chosen one.  

Here, our mathematical model which we wish to model is the transformation between two frames 
using the PnP algorithm. So we use RANSAC to model our transformation better by reducing the
chance using outlier measure in the modeling. But actually we earn another thing, at the 
end of the RANSAC operation we get the transformation between the frames and a 
subgroup of feature that support this transformation,
so we can trust that they are inliers. That way, we have got another
inliers' rejection policy which we call it `Consensus match`

So, how does that works? There are two main things which RANSAC method is built from:
1. Iteration number.
2. Operations performed in each iteration.

**1. Iteration number** 

For the iteration number, there is basically two methods. The one is defining some constant 
number of iteration and the second is defining two parameters of evaluating the process and
calculate the number of iteration they yield. One parameter is the `probability` of not getting
an outlier we wish to have and the second is the data's `outliers percentage`. We call it 
"Online estimation ransac". In our project I used the second option.

Now we will talk about the way we estimate our number of iteration as a function of the 
two parameters that mentioned above, the probability of not getting an outlier, and the 
outlier percentage. We want a set of points which include inliers only. Denote with 'e' 
the real probability of being an outlier and 's' the number of points in the set. The 
probability of being drawing a set that has only inliers is (1 - e)^s thus the probability
of drawing a set that contains, at least, one outlier is 1 - (1 - e)^s. So, with N1 iteration, 
the probability of getting at least one outlier in the all N1 iteration is (1 - (1 - e)^s )^N1.
Now, we denote with P the desired probability we want that our model will get only inliers, so
1 - P is the desired probability of getting at least one outlier thus want that the 'real' 
probability of getting at least one outlier will be lower than 1 - P:

<img src=README_Images/DeterministicApproach/RansacIter.png width="" height="">

In practice, we don't know the outliers' percentage of our data, so we estimate 'e' online. That
Means that we raise N1 at each iteration until it will hold the inequality mentioned above.

A little remark about computing the outliers' percentage at each iteration. 
In fact, we don't know the outliers' percentage, and we assume that at every
iteration there are some inliers which are outliers and vice versa. 
Therefore, we use an estimation for the outliers and inliers percentage for each iteration. 
In each iteration we sum, separately, all the outliers and inliers we have got, and we compute
the outliers' percentage by a simple mean.

```python
accum_outlrs_num += num_points - num_supp  # # accum for accumulated
accuum_inliers_num += num_supp
outliers_perc = min(accum_outliers_num / (accum_inliers_num + accum_outliers_num), outliers_perc)
```
> `outliers_perc` is the parameter of the outliers' percentage. Since we are using an estimation 
that in expectation it equals to the real outliers' percentage there maybe some iterations
that this estimation is totally wrong, where we get percentage that is bigger than one what can
cause numeric problem at the iteration number estimation we had mentioned above.

**2. Operations performed in each iteration**

At each iteration:
1. Draw a set of pixels at the frame '1' left image.
2. Compute the transformation using `P4P`
3. Supports detection - `Consensus match` method
4. Save the transformation and the supporters set if it's the best.

At the end of that iteration:
1. Compute the transformation using the inliers set by `PnP`

> Notice: At the first iteration we use P4P and at the last one we use PnP which uses more points.

Here we can see the influence of the RANSAC on finding supporters. Each column is 
left 0 and left1 images:

<img src=README_Images/DeterministicApproach/RansacCompareSupporters.png width="" height="">

###### Implementation
PnP using RANSAC function:
```python
online_est_pnp_ransac(PNP_NUM_PTS, pair0_p3d_pts,
                      M1, left0_matches_coor,
                      M2, right0_matches_coor,
                      left1_matches_coor,
                      M2, right1_matches_coor,
                      K, acc=SUPP_ERR)
```

----
The last issue we have in the Ransac is the `Consensus match` in step 3. As mentioned above,
every RANSAC method includes the step of checking the supporters of the specific modeling, In
my code is the Consensus match. 

#### Consensus match
So far, we have a tracking between 2 frames in the all 4 images, so we use this great idea to 
define a transformation supporter. Given a transformation, computed by the P4P algorithm, we
go through all the 3d points and projects them on frame 1 and check if its distance
from the feature we tracked is small, up to some threshold. If so, we define it as a supporter.
By Defining a supporter that way we get that a supporter is a feature that agree in all 4 images
what makes it a very strong outliers' rejection.

<img src=README_Images/DeterministicApproach/ConsensusMatch.png width="350" height="280">

> Courtesy of David Arnon and Refael Vivanti

>**Mathematical remark**. Notice that the 3d point is at the left0 camera coordinates
so in order to project it all 4 cameras we need to find each camera's projection matrix.
The projection matrix depends on the intrinsic and extrinsic matrices. The intrinsic
one is the same among all 4 and there is a difference at the extrinsic matrices.
The left0's extrinsic matrices is simply [I | 0], right0 was given to us by 
KITTI, left1 founded by PnP, so it remains to calculate the right1 only.
We can calculate the right1 by composing the transformation between right1 and
left 1, that we know by KITTI, with the transformation between left0 and left1.
<img src=README_Images/DeterministicApproach/composeCam.png width="330" height="90">
> 
> Courtesy of David Arnon and Refael Vivanti

And mathematically speaking:

<img src=README_Images/DeterministicApproach/ComposeCamMath.png>

###### Implementation
This can be done at the following function:
```python
from utils.utills import compose_transformations
# Compute second @ first
compose_transformations(first_ex_mat, second_ex_mat)
```

Consensus match code:

```python
from utils.utills import consensus_match
consensus_match(calib_mat,left1_ex_mat, right1_to_left1_ex_mat, 
                world_p3d_pts, left1_matches_coor, right1_matches_coor,
                acc)
```
> This function returns a boolean list where the ith element indicate whether the ith
> key point is a supporter or not

### Localization - make it all together
Now we can create a trajectory estimation. For every frame i we can compute its
relative transformation to its previous frame, so we get a list of relative 
transformation, by composing them all we will get a **global** transformations,
were global means in relate to the first camera, thus we can compute each camera's
location. We put all together, and we get the following trajectory estimation:


<img src=README_Images/DeterministicApproach/InitialEstimate.png width="500" height="360">

> We have tried several options that you can find at the "Comparisons" part.

----
Now that we have in hand the initial estimation we can move on to solve the 
Bundle Adjustment, but, as we said, the Bundle Adjustment is build from set of cameras
that each of them sees several landmarks and each landmark is seen by several cameras, so we 
need to create a connection between cameras and landmarks. So far, we had tracked 
feature between consecutive frame, now we extend our feature tracking across
multiple frames and implement a suitable database for the features' tracking. 
<img src=README_Images/Database/featureTracking.png>

> Courtesy of David Arnon and Refael Vivanti

## Building feature's tracking Database
So, we want to extend our feature tracking ability from consecutive frames 
to multiple frames tracking. We will do this inductively, given the ith frame
we track features between it and its previous frame so for each feature 
there are 2 options:
1. If it's a new feature - create a new track.
2. If it's an old feature - connect it to the existing track.

At the end of this process we have a set of tracks were each track corresponds to a 3d point
in the world, which we called it a landmark. 

In addition, for every track we save the frames it appears at and for 
each frame we save all tracks it sees. Thus, we have the suitable suit for the 
database.

###### Implementation
For this mission we defined several classes:
1. `DataBase`
2. `Feature`
3. `Frame`
4. `Track`

All those classes are stored at the `DataBaseDirectory` directory.

In order to create a `DataBase` object we need to do a preliminary calculation
of computing all the features tracked between consecutive frames. This can be
done by a function we had already for creating the trajectory's initial estimation.

```python
from utils.utills import find_features_in_consecutive_frames_whole_movie
# M1 = [I | 0]
find_features_in_consecutive_frames_whole_movie(first_left_ex_cam_mat=M1)
```
> This function returns 4 objects. The first is a list of tuples of lists where
> each tuple represent the features between two consecutive frames where the lists
> in the tuple are sharing indexes, means that the ith feature in the first list
> match to the ith feature in the second list. The second object is the inliers 
> percentage that of key frames that been tracked. The third  and the last
> objects are the global and the relative transformations of each frame.
 
Now, we can create a database object:
```python
from DataBaseDirectory.DataBase import DataBase
db = DataBase(consecutive_frame_features, inliers_percentage, global_trans,
              relative_trans)
```
Running that row will run the DataBase's object method `create_db` that performs
the inductive process of connecting tracks that is actually done by the following
function:
```python
db.find_tracks_between_consecutive_frame(first_frame_features_obj, 
                                         second_frame_features_obj,
                                         first_frame_id, second_frame_id)
```

### Evaluating tracking quality
We defined several measures for evaluating our tracking quality:

#### Tracking statistics
| Tracking statistics            | header |
|--------------------------------|--------|
| Total tracks number            | 280077 | 
| Frames number                  | 3450   |
| Max track                      | 126    | 
| Min track                      | 2      | 
| Mean track length              | 5.24   |
| Tracks number in average frame | 424.99 |

#### Example of tracking with length of 10

<img src=README_Images/Database/track.png >

> For convince we cropped the image to 100X100 for better view.

#### Connectivity graph
This graph represents, For each frame, the number of tracks outgoing
to the next frame, Means the number of tracks on the frame with links
also in the next frame.
<img src=README_Images/Database/Connectivity.png >

#### Track length histogram
<img src=README_Images/Database/Tracklengthhistogram.png width="500" height="310">

-----
Now we have our database in hand we can continue to main algorithm, the Bundle Adjustment.

## Back to the Bundle Adjustment
Because there is some noise in our measures, we want to add an uncertainty factor
to this process. Being more formally,
given a set of measures, denoted by Z, we want to find
a set of cameras, denoted by C, and a set of landmarks, denoted by Q, 
that maximize the conditional probability of C,Q under the condition of Z.

To continue from here we would assume the following assumptions:
- The measure is normally distributed around the "real" pixels with some covariance
- MLE estimation at the probability of some pair C,Q - means that we assume there is no pair
that different from others.
- The measures are independents in each other, 
each measure depends on its corresponding camera and landmark including 
some, normally distributed with zero mean and identity covariance, noise.

So by using those assumptions, bayes role and Cholesky decomposition we get:

<img src=README_Images/BundleAdjustmentPart/BundleFormal.png >

>delta z is the re projection error mentioned above and is defined to be the difference 
between the projection of landmark q_i on c_j and our measure z_{i,j}

### Bundle adjustment as a Least square problem

One can notice that this problem is the Least square problem but due to the fact that
the projection operation is not linear as it includes non-linear operation as 
dividing the projection result with the homogenous coordinate.

So solving Bundle adjustment, under our assumptions, is actually solving the Least square
problem. Recall that solving the Least square problem is done iteratively where in each
iteration we find the best step to step by in order to minimize our function. 
In practice, we will
use `GRSAM`'s implementation of the Levenberg-Marquardt algorithm which one can think
about it as a mixture of the Gradient descent and the Gauss-Newton algorithms.

Since the `Levenberg-Marquardt` algorithm there is a very importance of the initialization
of the cameras and landmarks poses that the algorithm starts with. The initial 
estimation will be explained later in details.

### Bundle adjustment freedom degrees
 
The last issue is the freedom degrees at the Bundle adjustment problem.
There are 7 freedom degree that are divided into 3 parts:
1. Scaling - scaling the whole system will not have influence on the solution.
2. Rotating - rotating the whole system.
3. Translation - moving the system to other location.


In the KITTI benchmark the scale is defined by the stereo camera baseline, so we
need to determine the rotation and translation. We will do this by setting the first
camera location.

For implementing the Bundle adjustment algorithm we will use as **Factor graph**
### Factor graph
A factor graph is a graph where each vertex represents some object that we want to 
find its values and each edge represents a constraint between two objects.
At the Bundle Adjustment problem we have:
- Vertices are Cameras or landmarks.
- Edges are the projection of a landmark to a camera.
As mentioned above Edge's constraint is the measure of the landmark on the camera. 

So actually we are converting our bayesian graph to the factor graph:

<img src=README_Images/BundleAdjustmentPart/BayedianGraphToFactorGraph.png width="600" height="200">

> Courtesy of David Arnon and Refael Vivanti 


#### Factor graph and GTSAM library
In our project, We use GTSAM - Georgia Tech Smoothing and Mapping Library -
for the factor graph optimization.

The main thing we need to notice is that so far we defined a transformation
, an extrinsic camera matrix, of a camera as a mapping from the world coordinates
to camera coordinates. At GTSAM things works the opposite, a transformation, or
as it called in GTSAM a `Pose3`, is a mapping from the camera coordinates to
the world coordinates. So in order to work with gtsam we convert our
transformation's directions. Mathematically we can do this by: 

<img src=README_Images/BundleAdjustmentPart/GtsamTrans.png>

This is done by the function:
```python
from utils.utills import convert_ex_cam_to_cam_to_world
cam_to_world_ex_cam = convert_ex_cam_to_cam_to_world(world_to_cam_ex_cam)
```

### Bundles "windows"
Due to the fact that solving the Least square problem using the Levenberg-Marquardt 
algorithm involve inverting a very large dimension covariance matrix 
which is not efficient and numerically unstable operation, we divide the whole trajectory 
to sub trajectories, creating a little bundle for each one and solving it.

We perform local Bundle Adjustment on a small window consisting of consecutive frames.
Each bundle ‘window’ starts and ends in special frames we call keyframes.
Each bundle window consists 2 key frames and all the frames between them.
It is important to notice that the last bundle window's key frame and the first
bundle window's key frame overlap

<img src=README_Images/BundleAdjustmentPart/BundleWindows.png >

> Courtesy of David Arnon and Refael Vivanti 

##### Choosing key frames
We choose the key frame iteratively by the following way. For the last chosen key frame
we look at all tracks in it and chose the **"median" track's length** to be the distance
between the current key frame to the next one, where the track's length
computed by The difference between the last frame in the track and the current key frame.
We quoted the word "median" because actually we take the track's length that is greater
than **82%** of the other tracks.

###### Implementation
Calling the following row will create a `BundleAdjustment` object
that contains a list of `BundleWindow` objects by the key frames choosing criteria 
mentioned above:
```python
import BundleAdjustmentDirectory.BundleAdjustment as BundleAdjustment
bundle_adjustment = BundleAdjustment.BundleAdjustment()
```

### Solving Bundles Windows and multiprocessing
Solving each bundle locally means that all cameras and landmarks in it are related
to the, by our choise, first camera. So at the end of solving all bundles we get 
a list of cameras that each of them is related to the previous one. In addition, all 
landmarks in each bundle are also related to the first key frame in it, so to
represent this system of cameras and landmarks in one coordinates system
we need to transform each bundle elements to the global system.

So each bundle is solved separately so this process of solving all bundles can be done 
by using multiprocessing. In our code we used 5 process.

###### Implementation
Calling the following rows will solve the Bundle Adjustment.

```python
import BundleAdjustmentDirectory.BundleAdjustment as BundleAdjustment

# For multiprocessing
bundle_adjustment.solve(BundleAdjustment.MULTI_PROCESSED)
# For iterative process
bundle_adjustment.solve(BundleAdjustment.ITERATIVE_PROCESS)
```

###### Run time issues
The iterative method runs in ~ 8 minutes

The multiprocessing method runs in ~ 2.8 minutes

Choosing key frames process ~ 40 seconds.

### Creating each bundle's Factor graph and choosing landmarks
For each bundle we need build a factor graph contains:
- All cameras between keyframes
- All landmarks at those frames

Adding cameras is quite simple and does not require any special approach, we
only need to compute the cameras relative to the first bundle transformation
that can be done easily by composing the current camera's global transformation with
the first camera's inverse global transformation.
Adding landmarks is not as adding cameras that, unlike adding cameras, 
some approaches may give us better results.
In our implementation we added the graph landmarks that appears in *all*
bundle's cameras, so we are passing through the tracks which appears at the first frame
and chose the ones that their last frame is greater than the last bundle's key frame
or equals to it. Now, we can create a `factor` of the factor graph.

For a given track, which passed the rejection above, we perform a `Triangulation` 
from the last frame and get a 3d point. This 3d point is used as an initial estimation
for that landmark, and we create a constraint between that landmark and all cameras
in the bundle (recall landmarks rejection policy we had made) by a measure of that 
landmarks that we get from our `DataBase` object.

Remarks:
- Each camera in the bundle get an initial estimation for its pose and an expected 
covariance and after the optimization it updates it also return relative pose's
covariance that will serve us later.
- We preform another rejection for landmarks. We reject very far landmarks
having in mind that such kind of landmarks are tends to be inaccurate and
in addition we make troubles at the optimization because their uncertainty is large
what reflected in numeric issues when inverting the covariance matrix at 
the optimization process. This is done by the `Triangulation` rejection policy we had mentioned
above.
- We assume that each bundle has a low error so the relative pose between the last key
frame and the first one is pretty accurate. The whole problem starts because
we are estimating a trajectory of 3450 frame, and about 432 key frames, therefore
even a small error accumulates into a large error. We will see in the future how 
fix this.

###### Implementation
Calling the following row will create a bundle:

```python
# Bundle window creation
from BundleAdjustmentDirectory.BundleAdjustment import BundleWindow
bundle_window = BundleWindow(first_key_frame, second_key_frame)
bundle_window.create_factor_graph()

```

Creating factors:
```python
import gtsam, numpy
# Projection factor
proj_covariance = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
proj_factor = gtsam.GenericStereoFactor3D(measurement, proj_uncertainty,
                                     cam_sym, landmark_sym, gtsam_calib_mat)

# Prior factor at the bundle's first camera
pose_covariance = numpy.array([(5 * numpy.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])
pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
pose_factor = gtsam.PriorFactorPose3(left_pose_sym, left_cam_pose, pose_uncertainty)

```

>At the `proj_covariance` we choose the identity covariance matrix meaning that
we assume that our detector is wrong up to 1 pixel (with std dist of 1) at the
left's x coordinate, right's x coordinates and the y coordinates on both images.
> 
>At the `pose_covariance` the first 3 values are for the euler angles: azimut, pitch and
roll, and we chose an uncertainty of 5 angles for each of them, while the second
3 are for the 3d location: x, y, z, here we chose 30cm uncertainty at the x axes, 
10 cm at the y axes and 1 meter at the z axes.

### Bundle Adjutment Results 
This graph shows the car trajectory over the driving scene. 
You can see that theirs a *pink* trajectory which is the initial estimation for 
the `Bundle Adjustment` algorithm. The red one is the cameras poses after the
optimization and the one in cyan is the ground-truth trajectory as received from
KITTI's benchmark.

<img src=README_Images/BundleAdjustmentPart/BundleResult.png >

---
So far, we have got a pretty good estimation for the car trajectory, It can be seeing
that the red trajectory is sitting almost perfectly at the cyan one besides some area
that it seems that the car had returned to the same road but the Bundle Adjustment
algorithm shows it as a different road. For making thing "tight" we'll add our final,
very powerful one, algorithm called "Loop closure", And from now one we are throwing 
away all are frames, that are not key frames, and landmarks, so we are left with
the key frames only, each one will be a vertex in a new `Factor Graph` 
called `Pose Graph`.

## Loop closure
Loop closure is an algorithm for finding "loops" at a given trajectory graph. 
By saying "loop" we mean two cameras that has the same pose (up to some distance).
The basic concept is that when we find a loop we actually add another constraint
to are graph and by finding good loops we can "repair" are trajectory retroactively.

<img src=README_Images/LoopClosure/LoopClosure.png width="300" height="300">

> Courtesy of Agrawal, Konolige

We can look at this process of finding loop as puzzle's closing circle. We are
building the puzzle frame one after the other and when we get the loop we connect
each other, and we close the loop.

<img src=README_Images/LoopClosure/puzzle.png width="400" height="200">

> Applying Information Theory to Efficient SLAM - M.Chli 
-----
At the next rows we will talk about how we find a loop, but before lets defined 
the graph that we will search loops at it which called `Pose Graph`.
### Pose Graph
Pose graph is no other than Factor Graph where:
- Vertex are camera poses
- Edges are relative pose between two edges.

#### Initializing pose graph
As mentioned above, from the Bundle Adjustment algorithm we receive a list 
of bundles where each bundle provides us with a relative pose between its two
key frames and a covariance of that relative pose. We first build the pose graph
with the key frames cameras as the vertex, the relative pose as edges, aka factors,
and we also use the covariance we got from the bundle for the factor uncertainty.

#### Pose Graph edge's error
As in the Bundle Adjustment's Factor graph, this graph also contains constraint
but here the constraint are poses, so we need to define error of those edges 
that "asks" what is the probability of one pose to be the other one,
After we will have that error, per edge, we can represent the total graph error
by the mahalanobis distance of that error from zero:

<img src=README_Images/BundleAdjustmentPart/PoseGraphError.png >


Here we calculate "How much two poses are closer" and by recalling that each pose
can be converted to a transformation matrix and vice versa we can define the following 
"metrix" :

<img src=README_Images/BundleAdjustmentPart/PoseGraphErrorDefined.png>


where `transformationToVector` convert a transformation matrix to a pose vector
with euler angles.

Now that we have the pose graph in our hands we can move on to find loops.

## Finding loops
In this process of finding loops we will actually need to find a transformation
between two cameras. As mentioned, this is done by the `consensus match`
method which is heavy operation. So, for avoiding perform this operation, for
a given camera, on every previous camera we will use a lighter operation
for finding candidates. We will follow those steps:
1. Find candidates by geometric intersection with a mahalanobis distance
estimation
2. Validate candidates by consensus match
3. Calculate edges and factors

### First step - Find candidates by geometric intersection 
as mentioned,  we defined the error as the mahalanobis distance between two poses,
for calculating this distance we will need to calculate the 
relative transformations, between those poses and the relative covariance
matrix of the second pose related to the first pose.
The related transformation is easy to compute and the relative covariance
we could calculate this by using marginalization and conditioning at the
Pose Graph covariance matrix, but it includes inverting the covariance matrix
which is a big matrix, so we will prefer using an estimation.

For conveniently, we denote current camera as C_n and previous camera C_i
amd we want to find their relative covariance

Relative covariance estimation - There are two options:
1. There is a one path only from previous camera to the current one.
2. There is more than one path - let's assume two only.

At the first case, we can use the covariance role which says that : 
<img src=README_Images/BundleAdjustmentPart/SumCov.png >

Where x and y represents our poses. We have to say that, of course, that summing
poses is not doing by vector summing but because we are dealing with small bundles, 
about 10-20 frames, we can assume that there is not a big difference between 
the vector poses, so we can use the regular sum as the expected sum. Another
thing is that summing over long path will result in a large error but 
the covariance values are pretty small, so we can assume that we can still work 
with that estimation. To conclude, in this case we can just sum the covariances
along the path and get ans estimation for the relative covariance.

At the second case, where we have two paths between C_i and C_n, its actually 
means that there are two estimation for the C_n - one from the first path
and one from the other path. Here we can look at the next covariance role:

<img src=README_Images/BundleAdjustmentPart/MultCov.png >

Since those measurements are independent we can look at their intersection 
probability as a multiplication, and in that case the covariance is given above.

But calculating the covariance by this way includes 3 operations of inverting
which are not recommended. Instead we will take the "little" covariance between
the two, and it will be an estimation. We can prove that in the worst case, 
where the two covariances are equals we give up at about 30% accuracy percentage
but at the standard case we lose about 11%. That means are estimation does 
not damage accuracy a lot.

To conclude, for finding the relative covariance of C_n in relate to C_i
we can just find the path with the shortest distance between C_i and C_n and sum
the covariance all over the path.

All that is done by calling the function:

```python
# Finds loops candidate by the mahalanobis distance condition
mahalanobis_dist_cand = pose_graph.find_loop_cand_by_mahalanobis_dist(cur_cam)

```

#### Finding the shortest path

The next issue is to find the shortest path between two poses at the graph. 
For that purpose, we 
convert our Pose Graph to a "Vertex" representation graph that represent 
the graph with adjacency matrix list, so we can apply the `Dijkstra`'s version
that includes `Min Heap` and we get a total run time of O(E * Log V)

```python
# Find the shortest path and estimate its relative covariance
shortest_path = vertex_graph.find_shortest_path(prev_cam, cur_cam)
estimated_rel_cov = vertex_graph.estimate_rel_cov(shortest_path)
```

`vertex_graph` is a VertexGraph object. `find_shortest_path` method is actually
a dijkstra implementation with `Min Heap` class that we create and can be found
at the `utils` directory. 


#### Edge's weighting
As we said before, we want to the find the shortest path in terms of the 
"smallest" covariance obtained. There are several ways to define a "size"
of a matrix:
1. Frobenius norm
2. Trace - In this method we ignore the "ellipse" shape
3. Determinant

Let's explain a little about the 3rd option. We can look at the transformation 
determinant as the space of the unit square under the transformation operation.
We can say that covariance is "small" if its ellipsis's space is smaller
(up to some mahalanobis distance). Now, if we look at the rectangle that
blocks the ellipsis, and we can say that its space is an approximation for 
the ellipsis. If we look at the 2d case we can see that the rectangle space
is a * b where a and b are the ellipsis axes. and by applying `SVD` on the
covariance matrix we will get that the values at the 
diagonal matrix in the SVD decomposition are a^2 and b^2, and by computing 
the covariance determinant through the SVD decomposition its easy to see that
the determinant of the covariance equals to a^2 * b^2, so the determinant root 
is a good approximation for the covariance "size".

Summarize first step - for some C_n we search for a candidate in all cameras C_i
between the first camera in the trajectory and C_n, We compute its relative pose,
finds a shortest path between them to compute the relative covariance for
the mahalanobis distance calculation, and we choose only the 3 best 
cameras of those who passed the distance threshold

#### Second and third step - Candidates validation and factors evaluation
Now that we have some good candidates we are ready to preform the consensus match
method. On every candidate, we preform the consensus match method that tells us the
match's inliers percentage between those images. We define some `INLIERS_THRESHOLD_PERC`
and we left with the candidates which passed the threshold. Now, if there is not
any loop, we stop that moment and moving forward to the next camera. If there
is some loop, we add it to the `Pose Graph` as a factor and to the `Vertex Graph`.

A small note, for computing the relative pose we are not satisfied by the 
result of the consensus match, and we are applying a little bundle, containing only
two frames, those that we found that moment, and it gives us the relative pose
and their relative covariance.

The above can be computed with:

```python
consen_frame_track_tuples = find_loop_cand_by_consensus(
                                        mahalab_cand_movie_ind,
                                        mahalab_dist, cur_frame_movie_ind,
                                        INLIERS_THRESHOLD_PERC)
```

Implementation remark, `find_loop_cand_by_consensu` actually return a list
of tuples where the first element is the previous camera that match to the `cur_frame`
and the second element is a list of tracks that founds at the consensus and will
serve us when computing the bundle.

All this process of loops finding can be done by:
```python
loop_prev_frames_tracks_tuples = pose_graph.find_loop_closure(cur_cam)
```

again `loop_prev_frames_tracks_tuples` contains tuples of previous keyframes and
tracks list.

### Optimization process
So we know how to find loops. Now we left with performing this process along the
whole pose graph. We start with the first key frames and for every key frame, we
find loops candidate, if we found loops, we add it to the factor and vertex graph and 
performing optimization.

This process is done by:
```python
# Performing loop closure on specific key frame
 pose_graph.loop_closure_for_specific_frame(cur_cam)
 
# performing loop closure for the whole pose graph
 pose_graph.loop_closure()
```

## Results
Finally, we get the following result:

<img src=README_Images/LoopClosure/LoopTraj.png width="560" height="420">


# Comparisons while working
## intial estimating trajectory
### Sift, Akaze, Brute Force, Knn, Flann
<img src=README_Images/DeterministicApproach/Comparisons/AkazeSift.png width="800" height="1000">

## Bundle adjustment
### Bundle percentage for key frame choosing
We check several percentages for the key frame choosing, here we represent the
percentage and it's influence at the bundle trajectory:

#### 0.65
<img src=README_Images/BundleAdjustmentPart/Comparisons/0.65.png width="400" height="300">

#### 0.7
<img src=README_Images/BundleAdjustmentPart/Comparisons/0.7.png width="400" height="300">

#### 0.8
<img src=README_Images/BundleAdjustmentPart/Comparisons/0.8.png width="400" height="300">

#### 0.85
<img src=README_Images/BundleAdjustmentPart/Comparisons/0.85.png width="400" height="300">

#### 0.9
<img src=README_Images/BundleAdjustmentPart/Comparisons/0.9.png width="400" height="300">


So we check the values at [0.855, 0.856, 0.857, 0.858, 0.859].
## Loop Closure

By manually examining the trajectory we have found that there are 3 possible areas
for loops as follows:

<img src=README_Images/LoopClosure/SuspectedLoops.png width="500" height="400">

So, we had printed cameras indicators at the first range, and we have got:

<img src=README_Images/LoopClosure/189and9.png width="570" height="440">

We can see that 189 and 9 are close, indeed when we look at their matching percentage
we get 95% percentage:

<img src=README_Images/LoopClosure/189and9per.png width="600" height="450">

Thus, we find all suspected areas for loop and find the mahalanobis distance and the inliers percentage
accordingly




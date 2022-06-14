# Slam project
This project were taken under the course 
"Computer vision based navigation" - 67604 that was passed at the 
Hebrew university of jerusalem at the year 2023 by Mr David Arnon and Dr 
Refael Vivanti.
This is my summarize in hebrew for te course:
[Course summarize](http://www.github.com)

## Overview
Slam, shortcut of **S**imultaneous **l**ocalization **a**nd **m**apping, 
is a family of problems that, for a moving object, 
try to localize it and creates a world mapping.
I accomplish this mission using the bundle adjustment algorithm which is 
an algorithm for solving the Slam issue. 
In this project I implement, that in some meaning is a refined version for the
"Frame slam" article ~~Add Link~~.

In this project we are using the concepts:
- Bundle Adjustment
- Loop closure
- triangulation
- pnp
- ransac
- consensus match

The data I am using is of Kitti's benchmark 
In this project we are trying to estimate a trajectory of a moving car
that has stereo camera on its roof, and it takes photos for some time.

~~Add some terms~~
- world coordinates
- cameras as poses of the car
- mapping - points cloud


## Kitti's Benchmark
Kitty is a ~~Add content about this company~~ company. 

Kitty uses a car includes some sensors that drives along some streets
in Germany . The sensors are stereo camera (color and black-white), gps,
lidar and some other.
This benchmark supplies a lot of ground truth which allows us to 
compare are results of our algorithms.
In my project I use only the black and white stereo cameras.

~~Add kitti's supplies~~
- Intrinsic cameras
- Extrinsic cameras Left and right
- 3450 Frames

## Bundle Adjustment
### Theory
As mentioned above, we are doing localization and mapping, localization
is finding object's location in "world" ~~Explain about it~~ coordinates
and mapping is creating some mapping of object's in the world.

Every camera in the whole trajectory Sees several objects (we call them landmarks)
in the world and every object in the world is seeing by several cameras. 
By saying "Seeing by a camera" we means that it can be projected to the camera plane.
So we can look at this structure as "Bundles" that suppose to fit perfectly. As 
we can see at the following image:

Thanks to Courtesy of Ackermann

<img src=README_Images/BundleAdjustmentPart/BundleStructure.png width="300" height="200">


So we want to fit the bundles, means that we want that all cameras and landmarks
poses would be in a place that the landmarks projections would fit to 
our measures. "measure" means pixels in some camera that some landmark 
appears at. Practically, bundle adjustment algorithm get as input, 
Cameras, Landmarks, measures and outputs cameras and landmarks poses that
minimize the re projection error.

Because there is some noise in our measures, we want to add an uncertainty factor
to this process so getting more formally, in the bundle adjustment algorithm
given a set of measures, denoted by Z, we want to find
a set of cameras, denoted by C, and a set of landmarks, denoted by Q, 
that maximize the conditional probability of C,Q under the condition of Z.

To continue from here we would assume the following assumptions:
- The measure is normally distributed around the "real" pixels with some covariance
- MLE estimation at the probability of some pair C,Q - means that we assume theirs no pair that different from other
- The measures are independents in each other, each measure depends on its corresponding camera and landmark including some normally distributed with zero mean and identity covariance noise  

so by using those assumptions, bayes role and Cholesky decomposition we get:

<img src=README_Images/BundleAdjustmentPart/BundleFormal.png width="460" height="205">

delta z is the re projection error mentioned above and is defined to be the difference 
between the projection of landmark q_i on c_j and our measure z_{i,j}

#### Bundle adjustment as a Least square problem

One can notice that this problem is the Least square problem but due to the fact that
the projection operation is not linear as it includes dividing the projection result
with the homogenous coordinate which is non-linear operation.

So solving Bundle adjustment, under our assumptions, is actually solving the Least square
problem. Recall that solving a Least square problem is done iteratively where in each
iteration we find the best step to step in to minimize our function. In practice we will
use gtsam's implementation of the Levenberg-Marquardt algorithm which one can think
about it as a mixture of Gradient descent and Gauss-Newton algorithms.

Since the Levenberg-Marquardt algorithm there is a very importance of the initialization
of the cameras and landmarks poses that the algorithm starts with. The initial 
estimation will be explained later in details.

#### Bundle adjustment freedom degrees
 
The last issue is the freedom degrees at the Bundle adjustment problem.
There are 7 freedom degree that are divided into 3 parts:
1. Scaling - scaling the whole system will not have influence on the solution.
2. Rotating - rotating the whole system.
3. Translation - moving the system to other location.

In the Kitty benchmark the scale is defined by the distance between the cameras, so we
need to determine the rotation and translation that we will do by setting the first
camera location

### Practical
For implementing the Bundle adjustment algorithm we will use as **Factor graph**
#### Factor graph
A factor graph is a graph where each vertex represents some object that we want to 
find its values and each edge represents a constraint between two objects.
At the Bundle Adjustment problem:
- Vertices are Cameras or landmarks.
- Edges are the projection of a landmark to a camera.
As mentioned above Edge's constraint is the measure of the landmark on the camera.

#### Bundles "windows"
Due to the fact that solving the Least square problem using the Levenberg-Marquardt 
algorithm involve inverting a very large dimension covariance matrix 
which is not efficient and numerically unstable, we divide the whole trajectory 
to sub trajectories, creating a little bundle for each one and solving it.

We perform local Bundle Adjustment on a small window consisting of consecutive frames.
Each bundle ‘window’ starts and ends in special frames we call keyframes.
Each bundle window consists 2 key frames and all the frames between them.
It is important to notice that the last bundle window's key frame and the first
bundle window's key frame overlap

<img src=README_Images/BundleAdjustmentPart/BundleWindows.png width="400" height="140">

##### Choosing key frames
We choose the key frame iteratively by the following way. For the last chosen key frame
we look at all tracks in it and chose the **"median" track's length** to be the distance
between the current key frame to the next one, where the track's length
computed by The difference between the last frame in the track and the current key frame.
We quoted the word "median" because actually we take the track's length that is greater
than **82%** of the other tracks. ~~Maybe change it condier knn~~



###### Implementation
Calling the following row will create a `BundleAdjustment` object
that contains a list of `BundleWindow` objects by the key frames choosing criteria 
mentioned above
```python
import BundleAdjustmentDirectory.BundleAdjustment as BundleAdjustment
bundle_adjustment = BundleAdjustment.BundleAdjustment()
```

#### Solving Bundles Windows and multiprocessing
Solving each bundle locally means that all cameras and landmarks in it are related
to the first camera (by our choice) so at the end of solving all bundles we get 
a list of cameras that each of them is related to the previous one. In addition, all 
landmarks in each bundle are also related to the first key frame in it, so when to
represent this system of cameras and landmarks we need to transform each bundle elements
to the global system.

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
The iterative method runs in ~ 10 minutes

The multiprocessing method runs in ~ 3.5 minutes

Choosing key frames process ~ 1.5 minute.

#### Creating each bundle's Factor graph and choosing landmarks
For each bundle we need build a factor graph contains:
- All cameras between keyframes
- All landmarks at those frames

Adding cameras is quite simple and does not require any special approach, we
only need to compute the cameras relative to the first bundle transformation
that can be done easily by composing the current camera's global transformation with
the first camera's inverse global transformation.
Adding landmarks is not as adding camera where, unlike adding cameras, 
some approaches may give us better results.
In our implementation we added the graph landmarks that appears in all
bundle's cameras, so we are passing through the tracks which appears at the first frame
and chose the ones that their last frame is greater than the last bundle's key frame
or equals to it. Now, we can create a `factor` of the factor graph.

For a given track, which passed the rejection above, we perform a `Triangulation` 
from the last frame and get a 3d point. This 3d point is used as an initial estimation
for that landmark, and we create a constraint between that landmark and all cameras
in the bundle (recall landmarks rejection policy we had made) by a measure of that 
landmarks that we get from our `DataBase` object.

Remark:
- We preform another rejection for landmarks. We reject very far landmarks
having in mind that such kind of landmarks are tends to be inaccurate and
in addition we make troubles at the optimization because their uncertainty is large
what reflected in numeric issues when inverting the covariance matrix at 
the optimization process. If the point is not that far, the landmark's left's 
x coordinate should be greater than the right one up to some threshold

The following code performs the last rejection: 

```python
# measure_xl is the measure's x coordinate at the frame's left image's
if abs(measure_xl - measure_xr) < DIST_THRESHOLD:
    return
```



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
# Factor creation
proj_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
factor = gtsam.GenericStereoFactor3D(measurement, proj_uncertainty,
                                     cam_sym, landmark_sym, gtsam_calib_mat)
```

At the `proj_unceirtainty` we choose the identity covariance matrix meaning that
we assume that our detector is wrong up to 1 pixel (with std dist of 1) at the
left's x coordinate, right's x coordinates and the y coordinates on both images.

### Results 
This graph shows the car trajectory over the driving scene. 
You can see that theirs a *pink* trajectory which is the initial estimation for 
the `Bundle Adjustment` algorithm. The red one is the cameras poses after performing
optimization and the one in cyan is the ground-truth trajectory as received from
Kitty's benchmark.

<img src=README_Images/BundleAdjustmentPart/BundleResult.png width="450" height="400">


# Helpers for file

###### Implementation
Calling the following rows will 

```python
# comments
Code

```

#### images
<img src=README_Images/BundleAdjustmentPart/BundleWindows.png width="400" height="140">











## Initial estimation for the bundle adjustment
#### Theory
Enter here : triangulation, pnp ,ransac ,consensus match

#### Practical

## Data base
knn - time 7:05 minutes

## Loop Closure
#### Theory
Explanation about Loop closure
#### Practical

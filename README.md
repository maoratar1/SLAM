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

So actually we are converting our bayesian graph to the factor graph:

Courtesy of David Arnon and Refael Vivanti 

<img src=README_Images/BundleAdjustmentPart/BayedianGraphToFactorGraph.png width="600" height="200">

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

<img src=README_Images/BundleAdjustmentPart/BundleWindows.png width="450" height="180">

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

Remarks:
- Each camera in the bundle get an initial estimation for its pose and an expected 
covariance and after the optimization it updates it also return relative pose's
covariance that will serve us later.
- We preform another rejection for landmarks. We reject very far landmarks
having in mind that such kind of landmarks are tends to be inaccurate and
in addition we make troubles at the optimization because their uncertainty is large
what reflected in numeric issues when inverting the covariance matrix at 
the optimization process. If the point is not that far, the landmark's left's 
x coordinate should be greater than the right one up to some threshold
- We assume that each bundle has a low error so the relative pose between the last key
frame and the first one is pretty accurate. The whole problem starts because
we are estimating a trajectory of 3450 frame, and about 432 key frames, therefore
even a small error accumulates into a large error. We will see in the future how 
fix this.
- 
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
# Projection factor
proj_covariance = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
proj_factor = gtsam.GenericStereoFactor3D(measurement, proj_uncertainty,
                                     cam_sym, landmark_sym, gtsam_calib_mat)

# Prior factor at the bundle's first camera
pose_covariance = numpy.array([(5 * numpy.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])
pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
pose_factor = gtsam.PriorFactorPose3(left_pose_sym, left_cam_pose, pose_uncertainty)

```

At the `proj_covariance` we choose the identity covariance matrix meaning that
we assume that our detector is wrong up to 1 pixel (with std dist of 1) at the
left's x coordinate, right's x coordinates and the y coordinates on both images.

At the `pose_covariance` the first 3 values are for the euler angles: azimut, pitch and
roll, and we chose an uncertainty of 5 angles for each of them, while the second
3 are for the 3d location: x, y, z, here we chose 30cm uncertainty at the x axes, 
10 cm at the y axes and 1 meter at the z axes.

### Results 
This graph shows the car trajectory over the driving scene. 
You can see that theirs a *pink* trajectory which is the initial estimation for 
the `Bundle Adjustment` algorithm. The red one is the cameras poses after performing
optimization and the one in cyan is the ground-truth trajectory as received from
Kitty's benchmark.

<img src=README_Images/BundleAdjustmentPart/BundleResult.png width="500" height="420">

---
Until now, we have got a pretty good estimation for the car trajectory, It can be seeing
that the red trajectory is sitting almost perfectly at the cyan one besides some area
that it seems that the driving car had returned to the same road but the Bundle Adjustment
algorithm shows it as a different road. For making thing "tight" we'll add our final,
very powerful one, algorithm called "Loop closure", And from now one we are throwing 
away all are frames, that are not key frames, and landmarks, so we are left with
the key frames only, each one will be a vertex in a new `Factor Graph` 
called `Pose Graph`.

## Loop closure
### Theory
Loop closure is an algorithm for finding "loops" at a given trajectory graph. 
By saying "loop" we mean two cameras that has the same pose (up to some distance).
The basic concept is that when we find a loop we actually add another constraint
to are graph and by finding good loops we can "repair" are trajectory retroactively.
At the next rows we will talk about how we find a loop, but before lets defined 
the graph that we will search loops at it which called `Pose Graph`.

#### Pose Graph
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

<img src=README_Images/BundleAdjustmentPart/PoseGraphError.png width="340" height="60">


Here we calculate "How much two poses are closer" and by recalling that each pose
can be converted to a transformation matrix and vice versa we can define the following 
"metrix" :

<img src=README_Images/BundleAdjustmentPart/PoseGraphErrorDefined.png width="490" height="60">


where `transformationToVector` convert a transformation matrix to a pose vector
with euler angles.

Now that we have the pose graph in our hands we can move on to find loops.

### Finding loops
In this process of finding loops we will actually need to find a transformation
between two cameras. As mentioned, this is done by the `consensus match`
method which is heavy operation. So, for avoiding perform this operation, for
a given camera, on every previous camera we will use a lighter operation
for finding candidates. We will follow those steps:
1. Find candidates by geometric intersection with a mahalanobis distance
estimation
2. Validate candidates by consensus match
3. Calculate edges and factors

#### First step - Find candidates by geometric intersection 
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
<img src=README_Images/BundleAdjustmentPart/SumCov.png width="570" height="60">

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

<img src=README_Images/BundleAdjustmentPart/MultCov.png width="585" height="110">


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

The next issue is to find the shortest path in the graph. For that purpose, we 
convert our Pose Graph to a "Vertex" representation graph that represent 
the graph with adjecancy matrix list, so we can apply the `Dijkstra`'s version
that includes `Min Heap` so we get a total run time of O(E * Log V)

```python
# Find the shortest path and estimate its relative covariance
shortest_path = vertex_graph.find_shortest_path(prev_cam, cur_cam)
estimated_rel_cov = vertex_graph.estimate_rel_cov(shortest_path)
```

`vertex_graph` is a VertexGraph object. `find_shortest_path` method is actually
a dijkstra implementation with `Min Heap` class that we create and can be found
at the `utils` directory. 


##### Edge's weighting
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
consen_frame_track_tuples = find_loop_cand_by_consensus(mahalab_cand_movie_ind,
                                        mahalab_dist,
                                        cur_frame_movie_ind,
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

### Results

<img src=README_Images/BundleAdjustmentPart/BundleWindows.png width="400" height="140">







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

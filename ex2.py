import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import utils
import numpy as np
import cv2

DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'
INLIER_COLOR = "orange"
OUTLIER_COLOR = "cyan"
IMAGE_HEIGHT = 376.0
# == 2.1 == #


def axes_of_matches(match, img1_kpts, img2_kpts):
    """
    Returns (x,y) values for each point from the two in the match object
    """
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    x1, y1 = img1_kpts[img1_idx].pt
    x2, y2 = img2_kpts[img2_idx].pt
    return x1, y1, x2, y2


def dev_from_pattern(match, img1_kpts, img2_kpts):
    """
    Computes the difference between match's y values
    """
    _, y1, _, y2 = axes_of_matches(match, img1_kpts, img2_kpts)
    return abs(y2 - y1)


def deviations_from_pattern(matches, img1_kpts, img2_kpts):
    """
    Apply the function dev_from_pattern for each match in matches
    """
    return [dev_from_pattern(match, img1_kpts, img2_kpts) for match in matches]


def create_hist(matches, img1_kpts, img2_kpts):
    """
    Creates a histogram of deviations from the stereo pattern
    """

    # Create a list of deviations for each match (len deviations = len matches)
    deviations = deviations_from_pattern(matches, img1_kpts, img2_kpts)

    # Plots the image
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(deviations, bins=50)

    ax.set_title("Histogram of deviations from pattern")
    plt.ylabel('Number of matches')
    plt.xlabel('Deviation from rectified stereo pattern')

    fig.savefig("VAN_ex/histogram.png")
    plt.close(fig)


def percentage_of_dev(matches, img1_kpts, img2_kpts):
    """
    Computes the percentage of matches the deviate by more than 2 pixels and prints it
    """
    deviations = deviations_from_pattern(matches, img1_kpts, img2_kpts)
    num_dev_cond = sum(map(lambda dev: dev > 2, deviations))
    matches_num = len(matches)
    print(f"Percentage of matches that deviate by more than 2 pixels: {num_dev_cond / matches_num * 100}%")


def rectified_stereo_pattern_rej(img1_kpts, img2_kpts, matches):  # Todo: Efficiency for further exercises
    """
    Apply the rectified stereo pattern rejection on image1 key d2_points and image 2 key d2_points
    :return: Inliers and outliers d2_points of the 2 KITTI
    """
    img1_inliers, img2_inliers, img1_outliers, img2_outliers = [], [], [], []

    for match in matches:
        _, y1, _, y2 = axes_of_matches(match, img1_kpts, img2_kpts)
        img1_idx, img2_idx = match.queryIdx, match.trainIdx
        if abs(y2 - y1) < 1:
            img1_inliers.append(img1_kpts[img1_idx].pt)
            img2_inliers.append(img2_kpts[img2_idx].pt)
        else:
            img1_outliers.append(img1_kpts[img1_idx].pt)
            img2_outliers.append(img2_kpts[img2_idx].pt)

    return np.array(img1_inliers), np.array(img2_inliers), np.array(img1_outliers), np.array(img2_outliers)


# == Helpers for uniform question == #
def rectified_stereo_pattern_rej2(img1_kpts, img2_kpts, matches):
    img1_inliers, img2_inliers, img1_outliers, img2_outliers = [], [], [], []

    for match in matches:
        _, y1, _, _ = axes_of_matches(match, img1_kpts, img2_kpts)
        y2 = np.random.uniform(0.0, IMAGE_HEIGHT)
        img1_idx, img2_idx = match.queryIdx, match.trainIdx
        if abs(y2 - y1) <= 1:
            img1_inliers.append(img1_kpts[img1_idx].pt)
            img2_inliers.append(img2_kpts[img2_idx].pt)
        else:
            img1_outliers.append(img1_kpts[img1_idx].pt)
            img2_outliers.append(img2_kpts[img2_idx].pt)

    return np.array(img1_inliers), np.array(img2_inliers), np.array(img1_outliers), np.array(img2_outliers)


def check_uniform(img1_kpts, img2_kpts, matches):
    iteration = 100000
    s_i = s_o = 0
    for i in range(iteration):
        img1_inliers, _, img1_outliers, _ = rectified_stereo_pattern_rej2(img1_kpts, img2_kpts, matches)
        s_i += len(img1_inliers)
        s_o += len(img1_outliers)

    a_i = s_i // iteration
    a_o = s_o // iteration
    print(f"Inliers and outliers average when y are distributed uniformly (over {iteration} iteration): ")
    print(f"Inliers: {a_i} Outliers: {a_o}")


# -- End for uniform helpers -- #


def draw_matches_with_rectified_stereo_pattern_rej(img1, img2, img1_inliers, img2_inliers,
                                                   img1_outliers, img2_outliers):
    """
    Draws the  inliers (matches that passed the rectified stereo pattern test) with orange
    and the outliers (the ones that failed the test) with cyan
    """

    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1
    fig.suptitle(f'Left/Right camera outliers({OUTLIER_COLOR}) and inliers({INLIER_COLOR}) \n'
                 f' {len(img1_outliers)} Outliers, {len(img1_inliers)} Inliers')

    # Left camera
    fig.add_subplot(rows, cols, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Left camera")
    plt.scatter(img1_inliers[:, 0], img1_inliers[:, 1], s=1, color=INLIER_COLOR)
    plt.scatter(img1_outliers[:, 0], img1_outliers[:, 1], s=1, color=OUTLIER_COLOR)

    # Right camera
    fig.add_subplot(rows, cols, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("Right camera")
    plt.scatter(img2_inliers[:, 0], img2_inliers[:, 1], s=1, color=INLIER_COLOR)
    plt.scatter(img2_outliers[:, 0], img2_outliers[:, 1], s=1, color=OUTLIER_COLOR)

    fig.savefig("VAN_ex/out-inliers.png")
    plt.close(fig)


def linear_least_square(l_cam_mat, r_cam_mat, kp1_xy, kp2_xy):
    """
    Linear least square procedure.
    :param l_cam_mat: Left camera matrix
    :param r_cam_mat: Right camera matrix
    :param kp1_xy: (x,y) for key point 1
    :param kp2_xy: (x,y) for key point 2
    :return: Solution for the equation Ax = 0
    """
    # Compute the matrix A
    mat = np.array([kp1_xy[0] * l_cam_mat[2] - l_cam_mat[0],
                    kp1_xy[1] * l_cam_mat[2] - l_cam_mat[1],
                    kp2_xy[0] * r_cam_mat[2] - r_cam_mat[0],
                    kp2_xy[1] * r_cam_mat[2] - r_cam_mat[1]])

    # Calculate A's SVD
    u, s, vh = np.linalg.svd(mat, compute_uv=True)

    # Last column of V is the result as a numpy object
    return vh[-1]


def triangulate(l_mat, r_mat, kp1_xy_lst, kp2_xy_lst):  # Todo : Efficiency in numpy and array create
    """
    Apply triangulation procedure
    :param l_mat: Left camera matrix
    :param r_mat: Right camera matrix
    :return: List of 3d d2_points in the world
    """
    kp_num = len(kp1_xy_lst)
    res = []
    for i in range(kp_num):
        p4d = linear_least_square(l_mat, r_mat, kp1_xy_lst[i], kp2_xy_lst[i])
        p3d = p4d[:3] / p4d[3]
        res.append(p3d)
    return np.array(res)


def draw_triangulations(p3d_pts, cv_p3d_pts):
    """
    Draws the 3d d2_points triangulations (Our and open-cv)
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    rows, cols = 1, 2
    fig.suptitle(f'Open-cv and our Triangulations compare')

    # Our results
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    ax.set_title("Our Result")
    ax.scatter3D(0, 0, 0, c='red', s=40)  # Camera
    ax.scatter3D(p3d_pts[:, 0], p3d_pts[:, 1], p3d_pts[:, 2])
    ax.set_xlim3d(10, -20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-100, 200)

    # Cv results
    ax = fig.add_subplot(rows, cols, 2, projection='3d')
    ax.set_title("Open-cv result")
    ax.scatter3D(0, 0, 0, c='red', s=40)  # Camera
    ax.scatter3D(cv_p3d_pts[:, 0], cv_p3d_pts[:, 1], cv_p3d_pts[:, 2])
    ax.set_xlim3d(-20, 10)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-100, 200)

    fig.savefig(f"VAN_ex/triangulations plot.png")
    plt.close(fig)


def compare_triangulations_pts_values_avg(p3d_pts, cv_p3d_pts):
    """
    Compares the triangulations (our and open-cv) by computing the average distance of all point
    :param p3d_pts: Our p3d d2_points
    :param cv_p3d_pts: cv d2_points
    """
    dist = np.linalg.norm(p3d_pts - cv_p3d_pts)
    print(f"Average of distances between matching d2_points: {dist / len(p3d_pts)}")


def triangulation_on_sequence_images():
    num_imgs = 1
    total_p3d_pts = []
    for i in range(num_imgs):
        # Read the KITTI
        img1, img2 = utils.read_images(utils.IDX + i)

        # Compute matching and inliers
        img1_kpts, _, img2_kpts, _, matches = utils.read_detect_and_match(img1, img2)
        img1_inliers, img2_inliers, _, _ = rectified_stereo_pattern_rej(img1_kpts, img2_kpts, matches)

        # Triangulate
        k, m1, m2 = utils.read_cameras()
        p3d_pts = triangulate(k @ m1, k @ m2, img1_inliers, img2_inliers)

        total_p3d_pts += p3d_pts.tolist()

    total_p3d_pts = np.array(total_p3d_pts)

    # Plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.axes(projection='3d')
    ax.set_title(f"Triangulation over {num_imgs} KITTI")
    ax.scatter3D(0, 0, 0, c='red', s=40)
    ax.scatter3D(total_p3d_pts[:, 0], total_p3d_pts[:, 1], total_p3d_pts[:, 2])
    x_dim, y_dim, z_dim = 10, 20, 10
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_xlim3d(10, -20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-20, 0)
    ax.view_init(30, 70)

    fig.savefig(f"VAN_ex/seqImgPlot.png")
    plt.close(fig)


def ex2():
    # Read KITTI
    img1, img2 = utils.read_images(utils.IDX)

    # Compute key d2_points and matches
    img1_kpts, _, img2_kpts, _, matches = utils.read_detect_and_match(img1, img2)

    # # 2.1 - Creates histogram and computes percentage
    create_hist(matches, img1_kpts, img2_kpts)
    percentage_of_dev(matches, img1_kpts, img2_kpts)

    # 2.2 - Rectified stereo test rejection
    img1_inliers, img2_inliers, img1_outliers, img2_outliers = rectified_stereo_pattern_rej(img1_kpts, img2_kpts,
                                                                                            matches)
    draw_matches_with_rectified_stereo_pattern_rej(img1_inliers, img2_inliers, img1_outliers, img2_outliers)

    # 2.3 Triangulation
    # Read cameras
    k, m1, m2 = utils.read_cameras()

    # Our triangulation
    p3d_pts = triangulate(k @ m1, k @ m2, img1_inliers, img2_inliers)

    # Cv triangulation
    cv_p4d_pts = cv2.triangulatePoints(k @ m1, k @ m2, img1_inliers.T, img2_inliers.T).T
    cv_p3d_pts = np.squeeze(cv2.convertPointsFromHomogeneous(cv_p4d_pts))

    # Compare Results. Plots and Average distance
    draw_triangulations(p3d_pts, cv_p3d_pts)
    compare_triangulations_pts_values_avg(p3d_pts, cv_p3d_pts)

    # Triangulate over sequence of KITTI
    triangulation_on_sequence_images()


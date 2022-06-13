import time
import matplotlib.pyplot as plt
import tqdm

from utils import utills
import numpy as np
import cv2
import cProfile
import pstats
from DataDirectory import Data

# == Constants == #

DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'
LEFT_CAM_TRANS_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/poses/00.txt'
INLIER_COLOR = "orange"
OUTLIER_COLOR = "cyan"
IMAGE_HEIGHT = 376.0
PNP_NUM_PTS = 4
SUPP_ERR = 2
MOVIE_LEN = utills.MOVIE_LEN
K, M1, M2 = Data.KITTI.get_K(), Data.KITTI.get_M1(), Data.KITTI.get_M2()


# == Functions == #
def create_pts_cloud(img1, img2):
    """
    Return d2_points cloud of two KITTI by:
        1. find key d2_points matches in the two KITTI
        2. reject key d2_points by the rectified policy rejection
        3. apply triangulation
    :return numpy array of 3d d2_points shape (Key point num) X 3
    """
    # Compute key d2_points and matches
    img1_kpts, _, img2_kpts, _, matches = utills.detect_and_match(img1, img2)

    # Rectified stereo test rejection
    inliers_matches_idx, _ = utills.rectified_stereo_pattern_rej(img1_kpts, img2_kpts, matches)
    inliers_matches = matches[inliers_matches_idx]
    img1_matches_coor, img2_matches_coor = utills.get_matches_coor(inliers_matches, img1_kpts, img2_kpts)

    # Triangulation
    pts_cloud = utills.triangulate(K @ M1, K @ M2, img1_matches_coor, img2_matches_coor)

    return pts_cloud


def rodriguez_to_mat(R_vec, t_vec):
    """
    Compute rotation matrix from rotation vector and returns it as
    numpy array [R|t] with shape 3 X 4
    :param R_vec: Rotation vector
    :param t_vec: Translation vector
    """
    R_mat, _ = cv2.Rodrigues(R_vec)
    return np.hstack((R_mat, t_vec))


def pnp(world_p3d_pts, img_proj_coor, calib_mat, flag):
    """
    Apply the pnp algorithm
    :param calib_mat: Image's calibration matrix
    :param flag: cv2 SOLVEPNP method
    :return: If pnp succeed returns Camera's Extrinsic [R|t] matrix else None
    """
    success, R_vec, t_vec = cv2.solvePnP(world_p3d_pts, img_proj_coor, calib_mat, None, flags=flag)
    ex_cam_mat = None
    if success:
        ex_cam_mat = rodriguez_to_mat(R_vec, t_vec)
    return ex_cam_mat


def relative_camera_pos(ex_cam_mat):
    """
    Finds and returns the Camera position at the "world" d2_points
    :param ex_cam_mat: [Rotation mat|translation vec]
    """
    # R = extrinsic_camera_mat[:, :3]
    # t = extrinsic_camera_mat[:, 3]
    return -1 * ex_cam_mat[:, :3].T @ ex_cam_mat[:, 3]


def relative_camera_pos_4(left0_ex_mat, right0_ex_mat, left1_ex_mat, right1_ex_mat):
    """
    Finds and returns the Camera position at the "world" d2_points of two stereo KITTI
    '0' index is for the first pair and '1' for the second pair
    """
    left0_pos = relative_camera_pos(left0_ex_mat)
    right0_pos = relative_camera_pos(right0_ex_mat)
    left1_pos = relative_camera_pos(left1_ex_mat)
    right1_pos = relative_camera_pos(right1_ex_mat)
    return left0_pos, right0_pos, left1_pos, right1_pos


def compose_transformations(first_ex_mat, second_ex_mat):
    """
    Compute the composition of two extrinsic camera matrices.
    first_cam_mat : A -> B
    second_cam_mat : B -> C
    composed mat : A -> C
    """
    # [R2 | t2] @ [ R1 | t1] = [R2 @ R1 | R2 @ t1 + t2]
    #             [000 | 1 ]
    hom1 = np.append(first_ex_mat, [np.array([0, 0, 0, 1])], axis=0)
    return second_ex_mat @ hom1


def find_supporters_4_frame(pair0_p3d_pts,
                            left1_proj_mat, left1_matches_coor,
                            right1_proj_mat, right1_matches_coor,
                            acc=SUPP_ERR):
    """
    Finds the supporter in all 4 KITTI
    :param pair0_p3d_pts: 3d d2_points from pair 0 triangulation
    :param left0_proj_mat: Left0 projection camera matrix: K[R|t]
    :param left0_matches_coor: Corresponds pixel location in the image plane to the 3d d2_points
    :param acc: Accuracy - distance from 3d point projection to correspond left0 pixel
    :return: Supporters indexes from left1 matches d2_points
    """
    # Finds the projection of the 3d d2_points to the KITTI plane
    left1_proj = project(pair0_p3d_pts, left1_proj_mat)
    right1_proj = project(pair0_p3d_pts, right1_proj_mat)

    # Finds correct projections
    left1_supp_indicator = check_projection(left1_proj, left1_matches_coor, acc)
    right1_supp_indicator = check_projection(right1_proj, right1_matches_coor, acc)

    # Finds the 3d d2_points that satisfies all the conditions
    all_supp_indicators = left1_supp_indicator & right1_supp_indicator  #Todo: checking now
    supporters_idx = np.where(all_supp_indicators == 1)[0]

    return supporters_idx


def project(p3d_pts, cam_proj_mat):
    """
    Project 3d point to the camera's image plane
    :return: numpy array of projections (num p3d point) X 2
    """
    hom_proj = p3d_pts @ cam_proj_mat[:, :3].T + cam_proj_mat[:, 3].T
    proj = hom_proj[:, :2] / hom_proj[:, [-1]]
    return proj


def check_projection(img_projected_pts, img_pts_coor, acc=SUPP_ERR):
    """
    Check the euclidean dist between the projected d2_points and correspond pixel locations
    and returns a boolean vector that indicates which row satisfy the condition
    :param acc: Euclidean distance
    :return:boolean vector
    """
    img_pts_diff = img_projected_pts - img_pts_coor  # (x1, y1), (x2, y2) -> (x1 - x2, y1 - y2)
    left0_dist = np.einsum("ij,ij->i", img_pts_diff, img_pts_diff)  # (x1 - x2)^2 + (y1 - y2)^2
    q_acc = acc ** 2
    return left0_dist <= q_acc


def create_rec_dic(pair_matches, pair_rec_matches_idx):
    """
    Create dictionary in the form that for each key point of the left image:
                    {index in the list of key d2_points: index in the matches list}
    :param pair_matches: matches between left(query) and right(train) KITTI
    :param pair_rec_matches_idx: list of matches indexes (from pair matches list) that passed the rectified test
    :return: Dictionary
    """
    rec1_dic = {}
    for idx in pair_rec_matches_idx:
        kpt_id = pair_matches[idx].queryIdx
        rec1_dic[kpt_id] = idx
    return rec1_dic


def find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic):
    """
    Finds matching key d2_points in left0 and left1 that passed the rectified test.
    :param left0_left1_matches: matches between left0 and left1 KITTI
    :param rec0_dic: Dictionary which contains the key d2_points indexes that passed the rectified test in pair0
    :param rec1_dic: same as above for pair1
    :return: 3 lists:
                    1) q_pair0: list of indexes from pair0 matches list that passed the rectified test in pair 0
                                and are matched in left0 and left1
                    2) q_pair1: same as above for pair1
                    3) q_left0_left1: same as above for left0_left1 matches list
    """
    q_pair0, q_pair1, q_left0_left1 = [], [], []

    for i, match in enumerate(left0_left1_matches):
        left0_kpt_idx = match.queryIdx
        left1_kpt_idx = match.trainIdx

        # Explanation:
        # Check if those key d2_points passed the rectified test in each pair
        # if so it adds to q_pair0 it's index in the list of pair 0 matches
        # so in index i, q_pair0[i] contains the index of the match in
        # pair 0 matches that match to pair 1 matches by the connector from
        # the matches between left0 and left1
        # for example if i'th match is <left0_kpt, left1_kpt> and left0_kpt and left1_kpt passed the rectified test
        # we get that pair0idx[i] = <left0_kpt, right0_kpt> (actually the pair0 index of this pair)
        #             pair1idx[i] = <left1_kpt, right1_kpt> (same as above)
        #             q_left0_left1[i] = <left0_kpt, left1_kpt> (same as above)
        if left0_kpt_idx in rec0_dic and left1_kpt_idx in rec1_dic:
            q_pair0.append(rec0_dic[left0_kpt_idx])
            q_pair1.append(rec1_dic[left1_kpt_idx])
            q_left0_left1.append(i)

    return q_pair0, q_pair1, q_left0_left1


def online_est_pnp_ransac(model_parm_num, world_p3d_pts,
                          left0_ex_mat, left0_matches_coor,
                          right0_ex_mat, right0_matches_coor,
                          left1_matches_coor,
                          right1_to_left1_ex_mat, right1_matches_coor,
                          calib_mat=K, acc=SUPP_ERR):  # Todo: check what to do with projecion on pair 0
    """
    Apply Ransac method to estimate the transformation using pnp with online computation of the iteration number
    :return: Extrinsic left1 camera matrix
    """
    # Basic variables
    max_supp, max_supp_group_idx = -1, None
    inliers_num, outliers_num = 0, 0
    first_loop_iter = 0
    first_loop_iter_est = lambda prob, outliers_perc: np.log(1 - prob) / np.log(
        1 - np.power(1 - outliers_perc, model_parm_num))
    outliers_perc, prob = 0.99, 0.99

    # Compute pair0 camera's projection matrices
    # left0_proj_mat = calib_mat @ left0_ex_mat
    # right0_proj_mat = calib_mat @ right0_ex_mat

    # Ransac loop
    while outliers_perc != 0 and first_loop_iter < first_loop_iter_est(prob, outliers_perc):

        # Get randomize d2_points with amount of "model_param_num" and estimate the model by them
        pts_idx = np.random.randint(0, len(left1_matches_coor), model_parm_num)
        left1_ex_mat = pnp(world_p3d_pts[pts_idx], left1_matches_coor[pts_idx], calib_mat, cv2.SOLVEPNP_AP3P)
        # Sanity check
        if left1_ex_mat is None:
            continue

        # Compute left1 and right1 projection matrices
        right1_to_left0_ex_mat = compose_transformations(left1_ex_mat, right1_to_left1_ex_mat)
        left1_proj_mat = calib_mat @ left1_ex_mat
        right1_proj_mat = calib_mat @ right1_to_left0_ex_mat

        # Find model's supporters
        supporters_idx = find_supporters_4_frame(world_p3d_pts,
                                                 left1_proj_mat, left1_matches_coor,
                                                 right1_proj_mat, right1_matches_coor,
                                                 acc=acc)

        # Check which model is the best
        num_supp = len(supporters_idx)
        if num_supp > max_supp:
            max_supp = num_supp
            max_supp_group_idx = supporters_idx

        first_loop_iter += 1

        # Estimate outliers percentage
        outliers_num += len(left1_matches_coor) - num_supp
        inliers_num += num_supp
        outliers_perc = min(outliers_num / (inliers_num + outliers_num), 0.99)

    # Refine the model by estimating the model by all d2_points
    best_T = extra_refine(max_supp_group_idx, world_p3d_pts, left1_matches_coor, right1_to_left1_ex_mat,
                          right1_matches_coor, calib_mat, acc, it_num=1)

    return best_T, max_supp_group_idx


def extra_refine(supp_idx, world_p3d_pts,
                 left1_matches_coor,
                 right1_to_left1_ex_mat, right1_matches_coor,
                 calib_mat=K, acc=SUPP_ERR, it_num=1):
    """
    Apply the model refining several times. Estimate model from inliers and inliers from model
    """

    left1_ex_mat = None
    for i in range(it_num):
        left1_ex_mat = pnp(world_p3d_pts[supp_idx], left1_matches_coor[supp_idx], calib_mat, flag=cv2.SOLVEPNP_ITERATIVE)
        # Sanity check
        if left1_ex_mat is None:
            continue

        # Compute left1 and right1 projection matrices
        right1_to_left0_ex_mat = compose_transformations(left1_ex_mat, right1_to_left1_ex_mat)
        left1_proj_mat = calib_mat @ left1_ex_mat
        right1_proj_mat = calib_mat @ right1_to_left0_ex_mat

        # Find model's supporters
        supp_idx = find_supporters_4_frame(world_p3d_pts,
                                           left1_proj_mat, left1_matches_coor,
                                           right1_proj_mat, right1_matches_coor,
                                           acc=acc)

    return left1_ex_mat


# == version 2 from running the computing for next exercises ====================================================
def compute_trans_between_left0_left1(left0, right0, left1, right1):
    """
    Compute the transformation T between left 0 and left1 KITTI
    :return: numpy array with shape 3 X 4
    """
    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, _, pair0_matches = utills.detect_and_match(left0, right0)
    pair0_rec_matches_idx, _ = utills.rectified_stereo_pattern_rej(left0_kpts, right0_kpts, pair0_matches)

    # Find matches in pair 1 with rectified test
    left1_kpts, left1_dsc, right1_kpts, right1_dsc, pair1_matches = utills.detect_and_match(left1, right1)
    pair1_rec_matches_idx, _ = utills.rectified_stereo_pattern_rej(left1_kpts, right1_kpts, pair1_matches)

    # Find matches between left0 and left1
    left0_left1_matches = utills.matching(left0_dsc, left1_dsc)

    # Find key pts that match in all 4 KITTI
    rec0_dic = create_rec_dic(pair0_matches, pair0_rec_matches_idx)  # dict of {left kpt idx: pair rec frame_idx}
    rec1_dic = create_rec_dic(pair1_matches, pair1_rec_matches_idx)
    q_pair0_idx, q_pair1_idx, q_left0_left1_idx = find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)

    # Pair0 triangulation
    left0_matches_coor, right0_matches_coor = utills.get_matches_coor(pair0_matches[q_pair0_idx], left0_kpts,
                                                                     right0_kpts)
    pair0_p3d_pts = utills.triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)

    # Compute left1 and right 1 matches d2_points in the image
    left1_matches_coor, right1_matches_coor = utills.get_matches_coor(pair1_matches[q_pair1_idx], left1_kpts,
                                                                     right1_kpts)

    # Find the best transformation between left0 and left1 with ransac
    best_left1_cam_mat, max_supp_group_idx = online_est_pnp_ransac(PNP_NUM_PTS, pair0_p3d_pts,
                                                                   M1, left0_matches_coor,
                                                                   M2, right0_matches_coor,
                                                                   left1_matches_coor,
                                                                   M2, right1_matches_coor,
                                                                   K, acc=SUPP_ERR)

    return best_left1_cam_mat


def whole_movie2(first_left_ex_mat=M1):
    # This function runs the whole movie with the function that
    # works on each 2 pairs every time means that it computes the i and i+1
    # every time
    """
    Compute the transformation of two consequence left KITTI in the whole movie
    :return:array of transformations where the i'th element is the transformation between i-1 -> i
    """
    T_arr = [first_left_ex_mat]

    for i in range(MOVIE_LEN - 1):
        if i % 100 == 0:
            print("Pair ", i + 1)
        left0, right0 = utills.read_images(utills.IDX + i)
        left1, right1 = utills.read_images(utills.IDX + (i + 1))

        left1_ex_mat = compute_trans_between_left0_left1(left0, right0, left1, right1)
        T_arr.append(left1_ex_mat)

    return T_arr
# ===== End of old code =============================


def compute_trans_between_cur_to_next(left0_kpts, left0_dsc, right0_kpts,
                                      pair0_matches, pair0_rec_matches_idx,
                                      left1_kpts, left1_dsc, right1_kpts,
                                      pair1_matches, pair1_rec_matches_idx):
    """
   Compute the transformation T between left 0 and left1 KITTI
   :return: numpy array with shape 3 X 4
   """

    # Find matches between left0 and left1
    left0_left1_matches = utills.matching(left0_dsc, left1_dsc)

    # Find key pts that match in all 4 KITTI
    rec0_dic = create_rec_dic(pair0_matches, pair0_rec_matches_idx)  # dict of {left kpt idx: pair rec frame_idx}
    rec1_dic = create_rec_dic(pair1_matches, pair1_rec_matches_idx)
    q_pair0_idx, q_pair1_idx, q_left0_left1_idx = find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)

    # Pair0 triangulation
    left0_matches_coor, right0_matches_coor = utills.get_matches_coor(pair0_matches[q_pair0_idx], left0_kpts,
                                                                     right0_kpts)
    pair0_p3d_pts = utills.triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)

    # Compute left1 and right 1 matches d2_points in the image
    left1_matches_coor, right1_matches_coor = utills.get_matches_coor(pair1_matches[q_pair1_idx], left1_kpts,
                                                                     right1_kpts)

    # Find the best transformation between left0 and left1 with ransac
    best_left1_cam_mat, max_supp_group_idx = online_est_pnp_ransac(PNP_NUM_PTS, pair0_p3d_pts,
                                                                   M1, left0_matches_coor,
                                                                   M2, right0_matches_coor,
                                                                   left1_matches_coor,
                                                                   M2, right1_matches_coor,
                                                                   K, acc=SUPP_ERR)

    return best_left1_cam_mat


def read_and_rec_match(idx):
    """
    Reads KITTI from pair idx, Finds matches with rectified test
    :param idx: Frame's index
    :return: key d2_points of the two KITTI and the matches
    """
    # Find matches in pair0 with rectified test
    left0, right0 = utills.read_images(idx)
    kernel_size = 10
    left0 = cv2.blur(left0, (kernel_size, kernel_size))
    right0 = cv2.blur(right0, (kernel_size, kernel_size))
    left0_kpts, left0_dsc, right0_kpts, _, pair0_matches = utills.detect_and_match(left0, right0)
    pair0_rec_matches_idx, _ = utills.rectified_stereo_pattern_rej(left0_kpts, right0_kpts, pair0_matches)
    return left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx


def whole_movie(first_left_ex_mat=M1):
    """
    Compute the transformation of two consequence left KITTI in the whole movie
    :return:array of transformations where the i'th element is the transformation between i-1 -> i
    """
    T_arr = [first_left_ex_mat]

    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = utills.read_and_rec_match(0, kernel_size=10)
    for i in tqdm.tqdm(range(1, MOVIE_LEN)):
        left1_kpts, left1_dsc, right1_kpts, pair1_matches, pair1_rec_matches_idx = utills.read_and_rec_match(i, kernel_size=10)

        left1_ex_mat = compute_trans_between_cur_to_next(left0_kpts, left0_dsc, right0_kpts,
                                                         pair0_matches, pair0_rec_matches_idx,
                                                         left1_kpts, left1_dsc, right1_kpts,
                                                         pair1_matches, pair1_rec_matches_idx)
        left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = left1_kpts, left1_dsc,\
                                                                                   right1_kpts, pair1_matches,\
                                                                                   pair1_rec_matches_idx
        T_arr.append(left1_ex_mat)

    return T_arr


def compute_whole_movie_time(first_left_ex_mat=M1):
    """
    Compute whole movie computation time
    """
    t0 = time.perf_counter()
    T_arr = whole_movie(first_left_ex_mat)  #Todo: check the difference between whole movie to consec
    # _, _, _, T_arr = utills.find_features_in_consecutive_frames_whole_movie()
    t1 = time.perf_counter() - t0
    return T_arr, t1 / 60


def left_cameras_trajectory(relative_T_arr):
    """
    Computes the left cameras 3d positions relative to the starting position
    :param T_arr: relative to first camera transformations array
    :return: numpy array with dimension num T_arr X 3
    """
    relative_cameras_pos_arr = []
    for t in relative_T_arr:
        relative_cameras_pos_arr.append(relative_camera_pos(t))
    return np.array(relative_cameras_pos_arr)


def left_cameras_relative_trans(T_arr):
    relative_T_arr = []
    last = T_arr[0]

    for t in T_arr:
        last = compose_transformations(last, t)
        relative_T_arr.append(last)

    return relative_T_arr


def get_ground_truth_transformations(left_cam_trans_path=LEFT_CAM_TRANS_PATH, movie_len=MOVIE_LEN):
    """
    Reads the ground truth transformations
    :return: array of transformation
    """
    T_ground_truth_arr = []
    with open(left_cam_trans_path) as f:
        lines = f.readlines()
    for i in range(movie_len):
        left_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
        T_ground_truth_arr.append(left_mat)
    return T_ground_truth_arr


def compare_trajectories():
    """
    Computes the trajectory of the left camera and plots it in 2d and 3d with the ground truth
    """
    T_arr = whole_movie(M1)
    left_camera_trajectory_plot(T_arr)

    T_ground_truth_arr = get_ground_truth_transformations()

    relative_trans = left_cameras_relative_trans(T_arr)
    relative_cameras_pos_arr = left_cameras_trajectory(relative_trans)
    ground_truth_relative_cameras_pos_arr = left_cameras_trajectory(T_ground_truth_arr)

    compare_left_cam_3d_trajectory_to_ground_truth(relative_cameras_pos_arr, ground_truth_relative_cameras_pos_arr)
    compare_left_cam_2d_trajectory_to_ground_truth(relative_cameras_pos_arr, ground_truth_relative_cameras_pos_arr)


def compare_trajectories_params(T_arr):
    """
    Computes the trajectory of the left camera and plots it in 2d and 3d with the ground truth
    """

    T_ground_truth_arr = get_ground_truth_transformations()

    relative_trans = left_cameras_relative_trans(T_arr)
    relative_cameras_pos_arr = left_cameras_trajectory(relative_trans)
    ground_truth_relative_cameras_pos_arr = left_cameras_trajectory(T_ground_truth_arr)

    compare_left_cam_3d_trajectory_to_ground_truth(relative_cameras_pos_arr, ground_truth_relative_cameras_pos_arr)
    compare_left_cam_2d_trajectory_to_ground_truth(relative_cameras_pos_arr, ground_truth_relative_cameras_pos_arr)


def check_bottleneck():
    """
    Check bottleneck at the code
    """
    profile = cProfile.Profile()
    profile.runcall(whole_movie, M1)
    ps = pstats.Stats(profile)
    ps.sort_stats('cumtime')
    ps.print_stats()


def check_feature_det_algs(alg, alg_name, match_method):
    utills.ALG = alg
    print(alg_name)
    T_arr, whole_time = compute_whole_movie_time()
    print("Time: ", whole_time)

    relative_trans = left_cameras_relative_trans(T_arr)
    relative_cameras_pos_arr = left_cameras_trajectory(relative_trans)

    T_ground_truth_arr = get_ground_truth_transformations()
    ground_truth_relative_cameras_pos_arr = left_cameras_trajectory(T_ground_truth_arr)

    compare_left_cam_2d_trajectory_to_ground_truth_params(relative_cameras_pos_arr, ground_truth_relative_cameras_pos_arr,
                                                   alg_name, match_method, whole_time)


# == Plot functions ==
def plot_3d_pts_cloud(pts1, pts2):
    """
    Plots 2 3d point clouds
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.axes(projection='3d')
    ax.set_title(f"3d Point clouds(red - current view, cyan - transformed from other view")
    ax.scatter3D(pts1[:, 0], pts1[:, 1], pts1[:, 2], c='red')
    ax.scatter3D(pts2[:, 0], pts2[:, 1], pts2[:, 2], c='cyan')
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-100, 100)

    fig.savefig(f"Results/2 pts cloud.png")
    plt.close(fig)


def draw_relative_3d_pos(left0_pos, right0_pos, left1_pos, right1_pos):
    """
    Draws the Cameras 3d positions in the "world"
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(projection='3d')
    ax.set_title("Relative 3d cameras positions (first pair - red, second pair - cyan)")
    ax.scatter3D(left0_pos[0], left0_pos[1], left0_pos[2], c='red')
    ax.scatter3D(right0_pos[0], right0_pos[1], right0_pos[2], c='red')
    ax.scatter3D(left1_pos[0], left1_pos[1], left1_pos[2], c='cyan')
    ax.scatter3D(right1_pos[0], right1_pos[1], right1_pos[2], c='cyan')
    ax.xlabel("x")
    ax.xlabel("height")
    ax.zlabel("Driving direction")
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)

    fig.savefig(f"Results/Relative 3d camera positions.png")
    plt.close(fig)


def draw_relative_2d_pos(left0_pos, right0_pos, left1_pos, right1_pos):
    """
    Draws the Cameras 2d positions in the "world"
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Our results
    ax = fig.add_subplot()
    ax.set_title("Relative 2d cameras positions (first pair - red, second pair - cyan)")
    ax.scatter(left0_pos[0], left0_pos[2], c='red')
    ax.scatter(right0_pos[0], right0_pos[2], c='red')
    ax.scatter(left1_pos[0], left1_pos[2], c='cyan')
    ax.scatter(right1_pos[0], right1_pos[2], c='cyan')
    plt.xlabel("x")
    plt.ylabel("Driving direction")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    fig.savefig(f"Results/Relative 2d camera positions.png")
    plt.close(fig)


def plot_supporters(left0, left1, left0_matches_coor, left1_matches_coor, supporters_idx):
    """
    Plot KITTI supporters on left0 and left1
    """
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1
    fig.suptitle(f'Left0 and Left1 supporters({len(supporters_idx)} / {len(left1_matches_coor)}, {INLIER_COLOR})')

    # Left1 camera
    fig.add_subplot(rows, cols, 2)
    plt.imshow(left1, cmap='gray')
    plt.title("Left1 camera")
    plt.scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    plt.scatter(left1_matches_coor[supporters_idx][:, 0],
                left1_matches_coor[supporters_idx][:, 1], s=3, color=INLIER_COLOR)

    # Left0 camera
    fig.add_subplot(rows, cols, 1)
    plt.imshow(left0, cmap='gray')
    plt.title("Left0 camera")
    plt.scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    plt.scatter(left0_matches_coor[supporters_idx][:, 0],
                left0_matches_coor[supporters_idx][:, 1], s=3, color=INLIER_COLOR)

    fig.savefig("Results/Left0 Left1 supporters.png")
    plt.close(fig)


def draw_left_cam_3d_trajectory(left_cameras_pos):
    """
    Draw left camera 3d trajectory
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(projection='3d')
    ax.set_title("Left cameras 3d trajectory")
    ax.scatter3D(left_cameras_pos[:, 0], left_cameras_pos[:, 1], left_cameras_pos[:, 2], s=1, c='red')

    fig.savefig("Results/Left cameras 3d trajectory.png")
    plt.close(fig)


def draw_left_cam_2d_trajectory(left_cameras_pos):
    """
    Draw left camera 2d trajectory
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory for {len(left_cameras_pos)} frames.")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red')

    fig.savefig(f"Results/Left cameras 2d 500 trajectory.png")
    plt.close(fig)


def draw_left_cam_2d_trajectory_addition_params(left_cameras_pos, alg_name, match_method, time):
    """
    Draw left camera 2d trajectory
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory for {len(left_cameras_pos)} frames. \n"
                 f"{alg_name} | {match_method} | Time: {time}")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red')

    fig.savefig(f"Results/Left cameras 2d trajectory {alg_name} {match_method}.png")
    plt.close(fig)


def compare_left_cam_3d_trajectory_to_ground_truth(left_cameras_pos, left_cameras_pos_gt):
    """
    Compare the left cameras relative 3d positions to the ground truth
    """
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.set_title(f"Left cameras 3d trajectory of {len(left_cameras_pos)} frames (ground truth - cyan)")
    ax.scatter3D(left_cameras_pos[:, 0], left_cameras_pos[:, 1], left_cameras_pos[:, 2], s=1, c='red')
    ax.scatter3D(left_cameras_pos_gt[:, 0], left_cameras_pos_gt[:, 1], left_cameras_pos_gt[:, 2], s=1, c='cyan')

    fig.savefig("Results/Compare Left cameras 3d trajectory.png")
    plt.close(fig)


def compare_left_cam_2d_trajectory_to_ground_truth_params(left_cameras_pos, left_cameras_pos_gt, alg_name, match_method, time):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory compared to ground truth of"
                 f" {len(left_cameras_pos)} frames (ground truth - cyan)\n"
                 f"{alg_name} | {match_method} | Time: {time}")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red')
    ax.scatter(left_cameras_pos_gt[:, 0], left_cameras_pos_gt[:, 2], s=1, c='cyan')

    fig.savefig(f"Results/Compare Left cameras 2d trajectory {alg_name} {match_method}.png")
    plt.close(fig)


def compare_left_cam_2d_trajectory_to_ground_truth(left_cameras_pos, left_cameras_pos_gt):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory of {len(left_cameras_pos)} frames (ground truth - cyan)")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red')
    ax.scatter(left_cameras_pos_gt[:, 0], left_cameras_pos_gt[:, 2], s=1, c='cyan')

    fig.savefig("Results/Compare Left cameras 2d trajectory.png")
    plt.close(fig)


def left_camera_trajectory_plot(T_arr):
    """
    Computes the trajectory of the left camera and plots it in 2d and 3d
    """
    relative_trans = left_cameras_relative_trans(T_arr)
    relative_cameras_pos_arr = left_cameras_trajectory(relative_trans)
    draw_left_cam_3d_trajectory(relative_cameras_pos_arr)
    draw_left_cam_2d_trajectory(relative_cameras_pos_arr)


def plot_supporters_compare_to_ransac(left0, left1,
                                      left0_matches_coor, left1_matches_coor,
                                      supporters_idx, ransac_supporters_idx):
    """
    Plots the supporters in left0 and left1 with and without ransac
    """
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 2
    fig.suptitle(f'Left 0 and left1 supporters with/out ransac')

    # Left0 camera without ransac
    fig.add_subplot(rows, cols, 1)
    plt.imshow(left0, cmap='gray')
    plt.title(f"Without ransac supporters({len(supporters_idx)} / {len(left1_matches_coor)}, {INLIER_COLOR})")
    plt.scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    plt.scatter(left0_matches_coor[supporters_idx][:, 0],
                left0_matches_coor[supporters_idx][:, 1], s=1, color=INLIER_COLOR)

    # Left1 camera without ransac
    fig.add_subplot(rows, cols, 3)
    plt.imshow(left1, cmap='gray')
    plt.scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    plt.scatter(left1_matches_coor[supporters_idx][:, 0],
                left1_matches_coor[supporters_idx][:, 1], s=1, color=INLIER_COLOR)

    # Left0 camera with ransac
    fig.add_subplot(rows, cols, 2)
    plt.imshow(left0, cmap='gray')
    plt.title(f"With ransac supporters({len(ransac_supporters_idx)} / {len(left1_matches_coor)}, {INLIER_COLOR})")
    plt.scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    plt.scatter(left0_matches_coor[ransac_supporters_idx][:, 0],
                left0_matches_coor[ransac_supporters_idx][:, 1], s=1, color=INLIER_COLOR)

    # Left1 camera with ransac
    fig.add_subplot(rows, cols, 4)
    plt.imshow(left1, cmap='gray')
    plt.scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    plt.scatter(left1_matches_coor[ransac_supporters_idx][:, 0],
                left1_matches_coor[ransac_supporters_idx][:, 1], s=1, color=INLIER_COLOR)

    fig.savefig("Results/Left0 Left1 supporters ransac compare.png")
    plt.close(fig)


# ===== Missions =====
def mission2_dot3():
    left0, right0 = utills.read_images(0)
    left1, right1 = utills.read_images(1)
    best_left1_cam_mat = compute_trans_between_left0_left1(left0, right0, left1, right1)
    draw_relative_2d_pos(relative_camera_pos(M1), relative_camera_pos(M2),
                         relative_camera_pos(best_left1_cam_mat), relative_camera_pos(compose_transformations(M2 ,best_left1_cam_mat)))
    print(f"Relative cameras positions:\n Left0: {relative_camera_pos(M1)}\n"
          f"Right0: {relative_camera_pos(M2)}\n Left1: {relative_camera_pos(best_left1_cam_mat)}\n"
          f"Right1: {relative_camera_pos(compose_transformations(M2 ,best_left1_cam_mat))}")


def mission2_dot4():
    # 2.4
    left0, right0 = utils.read_images(utils.IDX)
    left1, right1 = utils.read_images(utils.IDX + 1)
    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, _, pair0_matches = utils.detect_and_match(left0, right0)
    pair0_rec_matches_idx, _ = utils.rectified_stereo_pattern_rej(left0_kpts, right0_kpts, pair0_matches)

    # Find matches in pair 1 with rectified test
    left1_kpts, left1_dsc, right1_kpts, right1_dsc, pair1_matches = utils.detect_and_match(left1, right1)
    pair1_rec_matches_idx, _ = utils.rectified_stereo_pattern_rej(left1_kpts, right1_kpts, pair1_matches)

    # Find matches between left0 and left1
    left0_left1_matches = utils.matching(left0_dsc, left1_dsc)

    # Find key pts that match in all 4 KITTI
    rec0_dic = create_rec_dic(pair0_matches, pair0_rec_matches_idx)  # dict of {left kpt idx: pair rec frame_idx}
    rec1_dic = create_rec_dic(pair1_matches, pair1_rec_matches_idx)
    q_pair0_idx, q_pair1_idx, q_left0_left1_idx = find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)

    # Pair0 triangulation
    left0_matches_coor, right0_matches_coor = utils.get_matches_coor(pair0_matches[q_pair0_idx], left0_kpts,
                                                                     right0_kpts)
    pair0_p3d_pts = utils.triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)

    # Compute left1 and right 1 matches d2_points in the image
    left1_matches_coor, right1_matches_coor = utils.get_matches_coor(pair1_matches[q_pair1_idx], left1_kpts,
                                                                     right1_kpts)

    pts_idx = [0, 1, 2, 3]
    left1_ex_mat = pnp(pair0_p3d_pts[pts_idx], left1_matches_coor[pts_idx], K, cv2.SOLVEPNP_AP3P)

    # Compute left1 and right1 projection matrices
    right1_to_left0_ex_mat = compose_transformations(left1_ex_mat, M2)
    left1_proj_mat = K @ left1_ex_mat
    right1_proj_mat = K @ right1_to_left0_ex_mat

    # Find model's supporters
    supporters_idx = find_supporters_4_frame(pair0_p3d_pts,
                                             left1_proj_mat, left1_matches_coor,
                                             right1_proj_mat, right1_matches_coor,
                                             acc=2)

    # Find the best transformation between left0 and left1 with ransac
    best_left1_cam_mat, max_supp_group_idx = online_est_pnp_ransac(PNP_NUM_PTS, pair0_p3d_pts,
                                                                   M1, left0_matches_coor,
                                                                   M2, right0_matches_coor,
                                                                   left1_matches_coor,
                                                                   M2, right1_matches_coor,
                                                                   K, acc=SUPP_ERR)

    plot_supporters(left0, left1, left0_matches_coor, left1_matches_coor,
                    supporters_idx)
    plot_supporters_compare_to_ransac(left0, left1,
                                      left0_matches_coor, left1_matches_coor,
                                      supporters_idx, max_supp_group_idx)

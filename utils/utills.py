import heapq
import random

import cv2
import gtsam
import numpy as np
import tqdm

from DataDirectory import Data
from DataBaseDirectory import Feature, Trck

# == Databases ==
KITTI = Data.KITTI

K, M1, M2 = KITTI.get_K(), KITTI.get_M1(), KITTI.get_M2()

IDX = 000000
MATCHES_NUM = 20
RATIO = 0.75
MATCHES_NORM = cv2.NORM_L2
PASSED = "PASSED"
FAILED = "FAILED"
ALG = cv2.AKAZE_create()
LEFT_CAM_TRANS_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/poses/00.txt'
INLIER_COLOR = "orange"
OUTLIER_COLOR = "cyan"
IMAGE_HEIGHT = 376
IMAGE_WIDTH = 1241
PNP_NUM_PTS = 4
SUPP_ERR = 2
REC_ERR = 2
MOVIE_LEN = 3450


# == Ex1 == #
def feature_detection_and_description(img1, img2, alg):
    """
    Computes KITTI key d2_points and their's descriptors
    :param alg: Feature detecting and description algorithm
    :return: KITTI key d2_points and descriptors
    """
    img1_kpts, img1_dsc = alg.detectAndCompute(img1, None)
    img2_kpts, img2_dsc = alg.detectAndCompute(img2, None)
    return np.array(img1_kpts), np.array(img1_dsc), np.array(img2_kpts), np.array(img2_dsc)


def bf_matching(metric, img1_dsc, img2_dsc, crossCheck=True, sort=True):
    """
    Find Matches between two KITTI descriptors
    :param metric: distance function for computes distance between two descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: array of matches
    """
    bf = cv2.BFMatcher(metric, crossCheck=crossCheck)
    matches = bf.match(img1_dsc, img2_dsc)
    if sort:
        # Sort the matches from the best match to the worst - where best means it has the lowest distance
        matches = sorted(matches, key=lambda x: x.distance)
    return matches


def knn_flann_matching(img1_dsc, img2_dsc):
    """
    Find Matches between two KITTI descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: Array of matches
    """
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(img1_dsc, img2_dsc, k=2)
    return matches


def knn_matching(metric, img1_dsc, img2_dsc):
    """
    Find Matches between two KITTI descriptors
    :param metric: distance function for computes distance between two descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: array of matches
    """
    bf = cv2.BFMatcher(metric)
    matches = bf.knnMatch(img1_dsc, img2_dsc, 2)
    return matches


def matching(img1_dsc, img2_dsc):  # Todo: we left those options for further checking
    # matches = knn_matching(MATCHES_NORM, img1_dsc, img2_dsc)  # Notice does not return np array
    # matches = knn_flann_matching(img1_dsc, img2_dsc)
    # matches, _ = significance_test(matches, RATIO)  # Notice does not return np array
    matches = bf_matching(MATCHES_NORM, img1_dsc, img2_dsc, sort=False)
    return np.array(matches)


def detect_and_match(img1, img2):
    # Detects the image key d2_points and compute their descriptors
    img1_kpts, img1_dsc, img2_kpts, img2_dsc = feature_detection_and_description(img1, img2, ALG)

    # Matches between the KITTI and plots the matching
    matches = matching(img1_dsc, img2_dsc)
    return img1_kpts, img1_dsc, img2_kpts, img2_dsc, matches


def significance_test(matches, ratio):
    """
    Applying the significance test - rejects all matches that their distance ratio between the 1st and 2nd
    nearest neighbors is lower than predetermined RATIO.
    In practice:
        1. Finds 2 best matches in img2 for each descriptor in img1.
        2. Rejects all the matches that are not passing the test.
        3. Shows the new matches - that passed the test.
    :param ratio : The ratio threshold between the best and the second match
    :param rand: Boolean value that determines whether to choose part of the matches randomly or the best ones.
    """

    # Apply ratio test
    pass_test = []
    failed_test = []
    for first, second in matches:
        if first.distance < ratio * second.distance:
            pass_test.append(first)
        else:
            failed_test.append(first)

    return pass_test, failed_test


# == Ex2 == #

def axes_of_matches(match, img1_kpts, img2_kpts):
    """
    Returns (x,y) values for each point from the two in the match object
    """
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    x1, y1 = img1_kpts[img1_idx].pt
    x2, y2 = img2_kpts[img2_idx].pt
    return x1, y1, x2, y2


def rectified_stereo_pattern_rej(img1_kpts, img2_kpts, matches):  # Todo: Efficiency for further exercises
    """
    Apply the rectified stereo pattern rejection on image1 key d2_points and image 2 key d2_points
    :return: List of Inliers and outliers indexes of the 2 KITTI
    """
    inliers_matches_idx, outliers_matches_idx = [], []

    num_matches = len(matches)
    for i in range(num_matches):
        _, y1, _, y2 = axes_of_matches(matches[i], img1_kpts, img2_kpts)
        if abs(y2 - y1) <= REC_ERR:
            inliers_matches_idx.append(i)
        else:
            outliers_matches_idx.append(i)

    return inliers_matches_idx, outliers_matches_idx


def get_matches_coor(matches, img1_kpts, img2_kpts):
    """
    Returns 2 numpy arrays of matches d2_points in img1 and img2 accordingly
    """
    img1_matches_coor, img2_matches_coor = [], []
    for match in matches:
        img1_idx, img2_idx = match.queryIdx, match.trainIdx
        img1_matches_coor.append(img1_kpts[img1_idx].pt)
        img2_matches_coor.append(img2_kpts[img2_idx].pt)

    return np.array(img1_matches_coor), np.array(img2_matches_coor)


def get_matches_coor_and_kpts(matches, img1_kpts, img2_kpts):
    """
    Returns 2 numpy arrays with dimension of 2XN of matches d2_points and their coorsponding
    keypoint index in img1 and img2 accordingly
    """
    img1_matches_coor, img2_matches_coor = [], []
    for match in matches:
        img1_kpt_idx, img2_kpt_idx = match.queryIdx, match.trainIdx
        img1_x, img1_y = img1_kpts[img1_kpt_idx].pt[0], img1_kpts[img1_kpt_idx].pt[1]
        img2_x, img2_y = img2_kpts[img2_kpt_idx].pt[0], img2_kpts[img2_kpt_idx].pt[1]
        img1_matches_coor.append([img1_kpt_idx, img1_x, img1_y, img2_x, img2_y])

    return np.array(img1_matches_coor)


def get_inliers_and_outliers_coor_for_rec(inliers_matches, outliers_matches, img1_kpts, img2_kpts):
    img1_inliers, img2_inliers = get_matches_coor(inliers_matches, img1_kpts, img2_kpts)
    img1_outliers, img2_outliers = get_matches_coor(outliers_matches, img1_kpts, img2_kpts)
    return img1_inliers, img2_inliers, img1_outliers, img2_outliers


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


def triangulate(l_mat, r_mat, kp1_xy_lst, kp2_xy_lst):
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


# == Ex3 ==
def compute_trans_between_cur_to_next(left0_kpts, left0_dsc, right0_kpts,
                                      pair0_matches, pair0_rec_matches_idx,
                                      left1_kpts, left1_dsc, right1_kpts,
                                      pair1_matches, pair1_rec_matches_idx):
    """
   Compute the transformation T between left 0 and left1 KITTI
   :return: numpy array with shape 3 X 4
   """

    # Find matches between left0 and left1
    left0_left1_matches = matching(left0_dsc, left1_dsc)

    # Find key pts that match in all 4 KITTI
    rec0_dic = create_rec_dic(pair0_matches, pair0_rec_matches_idx)  # dict of {left kpt idx: pair rec id}
    rec1_dic = create_rec_dic(pair1_matches, pair1_rec_matches_idx)
    q_pair0_idx, q_pair1_idx, q_left0_left1_idx = find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)

    # Frame triangulation
    left0_matches_coor, right0_matches_coor = get_matches_coor(pair0_matches[q_pair0_idx], left0_kpts,
                                                               right0_kpts)
    pair0_p3d_pts = triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)

    # Compute left1 and right 1 matches d2_points in the image
    left1_matches_coor, right1_matches_coor = get_matches_coor(pair1_matches[q_pair1_idx], left1_kpts,
                                                               right1_kpts)

    # Find the best transformation between left0 and left1 with ransac
    best_left1_cam_mat, max_supp_group_idx = online_est_pnp_ransac(PNP_NUM_PTS, pair0_p3d_pts,
                                                                   M1, left0_matches_coor,
                                                                   M2, right0_matches_coor,
                                                                   left1_matches_coor,
                                                                   M2, right1_matches_coor,
                                                                   K, acc=SUPP_ERR)

    return best_left1_cam_mat, max_supp_group_idx


def read_and_rec_match(frame_num):
    """
    Reads KITTI from pair idx, Finds matches with rectified test
    :param idx: Frame's index
    :return: key d2_points of the two KITTI and the matches
    """
    # Find matches in frame with rectified test
    left_img, right_img = KITTI.get_image(frame_num)
    left0_kpts, left0_dsc, right0_kpts, _, pair0_matches = detect_and_match(left_img, right_img)
    pair0_rec_matches_idx, _ = rectified_stereo_pattern_rej(left0_kpts, right0_kpts, pair0_matches)
    return left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx


def whole_movie(first_left_ex_mat=M1):
    """
    Compute the transformation of two consequence left KITTI in the whole movie
    :return:array of transformations where the i'th element is the transformation between i-1 -> i
    """
    T_arr = [first_left_ex_mat]

    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = read_and_rec_match(IDX)
    for i in tqdm.tqdm(range(1, MOVIE_LEN)):
        left1_kpts, left1_dsc, right1_kpts, pair1_matches, pair1_rec_matches_idx = read_and_rec_match(IDX + i)

        left1_ex_mat = compute_trans_between_cur_to_next(left0_kpts, left0_dsc, right0_kpts,
                                                         pair0_matches, pair0_rec_matches_idx,
                                                         left1_kpts, left1_dsc, right1_kpts,
                                                         pair1_matches, pair1_rec_matches_idx)
        left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = left1_kpts, left1_dsc, \
                                                                                   right1_kpts, pair1_matches, \
                                                                                   pair1_rec_matches_idx
        T_arr.append(left1_ex_mat)

    return T_arr


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
        # Check if "left0_kpt_idx" and "left1_kpt_idx" are key d2_points that passed the rectified test in each pair.
        # If so:  It adds to q_pair0 the key d2_points index in the list of pair 0 matches
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
    while outliers_perc != 0 and first_loop_iter < first_loop_iter_est(prob, outliers_perc) and first_loop_iter < 1000:

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

    if len(supp_idx) < 6:  # Todo: TRyuing to dealing twith bad match
        return []

    for i in range(it_num):
        left1_ex_mat = pnp(world_p3d_pts[supp_idx], left1_matches_coor[supp_idx], calib_mat,
                           flag=cv2.SOLVEPNP_ITERATIVE)
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


def gtsam_relative_camera_pos(ex_cam_mat):
    """
    Finds and returns the Camera position at the "world" d2_points
    :param ex_cam_mat: [Rotation mat|translation vec]
    """
    # R = extrinsic_camera_mat[:, :3]
    # t = extrinsic_camera_mat[:, 3]
    return ex_cam_mat.translation()


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
    all_supp_indicators = left1_supp_indicator & right1_supp_indicator  # Todo: checking now
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
    Returns a boolean vector that indicates which row satisfy the distance condition
    :param acc: Euclidean distance
    :return:boolean vector
    """
    left0_dist = compute_square_dist(img_projected_pts, img_pts_coor)
    q_acc = acc ** 2
    return left0_dist <= q_acc


def compute_square_dist(pts_lst1, pts_lst2, dim="3d"):
    """
    Check the euclidean dist between the projected d2_points and correspond pixel locations
    :param pts_lst1:
    :param pts_lst2:
    :return:
    """
    pts_sub = pts_lst1 - pts_lst2  # (x1, y1), (x2, y2) -> (x1 - x2, y1 - y2)
    if dim == "2d":
        squared_dist = np.einsum("ij,ij->i", pts_sub, pts_sub)  # (x1 - x2)^2 + (y1 - y2)^2
    elif dim == "3d":
        squared_dist = np.linalg.norm(pts_sub, axis=1)
    return squared_dist


def euclidean_dist(pts_lst1, pts_lst2, dim="3d"):
    squared_dist = compute_square_dist(pts_lst1, pts_lst2, dim=dim)
    return np.sqrt(squared_dist)


def convert_trans_from_rel_to_global(T_arr):
    relative_T_arr = []
    last = T_arr[0]

    for t in T_arr:
        last = compose_transformations(last, t)
        relative_T_arr.append(last)

    return relative_T_arr


def convert_gtsam_trans_from_rel_to_global(T_arr):
    relative_T_arr = []
    last = T_arr[0]

    for t in T_arr:
        last = last.compose(t)
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


def gtsam_left_cameras_trajectory(relative_T_arr):
    """
    Computes the left cameras 3d positions relative to the starting position
    :param T_arr: relative to first camera transformations array
    :return: numpy array with dimension num T_arr X 3
    """
    global_cam_loc = []
    for t in relative_T_arr:
        global_cam_loc.append(gtsam_relative_camera_pos(t))
    return np.array(global_cam_loc)


# === Ex4 === #
def find_features_in_consecutive_frames_whole_movie(first_left_ex_cam_mat=M1):
    """
    Finds Features of two consecutive frames_in_window in the whole movie
    :return: Array which each row contains 2 arrays [frame0_features, frame1_features]
     frame0_features contains Feature objects that match to frame1_features and share
     indexes i.e the ith feature at frame0_features match to the ith feature at frame1_features
    """

    consecutive_frame_features = []
    relative_trans = [first_left_ex_cam_mat]

    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, \
    pair0_matches, pair0_rec_matches_idx = read_and_rec_match(frame_num=0)

    inliers_percentage = []
    for i in tqdm.tqdm(range(1, MOVIE_LEN)):
        left1_kpts, left1_dsc, right1_kpts, pair1_matches, pair1_rec_matches_idx = read_and_rec_match(frame_num=i)

        trans, frame0_features, frame1_features, frame0_inliers_percentage, supporters_idx = \
            find_features_in_consecutive_frames(
                left0_kpts, left0_dsc, right0_kpts,
                pair0_matches, pair0_rec_matches_idx,
                left1_kpts, left1_dsc, right1_kpts,
                pair1_matches, pair1_rec_matches_idx)

        frame0_features, frame1_features = frame0_features[supporters_idx], frame1_features[supporters_idx]

        left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = left1_kpts, left1_dsc, \
                                                                                   right1_kpts, pair1_matches, \
                                                                                   pair1_rec_matches_idx

        consecutive_frame_features.append([frame0_features, frame1_features])
        inliers_percentage.append(frame0_inliers_percentage)
        relative_trans.append(trans)

    global_trans = convert_trans_from_rel_to_global(relative_trans)

    return consecutive_frame_features, inliers_percentage, global_trans, relative_trans


def find_features_in_consecutive_frames(left0_kpts, left0_dsc, right0_kpts,
                                        pair0_matches, pair0_rec_matches_idx,
                                        left1_kpts, left1_dsc, right1_kpts,
                                        pair1_matches, pair1_rec_matches_idx):
    """
   Compute the transformation T between left 0 and left1 KITTI
   :return: Numpy array of Feature object from frame0 and frame1 that passed the consensus match
   """

    # Find matches between left0 and left1
    left0_left1_matches = matching(left0_dsc, left1_dsc)

    # Find key pts that match in all 4 KITTI
    rec0_dic = create_rec_dic(pair0_matches, pair0_rec_matches_idx)  # dict of {left kpt idx: pair rec id}
    rec1_dic = create_rec_dic(pair1_matches, pair1_rec_matches_idx)
    q_pair0_idx, q_pair1_idx, q_left0_left1_idx = find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)

    # Get frame 0 Feature objects (which passed the rec test)
    frame0_features = get_feature_obj(pair0_matches[q_pair0_idx], left0_kpts, right0_kpts)

    # Here we take only their d2_points
    left0_matches_coor = get_features_left_coor(frame0_features)
    right0_matches_coor = get_features_right_coor(frame0_features)

    # Frame 0 triangulation
    pair0_p3d_pts = triangulate(K @ M1, K @ M2, left0_matches_coor, right0_matches_coor)

    # Get frame 1 Feature objects  (which passed the rec test)
    frame1_features = get_feature_obj(pair1_matches[q_pair1_idx], left1_kpts, right1_kpts)

    # Notice that frame0_features and frame1_features are sharing indexes

    # Here we take only their d2_points
    left1_matches_coor = get_features_left_coor(frame1_features)
    right1_matches_coor = get_features_right_coor(frame1_features)

    # Finds Feature's indexes of frame0_features and frame1_features that passed the consensus match
    # with using Ransac method
    trans, max_supp_group_idx = online_est_pnp_ransac(PNP_NUM_PTS, pair0_p3d_pts,
                                                      M1, left0_matches_coor,
                                                      M2, right0_matches_coor,
                                                      left1_matches_coor,
                                                      M2, right1_matches_coor,
                                                      K, acc=SUPP_ERR)

    frame0_inliers_percentage = 100 * len(max_supp_group_idx) / len(frame0_features)

    return trans, frame0_features, frame1_features, frame0_inliers_percentage, max_supp_group_idx


def get_feature_obj(matches, img1_kpts, img2_kpts):
    """
    Same as "get_matches_coor_and_kpts" except that here we return a list of Feature object
    """
    frame_features = []
    for match in matches:
        img1_kpt_idx, img2_kpt_idx = match.queryIdx, match.trainIdx
        img1_x, img1_y = img1_kpts[img1_kpt_idx].pt[0], img1_kpts[img1_kpt_idx].pt[1]
        img2_x, img2_y = img2_kpts[img2_kpt_idx].pt[0], img2_kpts[img2_kpt_idx].pt[1]
        feature = Feature.Feature(img1_kpt_idx, img2_kpt_idx, img1_x, img2_x, img1_y, img2_y)
        frame_features.append(feature)

    return np.array(frame_features)


def get_features_left_coor(frame_features):
    """
    Returns left frame features d2_points
    """
    left_feature_coor = []
    for feature in frame_features:
        left_feature_coor.append(feature.get_left_coor())

    return np.array(left_feature_coor)


def get_features_right_coor(frame_features):
    """
    Returns right frame features d2_points
    """
    right_feature_coor = []
    for feature in frame_features:
        right_feature_coor.append(feature.get_right_coor())

    return np.array(right_feature_coor)


def get_rand_track(track_len, tracks):
    """
    Randomize a track with length of track_len
    """
    track = None
    found = False
    while not found:
        idx = random.randint(0, len(tracks) - 1)
        if tracks[idx].get_track_len() == track_len:
            track = tracks[idx]
            found = True

    return track


def create_gtsam_calib_cam_mat(calib_cam_mat):
    fx, fy, skew, cx, cy = calib_cam_mat[0, 0], calib_cam_mat[1, 1], calib_cam_mat[0, 1], \
                           calib_cam_mat[0, 2], calib_cam_mat[1, 2]
    baseline = Data.KITTI.get_M2()[0, 3]

    gtsam_K = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)

    return gtsam_K


def convert_ex_cam_to_cam_to_world(ex_cam):
    R_mat = ex_cam[:, :3]
    t_vec = ex_cam[:, 3]

    R = R_mat.T
    t = - R_mat.T @ t_vec

    ex_cam_mat_from_cam_to_world = np.hstack((R, t.reshape(3, 1)))  # Todo: check concatenation

    return ex_cam_mat_from_cam_to_world


def convert_gtsam_ex_cam_to_world_to_cam(gtsam_ex_cam):
    gtsam_R = gtsam_ex_cam.rotation()
    R_mat = np.hstack((gtsam_R.column(1).reshape(3, 1),
                       (gtsam_R.column(2).reshape(3, 1),
                       (gtsam_R.column(3).reshape(3, 1)))))

    t_vec = gtsam_ex_cam.translation()

    R = R_mat.T
    t = - R_mat.T @ t_vec

    ex_cam_mat_from_world_to_cam = np.hstack((R, t.reshape(3, 1)))  # Todo: check concatenation

    return ex_cam_mat_from_world_to_cam



# ===== Ex7
def create_empty_min_heap():
    return heapq.heapify([])


def mahalanobis_dist(delta, cov):
    r_squared = delta.T @ np.linalg.inv(cov) @ delta
    return r_squared ** 0.5


def compute_gtsam_rel_trans(first_cam_mat, second_cam_mat):
    # first_cam_to_world_ex_mat = convert_gtsam_ex_cam_to_world_to_cam(first_cam_mat)  # world -> first cam
    first_cam_to_world_ex_mat = first_cam_mat.inverse()  # world -> first cam

    # Compute transformation of : (world - > first cam) * (second cam -> world) = second cam-> first cam
    second_cam_rel_to_first_cam_trans = first_cam_to_world_ex_mat.compose(second_cam_mat)

    return second_cam_rel_to_first_cam_trans  # second cam-> first cam


def gtsam_cams_delta(first_cam_mat, second_cam_mat):
    gtsam_rel_trans = second_cam_mat.between(first_cam_mat)
    return gtsam_translation_to_vec(gtsam_rel_trans.rotation(), gtsam_rel_trans.translation())  # Todo : check if its ok


def rot_mat_to_euler_angles(R_mat):  # todo: change this function
    sy = np.sqrt(R_mat[0, 0] * R_mat[0, 0] + R_mat[1, 0] * R_mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        azimut = np.arctan2(R_mat[2, 1], R_mat[2, 2])
        pitch = np.arctan2(-R_mat[2, 0], sy)
        roll = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    else:
        azimut = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
        pitch = np.arctan2(-R_mat[2, 0], sy)
        roll = 0
    return np.array([azimut, pitch, roll])


def translation_to_vec(translation):
    R_mat, t_vec = translation[:, :3], translation[:, 3]
    loc = - R_mat.T @ t_vec
    euler_angles = rot_mat_to_euler_angles(R_mat)

    return np.hstack((euler_angles, loc))


def gtsam_translation_to_vec(R_mat, t_vec):
    np_R_mat = np.hstack((R_mat.column(1).reshape(3, 1), R_mat.column(2).reshape(3, 1), R_mat.column(3).reshape(3, 1)))
    euler_angles = rot_mat_to_euler_angles(np_R_mat)
    return np.hstack((euler_angles, t_vec))


def apply_full_consensus_match(first_frame, second_frame):
    left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = read_and_rec_match(first_frame)
    left1_kpts, left1_dsc, right1_kpts, pair1_matches, pair1_rec_matches_idx = read_and_rec_match(second_frame)

    trans, frame0_features, frame1_features, frame0_inliers_percentage, supporters_idx = \
                                find_features_in_consecutive_frames(left0_kpts, left0_dsc,
                                                                    right0_kpts,
                                                                    pair0_matches, pair0_rec_matches_idx,
                                                                    left1_kpts, left1_dsc,
                                                                    right1_kpts,
                                                                    pair1_matches, pair1_rec_matches_idx)

    return trans, frame0_features, frame1_features,  supporters_idx, frame0_inliers_percentage


def find_loop_candidate_by_consensus_match(mahalanobis_dist_cand_movie_ind, mahalanobis_dist_cand_pg_ind,
                                           cur_frame_movie_ind, threshold):
    passed_consensus_frame_track_tuples = []  # {pose graph prev frame index : tracks between prev frame and cur frame}
    cur_frame_left_kpts, cur_frame_left_dsc, cur_frame_right_kpts, \
    cur_frame_matches, cur_frame_rec_matches_idx = read_and_rec_match(cur_frame_movie_ind)

    for cand_ind, cand_at_movie_ind in enumerate(mahalanobis_dist_cand_movie_ind):
        cand_frame_left_kpts, cand_frame_left_dsc, cand_frame_right_kpts, \
        cand_frame_matches, cand_frame_rec_matches_idx = read_and_rec_match(cand_at_movie_ind)

        _, frame0_features, frame1_features, inliers_perc, supportes_idx = find_features_in_consecutive_frames(cand_frame_left_kpts, cand_frame_left_dsc,
                                                                    cand_frame_right_kpts,
                                                                    cand_frame_matches, cand_frame_rec_matches_idx,
                                                                    cur_frame_left_kpts, cur_frame_left_dsc,
                                                                    cur_frame_right_kpts,
                                                                    cur_frame_matches, cur_frame_rec_matches_idx)
        frame0_features, frame1_features = frame0_features[supportes_idx], frame1_features[supportes_idx]

        # Todo: check wether to return the inliers precentage or num
        if inliers_perc > threshold:
            # Data.DB.set_tracks_and_frames(frame0_features, frame1_features, cand, cur_frame_movie_ind)  # Todo: consider use it need to check its valid
            tracks = create_little_tracks(frame0_features, frame1_features, cand_at_movie_ind, cur_frame_movie_ind)
            passed_consensus_frame_track_tuples.append([mahalanobis_dist_cand_pg_ind[cand_ind], tracks])

    return passed_consensus_frame_track_tuples


def create_little_tracks(first_frame_features_obj, second_frame_features_obj, first_frame_id, second_frame_id):
    # Todo: consider change this to add those track sto the data base
    tracks = []

    for i in range(len(first_frame_features_obj)):
        first_frame_feature = first_frame_features_obj[i]
        second_frame_feature = second_frame_features_obj[i]

        track_idx = i
        track = Trck.Track(track_idx)

        # Creates the track and add its feature on frame i
        track.add_feature(first_frame_id, first_frame_feature)

        # Creates the track and add its feature on frame i + 1
        track.add_feature(second_frame_id, second_frame_feature)

        tracks.append(track)  # Todo: consider change to constant len and not applying append

    return tracks





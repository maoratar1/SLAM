import cProfile
import csv
import pstats

import numpy as np
import tqdm
from gtsam import symbol
from gtsam.utils import plot
from matplotlib import pyplot as plt

import DataDirectory.Data
import utils.plot
import utils.utills
from DataDirectory import Data
from PoseGraphDirectory import PoseGraph

# === Missions ===

    # Code for mission 7.5


def apply_loop_and_plot_traj(start_cam=0, end_cam=None, loaded_pose_graph=False, directed=True):
    """
    This function apply loop closure of cameras in range (start cam, end cam)
    and plot the camera's trajectory of the optimized cameras, initial estimate and the ground truth
    :param loaded_pose_graph: Bool variable for choosing whether to load a pose graph or create new one
    """
    if loaded_pose_graph:
        pose_graph = DataDirectory.Data.PG
    else:
        pose_graph = PoseGraph.create_pose_graph(directed)
        pose_graph.optimize()

    pose_graph.loop_closure(start_cam, end_cam)

    return plot_pose_graph_trajectory_before_and_after_optimization(pose_graph)


def plot_pose_graph_versions(num_versions=5):
    """
    Plots pose graph 2d Trajectories over the num versions of finding loops
    """
    pose_graph = PoseGraph.create_pose_graph(False)
    pose_graph.optimize()
    version = 1
    for i in tqdm.tqdm(range(pose_graph.get_num_poses())):
        founded_loops = pose_graph.loop_closure_for_specific_frame(cur_cam_pose_graph_ind=i)
        loops = pose_graph.get_loops()

        if founded_loops is not None and loops[-1][0] in {187, 190, 411, 419, 429}:
            print("version", version, ": ", loops[-1][0])

            marginals = pose_graph.marginals()
            result = pose_graph.get_optimized_values()
            plot.plot_trajectory(0, result, marginals=marginals, scale=1,
                                 title=f"Covariance poses for Pose graph after {version} loop.\n "
                                       f"Loop found at the {loops[-1][0]}th camera.",
                                 save_file=f"Results/Poses loop rel_covs{version}.png", d2_view=True)
            plt.tight_layout()

            if num_versions == version:
                break

            version += 1


def plot_pose_graph_absolute_location_error_before_and_after_opt():
    pose_graph = DataDirectory.Data.PG

    initial_estimate_cam_3d_loc, opt_cameras_3d_loc, cameras_gt_3d_loc = \
                                                            cameras_initial_est_and_ground_truth_locations(pose_graph)

    absolute_x = abs(cameras_gt_3d_loc[:, 0] - initial_estimate_cam_3d_loc[:, 0])
    absolute_y = abs(cameras_gt_3d_loc[:, 1] - initial_estimate_cam_3d_loc[:, 1])
    absolute_z = abs(cameras_gt_3d_loc[:, 2] - initial_estimate_cam_3d_loc[:, 2])

    absolute_x_loop = abs(cameras_gt_3d_loc[:, 0] - opt_cameras_3d_loc[:, 0])
    absolute_y_loop = abs(cameras_gt_3d_loc[:, 1] - opt_cameras_3d_loc[:, 1])
    absolute_z_loop = abs(cameras_gt_3d_loc[:, 2] - opt_cameras_3d_loc[:, 2])

    fig = plt.figure()
    plt.title(f"Pose graph absolute location error BEFORE Loop Closure")
    plt.plot(range(len(absolute_x)), absolute_x, label="X's error")
    plt.plot(range(len(absolute_y)), absolute_y, label="Y's error")
    plt.plot(range(len(absolute_z)), absolute_z, label="Z's error")
    plt.legend()

    fig.savefig(f"Results/Pose graph absolute location error BEFORE opt.png")
    plt.close(fig)

    fig = plt.figure()
    plt.title(f"Pose graph absolute location error AFTER Loop Closure optimization")
    plt.plot(range(len(absolute_x_loop)), absolute_x_loop, label="X's error")
    plt.plot(range(len(absolute_y_loop)), absolute_y_loop, label="Y's error")
    plt.plot(range(len(absolute_z_loop)), absolute_z_loop, label="Z's error")
    plt.ylim(0, 50)
    plt.legend()

    fig.savefig(f"Results/Pose graph absolute location error AFTER opt.png")
    plt.close(fig)


def plot_covariance_uncertainty():
    new_pose_graph = PoseGraph.create_pose_graph()
    new_pose_graph.optimize()

    pose_graph = Data.PG
    opt_covs = [pose_graph.marginals(optimized=True).marginalCovariance(symbol(PoseGraph.CAMERA_SYM, cam_ind)) for cam_ind in range(pose_graph.get_num_poses())]
    covs = [new_pose_graph.marginals(optimized=False).marginalCovariance(symbol(PoseGraph.CAMERA_SYM, cam_ind)) for cam_ind in range(pose_graph.get_num_poses())]

    det_opt_covs = [np.sqrt(np.linalg.det(opt_cov)) for opt_cov in opt_covs]
    det_covs = [np.sqrt(np.linalg.det(cov)) for cov in covs]

    fig = plt.figure()
    plt.title(f"Location uncertainty size before and after Loop Closure")
    plt.plot(range(len(det_covs)), det_covs, label="Before")
    plt.plot(range(len(det_opt_covs)), det_opt_covs, label="After")
    plt.yscale('log')
    plt.legend()

    fig.savefig(f"Results/Uncertainty before and after Loop.png")
    plt.close(fig)


def plot_pose_graph_trajectory_before_and_after_optimization(pose_graph):
    loops = pose_graph.get_loops()

    initial_estimate_cam_3d_loc, opt_cameras_3d_loc, cameras_gt_3d_loc = cameras_initial_est_and_ground_truth_locations(pose_graph)

    # Plot optimized trajectory without covariance
    utils.plot.plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=opt_cameras_3d_loc,
                                                                                  initial_estimate_poses=initial_estimate_cam_3d_loc,
                                                                                  cameras_gt=cameras_gt_3d_loc,
                                                                                  loops=loops, numbers=False,
                                                                                  mahalanobis_dist=PoseGraph.MAHALANOBIS_DIST_THRESHOLD,
                                                                                  inliers_perc=PoseGraph.INLIERS_THRESHOLD_PERC,
                                                                                  title="New ")

    return [PoseGraph.MAHALANOBIS_DIST_THRESHOLD, PoseGraph.INLIERS_THRESHOLD_PERC, len(loops),
            format(pose_graph.graph_error(optimized=False), ".2f"),
            format(pose_graph.graph_error(optimized=True), ".2f")]


def cameras_initial_est_and_ground_truth_locations(pose_graph):
    key_frames = pose_graph.get_key_frames()
    gtsam_optimized_cameras_poses = pose_graph.get_optimized_cameras_poses()
    gtsam_initial_est_cameras_poses = pose_graph.get_initial_est_cameras_poses()
    ground_truth = np.array(utils.utills.get_ground_truth_transformations())[key_frames]

    initial_estimate_cam_3d_loc = utils.utills.gtsam_left_cameras_trajectory(gtsam_initial_est_cameras_poses)
    opt_cameras_3d_loc = utils.utills.gtsam_left_cameras_trajectory(gtsam_optimized_cameras_poses)
    cameras_gt_3d_loc = utils.utills.left_cameras_trajectory(ground_truth)

    return initial_estimate_cam_3d_loc, opt_cameras_3d_loc, cameras_gt_3d_loc


def check_one_loop():
    pose_graph = DataDirectory.Data.PG
    # pose_graph = PoseGraphDirectory.PoseGraph.create_pose_graph()
    pose_graph.loop_closure_for_specific_frame(1)


def check_bottleneck(num_cams):
    """
    Check bottleneck at the code
    """
    pose_graph = DataDirectory.Data.PG

    profile = cProfile.Profile()
    profile.runcall(pose_graph.loop_closure, num_cams)
    ps = pstats.Stats(profile)
    ps.sort_stats('cumtime')
    ps.print_stats()


def plot_2_images(prev_cam_ind, cur_cam_ind, match=True):
    left0_matches_coor = left1_matches_coor = supporters_idx = frame0_inliers_percentage = None

    prev_frame_ind_whole_movie = Data.PG.get_key_frames()[prev_cam_ind]
    cur_frame_ind_whole_movie = Data.PG.get_key_frames()[cur_cam_ind]

    # Take the left images - we use the function read, so it won't apply blurring
    prev_img, cur_img = Data.KITTI.read_images(prev_frame_ind_whole_movie)[0], \
                        Data.KITTI.read_images(cur_frame_ind_whole_movie)[0]

    if match:
        _, frame0_features, frame1_features, supporters_idx, frame0_inliers_percentage = \
            utils.utills.apply_full_consensus_match(prev_frame_ind_whole_movie, cur_frame_ind_whole_movie)

        left0_matches_coor = utils.utills.get_features_left_coor(frame0_features)
        left1_matches_coor = utils.utills.get_features_left_coor(frame1_features)

    utils.plot.plot_supporters(prev_frame_ind_whole_movie, cur_frame_ind_whole_movie,
                               prev_cam_ind, cur_cam_ind,
                               prev_img, cur_img,
                               left0_matches_coor, left1_matches_coor, supporters_idx, frame0_inliers_percentage)


def check_several_mahalanobis_dit_and_inliers_perc(maha_dist_opt, inliers_perc_opt):

    csv_rows = []
    csv_rows.append(["Option", "Mahalanobis distance", "Inliers percentage", "Num loops", "Error before optimization",
                         "Error after optimization"])

    print("Start checking options: ")
    option_number = 1
    total_options_num = len(inliers_perc_opt) * len(maha_dist_opt)
    for inliers_perc in inliers_perc_opt:
        PoseGraph.INLIERS_THRESHOLD_PERC = inliers_perc
        for dist in maha_dist_opt:
            print(f"\tOption {option_number} / {total_options_num}")
            PoseGraph.MAHALANOBIS_DIST_THRESHOLD = dist
            csv_rows.append([option_number] + apply_loop_and_plot_traj(0, Data.PG.get_num_poses() - 1, loaded_pose_graph=False))
            option_number += 1

    with open('Results/m dist and inliers csv.csv', 'w') as f:
        writer = csv.writer(f)
        for row in csv_rows:
            writer.writerow(row)


def check_traj(min_key_frame, max_key_frame, loops, numbers=False):
    pose_graph = DataDirectory.Data.PG
    key_frames = pose_graph.get_key_frames()[min_key_frame: max_key_frame]
    ground_truth = np.array(utils.utills.get_ground_truth_transformations())[key_frames]
    cameras_gt_3d_loc = utils.utills.left_cameras_trajectory(ground_truth)

    gtsam_initial_est_cameras_poses = pose_graph.get_initial_est_cameras_poses()
    initial_estimate_cam_3d_loc = utils.utills.gtsam_left_cameras_trajectory(gtsam_initial_est_cameras_poses)

    utils.plot.plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras_gt=cameras_gt_3d_loc,
                                                                                  cameras=cameras_gt_3d_loc,
                                                                                  numbers=numbers,
                                                                                  loops=loops)


def get_cam_pose(camera_ind):
    return Data.PG.get_initial_estimate_values().atPose3(symbol(PoseGraph.CAMERA_SYM, camera_ind))


def trajectory_states_over_the_process():

    key_frames = Data.BA.get_key_frames()

    ground_truth_trans = np.array(utils.utills.get_ground_truth_transformations())[key_frames]
    ground_truth = utils.utills.left_cameras_trajectory(ground_truth_trans)

    initial_est = Data.DB.initial_estimate_poses()[key_frames]

    bundle_cam_trans = Data.BA.get_gtsam_cameras_global()
    bundle_opt = utils.utills.gtsam_left_cameras_trajectory(bundle_cam_trans)

    pose_graph = PoseGraph.create_pose_graph()
    pose_graph.optimize()
    pose_graph.loop_closure(0, 431)
    gtsam_optimized_cameras_poses = pose_graph.get_optimized_cameras_poses()
    loop_opt = utils.utills.gtsam_left_cameras_trajectory(gtsam_optimized_cameras_poses)

    utils.plot.trajectory_state_over_the_process(initial_est, bundle_opt, loop_opt, ground_truth)


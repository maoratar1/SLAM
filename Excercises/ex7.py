import cProfile
import csv
import pstats

import numpy as np
import tqdm
from gtsam import symbol
from matplotlib import pyplot as plt

import DataDirectory.Data
import utils.plot
import utils.utills
from DataDirectory import Data
from PoseGraphDirectory import PoseGraph

# === Missions ===

    # Code for mission 7.5


def plot_pose_graph_versions(num_versions=5):  #Todo: add marginals
    pose_graph = DataDirectory.Data.PG
    prev_num_loops = 0
    for i in tqdm.tqdm(pose_graph.get_num_poses()):
        pose_graph.loop_closure_for_specific_frame(cur_cam_pose_graph_ind=i)
        if len(pose_graph.get_loops()) > prev_num_loops:
            prev_num_loops = len(pose_graph.get_loops())
            plot_pose_graph_trajectory_before_and_after_optimization(pose_graph)
            num_versions -= 1

        if num_versions == 0:
            break


def plot_pose_graph_absolute_location_error_before_and_after_opt():
    pose_graph = DataDirectory.Data.PG
    pose_graph.loop_closure(0, pose_graph.get_loops()[-1])

    initial_estimate_cam_3d_loc, opt_cameras_3d_loc, cameras_gt_3d_loc = \
                                                            cameras_initial_est_and_ground_truth_locations(pose_graph)

    absolute_loc_err_before_opt = utils.utills.euclidean_dist(initial_estimate_cam_3d_loc, cameras_gt_3d_loc)
    absolute_loc_err_after_opt = utils.utills.euclidean_dist(opt_cameras_3d_loc, cameras_gt_3d_loc)

    fig = plt.figure()
    plt.title(f"Pose graph absolute location error before and after opt")
    plt.plot(len(absolute_loc_err_before_opt), absolute_loc_err_before_opt, s=1, c='red')
    plt.plot(len(absolute_loc_err_after_opt), absolute_loc_err_after_opt, s=1, c='blue')

    fig.savefig(f"Results/Pose graph absolute location error before and after opt.png")
    plt.close(fig)


def mission(start_cam, end_cam, loaded_pose_graph=False):
    if not loaded_pose_graph:
        pose_graph = PoseGraph.create_pose_graph()
        pose_graph.optimize()
    else:
        pose_graph = DataDirectory.Data.PG

    pose_graph.loop_closure(start_cam, end_cam)

    return plot_pose_graph_trajectory_before_and_after_optimization(pose_graph)


def plot_pose_graph_trajectory_before_and_after_optimization(pose_graph):
    loops = pose_graph.get_loops()

    initial_estimate_cam_3d_loc, opt_cameras_3d_loc, cameras_gt_3d_loc = cameras_initial_est_and_ground_truth_locations(pose_graph)

    # Plot optimized trajectory without covariance
    utils.plot.plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=opt_cameras_3d_loc,
                                                                                  initial_estimate_poses=initial_estimate_cam_3d_loc,
                                                                                  cameras_gt=cameras_gt_3d_loc,
                                                                                  loops=loops, numbers=False,
                                                                                  mahalanobis_dist=PoseGraph.MAHALANOBIS_DIST_THRESHOLD,
                                                                                  inliers_perc=PoseGraph.INLIERS_THRESHOLD_PERC)

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


def plot_2_images(prev_cam_ind, cur_cam_ind):
    prev_frame_ind_whole_movie = Data.PG.get_key_frames()[prev_cam_ind]
    cur_frame_ind_whole_movie = Data.PG.get_key_frames()[cur_cam_ind]

    # Take the left images - we use the function read, so it won't apply blurring
    prev_img, cur_img = Data.KITTI.read_images(prev_frame_ind_whole_movie)[0], \
                        Data.KITTI.read_images(cur_frame_ind_whole_movie)[0]
    _, frame0_features, frame1_features,  supporters_idx, frame0_inliers_percentage = utils.utills.apply_full_consensus_match(prev_frame_ind_whole_movie, cur_frame_ind_whole_movie)

    left0_matches_coor = utils.utills.get_features_left_coor(frame0_features)
    left1_matches_coor = utils.utills.get_features_left_coor(frame1_features)
    utils.plot.plot_supporters(prev_frame_ind_whole_movie, cur_frame_ind_whole_movie,
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
            csv_rows.append([option_number] + mission(0, 432, loaded_pose_graph=False))
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





    # # original_stdout = sys.stdout
    # with open('Results/m dist and inliers csv', 'w') as f:
    #     # sys.stdout = f  # Change the standard output to the file we created.
    #     writer = csv.writer(f)
    #     writer.writerow([PoseGraph.MAHALANOBIS_DIST_THRESHOLD, PoseGraph.INLIERS_THRESHOLD_PERC, len(loops),
    #                      pose_graph.graph_error(optimized=False), pose_graph.graph_error(optimized=True), ])
    #
    #     # writer.writerow(f"mahalanobis distance: {PoseGraph.MAHALANOBIS_DIST_THRESHOLD}, Inliers percentage: {PoseGraph.INLIERS_THRESHOLD_PERC}\n")
    #     # writer.writerow(f"\tNum loops: {len(loops)}\n")
    #     # writer.writerow(f"\tGraph error before optimization: {pose_graph.graph_error(optimized=False)}\n")
    #     # writer.writerow(f"\tGraph error after optimization: {pose_graph.graph_error(optimized=True)}\n")
    #     # writer.writerow("\n")
    #     # sys.stdout = original_stdout  # Reset the standard output to its original value
import numpy as np
from DataDirectory import Data
import PoseGraph
from utils import plot

import gtsam
from gtsam import symbol
from gtsam.utils import plot
from BundleAdjustmentDirectory import BundleWindow


def mission1():
    # Get the first bundle at the bundle adjustment
    # bundle_adjustment = BundleAdjustment.BundleAdjustment()
    # first_bundle = bundle_adjustment.get_bundles_lst()[0]
    # first_key_frame, second_key_frame = first_bundle.get_key_frames()

    first_bundle = BundleWindow.BundleWindow(0, 11, Data.DB.get_frames()[0: 12])
    first_key_frame, second_key_frame = first_bundle.get_key_frames()
    # Create factor graph and optimize the first bundle
    first_bundle.create_factor_graph()
    first_bundle.optimize()

    # Get marginals
    marginals = first_bundle.marginals()
    result = first_bundle.get_optimized_values()
    plot.plot_trajectory(0, result, marginals=marginals, scale=1, title="Covariance poses for first bundle",
                         save_file="Results/Poses cov.png")

    # Apply marginalization and conditioning at the last key frame
    keys = gtsam.KeyVector()
    keys.append(symbol(BundleWindow.CAMERA_SYM, first_key_frame))
    keys.append(symbol(BundleWindow.CAMERA_SYM, second_key_frame))
    information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    cond_cov_mat = np.linalg.inv(information_mat_first_second)

    first_camera = result.atPose3(symbol(BundleWindow.CAMERA_SYM, first_key_frame))
    second_camera = result.atPose3(symbol(BundleWindow.CAMERA_SYM, second_key_frame))
    relative_pose = first_camera.between(second_camera)

    print("Relative covariance between the frame poses:\n", cond_cov_mat, "\n")
    print("Relative poses of last key frames:\n", relative_pose)


def mission2():
    bundles_lst = Data.BA.get_bundles_lst()
    key_frames = Data.BA.get_key_frames()

    rel_poses_lst, cov_mat_lst = PoseGraph.compute_cov_rel_poses_for_bundles(bundles_lst, PoseGraph.MULTI_PROCESSED, 5)

    pose_graph = PoseGraph.PoseGraph(key_frames, rel_poses_lst, cov_mat_lst)

    pose_graph.optimize()

    initial_estimate_poses = pose_graph.get_initial_estimate_values()
    optimized_poses = pose_graph.get_optimized_values()

    scale = 1
    # Plot initial estimate trajectory
    gtsam.utils.plot.plot_trajectory(1, initial_estimate_poses, scale=scale, title="Initial estimate pose",
                                     save_file="Results/Initial estimate pose.png", d2_view=True)

    # Plot optimized trajectory without covariance
    gtsam.utils.plot.plot_trajectory(2, optimized_poses, scale=scale, title="Optimized poses",
                                     save_file="Results/Optimized poses.png", d2_view=True)
    # utils.plot.compare_left_cam_2d_trajectory_to_ground_truth(initial_estimate_poses, optimized_poses)

    # Optimized trajectory with covariance
    marginals = pose_graph.marginals()
    plot.plot_trajectory(3, optimized_poses, marginals=marginals,
                         title="Optimized poses with covariance", scale=scale,
                         save_file="Results/Optimized poses with covariance.png", d2_view=True)

    # Graph error before and after optimization
    print("Graph error BEFORE optimization: ", pose_graph.graph_error(optimized=False))
    print("Graph error AFTER optimization: ", pose_graph.graph_error(optimized=True))

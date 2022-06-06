import numpy as np
from matplotlib import pyplot as plt

from DataDirectory import Data
from utils import utills
from utils import plot
import utils
from BundleAdjustmentDirectory.BundleAdjustment import BundleAdjustment
from BundleAdjustmentDirectory import BundleAdjustment as BA

import gtsam
from gtsam import symbol
from gtsam.utils import plot


# == Missions ==
def mission1():
    # Get random track and it's frames_in_window
    track = Data.DB.get_rand_track(track_len=10)
    frames_in_track = Data.DB.get_frames()[track.get_frames_ids()]

    frame_idx_triangulate = -1

    total_proj_dist, factors, values = triangulate_from_specific_frame_and_project_on_track(track, frames_in_track)

    # Factor error
    factor_projection_errors = projection_factors_error(factors, values)

    # Plots re projection error
    utils.plot.plot_re_projection_error_graph(factor_projection_errors, frame_idx_triangulate, "")
    plot_factor_re_projection_error_graph(factor_projection_errors, total_proj_dist, frame_idx_triangulate)
    plot_factor_as_func_of_re_projection_error_graph(factor_projection_errors, total_proj_dist, frame_idx_triangulate)


def mission2():
    factor_error, optimized_factor_error = 0, 0

    # Create one bundle
    key_frames = [0, 4]
    bundle_adjustment = BundleAdjustment()
    bundles_lst = bundle_adjustment.get_bundles_lst()[0]
    first_bundle = bundles_lst[0]

    # Create factor graph and compute factors error
    first_bundle.create_factor_graph()
    factor_error += first_bundle.graph_error(optimized=False)

    # Optimize and compute factors error after optimization
    first_bundle.optimize()
    optimized_factor_error = first_bundle.graph_error(optimized=True)

    print("First Bundle Errors:")
    print("Error before optimization: ", factor_error)
    print("Error after optimization: ", optimized_factor_error)

    # gtsam.utils.plot.set_axes_equal(1)
    gtsam.utils.plot.plot_trajectory(fignum=0, values=first_bundle.get_optimized_values(), save_file=f"VAN_ex/2 pts cloud.png")
    utils.plot.plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(
        cameras=[first_bundle.get_optimized_cameras_p3d()], landmarks=[first_bundle.get_optimized_landmarks_p3d()])


def mission3(method):
    # Solve BA
    bundle_adjustment = BundleAdjustment()
    bundle_adjustment.solve(method=method)
    gtsam_cameras_rel_to_bundle = bundle_adjustment.get_gtsam_cameras_rel_to_bundle()
    all_landmarks_rel_to_bundle = bundle_adjustment.get_all_landmarks_rel_to_bundle()
    cameras, landmarks = BA.convert_rel_cams_and_landmarks_to_global(gtsam_cameras_rel_to_bundle,  all_landmarks_rel_to_bundle)

    # Plot 2d trajectory of cameras and landmarks compared to ground truth
    key_frames = bundle_adjustment.get_key_frames()
    ground_truth = np.array(utills.get_ground_truth_transformations())[key_frames]
    cameras_gt_3d = utills.left_cameras_trajectory(ground_truth)
    cameras_3d = utills.gtsam_left_cameras_trajectory(cameras)

    initial_estimate = Data.DB.initial_estimate_poses()[key_frames]

    utils.plot.plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=cameras_3d,
                                                                                  landmarks=landmarks,
                                                                                  initial_estimate_poses=initial_estimate,
                                                                                  cameras_gt=cameras_gt_3d)

    # Present keyframe localization error in meters from ground truth
    euclidean_dist = utills.euclidean_dist(cameras_3d, cameras_gt_3d, dim="3d")
    utils.plot.plot_arr_values_as_function_of_its_indexes(euclidean_dist,
                                                          title=f"Euclidean distance error for {len(euclidean_dist)} cameras",
                                                          file_path="Euclidean distance error")


def triangulate_from_specific_frame_and_project_on_track(track, frames_in_track, frame_idx_triangulate=-1):
    """
    At this mission we:
    1. Randomize a track with len of track_len
    2. Triangulate a point from the "frame_idx_triangulate" frame in the track
    3. Projects it to all frames_in_window in the track
    4. Computes their reprojection error on each frame
    5. Finally, plot the results
    :param db: Database
    :param frame_idx_triangulate: frame's index to triangulate from
    """
    # Factors list and values
    factors = []
    values = gtsam.Values()
    frame_to_triangulate_from = frames_in_track[frame_idx_triangulate]
    first_frame = frames_in_track[0]

    # Locations in frames_in_window
    left_locations = track.get_left_locations_in_all_frames()
    right_locations = track.get_right_locations_in_all_frames()

    # Last frame locations for triangulations
    last_left_img_coor = left_locations[frame_idx_triangulate]
    last_right_img_coor = right_locations[frame_idx_triangulate]

    # Triangulation
    gtsam_calib_mat = utills.create_gtsam_calib_cam_mat(utills.K)
    first_frame_cam_to_world_ex_mat = utills.convert_ex_cam_to_cam_to_world(first_frame.get_ex_cam_mat())  # first cam -> world

    camera_relate_to_first_frame_trans = utills.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                        frame_to_triangulate_from.get_ex_cam_mat())

    cur_cam_pose = utills.convert_ex_cam_to_cam_to_world(camera_relate_to_first_frame_trans)
    gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)
    gtsam_frame_to_triangulate_from = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)
    xl, xr, y = last_left_img_coor[0], last_right_img_coor[0], last_left_img_coor[1]
    gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(xl, xr, y)

    gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

    # Update values dictionary
    p3d_sym = symbol("q", 0)
    values.insert(p3d_sym, gtsam_p3d)

    left_projections = []
    right_projections = []

    for i, frame in enumerate(frames_in_track):

        # Create camera symbol and update values dictionary
        left_pose_sym = symbol("c", frame.get_id())

        camera_relate_to_first_frame_trans = utills.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                            frame.get_ex_cam_mat())

        cur_cam_pose = utills.convert_ex_cam_to_cam_to_world(camera_relate_to_first_frame_trans)
        gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

        values.insert(left_pose_sym, gtsam_left_cam_pose)

        # Measurement values
        measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
        gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        gtsam_frame = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)

        # Project p34 on frame
        gtsam_projected_stereo_point2 = gtsam_frame.project(gtsam_p3d)
        xl, xr, y = gtsam_projected_stereo_point2.uL(), gtsam_projected_stereo_point2.uR(), gtsam_projected_stereo_point2.v()

        # Factor creation
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol("c", frame.get_id()), p3d_sym, gtsam_calib_mat)

        factors.append(factor)

        # Projections list
        left_projections.append([xl, y])
        right_projections.append([xr, y])

    # Compute projection euclidean error
    left_proj_dist = utills.euclidean_dist(np.array(left_projections), np.array(left_locations))
    right_proj_dist = utills.euclidean_dist(np.array(right_projections), np.array(right_locations))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    return total_proj_dist, factors, values


def projection_factors_error(factors, values):
    errors = []
    for factor in factors:
        errors.append(factor.error(values))

    return np.array(errors)


def plot_factor_re_projection_error_graph(factor_projection_errors, total_proj_dist,  frame_idx_triangulate):
    """
    Plots re projection error
    """
    exponent_vals = np.exp(0.5 * total_proj_dist)
    fig, ax = plt.subplots(figsize=(10, 7))

    frame_title = "Last" if frame_idx_triangulate == -1 else "First"

    ax.set_title(f"Factor Re projection error from {frame_title} frame")
    plt.scatter(range(len(factor_projection_errors)), factor_projection_errors, label="Factor")
    plt.scatter(range(len(exponent_vals)), exponent_vals, label="exp(0.5* ||z - proj(c, q)||)")
    plt.legend(loc="upper right")
    plt.ylabel('Error')
    plt.xlabel('Frames')

    fig.savefig(f"VAN_ex/Factor Re projection error graph for {frame_title} frame.png")
    plt.close(fig)


def plot_factor_as_func_of_re_projection_error_graph(factor_projection_errors, total_proj_dist,  frame_idx_triangulate):
    """
    Plots re projection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    frame_title = "Last" if frame_idx_triangulate == -1 else "First"

    ax.set_title(f"Factor error as a function of a Re projection error graph for {frame_title} frame")
    plt.plot(total_proj_dist, factor_projection_errors, label="Factor")
    plt.plot(total_proj_dist, 0.5 * total_proj_dist ** 2, label="0.5x^2")
    plt.plot(total_proj_dist, total_proj_dist ** 2, label="x^2")
    plt.legend(loc="upper left")
    plt.ylabel('Factor error')
    plt.xlabel('Re projection error')

    fig.savefig(f"VAN_ex/Factor error as a function of a Re projection error graph for {frame_title} frame.png")
    plt.close(fig)


# def check_bottleneck(db):
#     """
#     Check bottleneck at the code
#     """
#     profile = cProfile.Profile()
#     profile.runcall(bundle_adjustment_iterative, db)
#     ps = pstats.Stats(profile)
#     ps.sort_stats('cumtime')
#     ps.print_stats()
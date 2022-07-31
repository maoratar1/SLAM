import numpy as np
import tqdm
from matplotlib import pyplot as plt

from DataDirectory import Data
from BundleAdjustmentDirectory import BundleAdjustment
import utils
from utils import utills


def plot_pose_graph_absolute_location_error_before_and_after_opt():
    """
    Plots pose graph absolute location error before and after loop optimization
    """
    pose_graph = Data.PG

    initial_trans_loc_diff, opt_trans_loc_diff = utils.utills.cameras_initial_est_and_ground_truth_locations(pose_graph)

    norm = np.sqrt(initial_trans_loc_diff[:, 0] ** 2 + initial_trans_loc_diff[:, 1] ** 2 + initial_trans_loc_diff[:, 2] ** 2)
    norm_loop = np.sqrt(opt_trans_loc_diff[:, 0] ** 2 + opt_trans_loc_diff[:, 1] ** 2 + opt_trans_loc_diff[:, 2] ** 2)

    fig = plt.figure()
    plt.title(f"Pose graph absolute location error BEFORE Loop Closure")
    plt.plot(range(len(initial_trans_loc_diff[:, 0])), abs(initial_trans_loc_diff[:, 0]), label="X's error")
    plt.plot(range(len(initial_trans_loc_diff[:, 1])), abs(initial_trans_loc_diff[:, 1]), label="Y's error")
    plt.plot(range(len(initial_trans_loc_diff[:, 2])), abs(initial_trans_loc_diff[:, 2]), label="Z's error")
    plt.plot(range(len(norm)), norm, label="Norm")

    plt.ylabel("Absolute Error(m)")
    plt.xlabel("Frame")
    plt.legend()

    fig.savefig(f"Results/Pose graph absolute location error BEFORE opt.png")
    plt.close(fig)

    fig = plt.figure()
    plt.title(f"Pose graph absolute location error AFTER Loop Closure optimization")
    plt.plot(range(len(opt_trans_loc_diff[:, 0])), abs(opt_trans_loc_diff[:, 0]), label="X's error")
    plt.plot(range(len(opt_trans_loc_diff[:, 1])), abs(opt_trans_loc_diff[:, 1]), label="Y's error")
    plt.plot(range(len(opt_trans_loc_diff[:, 2])), abs(opt_trans_loc_diff[:, 2]), label="Z's error")
    plt.plot(range(len(norm_loop)), norm_loop, label="Norm")
    plt.ylim(0, 50)
    plt.ylabel("Absolute Error(m)")
    plt.xlabel("Frame")
    plt.legend()

    fig.savefig(f"Results/Pose graph absolute location error AFTER opt.png")
    plt.close(fig)


def plot_pose_graph_absolute_angles_error_before_and_after_opt():
    """
    Plots pose graph absolute angles error before and after loop optimization
    """
    pose_graph = Data.PG

    initial_trans_angles_diff, opt_trans_angles_diff = utils.utills.cameras_initial_est_and_ground_truth_angles(pose_graph)

    fig = plt.figure()
    plt.title(f"Pose graph absolute angles error BEFORE Loop Closure")
    plt.plot(range(len(initial_trans_angles_diff[:, 0])), abs(initial_trans_angles_diff[:, 0]) * 180 / np.pi, label="YZ plane error")
    plt.plot(range(len(initial_trans_angles_diff[:, 1])), abs(initial_trans_angles_diff[:, 1]) * 180 / np.pi, label="XZ plane error")
    plt.plot(range(len(initial_trans_angles_diff[:, 0])), abs(initial_trans_angles_diff[:, 2]) * 180 / np.pi, label="XY plane error")

    plt.ylabel("Absolute Error(Deg)")
    plt.xlabel("Frame")
    plt.legend()

    fig.savefig(f"Results/Pose graph absolute angles error BEFORE opt.png")
    plt.close(fig)

    fig = plt.figure()
    plt.title(f"Pose graph absolute angles error AFTER Loop Closure optimization")
    plt.plot(range(len(opt_trans_angles_diff[:, 0])), abs(opt_trans_angles_diff[:, 0] * 180 / np.pi), label="YZ plane error")
    plt.plot(range(len(opt_trans_angles_diff[:, 1])), abs(opt_trans_angles_diff[:, 1] * 180 / np.pi), label="XZ plane error")
    plt.plot(range(len(opt_trans_angles_diff[:, 2])), abs(opt_trans_angles_diff[:, 2] * 180 / np.pi), label="XY plane error")
    plt.ylim(0, 50)
    plt.ylabel("Absolute Error(Deg)")
    plt.xlabel("Frame")
    plt.legend()

    fig.savefig(f"Results/Pose graph absolute angles error AFTER opt.png")
    plt.close(fig)


def plot_pnp_absolute_location_error():
    """
    Plots PnP absolute location error
    """
    db = Data.DB
    trans_diff_loc = utils.utills.cameras_rel_pnp_est_and_ground_truth_locations(db)
    norm = np.sqrt(trans_diff_loc[:, 0] ** 2 + trans_diff_loc[:, 1] ** 2 + trans_diff_loc[:, 2] ** 2)

    fig = plt.figure()
    plt.title(f"PnP absolute location error")
    plt.plot(range(len(trans_diff_loc[:, 0])), abs(trans_diff_loc[:, 0]), label="X's error")
    plt.plot(range(len(trans_diff_loc[:, 1])), abs(trans_diff_loc[:, 1]), label="Y's error")
    plt.plot(range(len(trans_diff_loc[:, 2])), abs(trans_diff_loc[:, 2]), label="Z's error")
    plt.plot(range(len(norm)), norm, label="Norm")

    plt.ylabel("Absolute Error(m)")
    plt.xlabel("Frame")
    plt.legend()

    fig.savefig(f"Results/PnP absolute location error.png")
    plt.close(fig)


def plot_pnp_absolute_angles_error():
    """
    Plots PnP absolute angles error
    """
    db = Data.DB

    trans_diff_angles = utils.utills.cameras_rel_pnp_est_and_ground_truth_angles(db)

    fig = plt.figure()
    plt.title(f"PnP absolute angles error")
    plt.plot(range(len(trans_diff_angles[:, 0])), abs(trans_diff_angles[:, 0]) * 180 / np.pi, label="YZ plane error")
    plt.plot(range(len(trans_diff_angles[:, 1])), abs(trans_diff_angles[:, 1]) * 180 / np.pi, label="XZ plane error")
    plt.plot(range(len(trans_diff_angles[:, 2])), abs(trans_diff_angles[:, 2]) * 180 / np.pi, label="XY plane error")

    plt.ylabel("Absolute Error(Deg)")
    plt.xlabel("Frame")
    plt.legend(loc='upper left')

    fig.savefig(f"Results/PnP absolute angles.png")
    plt.close(fig)


def rel_pnp_loc_error_over_seq(seq_len):
    """
    Plots PnP locations error over sequence with length of "seq_len"
    """
    db = Data.BA
    initial_estimate_cam_3d_loc, cameras_gt_3d_loc = utils.utills.cameras_rel_pnp_est_and_ground_truth_locations(db)

    perc_err_x = []
    perc_err_y = []
    perc_err_z = []

    for i in tqdm.tqdm(range(0, len(initial_estimate_cam_3d_loc) - seq_len)):
        absolute_x = abs(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 0]) - diff_along_seq(initial_estimate_cam_3d_loc[i: i + seq_len, 0]))
        perc_err_x.append(sum(absolute_x) / sum(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 0])))
        absolute_y = abs(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 1]) - diff_along_seq(initial_estimate_cam_3d_loc[i: i + seq_len, 1]))
        perc_err_y.append(sum(absolute_y) / sum(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 1])))
        absolute_z = abs(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 2]) - diff_along_seq(initial_estimate_cam_3d_loc[i: i + seq_len, 2]))
        perc_err_z.append(sum(absolute_z) / sum(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 2])))
        # norm = np.sqrt(absolute_x ** 2 + absolute_y ** 2 + absolute_z ** 2)

    fig = plt.figure()
    plt.title(f"Bundle Adjustment Relative Location Error Percentage Of Sequence Of {seq_len} Length")
    plt.plot(range(len(perc_err_x)), perc_err_x, label="X's relative error percentage")
    plt.plot(range(len(perc_err_y)), perc_err_y, label="Y's relative error percentage")
    plt.plot(range(len(perc_err_z)), perc_err_z, label="Z's relative error percentage")
    # plt.plot(range(len(norm)), norm, label="relative error's norm percentage")

    plt.ylabel("Error Percentage")
    plt.xlabel("Starting Frame")
    plt.legend(loc="upper right")

    fig.savefig(f"Results/BA percentage location error seq {seq_len}.png")
    plt.close(fig)


def diff_along_seq(pos):
    """
    Compute accumulative diff
    :param pos:
    :return:
    """
    dists = np.array([abs(pos[i] - pos[i - 1]) for i in range(1, len(pos))])
    return dists


def rel_pnp_angles_error_over_seq(seq_len):
    """
    Plots PnP angles error over sequence with length of "seq_len"
    """
    db = Data.DB
    initial_estimate_cam_3d_angles, cameras_gt_3d_angles = \
        utils.utills.cameras_rel_pnp_est_and_ground_truth_angles(db)

    perc_err_x = []
    perc_err_y = []
    perc_err_z = []

    for i in range(0, len(initial_estimate_cam_3d_angles) - seq_len):
        absolute_x = abs(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 0]) - diff_along_seq(
            initial_estimate_cam_3d_angles[i: i + seq_len, 0]))
        perc_err_x.append(sum(absolute_x) / sum(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 0])))
        absolute_y = abs(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 1]) - diff_along_seq(
            initial_estimate_cam_3d_angles[i: i + seq_len, 1]))
        perc_err_y.append(sum(absolute_y) / sum(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 1])))
        absolute_z = abs(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 2]) - diff_along_seq(
            initial_estimate_cam_3d_angles[i: i + seq_len, 2]))
        perc_err_z.append(sum(absolute_z) / sum(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 2])))
        # norm = np.sqrt(absolute_x ** 2 + absolute_y ** 2 + absolute_z ** 2)

    fig = plt.figure()
    plt.title(f"Bundle Adjustment Relative Angles Error Percentage Of Sequence Of {seq_len} Length")
    plt.plot(range(len(perc_err_x)), perc_err_x, label="YZ plane relative error percentage")
    plt.plot(range(len(perc_err_y)), perc_err_y, label="XY relative error percentage")
    plt.plot(range(len(perc_err_z)), perc_err_z, label="XZ relative error percentage")

    plt.xlabel("Starting Frame")
    plt.ylabel("Error Percentage")
    plt.legend(loc='upper left')

    fig.savefig(f"Results/BA percentage angles error seq {seq_len}.png")
    plt.close(fig)


def rel_ba_loc_error_over_seq(seq_len):
    """
    Plots Bundle Adjustment location error over sequence with length of "seq_len"
    """
    bundle_adjustment = Data.BA
    gtsam_cameras_rel_to_bundle = bundle_adjustment.get_gtsam_cameras_rel_to_bundle()
    all_landmarks_rel_to_bundle = bundle_adjustment.get_all_landmarks_rel_to_bundle()
    cameras, landmarks = BundleAdjustment.convert_rel_cams_and_landmarks_to_global(gtsam_cameras_rel_to_bundle,
                                                                     all_landmarks_rel_to_bundle)

    # Plot 2d trajectory of cameras and landmarks compared to ground truth
    key_frames = bundle_adjustment.get_key_frames()
    ground_truth = np.array(utills.get_ground_truth_transformations())[key_frames]
    cameras_gt_3d_loc = utills.left_cameras_trajectory(ground_truth)
    cameras_3d = utills.gtsam_left_cameras_trajectory(cameras)

    perc_err_x = []
    perc_err_y = []
    perc_err_z = []

    for i in tqdm.tqdm(range(0, len(cameras_3d) - seq_len)):
        absolute_x = abs(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 0]) - diff_along_seq(cameras_3d[i: i + seq_len, 0]))
        perc_err_x.append(sum(absolute_x) / sum(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 0])))
        absolute_y = abs(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 1]) - diff_along_seq(cameras_3d[i: i + seq_len, 1]))
        perc_err_y.append(sum(absolute_y) / sum(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 1])))
        absolute_z = abs(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 2]) - diff_along_seq(cameras_3d[i: i + seq_len, 2]))
        perc_err_z.append(sum(absolute_z) / sum(diff_along_seq(cameras_gt_3d_loc[i: i + seq_len, 2])))
        # norm = np.sqrt(absolute_x ** 2 + absolute_y ** 2 + absolute_z ** 2)

    fig = plt.figure()
    plt.title(f"Bundle Adjustment Relative Location Error Percentage Of Sequence Of {seq_len} Length")
    plt.plot(range(len(perc_err_x)), perc_err_x, label="X's relative error percentage")
    plt.plot(range(len(perc_err_y)), perc_err_y, label="Y's relative error percentage")
    plt.plot(range(len(perc_err_z)), perc_err_z, label="Z's relative error percentage")
    # plt.plot(range(len(norm)), norm, label="relative error's norm percentage")

    plt.ylabel("Error Percentage")
    plt.xlabel("Starting Frame")
    plt.legend(loc="upper right")

    fig.savefig(f"Results/BA percentage location error seq {seq_len}.png")
    plt.close(fig)


def rel_ba_angles_error_over_seq(seq_len):
    """
    Plots Bundle Adjustment angles error over sequence with length of "seq_len"
    """
    bundle_adjustment = Data.BA
    gtsam_cameras_rel_to_bundle = bundle_adjustment.get_gtsam_cameras_rel_to_bundle()
    all_landmarks_rel_to_bundle = bundle_adjustment.get_all_landmarks_rel_to_bundle()
    cameras, landmarks = BundleAdjustment.convert_rel_cams_and_landmarks_to_global(gtsam_cameras_rel_to_bundle,
                                                                          all_landmarks_rel_to_bundle)

    # Plot 2d trajectory of cameras and landmarks compared to ground truth
    key_frames = bundle_adjustment.get_key_frames()
    ground_truth = np.array(utills.get_ground_truth_transformations())[key_frames]
    cameras_gt_3d_angles = utills.left_cameras_trajectory(ground_truth)
    cameras_3d = utills.gtsam_left_cameras_trajectory(cameras)

    perc_err_x = []
    perc_err_y = []
    perc_err_z = []

    for i in range(0, len(cameras_3d) - seq_len):
        absolute_x = abs(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 0]) - diff_along_seq(
            cameras_3d[i: i + seq_len, 0]))
        perc_err_x.append(sum(absolute_x) / sum(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 0])))
        absolute_y = abs(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 1]) - diff_along_seq(
            cameras_3d[i: i + seq_len, 1]))
        perc_err_y.append(sum(absolute_y) / sum(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 1])))
        absolute_z = abs(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 2]) - diff_along_seq(
            cameras_3d[i: i + seq_len, 2]))
        perc_err_z.append(sum(absolute_z) / sum(diff_along_seq(cameras_gt_3d_angles[i: i + seq_len, 2])))
        # norm = np.sqrt(absolute_x ** 2 + absolute_y ** 2 + absolute_z ** 2)

    fig = plt.figure()
    plt.title(f"Bundle Adjustment Relative Angles Error Percentage Of Sequence Of {seq_len} Length")
    plt.plot(range(len(perc_err_x)), perc_err_x, label="YZ plane relative error percentage")
    plt.plot(range(len(perc_err_y)), perc_err_y, label="XY relative error percentage")
    plt.plot(range(len(perc_err_z)), perc_err_z, label="XZ relative error percentage")

    plt.xlabel("Starting Frame")
    plt.ylabel("Error Percentage")
    plt.legend(loc='upper left')

    fig.savefig(f"Results/BA percentage angles error seq {seq_len}.png")
    plt.close(fig)



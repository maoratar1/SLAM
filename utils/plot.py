import cv2
import matplotlib.pyplot as plt

import BundleAdjustmentDirectory.BundleAdjustment
from DataDirectory import Data
import utils.utills as utills

INLIER_COLOR = "orange"
OUTLIER_COLOR = "cyan"

def plot_frame1(frame_num):
    rows, cols = 1, 2

    fig, ax_arr = plt.subplots(rows, cols)
    fig.set_figwidth(10)
    fig.set_figheight(7)
    fig.tight_layout()

    fig.suptitle(f"Frame {frame_num}")
    # plt.rcParams["font.size"] = "10"
    plt.subplots_adjust(wspace=0.0, hspace=0.1)

    ax_arr[0].set_title("Left")
    ax_arr[1].set_title("Right")

    left_img, right_img = Data.KITTI.read_images(frame_num)

    ax_arr[0].axes.xaxis.set_visible(False)
    ax_arr[0].set_yticklabels([])
    ax_arr[0].imshow(left_img, cmap='gray')

    ax_arr[1].axes.xaxis.set_visible(False)
    ax_arr[1].set_yticklabels([])
    ax_arr[1].imshow(right_img, cmap='gray')

    fig.tight_layout()
    fig.savefig(f'Results/frame{frame_num}.png')
    plt.close(fig)


def plot_frame(frame_num):
    rows, columns = 1, 2
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Frame {frame_num}")

    fig.add_subplot(rows, columns, 1)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()

    left_img, right_img = Data.KITTI.read_images(frame_num)

    plt.imshow(left_img, cmap='gray')
    plt.axis('off')
    plt.title("Left camera")

    fig.add_subplot(rows, columns, 2)

    plt.imshow(right_img, cmap='gray')
    plt.axis('off')
    plt.title("Right camera")

    plt.subplots_adjust(wspace=0.0, hspace=0.1)

    fig.savefig(f'Results/frame{frame_num}.png')
    plt.close(fig)


def plot_arr_values_as_function_of_its_indexes(values, title, file_path):
    """
    Plot arr values where the x axes is the array's indxes
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"{title}.")
    ax.scatter(range(len(values)), values, s=1, c='red')

    fig.savefig(f"Results/{file_path}.png")
    plt.close(fig)

# Ex2
from matplotlib.patches import ConnectionPatch


def plot_matches(img1, img1_kpts, img2, img2_kpts, matches):
    """
    Draws a line between matches in img1 and img2
    """
    result = cv2.drawMatches(img1, img1_kpts, img2, img2_kpts, matches, None, flags=2)
    # Display the best matching d2_points
    plt.rcParams['figure.figsize'] = [20.0, 10.0]
    plt.title(f'Best Matching Points. Num matches:{len(matches)}')
    plt.imshow(result)
    plt.show()


def plot_tracking(left0, right0, left1, right1,
                  left0_matches_coor, right0_matches_coor, left1_matches_coor, right1_matches_coor):
    """
    Draw a tracking of a keypoint in 4 KITTI
    """
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 2

    fig.suptitle(f"{len(left0_matches_coor)} point tracking")

    # Left0 camera
    ax1 = fig.add_subplot(rows, cols, 1)
    plt.imshow(left0, cmap='gray')
    plt.title("Left 0")

    # Right0 camera
    ax2 = fig.add_subplot(rows, cols, 2)
    plt.imshow(right0, cmap='gray')
    plt.title("Right 0")

    # Left1 camera
    ax3 = fig.add_subplot(rows, cols, 3)
    plt.imshow(left1, cmap='gray')
    plt.title("Left 1")

    # Right1 camera
    ax4 = fig.add_subplot(rows, cols, 4)
    plt.imshow(right1, cmap='gray')
    plt.title("Right 1")

    for i in range(len(left0_matches_coor)):
        left0_right0_line = ConnectionPatch(xyA=(left0_matches_coor[i][0], left0_matches_coor[i][1]),
                               xyB=(right0_matches_coor[i][0], right0_matches_coor[i][1]),
                               coordsA="data", coordsB="data",
                               axesA=ax1, axesB=ax2, color="red")

        left0_left1_line = ConnectionPatch(xyA=(left0_matches_coor[i][0], left0_matches_coor[i][1]),
                          xyB=(left1_matches_coor[i][0], left1_matches_coor[i][1]),
                          coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax3, color="red")

        left1_right1_line = ConnectionPatch(xyA=(left1_matches_coor[i][0], left1_matches_coor[i][1]),
                               xyB=(right1_matches_coor[i][0], right1_matches_coor[i][1]),
                               coordsA="data", coordsB="data",
                               axesA=ax3, axesB=ax4, color="red")
        ax2.add_artist(left0_right0_line)
        ax3.add_artist(left0_left1_line)
        ax4.add_artist(left1_right1_line)

    fig.savefig("Results/tracking.png")
    plt.close(fig)


# Ex3
def plot_triangulations(p3d_pts, cv_p3d_pts):
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
    ax.set_xlim3d(-20, 10)
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

    fig.savefig(f"Results/triangulations plot.png")
    plt.close(fig)


def plot_left_cam_2d_trajectory(left_cameras_pos):
    """
    Draw left camera 2d trajectory
    :param left_cameras_pos : numpy array of *3d* points
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory for {len(left_cameras_pos)} frames_ind.")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red')

    fig.savefig(f"Results/Left cameras 2d trajectory.png")
    plt.close(fig)


def compare_left_cam_2d_trajectory_to_ground_truth(left_cameras_pos, left_cameras_pos_gt):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory of {len(left_cameras_pos)} frames_ind (ground truth - cyan)")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red')
    ax.scatter(left_cameras_pos_gt[:, 0], left_cameras_pos_gt[:, 2], s=1, c='cyan')

    fig.savefig("Results/Compare Left cameras 2d trajectory.png")
    plt.close(fig)


def compare_left_cam_2d_trajectory_to_ground_truth_params(left_cameras_pos, left_cameras_pos_gt, alg_name, match_method, time):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory compared to ground truth of"
                 f" {len(left_cameras_pos)} frames_ind (ground truth - cyan)\n"
                 f"{alg_name} | {match_method} | Time: {time}")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red', label="Initial estimation")
    ax.scatter(left_cameras_pos_gt[:, 0], left_cameras_pos_gt[:, 2], s=1, c='cyan'.title(), label="Cameras ground truth")
    plt.legend(loc='upper left')

    fig.savefig(f"Results/Compare Left cameras 2d trajectory {alg_name} {match_method}.png")
    plt.close(fig)


def plot_supporters(first_frame_ind, second_frame_ind, prev_frame_pg_ind, cur_frame_pg_ind,
                    left0, left1, left0_matches_coor=None, left1_matches_coor=None,
                    supporters_idx=None, inliers_perc=None):
    """
    Plot KITTI supporters on left0 and left1
    """
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1

    perc_title = ""
    if left1_matches_coor is not None:
        inliers_perc = format(inliers_perc, ".2f")
        perc_title = f"supporters({inliers_perc}% inliers, {INLIER_COLOR})"

    fig.suptitle(f'frame {prev_frame_pg_ind}(movie ind: {first_frame_ind}) and frame {cur_frame_pg_ind} (moive ind:'
                 f'{second_frame_ind})' 
                 f' {perc_title}')

    # Left1 camera
    fig.add_subplot(rows, cols, 2)
    plt.imshow(left1, cmap='gray')
    plt.title(f"frame {cur_frame_pg_ind}")

    if left1_matches_coor is not None:
        plt.scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=3, color=OUTLIER_COLOR)
        plt.scatter(left1_matches_coor[:, 0][supporters_idx], left1_matches_coor[:, 1][supporters_idx], s=3, color=INLIER_COLOR)

    # Left0 camera
    fig.add_subplot(rows, cols, 1)
    plt.imshow(left0, cmap='gray')
    plt.title(f"frame {prev_frame_pg_ind}")

    if left0_matches_coor is not None:
        plt.scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=3, color=OUTLIER_COLOR)
        plt.scatter(left0_matches_coor[:, 0][supporters_idx], left0_matches_coor[:, 1][supporters_idx], s=3,
                    color=INLIER_COLOR)

    fig.savefig(f"Results/Left {prev_frame_pg_ind} and {cur_frame_pg_ind} frames_ind supporters.png")
    plt.close(fig)


def plot_supporters_with_outliers(first_frame_ind, second_frame_ind, left0, left1,
                                  left0_matches_coor=None, left1_matches_coor=None,
                                  inliers_perc=None): # Todo: change to the version of ex3
    """
    Plot KITTI supporters on left0 and left1
    """
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1

    perc_title = ""
    if left1_matches_coor is not None:
        inliers_perc = "{:.2f}".format(inliers_perc)
        perc_title = f"supporters({inliers_perc}, {INLIER_COLOR})"

    fig.suptitle(f'frame {first_frame_ind} and frame {second_frame_ind} {perc_title}')

    # Second camera
    fig.add_subplot(rows, cols, 2)
    plt.imshow(left1, cmap='gray')
    plt.title(f"frame {second_frame_ind}")

    if left1_matches_coor is not None:
        plt.scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
        plt.scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=3, color=INLIER_COLOR)

    # First camera
    fig.add_subplot(rows, cols, 1)
    plt.imshow(left0, cmap='gray')
    plt.title(f"frame {first_frame_ind}")
    if left0_matches_coor is not None:
        plt.scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
        plt.scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=3, color=INLIER_COLOR)

    fig.savefig("Results/Left0 Left1 supporters.png")
    plt.close(fig)


def plot_supporters_compare_to_ransac(left0, left1,
                                      left0_matches_coor, left1_matches_coor,
                                      supporters_idx, ransac_supporters_idx):
    """
    Plots the supporters in left0 and left1 with and without ransac
    """
    rows, cols = 2, 2
    fig, ax_arr = plt.subplots(rows, cols)
    fig.set_figwidth(14)
    fig.set_figheight(7)
    plt.tight_layout()
    fig.suptitle(f'Left 0 and left1 supporters with/out ransac')

    # plt.rcParams["font.size"] = "10"
    plt.subplots_adjust(wspace=0.1, hspace=-0.55)
    ax_arr[0, 0].set_title(
        f"Without ransac supporters({len(supporters_idx)} / {len(left1_matches_coor)}, {INLIER_COLOR})")
    ax_arr[0, 1].set_title(
        f"With ransac supporters({len(ransac_supporters_idx)} / {len(left1_matches_coor)}, {INLIER_COLOR})")

    ax_arr[0, 0].axes.xaxis.set_visible(False)
    ax_arr[0, 0].axes.yaxis.set_label_text(f"Left 1")
    ax_arr[0, 0].set_yticklabels([])
    ax_arr[0, 0].imshow(left1, cmap='gray')
    ax_arr[0, 0].scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    ax_arr[0, 0].scatter(left1_matches_coor[supporters_idx][:, 0],
                left1_matches_coor[supporters_idx][:, 1], s=1, color=INLIER_COLOR)

    ax_arr[1, 0].axes.xaxis.set_visible(False)
    ax_arr[1, 0].axes.yaxis.set_label_text(f"Left 0")
    ax_arr[1, 0].set_yticklabels([])
    ax_arr[1, 0].imshow(left0, cmap='gray')
    ax_arr[1, 0].scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    ax_arr[1, 0].scatter(left0_matches_coor[supporters_idx][:, 0],
                         left0_matches_coor[supporters_idx][:, 1], s=1, color=INLIER_COLOR)

    ax_arr[0, 1].axes.xaxis.set_visible(False)
    ax_arr[0, 1].axes.yaxis.set_label_text(f"Left 1")
    ax_arr[0, 1].set_yticklabels([])
    ax_arr[0, 1].imshow(left1, cmap='gray')
    ax_arr[0, 1].scatter(left1_matches_coor[:, 0], left1_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    ax_arr[0, 1].scatter(left1_matches_coor[ransac_supporters_idx][:, 0],
                         left1_matches_coor[ransac_supporters_idx][:, 1], s=1, color=INLIER_COLOR)

    ax_arr[1, 1].axes.xaxis.set_visible(False)
    ax_arr[1, 1].axes.yaxis.set_label_text(f"Left 0")
    ax_arr[1, 1].set_yticklabels([])
    ax_arr[1, 1].imshow(left0, cmap='gray')
    ax_arr[1, 1].scatter(left0_matches_coor[:, 0], left0_matches_coor[:, 1], s=1, color=OUTLIER_COLOR)
    ax_arr[1, 1].scatter(left0_matches_coor[ransac_supporters_idx][:, 0],
                         left0_matches_coor[ransac_supporters_idx][:, 1], s=1, color=INLIER_COLOR)


    fig.savefig("Results/Left0 Left1 supporters ransac compare.png")
    plt.close(fig)


def plot_compare_left_cam_2d_trajectory_to_ground_truth_params(left_cameras_pos, left_cameras_pos_gt, alg_name, match_method, time):
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


# Ex4
def plot_track(track, first_frame=0, last_frame=-1):
    """
    Plots a track all over it's frames_in_window with a line connecting the track
    :param track:
    :return:
    """

    track_id = track.get_id()
    if last_frame > track.get_last_frame_ind():
        last_frame = -1
    if first_frame < track.get_first_frame_ind():
        first_frame = 0

    frames_ids = track.get_frames_ids()[first_frame: last_frame]
    frames_coor = track.get_frames_features_dict()

    # rows, cols = len(frames_coor) // 3, 3
    rows, cols = len(frames_ids), 2
    crop_size = 100

    fig, ax_arr = plt.subplots(rows, cols)
    fig.set_figwidth(4)
    fig.set_figheight(12)

    fig.suptitle(f"Track(id:{track_id}) with len {len(frames_ids)}\n Image cropped to {crop_size}X{crop_size}")
    plt.rcParams["font.size"] = "10"
    plt.subplots_adjust(wspace=-0.55, hspace=0.1)

    ax_arr[0, 0].set_title("Left")
    ax_arr[0, 1].set_title("Right")

    prev_l_coor, prev_left, prev_l_ax = None, None, None
    prev_r_coor, prev_right, prev_r_ax = None, None, None

    i = 0

    for frame_id in frames_ids:
        feature = frames_coor[frame_id]
        l_coor = feature.get_left_coor()
        r_coor = feature.get_right_coor()

        left_img, right_img = Data.KITTI.read_images(frame_id)
        cropped_left_img, l_coor = crop_image(l_coor, left_img, crop_size)
        cropped_right_img, r_coor = crop_image(r_coor, right_img, crop_size)

        ax_arr[i, 0].axes.xaxis.set_visible(False)
        ax_arr[i, 0].axes.yaxis.set_label_text(f"frame {frame_id}")
        ax_arr[i, 0].set_yticklabels([])
        ax_arr[i, 0].imshow(cropped_left_img, cmap='gray')
        ax_arr[i, 0].scatter(l_coor[0], l_coor[1], c='red', s=10)

        ax_arr[i, 1].axes.xaxis.set_visible(False)
        ax_arr[i, 1].set_yticklabels([])
        ax_arr[i, 1].imshow(cropped_right_img, cmap='gray')
        ax_arr[i, 1].scatter(r_coor[0], r_coor[1], c='red', s=10)

        if i == 0:
            prev_l_coor, prev_left, prev_l_ax = l_coor, cropped_left_img, ax_arr[i, 0]
            prev_r_coor, prev_right, prev_r_ax = r_coor, cropped_right_img, ax_arr[i, 1]

        left0_left1_line = ConnectionPatch(xyA=(prev_l_coor[0], prev_l_coor[1]),
                                           xyB=(l_coor[0], l_coor[1]),
                                           coordsA="data", coordsB="data",
                                           axesA=prev_l_ax, axesB=ax_arr[i, 0], color="cyan")

        right0_right1_line = ConnectionPatch(xyA=(prev_r_coor[0], prev_r_coor[1]),
                                             xyB=(r_coor[0], r_coor[1]),
                                             coordsA="data", coordsB="data",
                                             axesA=prev_r_ax, axesB=ax_arr[i, 1], color="cyan")

        ax_arr[i, 0].add_artist(left0_left1_line)
        ax_arr[i, 1].add_artist(right0_right1_line)

        prev_l_coor, prev_left, prev_l_ax = l_coor, cropped_left_img, ax_arr[i, 0]
        prev_r_coor, prev_right, prev_r_ax = r_coor, cropped_right_img, ax_arr[i, 1]

        i += 1

    fig.savefig(f"Results/track {track_id}.png")
    plt.close(fig)


def crop_image(coor, img, crop_size):
    """
    Crops image "img" to size of "crop_size" X "crop_size" around the d2_points "coor"
    :return: Cropped image
    """
    r_x = int(min(utills.IMAGE_WIDTH, coor[0] + crop_size))
    l_x = int(max(0, coor[0] - crop_size))
    u_y = int(max(0, coor[1] - crop_size))
    d_y = int(min(utills.IMAGE_HEIGHT, coor[1] + crop_size))

    return img[u_y: d_y, l_x: r_x], [crop_size, crop_size]

# Ex5
def plot_re_projection_error_graph(total_proj_dist, frame_idx_triangulate, title):
    """
    Plots re projection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    frame_title = "Last" if frame_idx_triangulate == -1 else "First"

    ax.set_title(f"{title} Re projection error from {frame_title} frame")
    plt.scatter(range(len(total_proj_dist)), total_proj_dist)
    plt.ylabel('Error')
    plt.xlabel('Frames')

    fig.savefig(f"Results/{title} Re projection error graph for {frame_title} frame.png")
    plt.close(fig)


def plot_cam_and_landmarks_2d_trajectory(cameras, landmarks):
    """
    Draw left camera 2d trajectory
    :param cameras : numpy array of *3d* points
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras and landmarks 2d trajectory for {len(cameras)} cameras.")
    ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red')
    ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='blue')

    fig.savefig(f"Results/Left cameras 2d trajectory.png")
    plt.close(fig)


def plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=None, landmarks=None,
                                                                       initial_estimate_poses=None, cameras_gt=None,
                                                                       title="",
                                                                       loops=None, numbers=False,
                                                                       mahalanobis_dist=None, inliers_perc=None):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    first_legend = []

    landmarks_title = "and landmarks " if landmarks is not None else ""
    loops_title = ""
    dist_title = "Dist = squared mahalanobis distance " if mahalanobis_dist is not None else ""

    ax.set_title(f"{title} Left cameras {landmarks_title}2d trajectory of {len(cameras_gt)} bundles.\n{dist_title}"
                 f"{loops_title}")

    if landmarks is not None:
        a = ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='orange', label="Landmarks")
        first_legend.append(a)

    if initial_estimate_poses is not None:
        first_legend.append(ax.scatter(initial_estimate_poses[:, 0], initial_estimate_poses[:, 2], s=1, c='pink', label="Initial estimate"))

    if cameras is not None:
        first_legend.append(ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red', label="Optimized cameras"))

    if cameras_gt is not None:
        first_legend.append(ax.scatter(cameras_gt[:, 0], cameras_gt[:, 2], s=1, c='cyan', label="Cameras ground truth"))

    # Mark loops
    if loops is not None:
        for cur_cam, prev_cams in loops:
            y_diff = 0 if abs(cameras[:, 0][cur_cam] - cameras[:, 0][cur_cam - 1]) < 2 else 15
            x_diff = 0 if abs(cameras[:, 2][cur_cam] - cameras[:, 2][cur_cam - 1]) < 2 else 20

            if numbers:
                ax.text(cameras[:, 0][cur_cam] - x_diff, cameras[:, 2][cur_cam] - y_diff, cur_cam, size=7, fontweight="bold")
            ax.scatter(cameras[:, 0][cur_cam], cameras[:, 2][cur_cam], s=3, c='black')

            if numbers:
                for prev_cam in prev_cams:
                    ax.text(cameras[:, 0][prev_cam], cameras[:, 2][prev_cam], prev_cam, size=7, fontweight="bold")
            ax.scatter(cameras[:, 0][prev_cams], cameras[:, 2][prev_cams], s=1, c='black')

    if landmarks is not None:
        ax.set_xlim(-250, 350)
        ax.set_ylim(-100, 430)

    if loops is not None:
        plt.subplots_adjust(left=0.25, bottom=0.08, right=0.95, top=0.9)

    landmarks_txt = "and landmarks" if landmarks is not None else ""
    mahalanobis_dist_and_inliers = f"Dist: {mahalanobis_dist}; Inliers: {inliers_perc}%\n"

    len_loops = None
    if loops is not None:
        loops_details = "\n".join([str(i) + ")  " + str(cur_cam) + ": " + ",".join([str(prev_cam) for prev_cam in prev_cams])
                                   for i, (cur_cam, prev_cams) in enumerate(loops)])

        loops_txt = mahalanobis_dist_and_inliers + loops_details
        # y = -65 + 9 * len(loops)
        plt.text(-360, -67, loops_txt, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        len_loops = len(loops)

    first_legend = plt.legend(handles=first_legend, loc='upper left', prop={'size': 7})
    plt.gca().add_artist(first_legend)

    fig.savefig(f"Results/{title} Left {len(cameras_gt)} cameras {landmarks_txt} 2d trajectory "
                f"m dist {mahalanobis_dist_and_inliers} loops {len_loops} "
                f"{BundleAdjustmentDirectory.BundleAdjustment.PERCENTAGE}.png")
    plt.close(fig)


def trajectory_state_over_the_process(initial_est, bundle_opt, loop_opt, ground_truth):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras (key frames) 2d trajectory over the process")
    ax.scatter(initial_est[:, 0], initial_est[:, 2], s=1, c='pink', label="1. Initial estimation")
    ax.scatter(bundle_opt[:, 0], bundle_opt[:, 2], s=1, c='orangered', label="2. Bundle Adjusment")
    ax.scatter(loop_opt[:, 0], loop_opt[:, 2], s=1, c='darkviolet', label="3. Loop Closure")
    ax.scatter(ground_truth[:, 0], ground_truth[:, 2], s=1, c='cyan', label="4. Ground truth")
    plt.legend(loc='upper left', prop={'size': 7})

    fig.savefig("Results/Left cameras 2d trajectory of the all process.png")
    plt.close(fig)


def plot_2_images_pg_ind(prev_cam_ind, cur_cam_ind, match=True):
    """
    This function plot 2 images with indexes from the pose graph
    """
    left0_matches_coor = left1_matches_coor = supporters_idx = frame0_inliers_percentage = None

    prev_frame_ind_whole_movie = Data.PG.get_key_frames()[prev_cam_ind]
    cur_frame_ind_whole_movie = Data.PG.get_key_frames()[cur_cam_ind]

    # Take the left images - we use the function read, so it won't apply blurring
    prev_img, cur_img = Data.KITTI.read_images(prev_frame_ind_whole_movie)[0], \
                        Data.KITTI.read_images(cur_frame_ind_whole_movie)[0]

    if match:
        _, frame0_features, frame1_features, supporters_idx, frame0_inliers_percentage = \
            utills.apply_full_consensus_match(prev_frame_ind_whole_movie, cur_frame_ind_whole_movie)

        left0_matches_coor = utills.get_features_left_coor(frame0_features)
        left1_matches_coor = utills.get_features_left_coor(frame1_features)

    plot_supporters(prev_frame_ind_whole_movie, cur_frame_ind_whole_movie,
                               prev_cam_ind, cur_cam_ind,
                               prev_img, cur_img,
                               left0_matches_coor, left1_matches_coor, supporters_idx, frame0_inliers_percentage)


def plot_2_images_movie_ind(first_frame_ind, second_frame_ind, match=True):
    """
    This function plot 2 images with indexes from the pose graph
    """
    left0_matches_coor = left1_matches_coor = supporters_idx = frame0_inliers_percentage = None

    # Take the left images - we use the function read, so it won't apply blurring
    prev_img, cur_img = Data.KITTI.read_images(first_frame_ind)[0], \
                        Data.KITTI.read_images(second_frame_ind)[0]

    if match:
        _, frame0_features, frame1_features, supporters_idx, frame0_inliers_percentage = \
            utills.apply_full_consensus_match(first_frame_ind, second_frame_ind)

        left0_matches_coor = utills.get_features_left_coor(frame0_features)
        left1_matches_coor = utills.get_features_left_coor(frame1_features)

    plot_supporters(first_frame_ind, second_frame_ind,
                         first_frame_ind, second_frame_ind,
                         prev_img, cur_img,
                         left0_matches_coor, left1_matches_coor, supporters_idx, frame0_inliers_percentage)
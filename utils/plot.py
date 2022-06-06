import cv2
import matplotlib.pyplot as plt


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
    ax.set_title(f"Left cameras 2d trajectory for {len(left_cameras_pos)} frames.")
    ax.scatter(left_cameras_pos[:, 0], left_cameras_pos[:, 2], s=1, c='red')

    fig.savefig(f"Results/Left cameras 2d trajectory.png")
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
                                                                       title=""):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_title(f"{title} Left cameras and landmarks 2d trajectory of {len(initial_estimate_poses)} bundles")

    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='orange', label="Landmarks")

    if cameras is not None:
        ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red', label="Cameras")

    if cameras_gt is not None:
        ax.scatter(cameras_gt[:, 0], cameras_gt[:, 2], s=1, c='cyan', label="Cameras ground truth")

    if initial_estimate_poses is not None:
        ax.scatter(initial_estimate_poses[:, 0], initial_estimate_poses[:, 2], s=1, c='pink', label="Initial estimate")

    ax.legend(loc="upper left")
    ax.set_xlim(-250, 350)
    ax.set_ylim(-100, 550)

    fig.savefig(f"Results/{title} Left cameras and landmarks 2d trajectory.png")
    plt.close(fig)




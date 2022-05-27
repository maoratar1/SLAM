"""
This file responsible on loading Kitti's data and the Data base
"""
import cv2

from DataDirectory import KittiData

# Load or create Kitti's DataDirectory
print("Trying to load an existing object of Kitti's data")
try:
    KITTI = KittiData.load(KittiData.LOADED_KITTI_DATA)
    print(f"\tKitti's data object exists and was loaded from the path: {KittiData.LOADED_KITTI_DATA}")
except:
    print("\tKitti's data did not created yed, Let's create it now:")
    KITTI = KittiData.KittiData()
    KittiData.save(KittiData.LOADED_KITTI_DATA, KITTI)
    print(f"\tKitti's data object created and it saved at : {KittiData.LOADED_KITTI_DATA}")

print("")

from utils import utills
from utils import plot
from DataBaseDirectory import DataBase
import ex3

# Load or create DataDirectory base
print("Trying to load an existing object of Data base")
try:
    DB = DataBase.load(DataBase.LOADED_DB_PATH)
    print(f"\tData base object exists and was loaded from the path: {DataBase.LOADED_DB_PATH}")
except:
    print("\tData base has not been created yet, Let's create it now:")
    consecutive_frame_features, inliers_percentage, global_trans, relative_trans = utills.find_features_in_consecutive_frames_whole_movie()

    # print("Ex3 trans")
    # ex3_trans = utills.convert_trans_from_rel_to_global(utills.whole_movie())
    DB = DataBase.DataBase(consecutive_frame_features, inliers_percentage, global_trans, relative_trans)

    # DB = DataBase.DataBase(consecutive_frame_features, inliers_percentage, global_trans, relative_trans)
    DataBase.save(DataBase.LOADED_DB_PATH, DB)
    print(f"\tDataDirectory base object created and it saved at : {DataBase.LOADED_DB_PATH}")


# global_trans_traj = utills.left_cameras_trajectory(global_trans)
# T_ground_truth_arr = utills.get_ground_truth_transformations()
# ground_truth_relative_cameras_pos_arr = utills.left_cameras_trajectory(T_ground_truth_arr)
#
# plot.compare_left_cam_2d_trajectory_to_ground_truth(global_trans_traj,
#                                                           ground_truth_relative_cameras_pos_arr)

# T_arr, whole_time = ex3.compute_whole_movie_time()
# ex3_trans = ex3.left_cameras_relative_trans(T_arr)
#
# ex3_global = ex3.left_cameras_trajectory(ex3_trans)
# db_global = DB.initial_estimate_poses()
#
# plot.compare_left_cam_2d_trajectory_to_ground_truth(db_global,
#                                                           ex3_global)
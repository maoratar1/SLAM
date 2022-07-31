"""
This file responsible on loading Kitti's data and the Data base
"""

from DataDirectory import KittiData

LOAD_KITTI = 1
LOAD_DB = 1
LOAD_BA = 0
LOAD_PG = 0

KITTI_NAME = "KITTI"


if LOAD_KITTI:
    # Load or create KITTI's DataDirectory
    print("Trying to load an existing object of KITTI's data")
    try:
        KITTI = KittiData.load(KittiData.LOADED_KITTI_DATA)
        print(f"\tKITTI's data object exists and was loaded from the path: {KittiData.LOADED_KITTI_DATA}")
    except:
        print("\tKITTI's data did not created yet, Let's create it now:")
        KITTI = KittiData.KittiData()
        KittiData.save(KittiData.LOADED_KITTI_DATA, KITTI)
        print(f"\tKITTI's data object created and saved at : {KittiData.LOADED_KITTI_DATA}")

    print("")

from utils import utills
from DataBaseDirectory import DataBase
from BundleAdjustmentDirectory import BundleAdjustment
from PoseGraphDirectory import PoseGraph

if LOAD_DB:
    # Load or create DataDirectory base
    print("Trying to load an existing object of Data base")
    try:
        DB = DataBase.load(DataBase.LOADED_DB_PATH)
        print(f"\tData base object exists and was loaded from the path: {DataBase.LOADED_DB_PATH}")
    except:
        print("\tData base has not been created yet, Let's create it now:")
        consecutive_frame_features, inliers_percentage, global_trans, relative_trans = utills.find_features_in_consecutive_frames_whole_movie()
        DB = DataBase.DataBase(consecutive_frame_features, inliers_percentage, global_trans, relative_trans)
        DataBase.save(DataBase.LOADED_DB_PATH, DB)
        print(f"\tDataDirectory base object created and saved at : {DataBase.LOADED_DB_PATH}")

    print("")

if LOAD_BA:
    print("Trying to load an existing object of Bundle adjustment")
    try:
        BA = BundleAdjustment.load(BundleAdjustment.LOADED_BA_PATH)
        print(f"\tBundle adjustment object exists and was loaded from the path: {BundleAdjustment.LOADED_BA_PATH}")
    except:
        print("\tBundle adjustment object did not created yet, Let's create it now:")
        bundle_adjustment = BundleAdjustment.BundleAdjustment()
        bundle_adjustment.solve(BundleAdjustment.ITERATIVE_PROCESS)
        BA = bundle_adjustment
        BundleAdjustment.save(BundleAdjustment.LOADED_BA_PATH, BA)
        print(f"\tBundle adjustment object created and saved at : {BundleAdjustment.LOADED_BA_PATH}")

    print("")

if LOAD_PG:
    print("Trying to load an existing object of Pose Graph")
    try:
        PG = PoseGraph.load(PoseGraph.LOADED_POSE_GRAPH_PATH)
        print(f"\tPose graph object exists and was loaded from the path: {PoseGraph.LOADED_POSE_GRAPH_PATH}")
    except:
        print("\tPose graph object did not created yed, Let's create it now:")
        PG = PoseGraph.create_pose_graph()
        PG.optimize()
        PG.loop_closure()
        PoseGraph.save(PoseGraph.LOADED_POSE_GRAPH_PATH, PG)
        print(f"\tPose graph object created and saved at : {PoseGraph.LOADED_POSE_GRAPH_PATH}")

print("")

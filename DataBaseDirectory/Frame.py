import numpy as np
from DataDirectory import Data


class Frame:
    """
    This class represents a frame
    """

    def __init__(self, id, ex_cam_mat, calib_cam_mat=Data.KITTI.get_K()):
        self.__id = id
        self.__tracks_ids = []  # Track's ids in this frame
        self.__ex_cam_mat = ex_cam_mat  # [R|t]
        self.__calib_cam_mat = calib_cam_mat
        self.__pose = self.compute_pose(ex_cam_mat)

    def add_track(self, track_id):
        """
        Adds track to frame
        """
        self.__tracks_ids.append(track_id)

    def get_id(self):
        """
        Returns frame's id
        """
        return self.__id

    def get_tracks_ids(self):
        """
        Returns track's ids list
        """
        return np.array(self.__tracks_ids).astype(int)

    def get_tracks_num(self):
        """
        Return tracks number at this frame
        """
        return len(self.__tracks_ids)

    def get_ex_cam_mat(self):
        return self.__ex_cam_mat

    def get_calib_cam_mat(self):
        return self.__calib_cam_mat

    def get_pose(self):
        return self.__pose

    def compute_pose(self, ex_cam_mat):
        """
        Finds and returns the Camera position at the "world" d2_points
        :param ex_cam_mat: [Rotation mat|translation vec]
        """
        # R = extrinsic_camera_mat[:, :3]
        # t = extrinsic_camera_mat[:, 3]
        # result : -R.T @ t
        return -1 * ex_cam_mat[:, :3].T @ ex_cam_mat[:, 3]


import gtsam
import numpy as np
from gtsam import symbol

from DataDirectory import Data
from utils import utills

LAND_MARK_SYM = "q"
CAMERA_SYM = "c"
P3D_MAX_DIST = 80


class BundleWindow:

    def __init__(self, first_key_frame, second_key_frame, all_frame_between=True, little_bundle_tracks=None):  # Todo: Delete the frame in windows and add instead calling to Data.DB.get_frames()
        self.__first_key_frame = first_key_frame
        self.__second_key_frame = second_key_frame
        self.__all_frames_between = all_frame_between

        if all_frame_between:  # Todo: consider change to constructor get list of frames_ind indexes
            self.__bundle_frames = range(first_key_frame, second_key_frame + 1)
            self.__bundle_len = 1 + second_key_frame - first_key_frame
        else:
            self.__bundle_frames = [first_key_frame, second_key_frame]
            self.__bundle_len = 2
            self.__little_bundle_tracks = little_bundle_tracks

        self.optimizer = None
        self.__initial_estimate = gtsam.Values()
        self.__optimized_values = None
        self.__camera_sym = set()  # {CAMERA_SYM + frame index : gtsam symbol} for example {"q1": symbol(q1, place)}
        self.__landmark_sym = set()

        self.graph = gtsam.NonlinearFactorGraph()

    def create_factor_graph(self):
        """
        Creates the factor graph for the bundle window
        """
        # Todo: 1. Consider change this such that it will get the all tracks in the frame in the bundle and run all over them
        #  instead of receive a track from each frame
        #  2. Check the change of getting the relative trans array instead of the relatiive to first
        gtsam_calib_mat = utills.create_gtsam_calib_cam_mat(utills.K)

        frames_in_window = Data.DB.get_frames()[self.__bundle_frames]

        first_frame = frames_in_window[0]

        # Compute first frame extrinsic matrix that takes a point at the camera coordinates and map it to
        # the world coordinates where "world" here means frame 0 of the whole movie coordinates
        first_frame_cam_to_world_ex_mat = utills.convert_ex_cam_to_cam_to_world(
            first_frame.get_ex_cam_mat())  # first cam -> world

        # For each frame - create initial estimation for it's pose
        gtsam_left_cam_pose = None

        for i, frame in enumerate(frames_in_window):

            # Create camera symbol and update values dictionary
            left_pose_sym = symbol(CAMERA_SYM, frame.get_id())
            self.__camera_sym.add(left_pose_sym)

            # Compute transformation of : (world - > cur cam) * (first cam -> world) = first cam -> cur cam
            camera_relate_to_first_frame_trans = utills.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                                frame.get_ex_cam_mat())

            # Convert this transformation to "cur cam -> first cam" as in gtsam
            cur_cam_pose = utills.convert_ex_cam_to_cam_to_world(camera_relate_to_first_frame_trans)
            gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)
            self.__initial_estimate.insert(left_pose_sym, gtsam_left_cam_pose)

            # Initialize constraints for first pose
            if i == 0:
                # sigmas array: first 3 for angles second 3 for location
                # I chose those values by assuming that theirs 1 angles uncertainty at the angles,
                # about 30cm at the x axes, 10cm at the y axes and 1 meter at the z axes which is the moving direction
                sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])

                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)  # todo: check choice of diagonal
                # Initial pose
                factor = gtsam.PriorFactorPose3(left_pose_sym, gtsam_left_cam_pose, pose_uncertainty)
                self.graph.add(factor)

        # For each track create measurements factors
        if self.__all_frames_between:
            tracks_in_frame = Data.DB.get_tracks_at_frame(first_frame.get_id())
            self.compute_tracks_and_add_factors(tracks_in_frame, gtsam_left_cam_pose, gtsam_calib_mat)
        else:
            self.compute_tracks_and_add_factors(self.__little_bundle_tracks, gtsam_left_cam_pose, gtsam_calib_mat)

    def compute_tracks_and_add_factors(self, tracks_in_frame, gtsam_left_cam_pose, gtsam_calib_mat):
        for track in tracks_in_frame:
            # Check that this track has bot been computed yet and that it's length is satisfied
            if track.get_last_frame_ind() < self.__second_key_frame:
                continue

            # Create a gtsam object for the last frame for making the projection at the function "add_factors"
            gtsam_last_cam = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)
            self.add_factors(track, self.__first_key_frame, self.__second_key_frame, gtsam_last_cam,
                             gtsam_calib_mat)  # Todo: as before

    def add_factors(self, track, first_frame_ind, last_frame_ind, gtsam_frame_to_triangulate_from, gtsam_calib_mat,
                    frame_idx_triangulate=-1):
        """
        Adds factors for a track
        """

        # Track's locations in frames_in_window
        if self.__all_frames_between:
            left_locations = track.get_left_locations_in_specific_frames(range(first_frame_ind, last_frame_ind + 1))
            right_locations = track.get_right_locations_in_specific_frames(range(first_frame_ind, last_frame_ind + 1))
        else:
            left_locations = track.get_left_locations_in_specific_frames([first_frame_ind, last_frame_ind])
            right_locations = track.get_right_locations_in_specific_frames([first_frame_ind, last_frame_ind])

        # Track's location at the Last frame for triangulations
        last_left_img_loc = left_locations[frame_idx_triangulate]
        last_right_img_loc = right_locations[frame_idx_triangulate]

        # Create Measures of last frame for the triangulation
        measure_xl, measure_xr, measure_y = last_left_img_loc[0], last_right_img_loc[0], last_left_img_loc[1]
        gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        if measure_xl < measure_xr or measure_xl - measure_xr < 2:  # Todo: check this option
            return

        # Triangulation from last frame
        gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

        # if gtsam_p3d[2] <= 0 or gtsam_p3d[2] >= 300:  # Todo: check limits  55%: 5:46
        #     return

        # Add landmark symbol to "values" dictionary
        p3d_sym = symbol(LAND_MARK_SYM, track.get_id())
        self.__landmark_sym.add(p3d_sym)
        self.__initial_estimate.insert(p3d_sym, gtsam_p3d)

        for i, frame_ind in enumerate(self.__bundle_frames):
            # Measurement values
            measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
            gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

            # Factor creation
            projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
            factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                                 symbol(CAMERA_SYM, frame_ind), p3d_sym, gtsam_calib_mat)

            # Add factor to the graph
            self.graph.add(factor)

    def optimize(self):
        """
        Apply optimization with Levenberg marquardt algorithm
        """
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.__initial_estimate)
        self.__optimized_values = self.optimizer.optimize()

    def update_optimization(self, values):
        """
        Updates the inital estimation to "values"
        """
        self.__initial_estimate = values

    def graph_error(self, optimized=True):
        """
        Return log og the graph error
        :param optimized:
        :return:
        """
        if not optimized:
            error = self.graph.error(self.__initial_estimate)
        else:
            error = self.graph.error(self.__optimized_values)

        return np.log(error)  # Todo: here we returns the reprojection error probably

    def get_initial_estimate_values(self):
        """
        Returns initial estimation values
        """
        return self.__initial_estimate

    def get_optimized_values(self):
        """
        Return optimized values
        """
        return self.__optimized_values

    def get_cameras_symbols_lst(self):
        """
        Return cameras symbols list
        """
        return self.__camera_sym

    def get_landmarks_symbols_lst(self):
        """
        Returns landmarks symbols list
        """
        return self.__landmark_sym

    def get_optimized_cameras_poses(self):
        """
        Returns optimized cameras poses
        """
        cameras_poses = []
        for camera_sym in self.__camera_sym:
            cam_pose = self.__optimized_values.atPose3(camera_sym)
            cameras_poses.append(cam_pose)

        return cameras_poses

    def get_optimized_cameras_p3d(self):
        """
        Returns optimized cameras 3d points
        """
        cam_pose = self.__optimized_values.atPose3(symbol(CAMERA_SYM, self.__second_key_frame))
        return cam_pose

    def get_optimized_landmarks_p3d(self):
        """
        Returns optimized cameras 3d points
        """
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmark = self.__optimized_values.atPoint3(landmark_sym)
            landmarks.append(landmark)

        return landmarks

    def get_initial_estimate_cameras_p3d(self):
        """
        Returns initial estimation for the bundle last frame's position
        """
        cam_pose = self.__initial_estimate.atPose3(symbol(CAMERA_SYM, self.__second_key_frame)).inverse()
        return cam_pose

    def get_initial_estimate_landmarks_p3d(self):
        """
        Returns initial estimation for the bundle landmarks
        :return:
        """
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmark = self.__initial_estimate.atPoint3(landmark_sym)
            landmarks.append(landmark)

        return landmarks

    def get_key_frames(self):
        """
        Returns first and last key frames_ind
        """
        return self.__first_key_frame, self.__second_key_frame

    def marginals(self):
        return gtsam.Marginals(self.graph, self.__optimized_values)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["optimizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add optimizer back since it doesn't exist in the pickle
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.__initial_estimate)

    # === Another options for creating the factor graph
    def create_factor_graph_opt2_all_tracks(self):
        # Compute all tracks
        gtsam_calib_mat = utills.create_gtsam_calib_cam_mat(utills.K)

        frames_in_window = Data.DB.get_frames()[self.__first_key_frame: self.__second_key_frame + 1]
        first_frame = frames_in_window[0]

        # Compute first frame extrinsic matrix that takes a point at the camera coordinates and map it to
        # the world coordinates where "world" here means frame 0 of the whole movie coordinates
        first_frame_cam_to_world_ex_mat = utills.convert_ex_cam_to_cam_to_world(
            first_frame.get_ex_cam_mat())  # first cam -> world

        # For each frame - create initial estimation for it's pose
        gtsam_left_cam_pose = None

        tracks_ids_in_frame = set()

        for i, frame in enumerate(frames_in_window):
            tracks_ids_in_frame.update(frame.get_tracks_ids())

            # Create camera symbol and update values dictionary
            left_pose_sym = symbol(CAMERA_SYM, frame.get_id())
            self.__camera_sym.add(left_pose_sym)

            # Compute transformation of : (world - > cur cam) * (first cam -> world) = first cam -> cur cam
            camera_relate_to_first_frame_trans = utills.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                                frame.get_ex_cam_mat())

            # Convert this transformation to: cur cam -> first cam
            cur_cam_pose = utills.convert_ex_cam_to_cam_to_world(camera_relate_to_first_frame_trans)
            gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)
            self.__initial_estimate.insert(left_pose_sym, gtsam_left_cam_pose)

            # Initialize constraints for first pose
            if i == 0:
                # sigmas array: first 3 for angles second 3 for location
                # sigmas = np.array([10 ** -3, 10 ** -3, 10 ** -3, 10 ** -2, 10 ** -2, 10 ** -2])
                # sigmas = np.array([0] * 6)
                sigmas = np.array([(3 * np.pi / 180) ** 2] * 3 + [1.0, 0.01, 1.0])
                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)  # todo: check choice of diagonal
                # Initial pose
                factor = gtsam.PriorFactorPose3(left_pose_sym, gtsam_left_cam_pose, pose_uncertainty)
                self.graph.add(factor)

        tracks_ids_in_frame_lst = list(tracks_ids_in_frame)
        tracks_in_frame = Data.DB.get_tracks()[tracks_ids_in_frame_lst]

        for track in tracks_in_frame:
            # Check that this track has bot been computed yet and that it's length is satisfied
            # if track.get_id() in self.__computed_tracks or track.get_last_frame_ind() < self.__second_key_frame:
            #     continue

            # Create a gtsam object for the last frame for making the projection at the function "add_factors"
            gtsam_last_cam = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)
            first_frame = max(self.__first_key_frame, track.get_first_frame_ind())
            last_frame = min(self.__second_key_frame, track.get_last_frame_ind())
            self.add_factors(track, first_frame, last_frame, gtsam_last_cam, gtsam_calib_mat)  # Todo: as before

            # self.__computed_tracks.add(track.get_id())

    def create_factor_graph_opt3_rel_trans(self):
        # Todo: 1. Check this with rel trans
        # instead of receive a track from each frame
        # 2. Check the change of getting the relative trans array instead of the relatiive to first
        gtsam_calib_mat = utills.create_gtsam_calib_cam_mat(utills.K)

        frames_in_window = Data.DB.get_frames()[self.__first_key_frame: self.__second_key_frame + 1]
        first_frame = frames_in_window[0]

        # Compute first frame extrinsic matrix that takes a point at the camera coordinates and map it to
        # the world coordinates where "world" here means frame 0 of the whole movie coordinates
        rel_trans = Data.DB.get_relative_cam_trans()[self.__first_key_frame: self.__second_key_frame + 1]
        cams_rel_to_bundle_first_cam = utills.convert_trans_from_rel_to_global(rel_trans)

        # For each frame - create initial estimation for it's pose
        cur_cam_pose = None
        for i, frame in enumerate(frames_in_window):

            # Create camera symbol and update values dictionary
            left_pose_sym = symbol(CAMERA_SYM, frame.get_id())
            self.__camera_sym.add(left_pose_sym)

            # Initialize constraints for first pose
            if i == 0:
                # sigmas array: first 3 for angles second 3 for location
                sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1.0, 0.3, 1.0])
                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)  # todo: check choice of diagonal
                # Initial pose
                factor = gtsam.PriorFactorPose3(left_pose_sym, gtsam.Pose3(), pose_uncertainty)
                self.graph.add(factor)

            # Convert this transformation to: cur cam -> first cam
            cur_cam_pose = utills.convert_ex_cam_to_cam_to_world(cams_rel_to_bundle_first_cam[i])
            self.__initial_estimate.insert(left_pose_sym, gtsam.Pose3(cur_cam_pose))

        gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

        # For each track create measurements factors
        tracks_in_frame = Data.DB.get_tracks_at_frame(first_frame.get_id())

        for track in tracks_in_frame:
            # Check that this track has bot been computed yet and that it's length is satisfied
            if track.get_last_frame_ind() < self.__second_key_frame:
                continue

            # Create a gtsam object for the last frame for making the projection at the function "add_factors"
            gtsam_last_cam = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)
            self.add_factors(track, self.__first_key_frame, self.__second_key_frame, gtsam_last_cam,
                             gtsam_calib_mat)  # Todo: as before

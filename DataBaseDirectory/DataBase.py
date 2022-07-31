import random
import numpy as np

from DataBaseDirectory import Trck as T
from DataBaseDirectory import Frame as Fr
import pandas as pd
import pickle
DB_NAME = "db_akaze_bf_3"
LOADED_DB_PATH = f'DataBaseDirectory/{DB_NAME}.pickle'
TRACK_NAME = "TRACK"
DB_TRACK_PATH = r'Results/Track frames_in_window.csv'
DB_FRAME_PATH = r'Results/Frame tracks.csv'


def save(path, db):
    """
    Saves the Database "db" in path "path"
    :param db: DataDirectory base
    """
    filehandler = open(path, 'wb')
    pickle.dump(db, filehandler)
    filehandler.close()


def load(saved_db_path):
    """
    Loads that Database from "saved_db_path"
    :param saved_db_path:
    :return: Database
    """
    filehandler = open(saved_db_path, 'rb')
    db = pickle.load(filehandler)
    filehandler.close()
    return db


class DataBase:

    def __init__(self, prev_and_next_frame_features, inliers_percentage, global_cam_trans, relative_cam_trans):
        """
        Creates database
        :param prev_and_next_frame_features: Array of [frame_i_features, frame_i+1_features] where frame_i_arr contains Feature object
        :param inliers_percentage: Array which it's i'th index contains i'th frame inliers percentage
        """
        self.__tracks = []
        self.__frames = []
        self.__last_kpts = dict()  # {kpt: track object} saves the last kpts that have track on the last frame computed
        self.__inliers_percentage_per_frame = inliers_percentage
        self.__prev_and_next_frame_features = prev_and_next_frame_features
        self.__relative_cam_trans = np.array(relative_cam_trans)
        self.create_db(prev_and_next_frame_features, global_cam_trans)

        # self.__frames_at_track = self.create_table_frames_at_track()  # table where each row represents a track with it's frames_in_window
        # self.__tracks_at_frame = self.create_table_tracks_at_frame()  # table where each row represents a frame with it's tracks

    def create_db(self, prev_and_next_frame_features, global_cam_trans):
        """
        This function creates the Database i.e it iterates over the array "prev_and_next_frame_features" array
        and computes tracks all over the consequence frames_in_window
        :param prev_and_next_frame_features: Array which each row contains 2 arrays [frame0_features, frame1_features]
        frame0_features contains Feature objects that match to frame1_features, and they are sharing
        indexes. Meaning that ith feature at frame0_features are match to ith frame1_features
        """

        # Create frame 0 because in the "set_tracks_and_frames" function we always look on 2 pairs of frames_in_window
        # "frame_i" and "frame_(i+1)" and in "set_tracks_and_frames" we create frame i+1
        frame = Fr.Frame(0, global_cam_trans[0])
        self.add_frame(frame)

        # Checks the connections between each pairs of frames_in_window
        for i, arr in enumerate(prev_and_next_frame_features):
            # Notice that those arrays are sharing the same indexes. Meaning that in the ith index
            # frame_i_features_obj[i] contains a Feature obj in the left image of frame_i
            # which matches to the Feature object frame_i_following_features_obj[i]
            frame_i_features_obj = arr[0]
            frame_i_following_features_obj = arr[1]
            self.set_tracks_and_frames(frame_i_features_obj, frame_i_following_features_obj, i, global_cam_trans)

        self.__frames = np.array(self.__frames)
        self.__tracks = np.array(self.__tracks)

    def set_tracks_and_frames(self, first_frame_features_obj, second_frame_features_obj, first_frame_id, global_trans):
        """
        This function receives pair of Featurs objects list of frames_in_window, frame i and frame i+1
        that match in their indexes i.e the i'th index in the first list is a feature that match
        to the ith feature in the second array
        :param first_frame_features_obj: List of features object of frame i
        :param second_frame_features_obj:List of features object of frame i+1
        :param first_frame_id: Equals to i.
        :return:
        """
        second_frame_id = first_frame_id + 1

        # Creates frame i+1
        frame = Fr.Frame(second_frame_id, global_trans[second_frame_id])
        self.add_frame(frame)

        self.__last_kpts = self.find_tracks_between_consecutive_frame(first_frame_features_obj,
                                                                      second_frame_features_obj,
                                                                      first_frame_id, second_frame_id)

    def find_tracks_between_consecutive_frame(self, first_frame_features_obj, second_frame_features_obj,
                                              first_frame_id, second_frame_id):

        new_dic = {}  # Dictionary of the key d2_points the has tracks at the new frame "second frame"
        # We create a new one such that we can remove all the keypoints that has a track at
        # the first frame but does not continue to the next frame "second frame"

        for i in range(len(first_frame_features_obj)):
            first_frame_feature = first_frame_features_obj[i]
            second_frame_feature = second_frame_features_obj[i]

            first_frame_feature_kpt = first_frame_feature.get_left_kpt()
            second_frame_feature_kpt = second_frame_feature.get_left_kpt()

            # Check if the key point in frame i continues an existing track
            # if so: add the kpt in the frame i + 1 to the  existing track
            # else: creates a new track by adding the d2_points of frame i and frame i + 1
            if first_frame_feature_kpt in self.__last_kpts:

                # Updates the dictionary of last key point to be to the key point of the second frame
                track_idx = self.__last_kpts[first_frame_feature_kpt]
                new_dic[second_frame_feature_kpt] = track_idx

                # Updates the track with keypoint in frame i + 1
                track = self.__tracks[track_idx]
                track.add_feature(second_frame_id, second_frame_feature)

                # Adds the track index to the frame i + 1
                frame = self.__frames[second_frame_id]
                frame.add_track(track_idx)

            else:  # Creates a new track with frame i and frame i + 1
                # Updates the dictionary of last key point to be to the key point of the second frame
                track_idx = len(self.__tracks)
                track = T.Track(track_idx)
                new_dic[second_frame_feature_kpt] = track_idx

                self.add_track(track)

                # Creates the track and add its feature on frame i
                track.add_feature(first_frame_id, first_frame_feature)

                frame = self.__frames[first_frame_id]
                frame.add_track(track_idx)

                # Creates the track and add its feature on frame i + 1
                track.add_feature(second_frame_id, second_frame_feature)

                frame = self.__frames[second_frame_id]
                frame.add_track(track_idx)

        return new_dic

    def add_frame(self, frame):
        """
        Adds frame to frame list
        """
        self.__frames.append(frame)

    def add_track(self, track):
        """
        Adds track to track list
        """
        self.__tracks.append(track)

    def create_table_frames_at_track(self): # Todo: add it later for efficiency
        """
        Creates a table where each row represents a track and the columns represents the frame which the track
        in them
        """
        tracks = {}
        for track in self.__tracks:
            tracks[track.get_id()] = track.get_frames_ids()

        df = pd.DataFrame(tracks)
        index = df.index
        index.name = "Tracks\Frames"

        return df

    def create_table_tracks_at_frame(self):  # Todo: add it later for efficiency
        """
        Creates a table where each col represents a frame and the it's values represents the track which the frame
        in them
        """
        frames = {}
        for frame in self.__frames:
            frames[frame.get_id()] = frame.get_tracks_ids()

        df = pd.DataFrame(frames)

        index = df.index
        index.name = "Frames\Tracks"

        return df

    def get_track_features_in_frame(self, frame_id, track_id):
        """
        Returns the track, with track_id, locations in frame with frame frame_idx
        """
        track = self.__tracks[track_id]
        return track.get_feature_at_frame(frame_id)

    def get_num_tracks(self):
        """
        Returns number of tracks
        """
        return len(self.__tracks)

    def get_num_frames(self):
        """
        Returns number of frames_in_window
        """
        return len(self.__frames)

    # def get_track_frame_table(self):
    #     """
    #     Return the track's frames_in_window table
    #     """
    #     return self.__frames_at_track
    #
    # def get_frame_track_table(self):
    #     """
    #     Returns the frame's tracks table
    #     """
    #     return self.__tracks_at_frame

    def get_tracks(self):
        """
        Returns the tracks list
        """
        return self.__tracks

    # def get_tracks_at_frame_with_pd(self, frames_ids):  #Todo: use it later when the pandas will be ready
    #     """
    #
    #     :param frames_ids: List of frame ids
    #     :return:
    #     """
    #     return self.__tracks_at_frame[frames_ids]

    def get_tracks_at_frame(self, frames_id):
        """

        :param frames_ids: List of frame ids
        :return:
        """
        frame = self.__frames[frames_id]
        return self.__tracks[frame.get_tracks_ids()]

    def get_frames_at_track(self, track_id):
        """

        :param tracks_ids: List of tracks ids
        :return:
        """
        track = self.__tracks[track_id]
        return self.__frames[track.get_frames_ids()]

    # def get_frames_at_track_with_pd(self, tracks_ids): #Todo: useit later when the pandas will be ready
    #     """
    #
    #     :param tracks_ids: List of tracks ids
    #     :return:
    #     """
    #     return self.__frames_at_track[tracks_ids]

    def get_frames(self):
        """
        Returns the frames_in_window list
        """
        return self.__frames

    def get_inliers_percentage_per_frame(self):
        """
        Return inliers percentage per frame
        """
        return self.__inliers_percentage_per_frame

    def create_tables_csv_format(self, header):
        """
        Create Frame-tracks and Track-frames_in_window table in csv format
        """
        if header is TRACK_NAME:
            return self.__frames_at_track.to_csv(DB_TRACK_PATH)
        else:
            return self.__tracks_at_frame.to_csv(DB_FRAME_PATH)

    def get_prev_and_next_frame_features(self):
        return self.__prev_and_next_frame_features

    def get_rand_track(self, track_len):
        """
        Randomize a track with length of track_len
        """
        track = None
        found = False
        while not found:
            idx = random.randint(0, len(self.__tracks) - 1)
            if self.__tracks[idx].get_track_len() == track_len:
                track = self.__tracks[idx]
                found = True

        return track

    def get_relative_cam_trans(self):
        return self.__relative_cam_trans

    def initial_estimate_poses(self):
        loc = []
        for frame in self.__frames:
            loc.append(frame.get_pose())

        return np.array(loc)

    def get_tracks_with_specific_len(self, track_len):
        return self.__tracks[list(map(lambda track: track.get_track_len() == track_len, self.__tracks))]

    def tracks_median_len(self, percentage):
        tracks_lens = [0] * len(self.__tracks)

        for i, track in enumerate(self.__tracks):
            tracks_lens[i] = track.get_track_len()

        tracks_lens.sort()
        median_ind = int(percentage * len(tracks_lens))
        return tracks_lens[median_ind]


# === Another option for setting tracks - written for finding tracks in little bundle at ex7
#     def set_tracks_and_frames(self, first_frame_features_obj, second_frame_features_obj, first_frame_id, global_trans):
#         """
#         This function receives pair of Featurs objects list of frames_in_window, frame i and frame i+1
#         that match in their indexes i.e the i'th index in the first list is a feature that match
#         to the ith feature in the second array
#         :param first_frame_features_obj: List of features object of frame i
#         :param second_frame_features_obj:List of features object of frame i+1
#         :param first_frame_id: Equals to i.
#         :return:
#         """
#         second_frame_id = first_frame_id + 1
#
#         # Creates frame i+1
#         frame = Fr.Frame(second_frame_id, global_trans[second_frame_id])
#         self.add_frame(frame)
#
#         self.__last_kpts = self.find_tracks_between_consecutive_frame(first_frame_features_obj,
#                                                                       second_frame_features_obj,
#                                                                       first_frame_id, second_frame_id)
#
#     def find_tracks_between_consecutive_frame(self, first_frame_features_obj, second_frame_features_obj,
#                                               first_frame_id, second_frame_id):
#
#         new_dic = {}  # Dictionary of the key d2_points the has tracks at the new frame "second frame"
#         # We create a new one such that we can remove all the keypoints that has a track at
#         # the first frame but does not continue to the next frame "second frame"
#
#         for i in range(len(first_frame_features_obj)):
#             first_frame_feature = first_frame_features_obj[i]
#             second_frame_feature = second_frame_features_obj[i]
#
#             first_frame_feature_kpt = first_frame_feature.get_left_kpt()
#             second_frame_feature_kpt = second_frame_feature.get_left_kpt()
#
#             # Check if the key point in frame i continues an existing track
#             # if so: add the kpt in the frame i + 1 to the  existing track
#             # else: creates a new track by adding the d2_points of frame i and frame i + 1
#             if first_frame_feature_kpt in self.__last_kpts:
#
#                 # Updates the dictionary of last key point to be to the key point of the second frame
#                 track_idx = self.__last_kpts[first_frame_feature_kpt]
#                 new_dic[second_frame_feature_kpt] = track_idx
#
#                 # Updates the track with keypoint in frame i + 1
#                 track = self.__tracks[track_idx]
#                 track.add_feature(second_frame_id, second_frame_feature)
#
#                 # Adds the track index to the frame i + 1
#                 frame = self.__frames[second_frame_id]
#                 frame.add_track(track_idx)
#
#             else:  # Creates a new track with frame i and frame i + 1
#                 # Updates the dictionary of last key point to be to the key point of the second frame
#                 track_idx = len(self.__tracks)
#                 track = T.Track(track_idx)
#                 new_dic[second_frame_feature_kpt] = track_idx
#
#                 self.add_track(track)
#
#                 # Creates the track and add its feature on frame i
#                 track.add_feature(first_frame_id, first_frame_feature)
#
#                 frame = self.__frames[first_frame_id]
#                 frame.add_track(track_idx)
#
#                 # Creates the track and add its feature on frame i + 1
#                 track.add_feature(second_frame_id, second_frame_feature)
#
#                 frame = self.__frames[second_frame_id]
#                 frame.add_track(track_idx)
#
#         return new_dic
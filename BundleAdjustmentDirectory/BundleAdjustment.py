import concurrent.futures
import pickle

import numpy as np
import tqdm
from DataDirectory import Data
from utils import utills
from BundleAdjustmentDirectory.BundleWindow import BundleWindow
import gtsam

LOADED_BA_PATH = r'BundleAdjustmentDirectory/bundle_adjustment_3.pickle'
ITERATIVE_PROCESS = "ITERATIVE"
MULTI_PROCESSED = "MULTI PROCESSED"
PERCENTAGE = 0.82


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


def choose_key_frames_by_time(frames, time=5):
    """
    Chooses keyframes by the time passed frames_ind
    :param frames:
    :return:
    """
    key_frames = []
    print("Choosing key frames_ind")
    for i in tqdm.tqdm(range(len(frames))):
        if i % time == 0:
            key_frames.append(i)

    return key_frames


def choose_key_frames_by_median_track_len(frames):
    """
    Choose keyframes by the median track len's from the last frame
    """
    key_frames = [0]
    n = len(frames)
    print("Choosing key frames_ind")
    pbar = tqdm.tqdm(total=len(frames))
    while key_frames[-1] < n - 1:
        last_key_frame = key_frames[-1]
        frame = frames[last_key_frame]
        tracks = Data.DB.get_tracks_at_frame(frame.get_id())

        tracks_lens = []
        for track in tracks:
            tracks_lens.append(track.get_track_len())

        tracks_lens.sort()

        new_key_frame = tracks_lens[len(tracks_lens) // 2] + last_key_frame
        key_frames.append(min(new_key_frame, n - 1))
        pbar.update(new_key_frame - last_key_frame + 1)

    pbar.close()

    return key_frames


def choose_key_frames_by_mean(frames):
    """
    Choose keyframes by the median track len's from the last frame
    """
    key_frames = [0]
    n = len(frames)
    print("Choosing key frames_ind")
    pbar = tqdm.tqdm(total=len(frames))
    while key_frames[-1] < n - 1:
        last_key_frame = key_frames[-1]
        frame = frames[last_key_frame]
        tracks = Data.DB.get_tracks_at_frame(frame.get_id())

        tracks_lens = 0
        for track in tracks:
            tracks_lens += track.get_track_len()

        mean_track_len = tracks_lens // len(tracks)

        new_key_frame = mean_track_len + last_key_frame
        key_frames.append(min(new_key_frame, n - 1))
        pbar.update(new_key_frame - last_key_frame + 1)

    pbar.close()

    return key_frames


def choose_key_frames_by_median_track_len2(frames, percentage):
    """
    Choose keyframes by the median track len's from the last frame
    """
    key_frames = [0]
    n = len(frames)
    print("Choosing key frames")
    pbar = tqdm.tqdm(total=len(frames))
    while key_frames[-1] < n - 1:
        last_key_frame = key_frames[-1]
        frame = frames[last_key_frame]
        tracks = Data.DB.get_tracks_at_frame(frame.get_id())

        tracks_lens = []
        for track in tracks:
            tracks_lens.append(track.get_last_frame_ind())

        tracks_lens.sort()

        index = int(len(tracks_lens) * percentage)
        new_key_frame = tracks_lens[index]
        key_frames.append(min(new_key_frame, n - 1))
        pbar.update(new_key_frame - last_key_frame + 1)

    pbar.close()

    return key_frames


def create_bundle_windows(key_frames):
    """
    Create BundleWindow object by the keyframe
    :param key_frames: list of frames_ind where the i'th and (i-1)'th elements represents the bundles frames_ind
    :return: bundle's list
    """
    print("Creating bundle windows")

    bundle_window_lst = []

    for i in tqdm.tqdm(range(1, len(key_frames))):
        first_key = key_frames[i - 1]
        second_key = key_frames[i]
        bundle_window = BundleWindow(first_key, second_key)
        bundle_window_lst.append(bundle_window)

    return bundle_window_lst


def create_and_solve_bundle(first_key, second_key):
    """
    Create Bundle and solve each one - for multiprocessing
    """
    bundle_window = BundleWindow(first_key, second_key)
    bundle_window.create_factor_graph()
    bundle_window.optimize()
    return bundle_window.get_optimized_cameras_p3d(), bundle_window.get_optimized_landmarks_p3d()


def convert_bundle_rel_landmark_to_global(global_first_cam_in_bundle, bundle_landmarks):
    """
    Convert bundle's landmarks from bundle relative to global
    :param global_first_cam_in_bundle: global first camera in bundle
    :param bundle_landmarks: as is
    :return:
    """
    bundle_global_landmarks = []
    for landmark in bundle_landmarks:
        bundle_global_landmarks.append(
            global_first_cam_in_bundle.transformFrom(landmark))

    return bundle_global_landmarks


def convert_rel_landmarks_to_global(cameras, landmarks):
    """
    Convert relative to each bundle landmarks to the global coordinate system
    :param cameras: list of cameras
    :param landmarks: list of landmarks lists
    :return: one list of the whole global landmarks
    """
    global_landmarks = []
    for bundle_camera, bundle_landmarks in zip(cameras, landmarks):
        bundle_global_landmarks = convert_bundle_rel_landmark_to_global(bundle_camera, bundle_landmarks)
        global_landmarks += bundle_global_landmarks

    return np.array(global_landmarks)


def convert_rel_cams_and_landmarks_to_global(gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle):
    # Convert cameras and landmarks from relative to their bundle to be relative to world
    gtsam_cameras_rel_to_world = utills.convert_gtsam_trans_from_rel_to_global(gtsam_cameras_rel_to_bundle)
    landmarks_rel_to_world = convert_rel_landmarks_to_global(gtsam_cameras_rel_to_world,
                                                             all_landmarks_rel_to_bundle)

    return gtsam_cameras_rel_to_world, landmarks_rel_to_world

class BundleAdjustment:

    def __init__(self):
        # Create a bundle windows
        # self.__key_frames = get_loaded_key_frames()
        self.__key_frames = choose_key_frames_by_median_track_len2(Data.DB.get_frames(), PERCENTAGE)
        self.__bundles_lst = create_bundle_windows(self.__key_frames)
        self.__num_bundles = len(self.__bundles_lst)
        self.__gtsam_cameras_rel_to_bundle = None
        self.__all_landmarks_rel_to_bundle = None

    def solve(self, method=ITERATIVE_PROCESS, workers_num=5):
        """
        Solves the bundle adjustment for each bundle window
        :param method: Iterative or multiprocessing
        :param workers_num: num cpu's
        :return:
            gtsam_cameras_rel_to_world - list of gtsam objects that represents the cameras
            and their poses relative to the world (and not for the bundle)
            landmarks_rel_to_world - list of 3d points that represents landmarks relative to the worls
        """
        gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle = None, None

        # Choose running method
        print("For all bundles - Create factor graph and optimize:")
        if method is MULTI_PROCESSED:
            gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle = self.bundle_adjustment_multi_processed(
                workers_num)

        elif method is ITERATIVE_PROCESS:
            gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle = self.bundle_adjustment_iterative()

        self.__gtsam_cameras_rel_to_bundle = gtsam_cameras_rel_to_bundle
        self.__all_landmarks_rel_to_bundle = all_landmarks_rel_to_bundle

    def bundle_adjustment_multi_processed(self, workers_num):
        """
        Solve bundle with multiprocessing
        :param workers_num: Cpu's number
        :return: numpy array of cameras relative to their bundle and list of lists relative to their bundle
        """
        cameras = [gtsam.Pose3()]
        landmarks = []

        with concurrent.futures.ProcessPoolExecutor(workers_num) as executor:
            results = list(tqdm.tqdm(executor.map(self.solve_bundle_window, range(self.__num_bundles)),
                                     total=self.__num_bundles))  # [[Cameras, Landmarks], ...]

        for res in results:  # res = [Camera = 3d point numpy array , landmarks = [...] ]
            cameras.append(res[0])
            landmarks.append(res[1])

        return np.array(cameras), landmarks

    def solve_bundle_window(self, bundle_num):
        """
        Solve one bundle window
        :param bundle_num: bundle number at the bundles list
        :return: optimized camera pose and landmark
        """
        self.__bundles_lst[bundle_num].create_factor_graph()
        self.__bundles_lst[bundle_num].optimize()
        return self.__bundles_lst[bundle_num].get_optimized_cameras_p3d(), \
               self.__bundles_lst[bundle_num].get_optimized_landmarks_p3d()

    def bundle_adjustment_iterative(self):
        """
        Solve bundle adjustment with iterative method
        :return: cameras and landmarks
        """
        cameras = [gtsam.Pose3()]
        landmarks = []

        for i in tqdm.tqdm(range(len(self.__bundles_lst))):
            self.__bundles_lst[i].create_factor_graph()
            self.__bundles_lst[i].optimize()
            cameras.append(self.__bundles_lst[i].get_optimized_cameras_p3d())
            landmarks.append(self.__bundles_lst[i].get_optimized_landmarks_p3d())

        return np.array(cameras), landmarks

    def get_key_frames(self):
        """
        Returns keyframes
        """
        return self.__key_frames

    def get_bundles_lst(self):
        """
        Returns bundles lsd
        """
        return self.__bundles_lst

    def get_gtsam_cameras_rel_to_bundle(self):
        return self.__gtsam_cameras_rel_to_bundle

    def get_all_landmarks_rel_to_bundle(self):
        return self.__all_landmarks_rel_to_bundle

    def get_gtsam_cameras_global(self):
        return utills.convert_gtsam_trans_from_rel_to_global(self.__gtsam_cameras_rel_to_bundle)


import concurrent.futures
import pickle

import numpy as np
import tqdm
from DataDirectory import Data
from utils import utills
from BundleAdjustmentDirectory.BundleWindow import BundleWindow
import gtsam

LOADED_BA_PATH = r'BundleAdjustmentDirectory/bundle_adjustment.pickle'
ITERATIVE_METHOD = "ITERATIVE"
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
    Chooses keyframes by the time passed frames
    :param frames:
    :return:
    """
    key_frames = []
    print("Choosing key frames")
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
    print("Choosing key frames")
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
    print("Choosing key frames")
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
    :param key_frames: list of frames where the i'th and (i-1)'th elements represents the bundles frames
    :return: bundle's list
    """
    print("Creating bundle windows")

    bundle_window_lst = []
    frames = Data.DB.get_frames()

    for i in tqdm.tqdm(range(1, len(key_frames))):
        first_key = key_frames[i - 1]
        second_key = key_frames[i]
        bundle_window = BundleWindow(first_key, second_key, frames[first_key: second_key + 1])
        bundle_window_lst.append(bundle_window)

    return bundle_window_lst


def create_and_solve_bundle(first_key, second_key, frames):
    """
    Create Bundle and solve each one - for multiprocessing
    """
    bundle_window = BundleWindow(first_key, second_key, frames[first_key: second_key + 1])
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
    gtsam_cameras_rel_to_world = utills.gtsam_left_cameras_relative_trans(gtsam_cameras_rel_to_bundle)
    landmarks_rel_to_world = convert_rel_landmarks_to_global(gtsam_cameras_rel_to_world,
                                                             all_landmarks_rel_to_bundle)

    return gtsam_cameras_rel_to_world, landmarks_rel_to_world


def get_loaded_key_frames():
    return [0, 12, 23, 36, 47, 58, 66, 73, 91, 103, 111, 118, 130, 144, 150, 152, 155, 160, 167, 176, 184, 198, 208, 215, 221, 228, 235, 244, 251, 257, 264, 270, 277, 283, 287, 292, 301, 311, 322, 331, 341, 359, 370, 375, 381, 387, 401, 411, 418, 428, 438, 446, 461, 472, 518, 535, 542, 548, 556, 563, 569, 575, 581, 587, 593, 598, 605, 614, 621, 628, 635, 644, 651, 660, 675, 690, 697, 701, 705, 711, 726, 733, 743, 751, 762, 781, 794, 802, 807, 812, 824, 830, 833, 838, 844, 850, 857, 864, 870, 877, 885, 893, 900, 905, 909, 917, 923, 928, 933, 940, 950, 955, 962, 966, 971, 978, 983, 988, 996, 1001, 1010, 1016, 1021, 1030, 1038, 1049, 1059, 1073, 1078, 1082, 1089, 1101, 1114, 1123, 1134, 1139, 1147, 1162, 1172, 1177, 1189, 1205, 1215, 1220, 1227, 1242, 1250, 1259, 1266, 1273, 1285, 1295, 1302, 1317, 1332, 1348, 1354, 1359, 1364, 1371, 1378, 1383, 1390, 1395, 1400, 1404, 1407, 1411, 1416, 1420, 1424, 1429, 1433, 1438, 1443, 1447, 1452, 1457, 1461, 1467, 1473, 1482, 1491, 1499, 1503, 1506, 1509, 1516, 1531, 1546, 1553, 1560, 1572, 1582, 1589, 1597, 1605, 1617, 1627, 1636, 1651, 1662, 1668, 1677, 1685, 1692, 1697, 1702, 1707, 1713, 1719, 1724, 1729, 1737, 1744, 1754, 1761, 1767, 1773, 1783, 1791, 1797, 1802, 1807, 1812, 1818, 1826, 1833, 1840, 1846, 1851, 1855, 1859, 1866, 1873, 1884, 1894, 1902, 1912, 1924, 1932, 1939, 1947, 1954, 1959, 1968, 1976, 1982, 1990, 1994, 1998, 2008, 2014, 2023, 2031, 2041, 2051, 2061, 2068, 2075, 2088, 2099, 2107, 2115, 2124, 2133, 2139, 2145, 2150, 2157, 2162, 2173, 2184, 2200, 2213, 2218, 2227, 2236, 2246, 2254, 2261, 2272, 2282, 2290, 2298, 2306, 2313, 2319, 2326, 2336, 2346, 2354, 2367, 2381, 2387, 2395, 2405, 2420, 2429, 2439, 2447, 2455, 2463, 2471, 2479, 2487, 2494, 2505, 2515, 2522, 2531, 2542, 2556, 2565, 2572, 2578, 2587, 2596, 2602, 2608, 2617, 2626, 2635, 2641, 2645, 2649, 2654, 2667, 2678, 2685, 2689, 2693, 2701, 2706, 2715, 2721, 2727, 2733, 2740, 2748, 2758, 2766, 2771, 2775, 2781, 2787, 2793, 2801, 2812, 2820, 2825, 2830, 2838, 2849, 2855, 2863, 2870, 2876, 2883, 2891, 2898, 2904, 2915, 2924, 2930, 2934, 2941, 2950, 2956, 2966, 2975, 2982, 2991, 2996, 3003, 3012, 3019, 3032, 3040, 3049, 3056, 3062, 3070, 3085, 3091, 3096, 3106, 3118, 3126, 3136, 3146, 3153, 3158, 3162, 3165, 3170, 3175, 3181, 3184, 3187, 3190, 3193, 3196, 3199, 3202, 3206, 3213, 3216, 3220, 3229, 3236, 3248, 3258, 3264, 3271, 3278, 3288, 3295, 3305, 3325, 3334, 3342, 3357, 3371, 3378, 3384, 3389, 3403, 3413, 3421, 3430, 3438, 3449]


class BundleAdjustment:

    def __init__(self):
        # Create a bundle windows
        self.__key_frames = get_loaded_key_frames()
        # self.__key_frames = [0, 4]
        # self.__key_frames = choose_key_frames_by_median_track_len2(Data.DB.get_frames(), PERCENTAGE)  # Todo: improve that
        self.__bundles_lst = create_bundle_windows(self.__key_frames)
        self.__num_bundles = len(self.__bundles_lst)
        self.__gtsam_cameras_rel_to_bundle = None
        self.__all_landmarks_rel_to_bundle = None

    def solve(self, method=ITERATIVE_METHOD, workers_num=5):
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

        elif method is ITERATIVE_METHOD:
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


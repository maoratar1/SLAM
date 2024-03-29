import pickle

import gtsam
import numpy as np
from gtsam import symbol
import tqdm
import concurrent.futures

from BundleAdjustmentDirectory.BundleWindow import BundleWindow
from DataDirectory import Data
from PoseGraphDirectory import VertexGraph

import utils.utills

LOADED_POSE_GRAPH_PATH = r'PoseGraphDirectory/pose_graph_3_with_loop.pickle'

COV_DIM = 6
CAMERA_SYM = "c"
ITERATIVE_METHOD = "ITERATIVE"
MULTI_PROCESSED = "MULTI PROCESSED"
OPTIMIZED = "OPTIMIZED"
MAHALANOBIS_DIST_THRESHOLD = 50
MAX_CAND_NUM = 3
INLIERS_THRESHOLD_PERC = 85
VERTEX_METHOD = VertexGraph.ADJ_LST

DIRECTED = False


def save(path, pg):
    """
    Saves the Database "db" in path "path"
    :param db: DataDirectory base
    """
    filehandler = open(path, 'wb')
    pickle.dump(pg, filehandler)
    filehandler.close()


def load(saved_pg_path):
    """
    Loads that Database from "saved_db_path"
    :param saved_pg_path: Pose graph saved path
    :return: Database
    """
    filehandler = open(saved_pg_path, 'rb')
    db = pickle.load(filehandler)
    filehandler.close()
    return db


def create_pose_graph(directed=DIRECTED):
    """
    Compute covariance and relative poses from Bundle adjustment algorithm
    :return: Pose graph
    """
    bundles_lst = Data.BA.get_bundles_lst()
    key_frames = Data.BA.get_key_frames()

    rel_poses_lst, rel_cov_mat_lst = compute_cov_rel_poses_for_bundles(bundles_lst, ITERATIVE_METHOD)

    return PoseGraph(key_frames, rel_poses_lst, rel_cov_mat_lst, directed=directed)


def compute_cov_rel_poses_for_bundles(bundles, method=ITERATIVE_METHOD, workers_num=5):
    """
    Computes relative poses between key frames_ind and their relative covariance matrix
    :return: Relative poses and covariance matrices lists
    """

    rel_poses_lst, cov_mat_lst = [], []
    if method is ITERATIVE_METHOD:
        rel_poses_lst, cov_mat_lst = compute_cov_rel_poses_for_bundles_iterative(bundles)

    elif method is MULTI_PROCESSED:
        rel_poses_lst, cov_mat_lst = compute_cov_rel_poses_for_bundles_multiprocess(bundles, workers_num)

    return rel_poses_lst, cov_mat_lst


def compute_cov_rel_poses_for_bundles_iterative(bundles):
    """
    Computes relative poses between key frames_ind and their relative covariance matrix iteratively
    :return: Relative poses and covariance matrices lists
    """
    rel_poses_lst, cov_mat_lst = [], []

    print("\tCompute relative poses and covariance for each bundle")
    for i in tqdm.tqdm(range(len(bundles))):
        relative_pose, cond_cov_mat = compute_cov_rel_poses_for_one_bundle(bundles[i])

        cov_mat_lst.append(cond_cov_mat)
        rel_poses_lst.append(relative_pose)

    return rel_poses_lst, cov_mat_lst


def compute_cov_rel_poses_for_bundles_multiprocess(bundles, workers_num):
    """
    Computes relative poses between key frames_ind and their relative covariance matrix with multi processing
    :return: Relative poses and covariance matrices lists
    """
    rel_poses_lst, cov_mat_lst = [], []

    with concurrent.futures.ProcessPoolExecutor(workers_num) as executor:
        results = list(tqdm.tqdm(executor.map(compute_cov_rel_poses_for_one_bundle, bundles), total=len(bundles)))

    for res in results:  # [[relative_pose, cond_cov_mat], ...]
        rel_poses_lst.append(res[0])
        cov_mat_lst.append(res[1])

    return rel_poses_lst, cov_mat_lst


def compute_cov_rel_poses_for_one_bundle(bundle):
    """
    Computes relative poses between key frame and their relative covariance matrix for one bundle
    :return: Relative pose and covariance matrix
    """
    # Compute bundle marginals
    first_key_frame, second_key_frame = bundle.get_key_frames()
    optimized_values = bundle.get_optimized_values()
    try:
        marginals = gtsam.Marginals(bundle.graph, optimized_values)
    except:
        stop = 2
    # Apply marginalization and conditioning to compute the covariance of the last key frame pose
    # in relate to first key frame
    keys = gtsam.KeyVector()
    keys.append(symbol(CAMERA_SYM, first_key_frame))
    keys.append(symbol(CAMERA_SYM, second_key_frame))

    # This row apply Conditioning by ignoring rows and cols that includes the first key frame at the information matrix
    information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    # This row compute the relative covariance of the last key frame in relate to the first one
    rel_last_cam_cov_mat = np.linalg.inv(information_mat_first_second)

    # Compute relative pose
    gtsam_first_camera_pose = optimized_values.atPose3(symbol(CAMERA_SYM, first_key_frame))  # first_cam - > world
    gtsam_second_camera_pose = optimized_values.atPose3(symbol(CAMERA_SYM, second_key_frame))  # second_cam - > world
    gtsam_relative_pose = gtsam_first_camera_pose.between(gtsam_second_camera_pose)  # second_cam -> world

    return gtsam_relative_pose, rel_last_cam_cov_mat


def compute_rel_pose_and_cov_with_bundle(first_frame, last_frame, tracks=None):
    """
    Compute relative pose and covariance with bundle by a given tracks that found between those first and last frames
    :return: relative covariance and poses
    """
    bundle = BundleWindow(first_frame, last_frame, all_frame_between=False, little_bundle_tracks=tracks)
    bundle.create_factor_graph()
    bundle.optimize()
    relative_pose, rel_last_cam_cov_mat = compute_cov_rel_poses_for_one_bundle(bundle)

    return relative_pose, rel_last_cam_cov_mat


class PoseGraph:

    def __init__(self, key_frames, rel_poses, rel_covs, directed=False):
        self.__global_pose = []
        self.__cov = rel_covs
        self.__rel_poses = rel_poses
        self.__key_frames = key_frames
        self.__camera_sym = []
        self.optimizer = None
        self.__initial_estimate_before_loop_closure = gtsam.Values()
        self.__initial_estimate = gtsam.Values()
        self.__optimized_values = None
        self.__graph = gtsam.NonlinearFactorGraph()
        self.__loops = []  # Contains tuples of [frame_ind, [prev frame, ...] ]

        self.__vertex_graph = VertexGraph.VertexGraph(vertices_num=len(key_frames),
                                                      rel_covs=rel_covs,
                                                      method=VERTEX_METHOD,
                                                      directed=DIRECTED)

        self.create_factor_graph()

    def set_vertex_graph(self, graph):
        """
        Set vertex graph to a given one. Basically used for testing
        """
        self.__vertex_graph = graph

    def create_factor_graph(self):
        """
        Creates pose graph
        """

        # Create first camera symbol
        gtsam_cur_global_pose = gtsam.Pose3()
        first_left_cam_sym = symbol(CAMERA_SYM, 0)
        self.__camera_sym.append(first_left_cam_sym)

        self.__global_pose.append(gtsam_cur_global_pose)

        # Create first camera's pose factor
        sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])
        pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        factor = gtsam.PriorFactorPose3(first_left_cam_sym, gtsam_cur_global_pose, pose_uncertainty)
        self.__graph.add(factor)

        # Add initial estimate
        self.__initial_estimate.insert(first_left_cam_sym, gtsam_cur_global_pose)

        prev_sym = first_left_cam_sym

        # Create factor for each pose and add it to the graph
        for i in range(len(self.__rel_poses)):
            cur_sym = symbol(CAMERA_SYM, i + 1)
            self.__camera_sym.append(cur_sym)

            # Create factor
            noise_model = gtsam.noiseModel.Gaussian.Covariance(self.__cov[i])
            factor = gtsam.BetweenFactorPose3(prev_sym, cur_sym, self.__rel_poses[i], noise_model)
            self.__graph.add(factor)

            # Add initial estimate
            gtsam_cur_global_pose = gtsam_cur_global_pose.compose(self.__rel_poses[i])
            self.__global_pose.append(gtsam_cur_global_pose)
            self.__initial_estimate.insert(cur_sym, gtsam_cur_global_pose)

            # Update prev_sym to be the cur one for the next iteration
            prev_sym = cur_sym

        self.__initial_estimate_before_loop_closure = self.__initial_estimate

    def optimize(self, loop=False):
        """
        Apply optimization with Levenberg marquardt algorithm
        """
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.__graph, self.__initial_estimate)
        result = self.optimizer.optimize()
        self.__optimized_values = result

        if loop:
            self.__initial_estimate = result

    def graph_error(self, optimized=True):
        """
        Returns the graph error
        """
        if not optimized:
            error = self.__graph.error(self.__initial_estimate_before_loop_closure)
        else:
            error = self.__graph.error(self.__optimized_values)

        return error

    def get_initial_estimate_values(self):
        """
        Returns initial estimation values
        """
        return self.__initial_estimate_before_loop_closure

    def get_optimized_values(self):
        """
        Return optimized values
        """
        return self.__optimized_values

    def marginals(self, optimized=True):
        """
        Return graph's marginals for optimized values
        """
        if optimized:
            return gtsam.Marginals(self.__graph, self.__optimized_values)

        return gtsam.Marginals(self.__graph, self.__initial_estimate_before_loop_closure)

    def get_optimized_cameras_poses(self):
        """
        Returns optimized cameras poses
        """
        cameras_poses = []
        for camera_sym in self.__camera_sym:
            cam_pose = self.__optimized_values.atPose3(camera_sym)
            cameras_poses.append(cam_pose)

        return cameras_poses

    def get_initial_est_cameras_poses(self):
        """
        Returns optimized cameras poses
        """
        cameras_poses = []
        for camera_sym in self.__camera_sym:
            cam_pose = self.__initial_estimate_before_loop_closure.atPose3(camera_sym)
            cameras_poses.append(cam_pose)

        return cameras_poses

    def get_key_frames(self):
        """
        Returns key frames
        """
        return self.__key_frames

    def get_num_poses(self):
        """
        Return num poses
        """
        return len(self.__key_frames)

    def get_rel_cov(self):
        """
        Return a list of the relative cov of the initial pose graph
        """
        return self.__cov

    def __getstate__(self):
        """
        Method for that make it possible to pickle a PoseGraph object
        """
        state = self.__dict__.copy()
        del state["optimizer"]
        return state

    def __setstate__(self, state):
        """
        Method for that "returns" the optimizer filed of the pickled PoseGraph object
        """
        self.__dict__.update(state)
        # Add optimizer back since it doesn't exist in the pickle
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.__graph, self.__initial_estimate)

    # == Code for Loop Closure ==
    def find_loop_cand_by_mahalanobis_dist(self, cur_cam_pose_graph_ind):
        """
        Finds loops candidates of "cur_cam_pose_graph_ind" camera's previous cameras by the
         mahalanobis distance condition
        :return:
        """
        candidates = []
        cur_cam_mat = self.__initial_estimate.atPose3(symbol(CAMERA_SYM, cur_cam_pose_graph_ind))  # 'n' camera : cur_cam -> world

        for prev_cam_pose_graph_ind in range(cur_cam_pose_graph_ind):  # Run on the previous cameras 0 <= i < n

            prev_cam_mat = self.__initial_estimate.atPose3(
                symbol(CAMERA_SYM, prev_cam_pose_graph_ind))  # 'i' camera : prev_cam -> world

            # Find the shortest path and estimate its relative covariance
            shortest_path = self.__vertex_graph.find_shortest_path(prev_cam_pose_graph_ind, cur_cam_pose_graph_ind)
            estimated_rel_cov = self.__vertex_graph.estimate_rel_cov(shortest_path)

            # Compute Cams delta and their mahalanobis distance
            cams_delta = utils.utills.gtsam_cams_delta(prev_cam_mat, cur_cam_mat)
            mahalanobis_dist = utils.utills.mahalanobis_dist(cams_delta, estimated_rel_cov)

            if mahalanobis_dist < MAHALANOBIS_DIST_THRESHOLD:
                candidates.append([mahalanobis_dist, prev_cam_pose_graph_ind])

            # if prev_cam_pose_graph_ind == 52 and cur_cam_pose_graph_ind == 431:
            #     candidates.append([1, 52])
            #
            # if prev_cam_pose_graph_ind == 47 and cur_cam_pose_graph_ind == 427:
            #     candidates.append([1, 47])
            #
            # if prev_cam_pose_graph_ind == 44 and cur_cam_pose_graph_ind == 424:
            #     candidates.append([1, 44])
            #
            # if prev_cam_pose_graph_ind == 293 and cur_cam_pose_graph_ind == 419:
            #     candidates.append([1, 293])
            #
            # if prev_cam_pose_graph_ind == 289 and cur_cam_pose_graph_ind == 415:
            #     candidates.append([1, 289])
            #
            # if prev_cam_pose_graph_ind == 285 and cur_cam_pose_graph_ind == 411:
            #     candidates.append([1, 285])
            if 191 <= cur_cam_pose_graph_ind <= 194:
                print(cur_cam_pose_graph_ind, prev_cam_pose_graph_ind, mahalanobis_dist)

        # if there are candidates, choose the best MAX_CAND_NUM numbers
        if len(candidates) > 0:

            # print(cur_cam_pose_graph_ind, candidates)

            sorted_candidates = sorted(candidates, key=lambda elem: elem[0])  # Sort candidates by mahalanobis dist
            # Take only the MAX_CAND_NUM candidate number and the index from the original list (without dist)
            candidates = np.array(sorted_candidates[:MAX_CAND_NUM]).astype(int)[:, 1]

        return candidates

    def find_loop_closure(self, cur_cam_pose_graph_ind):
        """
        Finds loop closure for a given camera:
            1. Find candidates by mahalanobis distance condition
            For those who past the first step:
            2. Choose the candidates that their inliers percentage after applying a full consensus match are greater
            than some threshold
        :return:
        """
        mahalanobis_dist_cand_at_pg_ind = self.find_loop_cand_by_mahalanobis_dist(cur_cam_pose_graph_ind)

        passed_consensus_frame_track_tuples = []

        if len(mahalanobis_dist_cand_at_pg_ind) > 0:
            cur_frame_movie_ind = self.__key_frames[cur_cam_pose_graph_ind]
            mahalanobis_dist_cand_at_movie_ind = self.convert_pose_graph_ind_to_movie_ind(mahalanobis_dist_cand_at_pg_ind)

            passed_consensus_frame_track_tuples = utils.utills.find_loop_cand_by_consensus(mahalanobis_dist_cand_at_movie_ind,
                                                                                          mahalanobis_dist_cand_at_pg_ind,
                                                                                          cur_frame_movie_ind,
                                                                                          INLIERS_THRESHOLD_PERC)

            # consensus_match_candidates = mahalanobis_dist_cand[consensus_match_candidates_ind]  # Todo: maybe use it it update data base at find loop candiate function

        return passed_consensus_frame_track_tuples

    def convert_pose_graph_ind_to_movie_ind(self, pose_graph_ind_lst):
        """
        Converts camera's pose graph index to the movie's index
        """
        return np.array(self.__key_frames)[pose_graph_ind_lst]

    def add_loop_factors(self, loop_prev_frames_tracks_tuples, cur_frame):
        """
        Adds loops factor by receiving a list of previous frames and their tracks between them to the cur_frame
        """
        cur_frame_sym = symbol(CAMERA_SYM, cur_frame)
        cur_frame_movie_ind = self.__key_frames[cur_frame]

        cur_frame_loop = []

        for prev_frame, tracks in loop_prev_frames_tracks_tuples:
            cur_frame_loop.append(prev_frame)

            prev_frame_movie_ind = self.__key_frames[prev_frame]
            rel_pose, rel_last_cam_cov_mat = compute_rel_pose_and_cov_with_bundle(prev_frame_movie_ind,
                                                                                  cur_frame_movie_ind, tracks)

            # Create factor
            prev_frame_sym = symbol(CAMERA_SYM, prev_frame)
            noise_model = gtsam.noiseModel.Gaussian.Covariance(rel_last_cam_cov_mat)
            factor = gtsam.BetweenFactorPose3(prev_frame_sym, cur_frame_sym, rel_pose, noise_model)
            self.__graph.add(factor)
            self.__vertex_graph.add_edge(prev_frame, cur_frame, rel_last_cam_cov_mat)

        self.__loops.append([cur_frame, cur_frame_loop])

    def loop_closure_for_specific_frame(self, cur_cam_pose_graph_ind):
        """
        Find a loop, adds it as a factor and apply optimization of the pose graph on it
        :return [] if didnt find a loop else
        """
        loop_prev_frames_tracks_tuples = self.find_loop_closure(cur_cam_pose_graph_ind)

        if len(loop_prev_frames_tracks_tuples) == 0:  # Did not find a loop
            return

        prev_loops = []
        for loop in loop_prev_frames_tracks_tuples:
            prev_loops += [loop[0]]

        print("FOUNDED LOOP: ", cur_cam_pose_graph_ind, prev_loops)

        self.add_loop_factors(loop_prev_frames_tracks_tuples, cur_cam_pose_graph_ind)
        self.optimize(loop=True)
        return loop_prev_frames_tracks_tuples

    def loop_closure(self, start_cam=0, end_cam=None):
        """
        Apply loop closure for the cams in range(start_cam, end_cam)
        """
        if end_cam is None:
            end_cam = len(self.__key_frames) - 1

        print("Applying loop closure for whole trajectory")
        for i in tqdm.tqdm(range(start_cam, end_cam + 1)):
            self.loop_closure_for_specific_frame(cur_cam_pose_graph_ind=i)

    def get_loops(self):
        """
        Return loops
        """
        return self.__loops












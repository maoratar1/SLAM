import gtsam
import numpy as np
from gtsam import symbol
import tqdm
import concurrent.futures

CAMERA_SYM = "c"
ITERATIVE_METHOD = "ITERATIVE"
MULTI_PROCESSED = "MULTI PROCESSED"


def compute_cov_rel_poses_for_bundles(bundles, method=ITERATIVE_METHOD, workers_num=5):
    """
    Computes relative poses between key frames and their relative covariance matrix
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
    Computes relative poses between key frames and their relative covariance matrix iteratively
    :return: Relative poses and covariance matrices lists
    """
    rel_poses_lst, cov_mat_lst = [], []

    print("Compute relative poses and covariance for each bundle")
    for i in tqdm.tqdm(range(len(bundles))):
        relative_pose, cond_cov_mat = compute_cov_rel_poses_for_one_bundle(bundles[i])

        # Add the result to the list
        cov_mat_lst.append(cond_cov_mat)
        rel_poses_lst.append(relative_pose)

    return rel_poses_lst, cov_mat_lst


def compute_cov_rel_poses_for_bundles_multiprocess(bundles, workers_num):
    """
    Computes relative poses between key frames and their relative covariance matrix with multi processing
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
    result = bundle.get_optimized_values()
    marginals = gtsam.Marginals(bundle.graph, result)

    # Apply marginalization and conditioning to compute the covariance of the last key frame pose
    # in relate to first key frame
    keys = gtsam.KeyVector()
    keys.append(symbol(CAMERA_SYM, first_key_frame))
    keys.append(symbol(CAMERA_SYM, second_key_frame))
    information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    cond_cov_mat = np.linalg.inv(information_mat_first_second)

    # Compute relative pose
    first_camera_pose = result.atPose3(symbol(CAMERA_SYM, first_key_frame))
    second_camera_pose = result.atPose3(symbol(CAMERA_SYM, second_key_frame))
    relative_pose = first_camera_pose.between(second_camera_pose)

    return relative_pose, cond_cov_mat


class PoseGraph:

    def __init__(self, key_frames, rel_poses, cov):
        self.__global_pose = []
        self.__cov = cov
        self.__rel_poses = rel_poses
        self.__key_frames = key_frames
        self.__optimizer = None
        self.__initial_estimate = gtsam.Values()
        self.__optimized_values = None
        self.__graph = gtsam.NonlinearFactorGraph()

        self.create_factor_graph()

    def create_factor_graph(self):
        """
        Creates pose graph
        """

        # Create first camera symbol
        gtsam_cur_global_pose = gtsam.Pose3()
        first_left_cam_sym = symbol(CAMERA_SYM, self.__key_frames[0])

        self.__global_pose.append(gtsam_cur_global_pose)

        # Create first camera's pose factor
        sigmas = np.array([(3 * np.pi / 180)**2] * 3 + [1e-2, 1e-3, 1e-1])
        pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        factor = gtsam.PriorFactorPose3(first_left_cam_sym, gtsam_cur_global_pose, pose_uncertainty)
        self.__graph.add(factor)

        # Add initial estimate
        self.__initial_estimate.insert(first_left_cam_sym, gtsam_cur_global_pose)

        prev_sym = first_left_cam_sym

        # Create factor for each pose and add it to the graph
        for i in range(1, len(self.__rel_poses)):
            cur_sym = symbol(CAMERA_SYM, self.__key_frames[i])
            gtsam_cur_global_pose = gtsam_cur_global_pose.compose(self.__rel_poses[i])
            self.__global_pose.append(gtsam_cur_global_pose)

            # Create factor
            noise_model = gtsam.noiseModel.Gaussian.Covariance(self.__cov[i - 1])
            factor = gtsam.BetweenFactorPose3(prev_sym, cur_sym, self.__rel_poses[i], noise_model)
            self.__graph.add(factor)

            # Add initial estimate
            self.__initial_estimate.insert(cur_sym,  gtsam_cur_global_pose)

            prev_sym = cur_sym

    def optimize(self):
        """
        Apply optimization with Levenberg marquardt algorithm
        """
        self.__optimizer = gtsam.LevenbergMarquardtOptimizer(self.__graph, self.__initial_estimate)
        self.__optimized_values = self.__optimizer.optimize()

    def graph_error(self, optimized=True):
        """
        Returns the graph error
        :param optimized:
        :return:
        """
        if not optimized:
            error = self.__graph.error(self.__initial_estimate)
        else:
            error = self.__graph.error(self.__optimized_values)

        return error

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

    def marginals(self):
        """
        Return graph's marginals for optimized values
        """
        return gtsam.Marginals(self.__graph, self.__optimized_values)






import pickle
import numpy as np
import cv2
import tqdm

DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'
LOADED_KITTI_DATA = r'DataDirectory/KITTI_DATA.pickle'
LEFT_CAM_TRANS_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/poses/00.txt'
IDX = 000000
MOVIE_LEN = 3450


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


class KittiData:
    """
    Class for loading Kitti's data (Images, left cameras transformations and cameras matrices)
    once and saving them for further using
    """
    def __init__(self):
        self.__images = []
        self.load_images(MOVIE_LEN, kernel_size=10)

        self.__k, self.__m1, self.__m2 = self.read_cameras()

        self.__ground_truth_transformations = []
        self.read_ground_truth_transformations()

    def load_images(self, movie_len, kernel_size=0):
        """
        Loads KITTI
        """
        print(f"\t\tLoading images....")

        for i in tqdm.tqdm(range(movie_len)):
            left_img, right_img = self.read_images(frame_num=i, kernel_size=kernel_size)
            self.add_image([left_img, right_img])

    def add_image(self, image):
        """
        Add image to the list
        """
        self.__images.append(image)

    def get_image(self, ind):
        """
        Returns image in index ind
        """
        return self.__images[ind]

    def read_images(self, frame_num, kernel_size=0):
        """
        :param idx: Image's index in the Kitti dataset
        :return: left and right cameras photos
        """
        img_name = '{:06d}.png'.format(IDX + frame_num)
        img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
        img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)

        if kernel_size != 0:
            img1 = cv2.blur(img1, (kernel_size, kernel_size))
            img2 = cv2.blur(img2, (kernel_size, kernel_size))

        return img1, img2

    def read_cameras(self):
        """
        Reads First frame cameras intrinsic and extrinsic matrices
        :return:
        """
        print("\tLoading cameras matrices:\n\tK: Calibration camera\n\tM1: left camera extrinsic matrix\n\tM2: Right "
              "camera extrinsic matrix relative to the left camera")

        with open(DATA_PATH + 'calib.txt') as f:
            l1 = f.readline().split()[1:]  # skip first token
            l2 = f.readline().split()[1:]  # skip first token
            l1 = [float(i) for i in l1]
            m1 = np.array(l1).reshape(3, 4)
            l2 = [float(i) for i in l2]
            m2 = np.array(l2).reshape(3, 4)
            k = m1[:, :3]
            m1 = np.linalg.inv(k) @ m1
            m2 = np.linalg.inv(k) @ m2
            return k, m1, m2

    def read_ground_truth_transformations(self, left_cam_trans_path=LEFT_CAM_TRANS_PATH, movie_len=MOVIE_LEN):
        """
        Reads the ground truth transformations
        :return: array of transformation
        """
        print("\tLoading ground truth cameras extrinsic matrices...")
        with open(left_cam_trans_path) as f:
            lines = f.readlines()

        for i in range(movie_len):
            left_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
            self.__ground_truth_transformations.append(left_mat)

        return self.__ground_truth_transformations

    def get_K(self):
        """
        Returns camera's matrices
        :return:
        """
        return self.__k

    def get_M1(self):
        return self.__m1

    def get_M2(self):
        return self.__m2

    def get_ground_truth_transformations(self):
        return self.__ground_truth_transformations

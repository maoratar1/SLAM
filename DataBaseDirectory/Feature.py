

class Feature:
    """
    This class represents feature in frame
    """

    def __init__(self, kpt_left_idx, kpt_right_idx, x_left, x_right, y_left, y_right):
        self.__kpt_left = kpt_left_idx
        self.__kpt_right = kpt_right_idx
        self.__x_left = x_left
        self.__x_right = x_right
        self.__y_left = y_left
        self.__y_right = y_right

    def get_left_coor(self):
        """
        Return d2_points at the left image
        """
        return self.__x_left, self.__y_left

    def get_right_coor(self):
        """
        Returns d2_points at the right image
        """
        return self.__x_right, self.__y_right

    def get_left_kpt(self):
        """
        Returns key point in the left image (from the list of key d2_points that found in left image)
        """
        return self.__kpt_left

    def get_right_kpt(self):
        """
        Returns key point in the right image (from the list of key d2_points that found in right image)
        :return:
        """
        return self.__kpt_right


import tqdm

import utills
# == Helpers for debugging == #
def check1(pair0, left0_left1_matches):
    """
    check if the matches in pair0 and left0_left1_matches are the same
    """
    n = len(pair0)
    for i in range(n):
        p0_idx = pair0[i].queryIdx
        l0_l1_idx = left0_left1_matches[i].queryIdx

        if p0_idx != l0_l1_idx:
            print(f"Wrong {i}")


def check2(pair1, left0_left1_matches):
    """
    check if the matches in pair1 and left0_left1_matches are the same
    """
    n = len(pair1)
    for i in range(n):
        p1_idx = pair1[i].queryIdx
        l0_l1_idx = left0_left1_matches[i].trainIdx

        if p1_idx != l0_l1_idx:
            print(f"Wrong {i}")


def whole_movie(first_left_ex_mat=utills.M1):
    """
    Compute the transformation of two consequence left KITTI in the whole movie
    :return:array of transformations where the i'th element is the transformation between i-1 -> i
    """
    T_arr = [first_left_ex_mat]

    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = utills.read_and_rec_match(0, kernel_size=10)
    for i in tqdm.tqdm(range(1, utills.MOVIE_LEN)):
        left1_kpts, left1_dsc, right1_kpts, pair1_matches, pair1_rec_matches_idx = utills.read_and_rec_match(i)

        left1_ex_mat = utills.compute_trans_between_cur_to_next(left0_kpts, left0_dsc, right0_kpts,
                                                         pair0_matches, pair0_rec_matches_idx,
                                                         left1_kpts, left1_dsc, right1_kpts,
                                                         pair1_matches, pair1_rec_matches_idx)
        left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = left1_kpts, left1_dsc,\
                                                                                   right1_kpts, pair1_matches,\
                                                                                   pair1_rec_matches_idx
        T_arr.append(left1_ex_mat)

    return T_arr

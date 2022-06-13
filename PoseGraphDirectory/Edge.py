import numpy as np


def cov_weight(cov):
    # Simple way to compute det for symetric matrix
    return np.linalg.det(cov)


class Edge:

    def __init__(self, source, target, weight=None, cov=None):
        self.source = source
        self.target = target
        self.__cov = cov

        if weight is None:
            self.__weight = cov_weight(cov)
        else:
            self.__weight = weight

    def get_cov(self):
        return self.__cov

    def get_weight(self):
        return self.__weight

    def get_vertices(self):
        return self.source, self.target

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target

    def __hash__(self):
        return hash((self.source, self.target))

    # def get_edge_id(self):
    #     return self.__id


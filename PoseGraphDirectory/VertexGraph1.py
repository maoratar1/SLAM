import numpy as np
from PoseGraphDirectory import Edge

COV_DIM = 6


def cov_weight(cov):
    if cov[0][0] == float('inf'):
        return float('inf')
    return np.linalg.det(cov)


class VertexGraph1:

    def __init__(self, num_vertices, rel_covs):
        self.__num_vertices = num_vertices
        self.__graph = [[None for col in range(self.__num_vertices)] for row in range(self.__num_vertices)]
        self.create_vertex_graph(rel_covs)

    def add_edge(self, first_v, second_v, cov):
        self.__graph[first_v][second_v] = Edge.Edge(cov, )

    def create_vertex_graph(self, cov):
        for i in range(len(cov)):
            self.add_edge(i, i + 1, cov[i])

    def find_shortest_path_adjacency_mat(self, source, target):
        dists = [np.diag(float('inf'), axis1=COV_DIM, axis2=COV_DIM)] * self.__num_vertices
        calculated_vertices = [False] * self.__num_vertices
        dists[source] = np.zeros((COV_DIM, COV_DIM))

        while calculated_vertices[target] is False:

            min_dist_vertex = self.find_min_dist_vertex(calculated_vertices, dists)
            calculated_vertices[min_dist_vertex] = True

            for y in range(self.__num_vertices):
                if self.__graph[min_dist_vertex][y] is not None and calculated_vertices[y] is False and \
                        cov_weight(dists[y]) > \
                                    cov_weight(dists[min_dist_vertex] + self.__graph[min_dist_vertex][y].get_cov()):
                    dists[y] = dists[min_dist_vertex] + self.__graph[min_dist_vertex][y].get_cov()

        return dists[target]

    def find_min_dist_vertex(self, calculated_vertices, dists):
        minimum = float('inf')
        min_ind = None

        for u in range(self.__num_vertices):
            if cov_weight(dists[u]) < minimum and calculated_vertices[u] is False:
                minimum = cov_weight(dists[u])
                min_ind = u

        return min_ind

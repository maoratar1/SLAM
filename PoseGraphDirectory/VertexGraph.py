from collections import defaultdict

import numpy as np

from PoseGraphDirectory import Edge
from utils import MinHeap

COV_DIM = 6
ADJ_MAT = "Adjacency matrix"
ADJ_LST = "Adjacency list"
BFS = "BFS"


class VertexGraph:

    def __init__(self, vertices_num, rel_covs=None, method=ADJ_LST, directed=True):
        # Todo: for now graph is for testing in Test.py
        """
        :param rel_covs: Relative covariances between consecutive cameras
        :param method: Graph's method for searching the shortest path.
            their is 3 options:
                1. Ajacency matrix
                2. Ajacency list with Min-Heap
                3. BFS
        :param edges : list of tuples (i, i + 1) such that there is an edge between 'i' and 'i + 1' vetrtices
        """
        self.__method = method
        self.__directed = directed

        if self.__method == ADJ_LST:
            self.__edges = dict()  # Dictionary for getting edges at the adjacency list representation efficiently

        self.__v_num = vertices_num

        self.__rel_covs = rel_covs

        self.create_vertex_graph()

    # === General code for creating graph and adding edges ===

    def create_vertex_graph(self):
        """
        Creates the vertex graph
        :return:
        """
        if self.__method is ADJ_MAT:

            # Initialize the adjacency matrix and creates the graph
            self.__graph = [[Edge.Edge(row, col, None) for row in range(self.__v_num)] for col in range(self.__v_num)]

            if self.__rel_covs is not None:
                self.set_vertex_graph_adj_mat()

        elif self.__method is ADJ_LST:
            self.__graph = defaultdict(list)

            if self.__rel_covs is not None:
                self.set_vertex_graph_adj_lst()

        elif self.__method is BFS:
            self.__graph = [[] for _ in range(self.__v_num)]

            if self.__rel_covs is not None:
                self.create_vertex_graph_bfs()

    def find_shortest_path(self, source, target):
        """
        Finds the shortest path  between the first_v vertex and target vertex according to the graph method and
        Returns it.
        """
        if self.__method == ADJ_MAT:
            return self.find_shortest_path_adjacency_mat(source, target)
        elif self.__method == ADJ_LST:
            return self.find_shortest_path_adjacency_lst(source, target)
        elif self.__method == BFS:
            return self.find_shortest_path_bfs(source, target)

    def estimate_rel_cov(self, path):
        """
        Compute the estimated relative covariance between to cameras in the path the connecting them
        :param path: list of cameras indexes where the first index contains the first camera and the last index contains
         the last camera in the path
         :return estimated covariance
        """
        estimated_rel_cov = np.zeros((COV_DIM, COV_DIM))
        for i in range(1, len(path)):  # don't include first rel_covs at the path
            edge = self.get_edge_between_vertices(path[i - 1], path[i])
            estimated_rel_cov += edge.get_cov()
        return estimated_rel_cov

    def add_edge(self, source, target, cov):
        """
        Adds a directed edge of the form (first_v, target) with weight and covariance of cov
        """
        if self.__method is ADJ_MAT:
            self.add_edge_adj_mat(source, target, cov)
        elif self.__method is ADJ_LST:
            self.add_edge_adj_lst(source, target, cov)
        elif self.__method is BFS:
            self.add_edge_bfs(source, target)

    def get_edge_between_vertices(self, source, target):
        """
        Returns the edge between first_v and target
        """
        if self.__method is ADJ_MAT:
            return self.__graph[source][target]
        elif self.__method is ADJ_LST:
            return self.__edges[(source, target)]
        elif self.__method is BFS:
            return self.__edges[(source, target)]

    # == Method 1: Adjacency matrix : O(V^2)
    def add_edge_adj_mat(self, first_v, second_v, cov):
        edge = Edge.Edge(first_v, second_v, cov)
        self.__graph[first_v][second_v] = edge

        if not self.__directed:
            edge = Edge.Edge(second_v, first_v, cov)
            self.__graph[second_v][first_v] = edge

    def set_vertex_graph_adj_mat(self):
        """
       Creates the vertex graph - convert the basic pose graph to the vertex graph.
       The basic structure of the pose graph is as chain of cameras there for here we initialize
       only edges of consecutive vertex i and 'i+1'
       """
        for i in range(len(self.__rel_covs)):
            self.add_edge_adj_mat(i, i + 1, self.__rel_covs[i])

    def find_shortest_path_adjacency_mat(self, source, target):
        """
        Applys the Dijkstra algorithm for the adjacency matrix representation and finds the shortest path
        between first_v and target
        :return: Shortest path
        """
        dists = [float('inf')] * self.__v_num
        parents = [-1] * self.__v_num
        calculated_vertices = [False] * self.__v_num
        dists[source] = 0

        while calculated_vertices[target] is False:

            # Finds the vertex with the minimum distance from first_v and add it to the calculated vertices
            # at the first iteration it is the first_v itself
            min_dist_vertex = self.find_min_dist_vertex(calculated_vertices, dists)
            calculated_vertices[min_dist_vertex] = True

            # Go over all the vertices, v, and update the vertex's distance from the first_v by checking if
            # the distance from the first_v to it via the min_dist_vertex (first_v -> min_dist_vertex -> v)
            # is smaller than its current distance
            for v in range(self.__v_num):
                # First condition : If v is min_dist_vertex neighbor
                # Second condition : We want to update v's distance, so we check if it has been already calculated
                # Third condition : Check if the current distance is lower than the distance of the path that
                # pass through min_dist_vertex
                if self.__graph[min_dist_vertex][v].get_weight() > 0 \
                        and calculated_vertices[v] is False \
                        and dists[v] > dists[min_dist_vertex] + self.__graph[min_dist_vertex][v].get_weight():

                    # Update v's distance to the distance through the min_dist_vertex,
                    # and it's parents to the min_dist_vertex
                    dists[v] = dists[min_dist_vertex] + self.__graph[min_dist_vertex][v].get_weight()
                    parents[v] = min_dist_vertex

        # Compute path
        path = self.get_path(parents, target)

        return path

    def get_path(self, parents, target):
        """
        Recursive function for computing the path of target from the first_v
        :param parents: List of vertices parents
        :return: Path
        """
        # Base Case : If target is a first_v
        if parents[target] == -1:
            return [target]

        # Get recursively the path from first_v to target's parent
        return self.get_path(parents, parents[target]) + [target]

    def find_min_dist_vertex(self, calculated_vertices, dists):
        """
        Finds the vertex with the minimum distance from the first_v
        :param calculated_vertices: vertices that has been calculated already
        :param dists: List of vertices distances from the first_v
        :return: Vertex's index with the minimum distance between it and the first_v
        """
        minimum = float('inf')
        min_dist_vertex_ind = -1

        for u in range(len(dists)):
            if dists[u] < minimum and calculated_vertices[u] is False:
                minimum = dists[u]
                min_dist_vertex_ind = u

        return min_dist_vertex_ind

    # == Method 2: Adjacency lst with min heap : O(E + V * LOGV)
    def set_vertex_graph_adj_lst(self):
        """
        Set the vertex graph with the pose graph edges
        """
        for i in range(len(self.__rel_covs)):
            self.add_edge_adj_lst(i, i + 1, self.__rel_covs[i])

    def add_edge_adj_lst(self, first_v, second_v, cov):  # Todo: consider change the graph to include edges not numbers
        """
       Adds a directed edge of the form (first_v, target) with weight and covariance of cov
       """
        edge = Edge.Edge(first_v, second_v, cov)
        # self.__graph[first_v].insert(0, second_v)  # Todo: why to insert that way ?
        self.__graph[first_v].append(second_v)  # Todo: way to insert that way ?
        self.__edges[(first_v, second_v)] = edge

        if not self.__directed:
            edge = Edge.Edge(second_v, first_v, cov)
            self.__edges[(second_v, first_v)] = edge
            self.__graph[second_v].append(first_v)

    def find_shortest_path_adjacency_lst(self, source, target):
        """
        Apply the Dijkstra algorithm for the adjacency list representation and finds the shortest path
        between first_v and target
        :return: Shortest path
        """
        dist = [float('inf')] * self.__v_num
        min_heap = MinHeap.MinHeap()
        parents = [-1] * self.__v_num

        # Initialize min heap with all vertices with infinity distance
        for vertex in range(self.__v_num):
           min_heap.add_node(vertex, dist[vertex])

        # Initialize source distance to 0 and update the min heap
        dist[source] = 0
        min_heap.update_vertex_val(source, dist[source])

        # Min heap contains all vertices that has minimum distance from the first_v had *not* been computed yet,
        # so we run this loop until we find the target minimum distance
        while min_heap.in_heap(target):

            # Extract vertex with minimum distance - this is the part that is efficient than the
            # adjacency matrix representation
            min_dist_heap_node = min_heap.extract_min()  # Min heap nodes are tuples of: (vertex num, distance)
            min_dist_vertex = min_dist_heap_node[0]

            # Go over all the vertices, v, and update v's distance from the first_v by checking if
            # the distance from the first_v to it via the min_dist_vertex (first_v -> min_dist_vertex -> v)
            # is smaller than its current distance
            for neighbor in self.__graph[min_dist_vertex]:
                # First condition: Checks if its distance calculation had not been finalized and the -
                #                  means it is in the Min heap
                # Second condition: Check if the current distance is lower than the distance of the path that
                #                   pass through min_dist_vertex
                source_neighbor_dist = self.__edges[(min_dist_vertex, neighbor)].get_weight()

                if min_heap.in_heap(neighbor) and dist[min_dist_vertex] != float('inf') and\
                        source_neighbor_dist + dist[min_dist_vertex] < dist[neighbor]:

                    dist[neighbor] = source_neighbor_dist + dist[min_dist_vertex]
                    parents[neighbor] = min_dist_vertex

                    # Update distance value in min heap also
                    min_heap.update_vertex_val(neighbor, dist[neighbor])

        # Compute path
        path = self.get_path(parents, target)
        return path

    # == Method 3: BFS assumes unweighted undirected : O(E + V)
    def add_edge_bfs(self, source, target):
        pass

    def create_vertex_graph_bfs(self):
        pass

    def find_shortest_path_bfs(self, source, target):
        pass

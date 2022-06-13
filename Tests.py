from PoseGraphDirectory import Edge, VertexGraph


def find_shortest_path_adjacency_mat(source, target, num_vertices, vertex_graph):
    dists = [float('inf')] * num_vertices
    calculated_vertices = [False] * num_vertices
    dists[source] = 0

    while calculated_vertices[target] is False:
        min_dist_vertex = find_min_dist_vertex(calculated_vertices, dists, num_vertices)
        calculated_vertices[min_dist_vertex] = True

        for y in range(num_vertices):
            if vertex_graph[min_dist_vertex][y] > 0 and calculated_vertices[y] is False and \
                    dists[y] > dists[min_dist_vertex] + vertex_graph[min_dist_vertex][y]:
                dists[y] = dists[min_dist_vertex] + vertex_graph[min_dist_vertex][y]

    return dists[target]


def find_min_dist_vertex(calculated_vertices, dists, num_vertices):
    minimum = float('inf')
    min_ind = None

    for u in range(num_vertices):
        if dists[u] < minimum and calculated_vertices[u] is False:
            minimum = dists[u]
            min_ind = u

    return min_ind


def check_heap():
    source, num_vertices = 0, 9
    vertex_graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
                    [4, 0, 8, 0, 0, 0, 0, 11, 0],
                    [0, 8, 0, 7, 0, 4, 0, 0, 2],
                    [0, 0, 7, 0, 9, 14, 0, 0, 0],
                    [0, 0, 0, 9, 0, 10, 0, 0, 0],
                    [0, 0, 4, 14, 10, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 2, 0, 1, 6],
                    [8, 11, 0, 0, 0, 0, 1, 0, 7],
                    [0, 0, 2, 0, 0, 0, 6, 7, 0]
                    ]

    for i in range(num_vertices):
        print(i, find_shortest_path_adjacency_mat(source, i, num_vertices, vertex_graph))


def check_vertex_adj_mat():
    v_graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
               [4, 0, 8, 0, 0, 0, 0, 11, 0],
               [0, 8, 0, 7, 0, 4, 0, 0, 2],
               [0, 0, 7, 0, 9, 14, 0, 0, 0],
               [0, 0, 0, 9, 0, 10, 0, 0, 0],
               [0, 0, 4, 14, 10, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 2, 0, 1, 6],
               [8, 11, 0, 0, 0, 0, 1, 0, 7],
               [0, 0, 2, 0, 0, 0, 6, 7, 0]
               ]

    v_graph = [[Edge.Edge(None, weight=v_graph[row][col]) for col in range(9)] for row in range(9)]

    graph = VertexGraph.VertexGraph(9, None, graph=v_graph, method=VertexGraph.ADJ_MAT)
    for v in range(9):
        print(graph.find_shortest_path_adjacency_mat(0, v))


def check_vertex_graph_min_heap():
    graph = VertexGraph.VertexGraph(9, [], method=VertexGraph.ADJ_LST, directed=False)
    graph.add_edge_adj_lst(0, 1, 4, None)
    graph.add_edge_adj_lst(0, 7, 8, None)
    graph.add_edge_adj_lst(1, 2, 8, None)
    graph.add_edge_adj_lst(1, 7, 11, None)
    graph.add_edge_adj_lst(2, 3, 7, None)
    graph.add_edge_adj_lst(2, 8, 2, None)
    graph.add_edge_adj_lst(2, 5, 4, None)
    graph.add_edge_adj_lst(3, 4, 9, None)
    graph.add_edge_adj_lst(3, 5, 14, None)
    graph.add_edge_adj_lst(4, 5, 10, None)
    graph.add_edge_adj_lst(5, 6, 2, None)
    graph.add_edge_adj_lst(6, 7, 1, None)
    graph.add_edge_adj_lst(6, 8, 6, None)
    graph.add_edge_adj_lst(7, 8, 7, None)

    for v in range(9):
        print(graph.find_shortest_path_adjacency_lst(0, v))


def main():
    check_vertex_graph_min_heap()


if __name__ == '__main__':
    main()
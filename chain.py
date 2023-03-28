import numpy as np
import re
import matplotlib.pyplot as plt


def approximately_equal(point_1, point_2, epsilon=1e-12):
    dist = np.linalg.norm(point_1 - point_2, ord=2)
    return dist < epsilon


def chain_edges(edges, epsilon=1e-12):
    num_edges = len(edges)
    is_used = np.zeros(num_edges, dtype=np.uint8)
    free_vertex = edges[0][0]
    chain = [free_vertex]
    stop = False
    while not stop:
        stop = True
        for i in range(num_edges):
            if not is_used[i]:
                test_vertex = edges[i]
                if approximately_equal(test_vertex[0], free_vertex, epsilon=epsilon):
                    free_vertex = test_vertex[1]
                    chain.append(free_vertex)
                    stop = False
                    is_used[i] = True
                    break
                elif approximately_equal(test_vertex[1], free_vertex, epsilon=epsilon):
                    free_vertex = test_vertex[0]
                    chain.append(free_vertex)
                    stop = False
                    is_used[i] = True
                    break
    if not approximately_equal(chain[0], chain[-1]):
        raise ValueError('Chain input edges do not form a loop')
    return np.vstack(chain[:-1])


def first_nonzero(array):
    indices_tuple = np.nonzero(array == 0)
    return indices_tuple[0][0]


def chain_edges_multi_connected_components(edges, epsilon=1e-12):
    num_edges = len(edges)
    is_used = np.zeros(num_edges, dtype=np.uint8)

    chains = []
    while np.sum(is_used) != num_edges:
        free_edge_index = first_nonzero(is_used)
        free_vertex = edges[free_edge_index][0]
        chain = [free_vertex]
        stop = False
        while not stop:
            stop = True
            for i in range(num_edges):
                if not is_used[i]:
                    test_vertex = edges[i]
                    if approximately_equal(test_vertex[0], free_vertex, epsilon=epsilon):
                        free_vertex = test_vertex[1]
                        chain.append(free_vertex)
                        stop = False
                        is_used[i] = True
                        break
                    elif approximately_equal(test_vertex[1], free_vertex, epsilon=epsilon):
                        free_vertex = test_vertex[0]
                        chain.append(free_vertex)
                        stop = False
                        is_used[i] = True
                        break
        chains.append(np.vstack(chain))
    return chains


def read_edges_file(file):
    with open(file) as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        match = re.match(r'{{(\d*.\d*), (\d*.\d*)}, {(\d*.\d*), (\d*.\d*)}}', line)
        row = np.array([
            [float(match[1]), float(match[2])],
            [float(match[3]), float(match[4])]
        ])
        rows.append(row)
    edges = np.stack(rows)
    return edges


def write_vertices_file(file, vertices):
    with open(file, 'w') as f:
        f.write('{')
        for vertex in vertices:
            f.write(
                '{'
                + str(vertex[0])
                + ', '
                + str(vertex[1])
                + '},\n'
            )
        f.write('}')

    rows = []
    for line in lines:
        match = re.match(r'{{(\d*.\d*), (\d*.\d*)}, {(\d*.\d*), (\d*.\d*)}}', line)
        row = np.array([
            [float(match[1]), float(match[2])],
            [float(match[3]), float(match[4])]
        ])
        rows.append(row)
    edges = np.stack(rows)
    return edges


def main():
    upper_edges = read_edges_file('/Users/eric/Desktop/upper_edges.txt')
    lower_edges = read_edges_file('/Users/eric/Desktop/lower_edges.txt')

    upper_chain = chain_edges(upper_edges)
    lower_chains = chain_edges_multi_connected_components(lower_edges)
    # plt.plot(upper_chain[:, 0], upper_chain[:, 1])
    # plt.plot(lower_chains[0][:, 0], lower_chains[0][:, 1])
    # plt.plot(lower_chains[1][:, 0], lower_chains[1][:, 1])
    # plt.show()


if __name__ == '__main__':
    main()

# edges = np.array(
#     [
#         [
#             [1., 1.], [2., 2.]
#         ],
#         [
#             [3., 1.], [2., 2.]
#         ],
#         [
#             [1., 1.], [3., 1.]
#         ]
#     ]
# )

# chain = chain_edges(edges)
# chain_expected = np.array([0, 1, 2])

# edges = np.array(
#     [
#         [
#             [1., 1.], [2., 2.]
#         ],
#         [
#             [3., 1.], [2., 2.]
#         ],
#         [
#             [2., 0.], [1., 1.]
#         ],
#         [
#             [2., 0.], [3., 1.]
#         ],
#     ]
# )

# chain = chain_edges(edges)
# chain_expected = np.array([0, 1, 3, 2])
# print(chain)

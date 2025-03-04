import numpy as np
import numpy.typing as npt
import numba

from triangulation import (
    Triangulation
)


@numba.njit
def findfirst(array):
    for i in range(len(array)):
        if array[i]:
            return i
    return -1


def triangle_area(
    v1: npt.NDArray[np.float_],
    v2: npt.NDArray[np.float_],
    v3: npt.NDArray[np.float_]
):
    """triangle_area(v1::Vector{np.float64}, v2::Vector{np.float64}, v3::Vector{np.float64})

    Area of a triangle with vertices `v1`, `v2`, and `v3`."""
    return (-v2[0] * v1[1] + v3[0] * v1[1] + v1[0] * v2[1] - v3[0] * v2[1] - v1[0] * v3[1] + v2[0] * v3[1]) / 2.


# Interpolate the value of pde_solution to get its values on the omegas
@numba.njit
def cartesian_to_barycentric(x, y, x_1, y_1, x_2, y_2, x_3, y_3):
    det_T_inverse = 1 / ((x_1 - x_3) * (y_2 - y_3) + (x_3 - x_2) * (y_1 - y_3))
    lambda_1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) * det_T_inverse
    lambda_2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) * det_T_inverse
    return lambda_1, lambda_2


@numba.njit
def barycentric_to_cartesian(lambda_1, lambda_2, x_1, y_1, x_2, y_2, x_3, y_3):
    lambda_3 = 1 - lambda_1 - lambda_2
    x = lambda_1 * x_1 + lambda_2 * x_2 + lambda_3 * x_3
    y = lambda_1 * y_1 + lambda_2 * y_2 + lambda_3 * y_3
    return x, y


@numba.njit
def barycentric_interpolation(x, y, x_1, y_1, x_2, y_2, x_3, y_3, f_1, f_2, f_3):
    lambda_1, lambda_2 = cartesian_to_barycentric(x, y, x_1, y_1, x_2, y_2, x_3, y_3)
    lambda_3 = 1 - lambda_1 - lambda_2
    return lambda_1 * f_1 + lambda_2 * f_2 + lambda_3 * f_3


def degree_to_size(d: int):
    """Returns the number of coefficients on a triangle for the given degree"""
    return (d + 1) * (d + 2) // 2


def size_to_degree(m: int):
    """The inverse of the `degree_to_size`"""
    return (-3 + int(np.round(np.sqrt(1 + 8 * m)))) // 2


def linear_indices(d: int):
    """Creates the linear indices using the pattern

          v1 = 1
              /  \\
             2     5
            /        \\
           3     6     8
          /              \\
    v2 = 4-----7------9---10 = v3

    """
    m = degree_to_size(d)
    result = np.zeros((m, 3), dtype=np.int64)
    counter = 0
    for iter in range(d, -1, -1):
        result[counter:counter + iter + 1, 0] = np.arange(iter, -1, -1)
        result[counter:counter + iter + 1, 1] = np.arange(0, iter + 1)
        result[counter:counter + iter + 1, 2] = (d - iter) * np.ones(iter + 1, dtype=np.int64)
        counter += iter + 1
    return result


def locate(triple, triple_list):
    for i in range(len(triple_list)):
        if np.all(triple == triple_list[i, :]):
            return i
    return -1


def build_casteljau(d: int):
    m_new = degree_to_size(d - 1)   # Number of new coefficients
    ind_old = linear_indices(d)
    ind_new = linear_indices(d - 1)
    result = np.zeros((m_new, 3), dtype=np.int64)
    for j in range(m_new):
        result[j, 0] = locate(ind_new[j, :] + np.array([1, 0, 0]), ind_old)
        result[j, 1] = locate(ind_new[j, :] + np.array([0, 1, 0]), ind_old)
        result[j, 2] = locate(ind_new[j, :] + np.array([0, 0, 1]), ind_old)
    return result


def casteljau(b, B):
    """Takes `b`, the barycentric coordinates of a point v and `B` the `m` B-form
    coefficients listed linearly in a vector according to `linear_indices` of the
    polynomial to evaluate and performs one step of the de Casteljau algorithm,
    returning a new vector of B-form coefficients."""
    m_old = len(B)                 # Number of old coefficients
    d = size_to_degree(m_old)      # Inferred degree of the polynomial
    m_new = degree_to_size(d - 1)  # Number of new coefficients
    B_new = np.zeros(m_new, dtype=np.float64)
    ind_new = linear_indices(d - 1)
    ind_old = linear_indices(d)
    for j in range(m_new):
        # TODO: hardcode/metacode ind's for speed for low level m's
        ind_1 = locate(ind_new[j, :] + np.array([1, 0, 0]), ind_old)
        ind_2 = locate(ind_new[j, :] + np.array([0, 1, 0]), ind_old)
        ind_3 = locate(ind_new[j, :] + np.array([0, 0, 1]), ind_old)
        B_new[j] = B[ind_1] * b[1] + B[ind_2] * b[2] + B[ind_3] * b[3]
    return B_new


def spline_eval(b, B):
    while len(B) > 1:
        B = casteljau(b, B)
    return B[0]


def bary_grid(mesh_size: int):
    """bary_grid(mesh_size: int)

    Builds the standard domain points on a triangle using barycentric coordinates"""
    n = mesh_size - 1
    return linear_indices(n) / n


def sum_n(n: int):
    """sum_n(n: int)

    Sum of the first `n` natural numbers, i.e. n(n+1)/2"""
    return n * (n + 1) // 2


def e12(degree: int, offset: int = 0):
    """e12(degree: int, offset: int=0)

    Gives the indices of the B-form coefficients on a line parallel to
    the line from v1 to v2, with `offset` being the distance that line is
    from the edge e12."""
    start = sum_n(degree + 1) - sum_n(degree + 1 - offset)
    return np.arange(start, start + degree + 1 - offset)


def e23(degree: int, offset: int = 0):
    """e23(degree: int, offset: int=0)

    Gives the indices of the B-form coefficients on a line parallel to
    the line from v2 to v3, with `offset` being the distance that line is
    from the edge e23."""
    return [sum_n(degree + 1) - sum_n(k - 1) - offset - 1 for k in range(degree + 1, offset, -1)]


def e31(degree: int, offset: int = 0):
    """e31(degree: int, offset: int=0)

    Gives the indices of the B-form coefficients on a line parallel to
    the line from v3 to v1, with `offset` being the distance that line is
    from the edge e31."""
    return [sum_n(degree + 1) - sum_n(k) + offset for k in range(offset + 1, degree + 2)]


def boundary_marker_to_fvalue(b: int):
    """boundary_marker_to_fvalue(b: int)

    Plays the role of the function g, but takes the boundary markers as its argument."""
    if b > 1:
        return 0.
    elif b == 1:
        return 1.
    else:
        raise ValueError("Invalid boundary marker")


def Bindex(triangle: int, mindex: int, m: int):
    """Bindex(triangle: int, mindex: int, m: int)

    Index in the B-form coefficient corresponding to `triangle` and `mindex`
    which ranges from 1 to `m` for the `m` B-form coefficients on each triangle."""
    return (triangle - 1) * m + mindex


def traverse(edge: int, degree: int, offset: int = 0):
    """traverse(index: int, degree: int, offset: int=0)

    Returns the indices needed to traverse the appropriate edge based on `edge`
    (either 1, 2, or 3 corresponding to the edges e12, e23, or e31)
    by passing the arguments to the functions e12, e23, or e31."""
    if edge == 0:
        return e12(degree, offset)
    elif edge == 1:
        return e23(degree, offset)
    elif edge == 2:
        return e31(degree, offset)
    else:
        raise ValueError("Invalid index")


def odd_vertex(v: int):
    """odd_vertex(v: int)

    Each v-value represents the correspondence 1~e12, 2~e23, 3~e31. The function
    returns the odd_vertex out, meaning the vertex not a part of that edge."""
    if v == 0:
        return 2
    elif v == 1:
        return 0
    elif v == 2:
        return 1


def cycle(i: int, v: int):
    """cycle(i: int, v: int)

    Cycles the number `i` in the ordered triple (1, 2, 3) by v positions"""
    return (i + v) % 3


def constraints(T: Triangulation, d: int, r: int):
    """constraints(T: Triangulation, d: int, r: int)

    Finds the constraint matrices `H` and `G` and constraint vector `g` such that
    Hc = 0 and Gc = g, where c is the unrolled vector of B-form coefficients."""
    m = degree_to_size(d)
    n = T.num_triangles
    H = np.zeros((m * n, 3 * n * (d + 1)), dtype=np.float64)
    G = np.zeros((m * n, 3 * n * (d + 1)), dtype=np.int64)
    g = np.zeros(3 * n * (d + 1), dtype=np.float64)
    n_b = 0  # Number of boundary constraints
    n_i = 0  # Number of interior smoothness constraints
    for i in range(n):
        for v in range(3):  # Iterates over the edges v==1 ~ e12, v==2 ~ e23, v==3 ~ e31
            if T.topology[i, v] == 0:  # Boundary edge
                for k in traverse(v, d, 0):
                    n_b += 1
                    G[Bindex(i, k, m), n_b] = 1
                    g[n_b] = boundary_marker_to_fvalue(T.vertex_boundary_markers[T.triangles[v, i]])
            elif T.topology[i, v] > i:  # Interior edge, does not repeat triangles  # TODO: check
                opposite_tri = T.topology[i, v]
                v_tilde = findfirst(T.topology[:, opposite_tri] == i)
                odd_v = odd_vertex(v)
                odd_v_tilde = odd_vertex(v_tilde)

                lambda_1, lambda_2, lambda_3 = cartesian_to_barycentric(
                    T.coordinates[1, T.triangles[odd_v_tilde, opposite_tri]],
                    T.coordinates[2, T.triangles[odd_v_tilde, opposite_tri]],
                    T.coordinates[0, T.triangles[cycle(1, odd_v - 1), i]],
                    T.coordinates[1, T.triangles[cycle(1, odd_v - 1), i]],
                    T.coordinates[0, T.triangles[cycle(2, odd_v - 1), i]],
                    T.coordinates[1, T.triangles[cycle(2, odd_v - 1), i]],
                    T.coordinates[0, T.triangles[cycle(3, odd_v - 1), i]],
                    T.coordinates[1, T.triangles[cycle(3, odd_v - 1), i]]
                )
                for j in range(r):  # Iterate over smoothness levels
                    traversal = traverse(v, d, 0)
                    opposite_traversal = reversed(traverse(v_tilde, d, 0))
                    if j == 0:  # C0 smoothness conditions
                        for k in range(d + 1 - j):
                            n_i += 1
                            H[n_i, Bindex(i, traversal[k], m)] = 1.
                            H[n_i, Bindex(opposite_tri, opposite_traversal[k], m)] = -1.
                    if j == 1:  # C1 smoothness conditions
                        traversal_1 = traverse(v, d, 1)
                        opposite_traversal_1 = reversed(traverse(v_tilde, d, 1))
                        for k in range(d + 1 - j):
                            n_i += 1
                            H[n_i, Bindex(i, traversal_1[k], m)] = lambda_1
                            H[n_i, Bindex(i, traversal[k], m)] = lambda_2
                            H[n_i, Bindex(i, traversal[k + 1], m)] = lambda_3
                            H[n_i, Bindex(opposite_tri, opposite_traversal_1[k], m)] = -1.
    return H[1:n_i, :], G[1:n_b, :], g[1:n_b, :]


if __name__ == '__main__':
    d = 2
    m = degree_to_size(d)
    d_test = size_to_degree(m)
    print(f'd = {d}, m = {m}, d_test = {d_test}')

    indices = linear_indices(d)
    print(indices)

    build_casteljau(d)

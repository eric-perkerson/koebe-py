module BivariateSplines

using LinearAlgebra: dot
using SparseArrays
using ElectricalNetworks: Triangulation
using RecipesBase

# TODO: Replace the index location with saved data structures for performance

export barycentric, degree_to_size, size_to_degree, linear_indices, 
    casteljau, locate, spline_eval, bary_grid, 
    e12, e23, e31, constraints, Spline, x_derivative, y_derivative,
    inner_product_matrix, Bindex, stiffness_tensor, 
    build_D, build_casteljau, e12, uniform_triangles, sum_n, uniform_topology,
    traverse,
    traverse_triangles, refine_topology, refine_triangles, triangle_area,
    uniform_coordinates, refine_tri_coor_bdry, stiffness_matrix, uzawa

export refine_triangulation

"""triangle_area(v1::Vector{Float64}, v2::Vector{Float64}, v3::Vector{Float64})

Area of a triangle with vertices `v1`, `v2`, and `v3`."""
function triangle_area(v1::Vector{Float64}, v2::Vector{Float64}, v3::Vector{Float64})
    (-v2[1]*v1[2] + v3[1]*v1[2] + v1[1]*v2[2] - v3[1]*v2[2] - v1[1]*v3[2] + v2[1]*v3[2])/2.
end

"""barycentric(x::Float64, y::Float64, v1::Vector{Float64}, v2::Vector{Float64},
v3::Vector{Float64})

Computes the barycentric coordinates of the point (x, y) with
respect to the triangle with vertices v1, v2, and v3. Returns a
tuple (b1, b2, b3)"""
function barycentric(x::Float64,
                     y::Float64,
                     v1::Vector{Float64},
                     v2::Vector{Float64},
                     v3::Vector{Float64})

    x1 = v1[1]
    y1 = v1[2]
    x2 = v2[1]
    y2 = v2[2]
    x3 = v3[1]
    y3 = v3[2]

    A_2 = -x2*y1 + x3*y1 + x1*y2 - x3*y2 - x1*y3 + x2*y3
    b1 = ((x2 - x)*(y3 - y) - (x3 - x)*(y2 - y))/A_2
    b2 = ((x3 - x)*(y1 - y) - (x1 - x)*(y3 - y))/A_2
    b3 = ((x1 - x)*(y2 - y) - (x2 - x)*(y1 - y))/A_2
    return (b1, b2, b3)
end

function barycentric(x::Float64, y::Float64, triangle::Int, T::Triangulation)
    v1 = T.coordinates[:, T.triangles[1, triangle]]
    v2 = T.coordinates[:, T.triangles[2, triangle]]
    v3 = T.coordinates[:, T.triangles[3, triangle]]
    barycentric(x, y, v1, v2, v3)
end

"""Returns the number of coefficients on a triangle for the given degree"""
function degree_to_size(d::Int)
    div((d+1)*(d+2), 2)
end

"""The inverse of the `degree_to_size`"""
function size_to_degree(m::Int)
    Int((-3 + sqrt(9 - 4*2*(1-m)))/2)
end

"""Creates the linear indices using the pattern

      v1 = 1
          /  \\
         2     5
        /        \\
       3     6     8
      /              \\
v2 = 4-----7------9---10 = v3

"""
function linear_indices(d::Int)
    m = degree_to_size(d)
    result = zeros(Int, 3, m)
    counter = 1
    for iter = d:-1:0
        result[1, counter:counter+iter] = collect(iter:-1:0)
        result[2, counter:counter+iter] = collect(0:iter)
        result[3, counter:counter+iter] = (d - iter)*ones(Int, iter+1)
        counter += iter + 1
    end
    return result
end

function linear_indices_triangles()

end

function locate(triple::Vector{Int}, triple_list::Matrix{Int})
    for i = 1:size(triple_list, 2)
        if triple == triple_list[:, i]
            return i
        end
    end
    return 0
end

function build_casteljau(d::Int)
    m_old = degree_to_size(d)     # Number of old coefficients
    m_new = degree_to_size(d - 1) # Number of new coefficients
    ind_old = linear_indices(d)
    ind_new = linear_indices(d - 1)
    result = zeros(Int, 3, m_new)
    for j = 1:m_new
        result[1, j] = locate(ind_new[:, j] + [1, 0, 0], ind_old)
        result[2, j] = locate(ind_new[:, j] + [0, 1, 0], ind_old)
        result[3, j] = locate(ind_new[:, j] + [0, 0, 1], ind_old)
    end
    return result
end

"""Takes `b`, the barycentric coordinates of a point v and `B` the `m` B-form 
coefficients listed linearly in a vector according to `linear_indices` of the
polynomial to evaluate and performs one step of the de Casteljau algorithm, 
returning a new vector of B-form coefficients."""
function casteljau(b::Vector{Float64}, B::Vector{Float64})
    m_old = length(B)             # Number of old coefficients 
    d = size_to_degree(m_old)     # Inferred degree of the polynomial
    m_new = degree_to_size(d - 1) # Number of new coefficients 
    B_new = zeros(Float64, m_new)
    ind_new = linear_indices(d - 1)
    ind_old = linear_indices(d)
    for j = 1:m_new
        # TODO: hardcode/metacode ind's for speed for low level m's
        ind1 = locate(ind_new[:, j] + [1, 0, 0], ind_old)
        ind2 = locate(ind_new[:, j] + [0, 1, 0], ind_old)
        ind3 = locate(ind_new[:, j] + [0, 0, 1], ind_old)
        B_new[j] = B[ind1]*b[1] + B[ind2]*b[2] + B[ind3]*b[3]
    end
    return B_new
end

function spline_eval(b, B)
    m = length(B)
    while length(B) > 1
        B = casteljau(b, B)
    end
    return B[1]
end

struct Spline
    triangulation::Triangulation
    coefficients::Matrix{Float64}
end

@recipe function plot(spline; imagesize=(300, 300))
    seriestype := :surface
    fillalpha := .0
    legend := false
    framestyle := :none
    size --> imagesize
    make_x_y(region)
end

"""bary_grid(mesh_size::Int)

Builds the standard domain points on a triangle using barycentric coordinates"""
function bary_grid(mesh_size::Int)
    n = mesh_size - 1
    linear_indices(n) ./ n
end

"""sum_n(n::Int)

Sum of the first `n` natural numbers, i.e. n(n+1)/2"""
sum_n(n::Int) = div(n*(n+1), 2)

"""e12(degree::Int, offset::Int=0)

Gives the indices of the B-form coefficients on a line parallel to
the line from v1 to v2, with `offset` being the distance that line is 
from the edge e12."""
function e12(degree::Int, offset::Int=0)
    collect(range(sum_n(degree+1) - sum_n(degree+1-offset) + 1, step=1, length=degree+1-offset))
end

"""e23(degree::Int, offset::Int=0)

Gives the indices of the B-form coefficients on a line parallel to
the line from v2 to v3, with `offset` being the distance that line is 
from the edge e23."""
function e23(degree::Int, offset::Int=0)
    [sum_n(degree+1) - sum_n(k-1) - offset for k=degree+1:-1:1+offset]
end

"""e31(degree::Int, offset::Int=0)

Gives the indices of the B-form coefficients on a line parallel to
the line from v3 to v1, with `offset` being the distance that line is 
from the edge e31."""
function e31(degree::Int, offset::Int=0)
    [sum_n(degree+1) - sum_n(k) + 1 + offset for k = offset+1:degree+1]
end

"""boundary_marker_to_fvalue(b::Int)

Plays the role of the function g, but takes the boundary markers as its argument."""
function boundary_marker_to_fvalue(b::Int)
    if b > 1
        return(0.)
    elseif b == 1
        return(1.)
    else
        error("Invalid boundary marker")
    end
end

"""Bindex(triangle::Int, mindex::Int, m::Int)

Index in the B-form coefficient corresponding to `triangle` and `mindex`
which ranges from 1 to `m` for the `m` B-form coefficients on each triangle."""
function Bindex(triangle::Int, mindex::Int, m::Int)
    (triangle - 1)*m + mindex
end

"""traverse(index::Int, degree::Int, offset::Int=0)

Returns the indices needed to traverse the appropriate edge based on `edge` 
(either 1, 2, or 3 corresponding to the edges e12, e23, or e31) 
by passing the arguments to the functions e12, e23, or e31."""
function traverse(edge::Int, degree::Int, offset::Int=0)
    if edge == 1
        return e12(degree, offset)
    elseif edge == 2
        return e23(degree, offset)
    elseif edge == 3
        return e31(degree, offset)
    else
        error("Invalid index")
    end
end

"""odd_vertex(v::Int)

Each v-value represents the correspondence 1~e12, 2~e23, 3~e31. The function
returns the odd_vertex out, meaning the vertex not a part of that edge."""
function odd_vertex(v::Int)
    if v == 1
        return 3
    elseif v == 2
        return 1
    elseif v == 3
        return 2
    end
end

"""cycle(i::Int, v::Int)

Cycles the number `i` in the ordered triple (1, 2, 3) by v positions"""
function cycle(i::Int, v::Int)
    mod(i - 1 + v, 3) + 1
end

"""constraints(T::Triangulation, d::Int, r::Int)

Finds the constraint matrices `H` and `G` and constraint vector `g` such that 
Hc = 0 and Gc = g, where c is the unrolled vector of B-form coefficients."""
function constraints(T::Triangulation, d::Int, r::Int)
    m = degree_to_size(d)
    n = size(T.triangles, 2)
    H = zeros(Float64, 3*n*(d+1), m*n)
    G = zeros(Int, 3*n*(d+1), m*n)
    g = zeros(Float64, 3*n*(d+1))
    n_b = 0 # Number of boundary constraints
    n_i = 0 # Number of interior smoothness constraints
    for i = 1:size(T.triangles, 2)
        for v = 1:3 # Iterates over the edges v==1 ~ e12, v==2 ~ e23, v==3 ~ e31
            if T.topology[v, i] == 0 # Boundary edge
                for k = traverse(v, d, 0)
                    n_b += 1
                    G[n_b, Bindex(i, k, m)] = 1
                    g[n_b] = boundary_marker_to_fvalue(T.boundary_markers[T.triangles[v, i]])
                end
            elseif T.topology[v, i] > i # Interior edge, does not repeat triangles
                opposite_tri = T.topology[v, i]
                v_tilde = findfirst(T.topology[:, opposite_tri] .== i)
                odd_v = odd_vertex(v)
                odd_v_tilde = odd_vertex(v_tilde)
                λ1, λ2, λ3 = barycentric(
                    T.coordinates[1, T.triangles[odd_v_tilde, opposite_tri]],
                    T.coordinates[2, T.triangles[odd_v_tilde, opposite_tri]],
                    T.coordinates[:, T.triangles[cycle(1, odd_v - 1), i]],
                    T.coordinates[:, T.triangles[cycle(2, odd_v - 1), i]],
                    T.coordinates[:, T.triangles[cycle(3, odd_v - 1), i]]
                )
                for j = 0:r # Iterate over smoothness levels
                    traversal = traverse(v, d, 0)
                    opposite_traversal = reverse(traverse(v_tilde, d, 0))
                    if j == 0 # C0 smoothness conditions
                        for k = 1:d+1-j 
                            n_i += 1
                            H[n_i, Bindex(i, traversal[k], m)] = 1.
                            H[n_i, Bindex(opposite_tri, opposite_traversal[k], m)] = -1.
                        end
                    end
                    if j == 1 # C1 smoothness conditions
                        traversal_1 = traverse(v, d, 1)
                        opposite_traversal_1 = reverse(traverse(v_tilde, d, 1))
                        for k = 1:d+1-j 
                            n_i += 1
                            H[n_i, Bindex(i, traversal_1[k], m)] = λ1
                            H[n_i, Bindex(i, traversal[k], m)] = λ2
                            H[n_i, Bindex(i, traversal[k + 1], m)] = λ3
                            H[n_i, Bindex(opposite_tri, opposite_traversal_1[k], m)] = -1.
                        end
                    end
                end
            end
        end
    end
    return H[1:n_i, :], G[1:n_b, :], g[1:n_b, :]
end

"""Finds the directional derivative of the given B-form polynomial"""
function directional_derivative(v::Vector{Float64},
                                v1::Vector{Float64},
                                v2::Vector{Float64},
                                v3::Vector{Float64},
                                B::Vector{Float64},
                                d::Int,
                                m::Int)
    b1, b2, b3 = barycentric(v[1], v[2], v1, v2, v3) .- barycentric(0., 0., v1, v2, v3)
    d*casteljau([b1, b2, b3], B)
end

"""x_derivative(c::Vector{Float64}, T::Triangulation, d::Int, m::Int)

Finds the directional derivative of the given B-form polynomial"""
function x_derivative(c::Vector{Float64}, T::Triangulation, d::Int, m::Int)
    v = [1., 0.]
    n = size(T.triangles, 2)
    m_new = degree_to_size(d-1)
    Dx_c = zeros(Float64, n*m_new)
    for i = 1:n
        v1 = T.coordinates[:, T.triangles[1, i]]
        v2 = T.coordinates[:, T.triangles[2, i]]
        v3 = T.coordinates[:, T.triangles[3, i]]
        B = c[Bindex(i, 1, m):Bindex(i, m, m)]
        Dx_c[Bindex(i, 1, m_new):Bindex(i, m_new, m_new)] .= directional_derivative(v, v1, v2, v3, B, d, m)
    end
    return Dx_c
end

"""y_derivative(c::Vector{Float64}, T::Triangulation, d::Int, m::Int)

Finds the directional derivative of the given B-form polynomial"""
function y_derivative(c::Vector{Float64}, T::Triangulation, d::Int, m::Int)
    v = [0., 1.]
    n = size(T.triangles, 2)
    m_new = degree_to_size(d-1)
    Dy_c = zeros(Float64, n*m_new)
    for i = 1:n
        v1 = T.coordinates[:, T.triangles[1, i]]
        v2 = T.coordinates[:, T.triangles[2, i]]
        v3 = T.coordinates[:, T.triangles[3, i]]
        B = c[Bindex(i, 1, m):Bindex(i, m, m)]
        Dy_c[Bindex(i, 1, m_new):Bindex(i, m_new, m_new)] .= directional_derivative(v, v1, v2, v3, B, d, m)
    end
    return Dy_c
end

function inner_product_matrix(d::Int, m::Int)
    M = zeros(Int, m, m)
    ind = linear_indices(d)
    # denominator = (binomial(2d, d)*binomial(2d + 2, 2))
    for j = 1:m
        for i = j:m
            M[i, j] = binomial(ind[1, j] + ind[1, i], ind[1, j]) * 
                      binomial(ind[2, j] + ind[2, i], ind[2, j]) *
                      binomial(ind[3, j] + ind[3, i], ind[3, j]) # / denominator
            M[j, i] = M[i, j] # Faster not to check if off-diagonal
        end
    end
    return M
end

# function indicator(v::Vector{Int}, n::Int)
#     result = zeros(Int, n)
#     for i = 1:length(v)
#         result[v[i]] = 1
#     end
#     return result
# end

function build_D(v::Matrix{Int}, d::Tuple{Float64, Float64, Float64}, n::Int)
    k, m = size(v)
    result = zeros(Float64, n, m)
    for j = 1:m
        for i = 1:k
            result[v[i, j], j] = d[i]
        end
    end
    return result
end

"""Compute the stiffness matrix K as a tensor [K_1 ; K_2 ; ... ; K_n] for each of
the block diagonal matrices K_t for each triangle t = 1:n"""
function stiffness_tensor(T::Triangulation, d::Int, m::Int, index_struct::Matrix{Int})
    n = size(T.triangles, 2)
    K = zeros(Float64, m, m, n)
    M = inner_product_matrix(d-1, degree_to_size(d-1))
    for t = 1:n
        av = barycentric(0., 0., t, T)
        xv = barycentric(1., 0., t, T)
        yv = barycentric(0., 1., t, T)
        dx = xv .- av
        dy = yv .- av
        Dx = build_D(index_struct, dx, m)
        Dy = build_D(index_struct, dy, m)
        K[:, :, t] = Dx*M*Dx' + Dy*M*Dy'
    end
    return K
end

function stiffness_matrix(Kt)
    K = blockdiag([sparse(Kt[:, :, i]) for i = 1:size(Kt, 3)]...)
end

function uzawa_update!(c, K, uzawa_block, g_block, ϵ)
    c .= (K + (1/ϵ)uzawa_block*uzawa_block') \ (K*c + (1/ϵ)uzawa_block*g_block)
end

function uzawa(T, d, r, m, ϵ=.01, max_iters=500)
    c = zeros(m*size(T.triangles, 2))
    H, G, g = constraints(T, d, r) # Constraints are H*c = 0, G*c = g
    g = vec(g)
    # Hs = sparse(H)
    # Gs = sparse(G)
    # gs = sparse(g)

    index_struct = build_casteljau(d) # Sub-triangle struct by indices in standard ordering
    Kt = stiffness_tensor(T, d, m, index_struct) # Stiffness tensor K
    K = stiffness_matrix(Kt)

    uzawa_block = cat(G', H', dims=2)
    g_block = vcat(zeros(size(H, 1)), g)

    energy = []
    smoothness = []
    boundary = []
    for i = 1:max_iters
        uzawa_update!(c, K, uzawa_block, g_block, ϵ)
        push!(energy, c'*K*c)
        push!(smoothness, norm(H*c))
        push!(boundary, norm(G*c - g))
    end
    return c, energy, smoothness, boundary
end

"""Solves the dirichlet problem on the given triangulation `T` using a
spline of degree `d` with smoothness `r`. """
function solve_dirichlet(T::Triangulation, d::Int, r::Int)
    m = degree_to_size(d)
    n = size(T.triangles, 2)
    c = zeros(m*n)
    H, G, g = constraints(T, d, r)
    K = stiffness_matrix(T, )
    for i = 1:max_iter
        uzawa_update!(c, H, G, g)
    end
    return c
end

"""uniform_triangles(d::Int)

Generate the list of triangle indices relative to one particular triangle"""
function uniform_triangles(d::Int)
    triangles = zeros(Int, 3, d^2)
    top = e12(d, 0)
    bottom = e12(d, 1)
    counter = 1
    for i = 1:d # White triangle pass (think ∆-chessboard)
        triangles[1, counter] = top[i]
        triangles[2, counter] = top[i+1]
        triangles[3, counter] = bottom[i]
        counter += 1
    end
    for j = 2:d
        for i = 1:d-j+1 # Black triangle pass (think ∆-chessboard)
            triangles[1, counter] = top[i+1]
            triangles[2, counter] = bottom[i+1]
            triangles[3, counter] = bottom[i]
            counter += 1
        end
        top = bottom
        bottom = e12(d, j)
        for i = 1:d-j+1 # White triangle pass (think ∆-chessboard)
            triangles[1, counter] = top[i]
            triangles[2, counter] = top[i+1]
            triangles[3, counter] = bottom[i]
            counter += 1
        end
    end
    return triangles
end

"""Traverse the edge corresponding to `v` (v==1 ~ e12, v==2 ~ e23, v==3 ~ e31) with
refinement level `d` with all indices offset to correspond to triangle `t`."""
function traverse_triangles(v::Int, d::Int, t::Int=1)
    if v == 1
        return [i + (t-1)*d^2 for i = 1:d] # Triangle indices along edge e12
    elseif v == 2
        return [d^2 - i*(i - 1) + (t-1)*d^2 for i = d:-1:1] # Triangle indices along edge e23
    else
        return [d^2 + 1 - i^2 + (t-1)*d^2 for i = 1:d] # Triangle indices along edge e31
    end
end

"""uniform_topology(d::Int)

Creates the triangle topology for a single triangle with a uniform refinement corresponding
to degree `d`. Does NOT zero out boundary connections along edges e23 and e31. Use the 
functions zero23 and zero31 for this."""
function uniform_topology(d::Int)
    # c = [d^2 + 1 - i^2 for i = d:-1:1] # Triangle indices along edge e13
    c = reverse(traverse_triangles(3, d, 1)) # Triangle indices along edge e13
    hcat(vcat(zeros(Int, 1, d), collect(d+1:2d)', collect(d:2d-1)'), # First half-pass, all white ∆ on ∆-chessboard
        [vcat(collect(range(c[j]+1, length=2(d-j)))', # jth pass, black tris then white tris
            collect(range(c[j+1], length=2(d-j)))',
            vcat(collect(range(c[j], length=d-j)),
                collect(range(c[j+1]+d-1-j, length=d-j))
                )'
            ) for j = 1:d-1]...
        )
end

function refine_topology(T::Triangulation, d::Int)
    n = size(T.triangles, 2)
    topology_list = [uniform_topology(d) .+ d^2*(i-1) for i = 1:n] # for all triangles
    for i = 1:n
        for v = 1:3 # Iterates over the edges v==1 ~ e12, v==2 ~ e23, v==3 ~ e31
            if T.topology[v, i] == 0 # Boundary edge
                topology_list[i][v, traverse_triangles(v, d)] .= 0
            else # Interior edge
                opposite_tri = T.topology[v, i]
                opposite_edge = findfirst(T.topology[:, opposite_tri] .== i)
                topology_list[i][v, traverse_triangles(v, d)] = reverse(traverse_triangles(opposite_edge, d, opposite_tri))
                topology_list[opposite_tri][opposite_edge, traverse_triangles(opposite_edge, d)] = reverse(traverse_triangles(v, d, i))
            end
        end
    end
    topology = hcat(topology_list...)
    return topology
end

"""uniform_coordinates(d::Int)

Generates the barycentric coordinates for a single triangle with uniform refinement
of degree `d`."""
function uniform_coordinates(d::Int)
    linear_indices(d) ./ d
end

function refine_tri_coor_bdry(T::Triangulation, d::Int, m::Int)
    n_triangles = size(T.triangles, 2)
    n_vertices = size(T.coordinates, 2)
    uniform_triangles_mat = uniform_triangles(d)
    uniform_coordinates_mat = uniform_coordinates(d)
    old2new = zeros(Int, n_vertices) # Lookup table for which new indices correspond to the old ones
    used_vertices = falses(n_vertices) # If ith vertex has been used, used_vertices[i] == True
    tri_block_size = d^2 # Number of new triangles per old triangle

    old_vertex_locations = [1, d+1, m] # Within a particular triangle, the old vertex locations
    # inverse is used to look up linear indices in each triangle block to reverse the triangles-to-traverse vertices
    vec_mat = vec(uniform_triangles_mat) # Unrolled so findfirst returns an int instead of CartesianIndex
    inverse = hcat([[findfirst(vec_mat .== i) for i = traverse(v, d)] for v = 1:3]...)
    
    triangles_ = zeros(Int, 3, tri_block_size*n_triangles)
    coordinates_ = zeros(Float64, 2, sum_n(d+1)*n_triangles)
    boundary_markers_ = zeros(Int, sum_n(d+1)*n_triangles)
    current_vertex = 0
    for i = 1:n_triangles # i iterates over all triangles in T
        current_triangle_linear_indices = zeros(Int, m) # Linear indices of recycled vertices AND new vertices
        tri_used_vertices = used_vertices[T.triangles[:, i]] # Which vertices have been used?
        tri_used_edges = (T.topology[:, i] .< i) .& (T.topology[:, i] .!= 0)
        for v = 1:3
            # Update the used vertices
            tri_used_vertices[v] && (current_triangle_linear_indices[old_vertex_locations[v]] = old2new[T.triangles[v, i]])

            # Update the used edges
            if tri_used_edges[v]
                opposite_tri = T.topology[v, i]
                v_tilde = findfirst(T.topology[:, opposite_tri] .== i)
                current_triangle_linear_indices[traverse(v, d)] = reverse(triangles_[:, (opposite_tri-1)*tri_block_size+1:opposite_tri*tri_block_size][inverse[:, v_tilde]])
            end
        end
        
        # Add in new vertices
        tri_coordinates = T.coordinates[:, T.triangles[:, i]] * uniform_coordinates_mat
        for j = 1:m
            if current_triangle_linear_indices[j] == 0
                current_vertex += 1
                current_triangle_linear_indices[j] = current_vertex
                coordinates_[:, current_vertex] = tri_coordinates[:, j]
            end
        end

        # Compute the boundary markers
        for v = 1:3 # Iterates over the edges v==1 ~ e12, v==2 ~ e23, v==3 ~ e31
            if T.topology[v, i] == 0 # If e is a boundary edge
                boundary_markers_[current_triangle_linear_indices[traverse(v, d)]] .= T.boundary_markers[T.triangles[v, i]]
            end
        end

        used_vertices[T.triangles[:, i]] .= true
        old2new[T.triangles[:, i]] = current_triangle_linear_indices[old_vertex_locations]
        triangles_[:, (i-1)*tri_block_size+1:i*tri_block_size] = current_triangle_linear_indices[uniform_triangles_mat]
    end
    return triangles_, coordinates_[:, 1:current_vertex], boundary_markers_[1:current_vertex]
end

"""refine_triangulation(T::Triangulation, d::Int) refines a triangulation `T` """
function refine_triangulation(T::Triangulation, d::Int)
    m = degree_to_size(d)
    triangles, coordinates, boundary_markers = refine_tri_coor_bdry(T, d, m)
    topology = refine_topology(T, d)
    T = Triangulation(coordinates, triangles, boundary_markers, topology)
end

end # module
module TriVor

using RecipesBase
using StatsBase: mean
using Formatting

export Region, Triangulation, acute_triangulate, refine_triangulation, write_triangulation

mutable struct Region
    coordinates::Array{Float64, 2}
    exterior_vertices::Vector{Int}
    interior_vertices_by_hole::Array{Vector{Int}, 1}
    points_in_holes::Array{Float64, 2}
    function Region(c::Array{Float64, 2}, e::Vector{Int}, i::Array{Vector{Int}, 1})
        if size(c, 1) != 2
            error("Coordinates should be a 2 x n matrix")
        end

        if size(c, 2) != length(e) + sum(map(length, i))
            error("Number of coordinates given does not match the number of exterior and interior vertices given")
        end

        all_vertices = sort(cat(e, i..., dims=1))
        n = length(all_vertices)
        if all_vertices != collect(range(1, n, step=1))
            error("exterior_vertices and interior_vertices_by_hole should have indices 1 through n")
        end

        num_holes = length(i)
        points_in_holes = zeros(Float64, 2, num_holes)
        for j = 1:num_holes
            points_in_holes[:, j] = mapslices(mean, c[:, i[j]], dims=2)
        end

        new(c, e, i, points_in_holes)
    end
end

Region(c::Array{Array{Float64, 1}, 1}, e::Vector{Int}, i::Array{Vector{Int}, 1}) = Region(cat(c..., dims=2), e, i)

"""Build the x and y series for plotting using Plots. Insert NaNs
for discontinuous jumps between components."""
function build_plot_coordinates(region::Region)
    n_exterior = length(region.exterior_vertices)
    n_interior = sum(map(length, region.interior_vertices_by_hole))
    num_holes = length(region.interior_vertices_by_hole)
    n = n_exterior + n_interior + num_holes + 1
    x = zeros(Float64, n)
    y = zeros(Float64, n)
    counter = 1
    for i = 1:n_exterior
        x[counter] = region.coordinates[1, region.exterior_vertices[i]]
        y[counter] = region.coordinates[2, region.exterior_vertices[i]]
        counter += 1
    end
    x[counter] = NaN
    y[counter] = NaN
    counter += 1
    for j = 1:num_holes
        for i = 1:length(region.interior_vertices_by_hole[j])
            x[counter] = region.coordinates[1, region.interior_vertices_by_hole[j][i]]
            y[counter] = region.coordinates[2, region.interior_vertices_by_hole[j][i]]
            counter += 1
        end
        x[counter] = NaN
        y[counter] = NaN
        counter += 1
    end
    return (x, y)
end

@recipe function plot(region::Region; size = (300, 300))
    seriestype := :shape
    fillalpha := .0
    legend := false
    framestyle := :none
    size --> size
    build_plot_coordinates(region)
end

"""edges_wrap(polygon::Vector{Int}) takes a list of indices and returns
the list of pairs from one index to the next."""
function edges_wrap(polygon::Vector{Int})
    n = length(polygon)
    result = zeros(Int, 2, n)
    for i = 1:n-1
        result[:, i] = polygon[i:i+1]
    end
    result[1, n] = polygon[n]
    result[2, n] = polygon[1]
    return result
end

function region_edges(region::Region)
    edges = edges_wrap(region.exterior_vertices)
    boundary_markers = fill(1, size(edges, 2))
    num_holes = length(region.interior_vertices_by_hole)
    for i = 1:num_holes
        new_edges = edges_wrap(region.interior_vertices_by_hole[i])
        edges = hcat(edges, new_edges)
        boundary_markers = vcat(boundary_markers, fill(i+1, size(new_edges, 2)))
    end
    return edges, boundary_markers
end

"""make_poly_file(file_name::AbstractString, R::Region) takes the given region and
writes a .poly file that can be used by triangle.c and acute.c. It
assumes dimensions is 2, n_attr is 0, and n_boundary_markers is 0."""
function make_poly_file(file_name::AbstractString, region::Region)
    fmt = "%3.15f"
    fmtrfunc = generate_formatter(fmt)
    dimensions = 2
    n_attr = 0
    # vertices = region.coordinates # round and use "vertices" instead of "region.coordinates" later if necessary
    edges, boundary_markers = region_edges(region)
    num_holes = length(region.interior_vertices_by_hole)
    n_boundary_markers = num_holes + 1
    n = size(region.coordinates, 2)
    stream = open(file_name * ".poly", "w")
    write(stream, string(n) * " " * string(dimensions) * " " * string(n_attr) * " " * string(n_boundary_markers) * "\n")
    for i = 1:n
        write(stream, string(i) * " " * fmtrfunc(region.coordinates[1, i]) * " " * fmtrfunc(region.coordinates[2, i]) * "\n")
    end
    write(stream, string(n) * " " * string(n_boundary_markers) * "\n")
    for i = 1:size(edges, 2)
        write(stream, string(i) * " " * string(edges[1, i]) * " " * string(edges[2, i]) * " " * string(boundary_markers[i]) * "\n")
    end
    write(stream, string(num_holes) * "\n")
    for i = 1:num_holes
        write(stream, string(i) * " " * fmtrfunc(region.points_in_holes[1, i]) * " " * fmtrfunc(region.points_in_holes[2, i]) * "\n")
    end
    close(stream)
end

function read_poly_file(file_name)
    if file_name[end - 4:end] == ".poly"
        stream = open(file_name, "r")
    else
        stream = open(file_name * ".poly", "r")
    end

    # Parse the vertices/coordinates section of the .poly file
    header = readline(stream)
    num_vertices, _, _, num_bdry_markers = map(x -> parse(Int, x), split(header))
    vertices = zeros(2, num_vertices)
    vertex_bdry_markers = zeros(num_vertices)
    for i in 1:num_vertices
        line = readline(stream)
        _, x_str, y_str, bdry_marker_str = split(line)
        vertices[1, i] = parse(Float64, x_str)
        vertices[2, i] = parse(Float64, y_str)
        vertex_bdry_markers[i] = parse(Int, bdry_marker_str)
    end

    # Parse the edges/segments section of the .poly file
    header = readline(stream)
    num_edges, num_bdry_markers = map(x -> parse(Int, x), split(header))
    edges = zeros(Int, 2, num_edges)
    edge_bdry_markers = zeros(Int, num_edges)
    for i in 1:num_edges
        line = readline(stream)
        _, e1, e2, bdry_marker = map(x -> parse(Int, x), split(line))
        edges[1, i] = e1
        edges[2, i] = e2
        edge_bdry_markers[i] = bdry_marker
    end

    header = readline(stream)
    num_holes = parse(Int, header)
    holes = zeros(Float64, 2, num_holes)
    for i in 1:num_holes
        line = readline(stream)
        _, x_str, y_str = split(line)
        holes[1, i] = parse(Float64, x_str)
        holes[2, i] = parse(Float64, y_str)
    end
    return vertices, vertex_bdry_markers, edges, edge_bdry_markers, holes
end

"""acute_triangulate(region::Region; delete_files=true) makes an acute triangulation of a given region."""
function acute_triangulate(file::AbstractString; delete_files=true)
    # file = "acute_triangulation"
    # make_poly_file(file, region)
    # chmod(file * ".poly", 0o777)

    if !isfile("acute")
        error("The aCute program was not found")
    end

    acute = run(`./acute -q35 -U89 $(file)`)
    if acute.exitcode != 0
        error("Error in aCute")
    end

    file_stem = file[1:end-5]
    n_flags = 2
    n_attributes = 0
    _, _, _, _, holes = read_poly_file(file)
    num_holes = size(holes, 2)
    n_boundary_markers = num_holes + 1
    raw_node = readlines(file_stem * ".1.node")
    clean_node = map(x->split(x, " ", keepempty=false), raw_node)
    raw_ele = readlines(file_stem * ".1.ele")
    clean_ele = map(x->split(x, " ", keepempty=false), raw_ele)

    n_points = parse(Int, split(raw_node[1], " ", keepempty=false)[1])
    vertices = zeros(Float64, 2, n_points)
    boundary_markers = zeros(Int, n_points)
    for i = 1:n_points
        vertices[:, i] = map(x->parse(Float64, x), clean_node[i + 1][2:3])
        boundary_markers[i] = parse(Int, clean_node[i + 1][4])
    end

    n_triangles = parse(Int, split(raw_ele[1], " ", keepempty=false)[1])
    triangles = zeros(Int, 3, n_triangles)
    for i = 1:n_triangles
        triangles[:, i] = map(x->parse(Int, x), clean_ele[i + 1][2:4])
    end

    if delete_files
        rm(joinpath(pwd(), file_stem * ".1.node"))
        rm(joinpath(pwd(), file_stem * ".1.ele"))
        rm(joinpath(pwd(), file_stem * ".1.poly"))
    end
    return (vertices, triangles, boundary_markers)
end

mutable struct Triangulation
    coordinates::Matrix{Float64}
    triangles::Matrix{Int}
    boundary_markers::Vector{Int}
    topology::Matrix{Int}
    edges::Matrix{Int}
    edges_sorted_unique::Matrix{Int}
    function Triangulation(coordinates::Matrix{Float64}, triangles::Matrix{Int}, boundary_markers::Vector{Int})
        topology = triangulation_topology_builder(triangles)
        new(coordinates, triangles, boundary_markers, topology)
    end
    function Triangulation(coordinates::Matrix{Float64}, triangles::Matrix{Int}, boundary_markers::Vector{Int}, topology::Matrix{Int})
        new(coordinates, triangles, boundary_markers, topology)
    end
end

"""Build the x and y series for plotting using Plots. Insert NaNs
for discontinuous jumps between components."""
function build_plot_coordinates(triangulation::Triangulation)
    n_triangles = size(triangulation.triangles, 2)
    n = 4*n_triangles
    x = zeros(Float64, n)
    y = zeros(Float64, n)
    counter = 1
    for j = 1:n_triangles
        for i = 1:3
            x[counter] = triangulation.coordinates[1, triangulation.triangles[i, j]]
            y[counter] = triangulation.coordinates[2, triangulation.triangles[i, j]]
            counter += 1
        end
        x[counter] = NaN
        y[counter] = NaN
        counter += 1
    end
    return (x, y)
end

@recipe function plot(triangulation::Triangulation; size = (300, 300), fillalpha = 0.5)
    seriestype := :shape
    fillalpha --> fillalpha
    legend := false
    framestyle := :none
    size --> size
    build_plot_coordinates(triangulation)
end

"""edges_wrap_triangle(triangle::Vector{Int}) takes a list of indices and returns
the list of pairs from one index to the next."""
function edges_wrap_triangle(t::Vector{Int})
    Int[t[1] t[2] t[3] ; t[2] t[3] t[1]]
end

"""triangulation_topology_builder(triangles::Matrix{Int})
returns a list of which triangles are connected to which across
which edges."""
function triangulation_topology_builder(triangles::Matrix{Int})
    n = size(triangles, 2)
    edges = zeros(Int, 2, 3n)
    topology = zeros(Int, 3, n)
    @inline edges_index(triangle, edge) = 3*(triangle - 1) + edge
    for i = 1:n
        edges[:, 3(i-1)+1:3i] = edges_wrap_triangle(triangles[:, i])
    end
    for i = 1:n
        for k = 1:3
            if topology[k, i] != 0
                continue
            end
            current_edge = reverse(edges[:, edges_index(i, k)]) # Triangles are neighbors
            for j = 1:n
                for m = 1:3
                    if current_edge[1] == edges[1, edges_index(j, m)] && current_edge[2] == edges[2, edges_index(j, m)]
                        topology[k, i] = j
                        topology[m, j] = i
                    end
                end
            end
        end
    end
    return topology
end

"""Returns the number of coefficients on a triangle for the given degree"""
function degree_to_size(d::Int)
    div((d+1)*(d+2), 2)
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

"""sum_n(n::Int)

Sum of the first `n` natural numbers, i.e. n(n+1)/2"""
function sum_n(n::Int)
    div(n*(n+1), 2)
end

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
to degree `d`. Does NOT zero out boundary connections along edges e23 and e31."""
function uniform_topology(d::Int)
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
    refined_topology = hcat(topology_list...)
    return refined_topology
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
    old_to_new = zeros(Int, n_vertices) # Lookup table for which new indices correspond to the old ones
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
            tri_used_vertices[v] && (current_triangle_linear_indices[old_vertex_locations[v]] = old_to_new[T.triangles[v, i]])

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
        old_to_new[T.triangles[:, i]] = current_triangle_linear_indices[old_vertex_locations]
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

function write_triangulation_vertices(file_stem::AbstractString, T::Triangulation)
    fmt = "%3.15f"
    fmtrfunc = generate_formatter(fmt)

    num_vertices = size(T.coordinates, 2)
    dimension = 2
    num_attributes = 0
    num_bdry_markers = 1
    print("Writing vertices to $(pwd() * file_stem * ".node")", "\n")
    open(file_stem * ".node", "w") do stream
        write(stream, string(num_vertices) * " " * string(dimension) * " " * string(num_attributes) * " " * string(num_bdry_markers) * "\n")
        for i = 1:num_vertices
            write(stream, string(i) * " " * fmtrfunc(T.coordinates[1, i]) * " " * fmtrfunc(T.coordinates[2, i]) * " " * string(T.boundary_markers[i]) * "\n")
        end
    end
end

function write_triangulation_triangles(file_stem::AbstractString, T::Triangulation)
    num_triangles = size(T.triangles, 2)
    nodes_per_triangle = 3
    num_attributes = 0
    open(file_stem * ".ele", "w") do stream
        write(stream, string(num_triangles) * " " * string(nodes_per_triangle) * " " * string(num_attributes) * "\n")
        for i = 1:num_triangles
            write(stream, string(i) * " " * string(T.triangles[1, i]) * " " * string(T.triangles[2, i]) * " " * string(T.triangles[3, i]) * "\n")
        end
    end
end

function write_triangulation_topology(file_stem::AbstractString, T::Triangulation)
    num_triangles = size(T.topology, 2)
    nodes_per_triangle = 3
    num_attributes = 0
    open(file_stem * ".topo.ele", "w") do stream
        write(stream, string(num_triangles) * " " * string(nodes_per_triangle) * " " * string(num_attributes) * "\n")
        for i = 1:num_triangles
            write(stream, string(i) * " " * string(T.topology[1, i]) * " " * string(T.topology[2, i]) * " " * string(T.topology[3, i]) * "\n")
        end
    end
end

function write_triangulation(file_stem::AbstractString, T::Triangulation)
    write_triangulation_vertices(file_stem, T)
    write_triangulation_triangles(file_stem, T)
    write_triangulation_topology(file_stem, T)
end

end # module
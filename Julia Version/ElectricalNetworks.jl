module ElectricalNetworks

using StatsBase: mean
using LinearAlgebra: det
using RecipesBase
using Formatting
# using Plots

# TODO: Nonconvex constructor for points_in_holes
#        Check if rounding is necessary in writing .poly files
#        Bug in boundary_classes

export Region, Triangulation, acute_triangulate,
interior_and_boundary_edges, boundary_classes, chain_seeded, new_boundary_marker,
edge_builder!, vertex_topology_builder, circumcenter

# Helper functions
function unique_col(A::Matrix{T}) where {T <: Real}
    m, n = size(A)
    if n == 0
        return(A)
    end
    result = zeros(T, size(A))
    result[:, 1] = A[:, 1]
    counter = 1
    for j = 1:n # Iterate over columns of A
        repeated_flag = false
        for i = 1:counter # Iterate over current result
            if A[:, j] == result[:, i]
                repeated_flag = true
                break
            end
        end
        if repeated_flag
            continue
        else
            result[:, counter+1] = A[:, j]
            counter += 1
        end
    end
    return result[:, 1:counter]
end

struct Region
    coordinates::Array{Float64, 2}
    exterior_vertices::Vector{Int}
    interior_vertices_by_hole::Array{Vector{Int}, 1}
    points_in_holes::Array{Float64, 2}
    n_holes::Int
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

        n_holes = length(i)
        points_in_holes = zeros(Float64, 2, n_holes)
        for j = 1:n_holes
            points_in_holes[:, j] = mapslices(mean, c[:, i[j]], dims=2)
        end

        new(c, e, i, points_in_holes, n_holes)
    end
end

Region(c::Array{Array{Float64, 1}, 1}, e::Vector{Int}, i::Array{Vector{Int}, 1}) = Region(cat(c..., dims=2), e, i)

"""Build the x and y series for plotting using Plots. Insert NaNs
for discontinuous jumps between components."""
function make_x_y(region::Region)
    n_exterior = length(region.exterior_vertices)
    n_interior = sum(map(length, region.interior_vertices_by_hole))
    n = n_exterior + n_interior + region.n_holes + 1
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
    for j = 1:region.n_holes
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
    make_x_y(region)
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
    for i = 1:region.n_holes
        new_edges = edges_wrap(region.interior_vertices_by_hole[i])
        edges = hcat(edges, new_edges)
        boundary_markers = vcat(boundary_markers, fill(i+1, size(new_edges, 2)))
    end
    return edges, boundary_markers
end

"""edges_wrap_triangle(triangle::Vector{Int}) takes a list of indices and returns
the list of pairs from one index to the next."""
function edges_wrap_triangle(t::Vector{Int})
    Int[t[1] t[2] t[3] ; t[2] t[3] t[1]]
end

# edges_wrap_sorted(polygon) = mapslices(sort, edges_wrap(polygon), dims=1)

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
    n_boundary_markers = region.n_holes + 1
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
    write(stream, string(region.n_holes) * "\n")
    for i = 1:region.n_holes
        write(stream, string(i) * " " * fmtrfunc(region.points_in_holes[1, i]) * " " * fmtrfunc(region.points_in_holes[2, i]) * "\n")
    end
    close(stream)
end

"""acute_triangulate(region::Region; delete_files=true) makes an acute triangulation of a given region."""
function acute_triangulate(region::Region; delete_files=true)
    file = "acute_triangulation"
    make_poly_file(file, region)
    chmod(file * ".poly", 0o777)
    if !isfile("acute")
        error("The aCute program was not found")
    end

    acute = run(`./acute -q35 -U89 $(joinpath(pwd(), file * ".poly"))`)
    if acute.exitcode != 0
        error("Error in aCute")
    end

    n_flags = 2
    n_attributes = 0
    n_boundary_markers = region.n_holes + 1
    raw_node = readlines(file * ".1.node")
    clean_node = map(x->split(x, " ", keepempty=false), raw_node)
    raw_ele = readlines(file * ".1.ele")
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
        rm(joinpath(pwd(), file * ".1.node"))
        rm(joinpath(pwd(), file * ".1.ele"))
        rm(joinpath(pwd(), file * ".1.poly"))
        rm(joinpath(pwd(), file * ".poly"))
    end
    return (vertices, triangles, boundary_markers)
end

function vertex_topology_builder(n_vertices::Int, edges::Matrix{Int})
    n_edges = size(edges, 2) # Number of edges
    n_neighbors_list = zeros(Int, n_vertices) # Number of neighbors of each vertex
    max_valence_initial = 50 # Maximum possible valence to allocate memory for
    topology = zeros(Int, max_valence_initial, n_vertices)
    for i = 1:n_edges
        v1 = edges[1, i]
        v2 = edges[2, i]
        n_neighbors_list[v1] += 1
        n_neighbors_list[v2] += 1
        topology[n_neighbors_list[v1], v1] = v2
        topology[n_neighbors_list[v2], v2] = v1
    end
    max_valence_actual = maximum(n_neighbors_list)
    return topology[1:max_valence_actual, 1:n_vertices]
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

"""Build a datastructure where the ith row is all vertices connected
by an edge to vertex i"""
function vertex_topology_builder(T::Triangulation)
    vertex_topology_builder(size(T.coordinates, 2), T.edges)
end

function edge_builder!(T::Triangulation)
    triangles = T.triangles
    n_triangles = size(T.triangles, 2)
    edges = zeros(Int, 2, 3*n_triangles)
    for i = 1:n_triangles
        edges[:, 3(i-1)+1:3i] = edges_wrap(triangles[:, i])
    end
    # edges_sorted_unique = mapslices(sort, edges, dims=1)
    # edges_sorted_unique = unique_col(edges)
    T.edges = edges
    # T.edges_sorted_unique = edges_sorted_unique
end

"""Build the x and y series for plotting using Plots. Insert NaNs
for discontinuous jumps between components."""
function make_x_y(triangulation::Triangulation)
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
    make_x_y(triangulation)
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

"""new_boundary_marker(b1::Int, b2::Int)

Computes the new boundary marker to be placed on the new vertex between the
two vertices with the boundary markers `b1` and `b2`. Remember that 0 denotes
interior points, so `new_boundary_marker` should only return non-zero values
for matching non-zero values."""
function new_boundary_marker(b1::Int, b2::Int)
    if b1 == 0 || b2 == 0
        return(0)
    elseif b1 == b2
        return(b1)
    else
        return(0)
    end
end

"""refine_triangles(triangulation::Triangulation)
uses the Mitsuuroko (i.e. Triforce shape) refinement in order to preserve the property that all triangles are acute,
in which each triangle in the starting triangulation is replaced with four triangles, with the new vertices
at the midpoints of the edges of the original triangle."""
function refine_triangles(T::Triangulation)
    n_vertices = size(T.coordinates, 2)
    n = size(T.triangles, 2)
    new_vertex_1 = 0
    new_vertex_2 = 0
    new_vertex_3 = 0
    new_triangles = zeros(Int, 3, 4n) # New triangle list. This will be returned joined with new_vertices_record
    new_vertices_record = zeros(Int, 3, n) # Needed to keep track of which new vertices have already been created and for which tri
    LUT = zeros(Int, 3, 3n) # Look Up Table: newVertex, TCoordinates of edge to calculate midpoint later
    newest_vertex = n_vertices # Keeps track of the most recent vertex created
    new_boundary_markers = zeros(Int, 3n)

    for i = 1:n
        if T.topology[1, i] != 0 && T.topology[1, i] < i # Check if the vertex for this edge has already been created
            connected_triangle = T.topology[1, i]
            connected_edge = findfirst(T.topology[:, connected_triangle] .== i) # Either 1 for E12, 2 for E23, or 3 for E31
            new_vertex_1 = new_vertices_record[connected_edge, connected_triangle] # Looks up the previously created vertex
        else # creates new vertex if not already created
            new_vertex_1 = newest_vertex + 1
            new_vertices_record[1, i] = new_vertex_1
            newest_vertex = new_vertex_1
            LUT[1, newest_vertex - n_vertices] = newest_vertex
            LUT[2, newest_vertex - n_vertices] = T.triangles[1, i]
            LUT[3, newest_vertex - n_vertices] = T.triangles[2, i]
            if T.topology[1, i] == 0
                new_boundary_markers[newest_vertex - n_vertices] = new_boundary_marker(
                    T.boundary_markers[T.triangles[1, i]], T.boundary_markers[T.triangles[2, i]])
            else
                new_boundary_markers[newest_vertex - n_vertices] = 0
            end
        end
        if T.topology[2, i] != 0 && T.topology[2, i] < i
            connected_triangle = T.topology[2, i]
            connected_edge = findfirst(T.topology[:, connected_triangle] .== i) # Either 1 for E12, 2 for E23, or 3 for E31
            new_vertex_2 = new_vertices_record[connected_edge, connected_triangle] # Looks up the previously created vertex
        else
            new_vertex_2 = newest_vertex + 1
            new_vertices_record[2, i] = new_vertex_2
            newest_vertex = new_vertex_2
            LUT[1, newest_vertex - n_vertices] = newest_vertex
            LUT[2, newest_vertex - n_vertices] = T.triangles[2, i]
            LUT[3, newest_vertex - n_vertices] = T.triangles[3, i]
            if T.topology[2, i] == 0
                new_boundary_markers[newest_vertex - n_vertices] = new_boundary_marker(
                    T.boundary_markers[T.triangles[2, i]], T.boundary_markers[T.triangles[3, i]])
            else
                new_boundary_markers[newest_vertex - n_vertices] = 0
            end
        end
        if T.topology[3, i] != 0 && T.topology[3, i] < i
            connected_triangle = T.topology[3, i]
            connected_edge = findfirst(T.topology[:, connected_triangle] .== i) # Either 1 for E12, 2 for E23, or 3 for E31
            new_vertex_3 = new_vertices_record[connected_edge, connected_triangle] # Looks up the previously created vertex
        else
            new_vertex_3 = newest_vertex + 1
            new_vertices_record[3, i] = new_vertex_3
            newest_vertex = new_vertex_3
            LUT[1, newest_vertex - n_vertices] = newest_vertex
            LUT[2, newest_vertex - n_vertices] = T.triangles[3, i]
            LUT[3, newest_vertex - n_vertices] = T.triangles[1, i]
            if T.topology[3, i] == 0
                new_boundary_markers[newest_vertex - n_vertices] = new_boundary_marker(
                    T.boundary_markers[T.triangles[3, i]], T.boundary_markers[T.triangles[1, i]])
            else
                new_boundary_markers[newest_vertex - n_vertices] = 0
            end
        end

        new_triangles_index = 4*i - 3 # Starting index for inserting new triangles

        new_triangles[1, new_triangles_index] = T.triangles[1, i]
        new_triangles[2, new_triangles_index] = new_vertex_1
        new_triangles[3, new_triangles_index] = new_vertex_3
        new_triangles[1, new_triangles_index + 1] = T.triangles[2, i]
        new_triangles[2, new_triangles_index + 1] = new_vertex_2
        new_triangles[3, new_triangles_index + 1] = new_vertex_1
        new_triangles[1, new_triangles_index + 2] = T.triangles[3, i]
        new_triangles[2, new_triangles_index + 2] = new_vertex_3
        new_triangles[3, new_triangles_index + 2] = new_vertex_2
        new_triangles[1, new_triangles_index + 3] = new_vertex_1
        new_triangles[2, new_triangles_index + 3] = new_vertex_2
        new_triangles[3, new_triangles_index + 3] = new_vertex_3
    end
    return (newest_vertex, new_triangles, LUT, new_boundary_markers[1:newest_vertex - n_vertices])
end

"""refine_coordinates[{{coordinates,_Real,2},{n_vertices,_Integer},{LUT,_Integer,2}}] creates a
list of vertices coordinates for the refined triangulation. LUT stands for 'Look Up Table' for finding the new vertices."""
function refine_coordinates(old_coor::Matrix{Float64},
                                       n_all::Int,
                                       LUT::Matrix{Int})

    n_old = size(old_coor, 2)
    n_new = n_all - n_old
    all_coor = zeros(Float64, 2, n_all)
    for i = 1:n_old
        all_coor[1, i] = old_coor[1, i]
        all_coor[2, i] = old_coor[2, i]
    end
    for i = 1:n_new
        all_coor[1, i + n_old] = (old_coor[1, LUT[2, i]] + old_coor[1, LUT[3, i]])/2.
        all_coor[2, i + n_old] = (old_coor[2, LUT[2, i]] + old_coor[2, LUT[3, i]])/2.
    end
    return all_coor
end

# """refine_triangulation(T::Triangulation) uses the Mitsuuroko refinement
# (in order to preserve the property that all triangles are acute), in
# which each triangle in the starting triangulation is replaced with
# four triangles, with the new vertices at the midpoints of the edges
# of the original triangle."""
# function refine_triangulation(T::Triangulation)
#     n_new_vertices, all_triangles, LUT, new_boundary_markers = refine_triangles(T)
#     all_coor = refine_coordinates(T.coordinates, n_new_vertices, LUT)
#     all_boundary_markers = vcat(T.boundary_markers, new_boundary_markers)
#     Triangulation(all_coor, all_triangles, all_boundary_markers)
# end

"""InteriorAndBoundaryEdges(T::Triangulation)
creates a list of interior edges and a list of boundary edges."""
function interior_and_boundary_edges_table(T::Triangulation)
    n_triangles = size(T.triangles, 2)
    table = zeros(Int, 3, 3*n_triangles)
    counter = 0
    for i = 1:n_triangles
        for j = 1:2
            counter += 1
            if T.topology[j, i] == 0
                table[1, counter] = T.triangles[j, i]
                table[2, counter] = T.triangles[j + 1, i]
                table[3, counter] = 0
            else
                table[1, counter] = T.triangles[j, i]
                table[2, counter] = T.triangles[j + 1, i]
                table[3, counter] = 1
            end
        end
        j = 3
        counter += 1
        if T.topology[j, i] == 0
            table[1, counter] = T.triangles[j, i]
            table[2, counter] = T.triangles[1, i]
            table[3, counter] = 0
        else
            table[1, counter] = T.triangles[j, i]
            table[2, counter] = T.triangles[1, i]
            table[3, counter] = 1
        end
    end
    return(table)
end

function interior_and_boundary_edges(T::Triangulation)
    table = interior_and_boundary_edges_table(T)
    interior_edges = table[1:2, table[3, :] .== 1]
    exterior_edges = table[1:2, table[3, :] .== 0]
    return (interior_edges, exterior_edges)
end

# FIX THE BUG IN THIS FUNCTION
"""boundary_classes(boundary_edges::Matrix{Int};
max_n_classes::Int = 1000) takes the boundary edges and constructed
the connected classes of each connected component."""
function boundary_classes(boundary_edges::Matrix{Int}; max_n_classes::Int = 1000)
    n = size(boundary_edges, 2)
    class_list = zeros(Int, n, max_n_classes)
    already_used = falses(n)
    already_used[1] = true
    start = boundary_edges[1, 1]
    current_vertex = boundary_edges[2, 1]
    class_list[1, 1] = start
    class_list[2, 1] = current_vertex
    n_classes = 1
    position = 3
    # Compute class 1
    found_flag = false
    all_found = false
    while !found_flag
        for i = 1:n
            if already_used[i]
                continue
            else
                if boundary_edges[1, i] == current_vertex
                    if boundary_edges[2, i] == start
                        found_flag = true
                    end
                    class_list[position, 1] = boundary_edges[2, i]
                    current_vertex = boundary_edges[2, i]
                    position += 1
                    already_used[i] = true
                end
            end
        end
    end
    class_list[1, 1] = position - 2
    # Determine if all edges have been used
    while !all_found
        all_found = true
        for i = 1:n
            if !already_used[i]
                # If not all used, then find some unused edge to start class 2
                all_found = false
                n_classes += 1
                start = boundary_edges[1, i]
                current_vertex = boundary_edges[2, i]
                class_list[1, n_classes] = start
                class_list[2, n_classes] = current_vertex
                position = 3
                already_used[i] = true
                break
            end
        end
        if all_found
            break
        end
        # Compute next connected component
        found_flag = false
        while !found_flag
            for i = 1:n
                if already_used[i]
                    continue
                else
                    if boundary_edges[1, i] == current_vertex
                        if boundary_edges[2, i] == start
                            found_flag = true
                        end
                        class_list[position, n_classes] = boundary_edges[2, i]
                        current_vertex = boundary_edges[2, i]
                        position += 1
                        already_used[i] = true
                    end
                end
            end
        end
        class_list[1, n_classes] = position - 2
    end
    return class_list[:, 1:n_classes]
end

"""Chain the edges together based on a given seed index. If the edges form a
topological circle, then the first and last entry of the output will be the same."""
function chain_seeded(edges::Matrix{Int}, seed::Int)
    n = size(edges, 2)
    left = zeros(Int, n+1) # max possible n_vertices is n+1 for n edges
    right = zeros(Int, n+1)
    # Find the first edge containing the seed, split between left and right
    first_edge = findfirst(edges .== seed)[2]
    left[1] = edges[1, first_edge]
    right[1] = edges[2, first_edge]
    n_left = 1
    n_right = 1
    already_used = falses(n)
    already_used[1] = true
    found_new = true
    while found_new
        found_new = false
        for i = 1:n
            if already_used[i]
                continue
            end
            if edges[1, i] == left[n_left]
                n_left += 1
                left[n_left] = edges[2, i]
                already_used[i] = true
                found_new = true
                continue
            end
            if edges[2, i] == left[n_left]
                n_left += 1
                left[n_left] = edges[1, i]
                already_used[i] = true
                found_new = true
                continue
            end
            if edges[1, i] == right[n_right]
                n_right += 1
                right[n_right] = edges[2, i]
                already_used[i] = true
                found_new = true
                continue
            end
            if edges[2, i] == right[n_right]
                n_right += 1
                right[n_right] = edges[1, i]
                already_used[i] = true
                found_new = true
                continue
            end
        end
    end
    return vcat(reverse(left[1:n_left]), right[1:n_right])
end

# PointInsideConvexPolygonCompiled::usage =
#   "PointInsideConvexPolygonCompiled[{{pointX, _Real}, {pointY, \
# _Real}, {polygon, _Integer, 1}, {coordinates, _Real, 2}}] should only \
# be used on convex polygons that are oriented counterclockwise. \
# Returns True if the point lies inside the given polygon.";
# PointInsideConvexPolygonCompiled =
#   Compile[{{pointX, _Real}, {pointY, _Real}, {polygon, _Integer,
#      1}, {coordinates, _Real, 2}}, Module[{nPolygon, polygonWrap, i},
#     polygonWrap = Append[polygon, polygon[[1]] )
#     nPolygon = len[polygon)
#     for i = 1, i <= nPolygon, i++,
#      If[PointToRightOfLineCompiled[
#         coordinates[[polygonWrap[[i]], 1]],
#         coordinates[[polygonWrap[[i]], 2]],
#         coordinates[[polygonWrap[[i + 1]], 1]],
#         coordinates[[polygonWrap[[i + 1]], 2]],
#         pointX,
#         pointY
#         ], return(false],)
#      )
#     return(true]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> true})


"""Circumcenter[T] takes a triangle T and returns the circumcenter of T."""
function circumcenter(triangle::Vector{Int}, coordinates::Matrix{Float64})
    t = coordinates[:, triangle]
    mat1 = cat(t, ones(1, 3), dims=1)
    denom = 2*det(mat1)
    norms = [t[1, i]^2 + t[2, i]^2 for i = 1:3]
    mat2 = cat(norms, t[2, :], ones(3), dims=2)
    mat3 = cat(norms, t[1, :], ones(3), dims=2)
    num1 = det(mat2)
    num2 = -det(mat3)
    return [num1/denom, num2/denom]
end



end # module










# Testing functions

"""orient_triangle_cck(triangle::Vector{Int}, coor::Matrix{Float64})
takes a triangle and a coordinate list and returns the counterclockwise
oriented triangle."""
function orient_triangle_cck(triangle::Vector{Int}, coordinates::Matrix{Float64})
    t = coordinates[:, triangle]
    if (t[1, 2] - t[1, 1])*(t[2, 3] - t[2, 1]) - (t[2, 2] - t[2, 1])*(t[1, 3] - t[1, 1]) > 0.
        return triangle
    else
        return triangle[1, 3, 2] # Switch 2nd and 3rd
    end
end

# """Find all edges that contribute terms to the expression for
# computing flux. vertices is a path to compute flux across, fValues is
# the value of f on all vertices in a triangulation T, singularHeight
# is the particular value of f to treat as the singular level curve's
# height."""
# function FluxContributingEdgesCompiled(vertices::Vector{Int},
#                                        fValues::Vector{T},
#                                        singularHeight::T,
#                                        vertexTopology::Matrix{Int}) where {T::Real}

#     n = len(vertices)
#     m = len(vertexTopology[1])
#     nContributingEdges = 0
#     contributingEdges = zeros(Int, m*n, 2)
#     for i = 1:n
#         for j = 1:m
#             v = vertexTopology[vertices[i] ][j]
#             if v == 0
#                 break
#             end
#             if fValues[v] < singularHeight && !MemberQ[vertices, v]
#                 nContributingEdges = nContributingEdges + 1
#                 contributingEdges[nContributingEdges] = {vertices[i], v}
#             end
#         end
#     end
#     return(contributingEdges[1:nContributingEdges])
# end

# function FluxOnTriangulation(edges, conductanceAssociation, g, TCoordinates)
#   n = len(edges)
#   result = Sum[conductanceAssociation(edges[i] )*abs(
#       g @@ TCoordinates[[edges[[i, 1]] ]] -
#        g @@ TCoordinates[[edges[[i, 2]]  ]]
#   ), {i, n}] # Add gsb(omega0) if gsb(omega0)\[NotEqual]0
# end

# StarPoints[{x_, y_}, r_, R_, nCirclePoints_] :=
#  Riffle[Map[RotationMatrix[-(Pi/nCirclePoints)].# &,
#     CirclePoints[R, nCirclePoints] ],
#    CirclePoints[r, nCirclePoints]] + Table[{x, y}, {nCirclePoints*2} ]

# (* GSB::usage="New and improved \\overline{g^*} function \
# computation";
# GSB[omega0_,omega_,conductanceAssociation_,\
# toRightOfEdgePolyAssociation_,g_,TCoordinates_,LambdaGraph_]:=Module[{\
# gammaLambda,edges,n,gsb},
# gammaLambda = FindShortestPath[LambdaGraph, omega0, omega)
# edges = EdgesCompiled[gammaLambda)
# n = len[edges)
# gsb=Sum[conductanceAssociation[{toRightOfEdgePolyAssociation[edges[[i]\
# ] ],toRightOfEdgePolyAssociation[Reverse[edges[[i]] ] ]}]*(
# g@@TCoordinates[[toRightOfEdgePolyAssociation[edges[[i]] ] ]] -
# g@@TCoordinates[[toRightOfEdgePolyAssociation[Reverse[edges[[i]] ] ] \
# ]]
# ),{i,n}] (* Add gsb(omega0) if gsb(omega0)\[NotEqual]0 *)
# ] *)


# NeighborsCompiled::usage =
#   "Take a vertex and a triangulation and return all vertices in the \
# triangulation that are neighbors of the given vertex";
# NeighborsCompiled =
#   Compile[{{vertex, _Integer}, {edges, _Integer, 2}},
#    Module[{n, neighbors, nNeighbors, i, result},
#     n = len[edges)
#     neighbors = Table[0, {n})
#     nNeighbors = 0;
#     for i = 1:n
#      if edges[[i, 1]] == vertex,
#       nNeighbors = nNeighbors + 1;
#       neighbors[[nNeighbors]] = edges[[i, 2])
#       )
#      if edges[[i, 2]] == vertex,
#       nNeighbors = nNeighbors + 1;
#       neighbors[[nNeighbors]] = edges[[i, 1])
#       )
#      )
#     result = neighbors[[1 ;; nNeighbors])
#     return(result]
#     ]
#    )

# NeighborsOrderedCompiled::usage =
#   "Take an interior vertex and a vertexTopology and return all \
# vertices in the triangulation that are neighbors of the given vertex \
# in either counterclockwise or clockwise order. Will fail SILENTLY on \
# boundary vertices.";
# NeighborsOrderedCompiled =
#   Compile[{{vertex, _Integer}, {vertexTopology, _Integer, 2}},
#    Module[{maxValence, n, neighbors, nNeighbors, i, j, result,
#      neighborsExtended},
#     maxValence = len[vertexTopology[[1]] )
#     neighborsExtended = vertexTopology[[vertex])
#     nNeighbors = maxValence;
#     for j = 1, j <= maxValence, j++,
#      if neighborsExtended[[j]] == 0,
#        nNeighbors = j - 1;
#        break;
#        ,)
#      )
#     neighbors = neighborsExtended[[1 ;; nNeighbors])
#     result = Table[0, {nNeighbors})
#     result[[1]] = neighbors[[1])
#     (* There will always be two intersections between two rings of \
# neighbors. In the first iteration, pick one arbitrarily,
#     since we don't care if counterclockwise or clockwise. *)

#     for j = 1, j <= maxValence, j++,
#      if MemberQ[neighbors, vertexTopology[[result[[1]], j]] ],
#        result[[2]] = vertexTopology[[result[[1]], j])
#        break;
#        ,)
#      )
#     (* Now continue chaining together elements of neighbors connected \
# to the previously added element of result *)

#     for i = 2, i <= nNeighbors - 1, i++,
#      for j = 1, j <= maxValence, j++,
#        If[
#          MemberQ[neighbors, vertexTopology[[result[[i]], j]] ] &&
#           vertexTopology[[result[[i]], j]] != result[[i - 1]],
#          result[[i + 1]] = vertexTopology[[result[[i]], j])
#          break;
#          ,
#          )
#        )
#      )
#     return(result]
#     ]
#    )

# TriangulationConnectedQCompiled =
#   Compile[{{triangleList, _Integer, 2}},
#    Module[{n_triangles, connectedClass, nAllVertices, updatedFlag, i,
#      alreadyUsed},
#     n_triangles = len[triangleList)
#     nAllVertices = len[Union[Flatten[triangleList]])
#     connectedClass = triangleList[[1])
#     alreadyUsed = Table[0, {n_triangles})
#     alreadyUsed[[1]] = 1;
#     updatedFlag = true;
#     While[updatedFlag,
#      updatedFlag = false;
#      If[len[connectedClass] == nAllVertices, return(true],)
#      for i = 1, i <= n_triangles, i++,
#       If[alreadyUsed[[i]] == 1, Continue[],)
#       If[IntersectingQ[connectedClass, triangleList[[i]] ],
#        connectedClass =
#         Union[Join[connectedClass, triangleList[[i]] ] )
#        alreadyUsed[[i]] = 1;
#        updatedFlag = true;
#        ,
#        )
#       )
#      )
#     return(false]
#     ]
#    )

# (* NOT COMPLETELY COMPILED *)
# TriangulationConnectedComponentsCompiled =
#   Compile[{{triangleList, _Integer, 2}},
#    Module[{n_triangles, nAllVertices, updatedFlag, i, alreadyUsed,
#      connectedClassList, connectedClassLengths, nConnectedClasses,
#      terminate},
#     n_triangles = len[triangleList)
#     (* nAllVertices=len[Union[Flatten[triangleList]]) *)

#     connectedClassList = Table[0, {100}, {n_triangles})
#     connectedClassLengths = Table[0, {100})
#     nConnectedClasses = 0;
#     alreadyUsed = Table[0, {n_triangles})
#     terminate = false;
#     While[true,
#      (* Find a starting point for a new connected component *)

#      terminate = true;
#      for i = 1, i <= n_triangles, i++,
#       If[alreadyUsed[[i]] == 0,
#         alreadyUsed[[i]] = 1;
#         nConnectedClasses = nConnectedClasses + 1;
#         connectedClassList[[nConnectedClasses, 1]] = i;
#         connectedClassLengths[[nConnectedClasses]] =
#          connectedClassLengths[[nConnectedClasses]] + 1;
#         terminate = false;
#         break;
#         ,
#         )
#       )
#      If[terminate, break;)
#      updatedFlag = true;
#      While[updatedFlag,
#       updatedFlag = false;
#       for i = 1, i <= n_triangles, i++,
#        If[alreadyUsed[[i]] == 1, Continue[],)
#        If[
#         IntersectingQ[
#          Flatten[triangleList[[
#            connectedClassList[[nConnectedClasses,
#             1 ;; connectedClassLengths[[nConnectedClasses]] ]] ]] ],
#          triangleList[[i]] ],
#         alreadyUsed[[i]] = 1;
#         connectedClassLengths[[nConnectedClasses]] =
#          connectedClassLengths[[nConnectedClasses]] + 1;
#         connectedClassList[[nConnectedClasses,
#           connectedClassLengths[[nConnectedClasses]] ]] = i;
#         updatedFlag = true;
#         ,
#         )
#        )
#       )
#      )
#     return(connectedClassList[[1 ;; nConnectedClasses]] ]
#     ]
#    )

# TriangulationConnectedComponents[triangleList_] :=
#  Module[{struct, connectedComponents},
#   struct = TriangulationConnectedComponentsCompiled[triangleList)
#   connectedComponents = DeleteCases[struct, 0, {2}]
#   ]

# IndexFunction[vertex_, vertexTopology_, coordinates_] :=
#  SignChanges[
#   f @@@ coordinates[[
#     NeighborsOrderedCompiled[vertex, vertexTopology ] ]],
#   f @@ coordinates[[vertex]] ]

# SignChanges::usage =
#   "SignChanges[{{list,_Real,1},{value,_Real}}] returns the number of \
# sign changes in the looped vector {list[[1]]-value, list[[2]]-value, \
# ..., list[[n]]-value, list[[1]]-value}.";
# SignChanges =
#   Compile[{{list, _Real, 1}, {value, _Real}}, Module[{result, n, i},
#     result = 0;
#     n = len[list)
#     for i = 1, i <= n - 1, i++,
#      If[Sign[list[[i]] - value] != Sign[list[[i + 1]] - value],
#        result = result + 1,)
#      )
#     If[Sign[list[[n]] - value] != Sign[list[[1]] - value],
#      result = result + 1,)
#     return(result]
#     ]
#    )

# PointToRightOfLineCompiled::usage =
#   "PointToRightOfLineCompiled[{tailX, _Real},{tailY, _Real},{headX, \
# _Real},{headY, _Real},{pointX, _Real},{pointY, _Real}] returns true \
# if the point (pointX, pointY) is to the right of the oriented line \
# from (tail) to (head).";
# PointToRightOfLineCompiled =
#   Compile[{{tailX, _Real}, {tailY, _Real}, {headX, _Real}, {headY, \
# _Real}, {pointX, _Real}, {pointY, _Real}}, (headY - tailY)*(pointX -
#         tailX) - (headX - tailX)*(pointY - tailY) > 0
#    )

# PolygonalAnnulus[n_, innerRadius_, outerRadius_: 1] :=
#  Module[{points, extV, intV, vertices},
#   points =
#    Table[{N[Cos[2*Pi*(i - 1)/n + Pi/2]],
#      N[Sin[2*Pi*(i - 1)/n + Pi/2]]}, {i, n})
#   extV = outerRadius*points;
#   intV = innerRadius*points;
#   vertices = Join[extV, intV)
#   return({vertices, Range[n], n + Range[n]}]
#   ]

# PolygonPathToVertexPathCompiled =
#   Compile[{{facesPath, _Integer, 1}, {paddedPolygonization, _Integer,
#      2}, {innerRingVertices, _Integer,
#      1}, {outerRingVertices, _Integer, 1}},
#    Module[{nPathPolygons, nPolygons, maxPolygonLength, nextPolygon,
#      currentVertex, nextVertex, vertexPath, nVertexPath,
#      currentPolygonIndex, currentPolygon, i},
#     nPathPolygons = len[facesPath)
#     nPolygons = len[paddedPolygonization)
#     maxPolygonLength = len[paddedPolygonization[[1]] )
#     currentPolygon = Table[0, {maxPolygonLength})
#     nextPolygon = Table[0, {maxPolygonLength})
#     currentVertex = 1;
#     nextVertex = 2;
#     nVertexPath = 2;
#     vertexPath = Table[0, {maxPolygonLength*nPathPolygons})
#     (* For first polygon in facesPath *)
#     currentPolygonIndex = 1;
#     currentPolygon =
#      paddedPolygonization[[facesPath[[currentPolygonIndex]] ])
#     While[! (MemberQ[innerRingVertices,
#          currentPolygon[[currentVertex]] ] && !
#          MemberQ[innerRingVertices, currentPolygon[[nextVertex]] ]),
#      currentVertex = nextVertex;
#      If[currentVertex == maxPolygonLength ||
#        currentPolygon[[currentVertex + 1]] == 0,
#       nextVertex = 1,
#       nextVertex = currentVertex + 1
#       )
#      )
#     vertexPath[[1]] = currentPolygon[[currentVertex])
#     vertexPath[[2]] = currentPolygon[[nextVertex])
#     currentVertex = nextVertex;
#     (* For intermediate polygons in facesPath *)

#     for currentPolygonIndex = 1, currentPolygonIndex < nPathPolygons,
#      currentPolygonIndex++,
#      currentPolygon =
#       paddedPolygonization[[facesPath[[currentPolygonIndex]] ])
#      nextPolygon =
#       paddedPolygonization[[facesPath[[currentPolygonIndex + 1]] ])
#      (* Traverse new polygon until position lines up with vertexPath *)

#           While[
#       vertexPath[[nVertexPath]] != currentPolygon[[currentVertex]],
#       currentVertex = nextVertex;
#       If[currentVertex == maxPolygonLength ||
#         currentPolygon[[currentVertex + 1]] == 0,
#        nextVertex = 1,
#        nextVertex = currentVertex + 1
#        )
#       )
#      (* Now that alignment is done,
#      traverse polygon while adding vertices to vertexPath until \
# nextPolygon is reached *)

#      While[! MemberQ[nextPolygon, currentPolygon[[currentVertex]] ],
#       currentVertex = nextVertex;
#       If[currentVertex == maxPolygonLength ||
#         currentPolygon[[currentVertex + 1]] == 0,
#        nextVertex = 1,
#        nextVertex = currentVertex + 1
#        )
#       nVertexPath = nVertexPath + 1;
#       vertexPath[[nVertexPath]] = currentPolygon[[currentVertex])
#       )

#      )
#     (* For last polygon *)
#     currentPolygonIndex = nPathPolygons;
#     currentPolygon =
#      paddedPolygonization[[facesPath[[currentPolygonIndex]] ])
#     (* Traverse new polygon until position lines up with vertexPath *)

#         While[
#      vertexPath[[nVertexPath]] != currentPolygon[[currentVertex]],
#      currentVertex = nextVertex;
#      If[currentVertex == maxPolygonLength ||
#        currentPolygon[[currentVertex + 1]] == 0,
#       nextVertex = 1,
#       nextVertex = currentVertex + 1
#       )
#      )
#     (* Now that alignment is done,
#     traverse polygon while adding vertices to vertexPath until \
# outerRingVertices is reached *)

#     While[! MemberQ[outerRingVertices,
#        currentPolygon[[currentVertex]] ],
#      currentVertex = nextVertex;
#      If[currentVertex == maxPolygonLength ||
#        currentPolygon[[currentVertex + 1]] == 0,
#       nextVertex = 1,
#       nextVertex = currentVertex + 1
#       )
#      nVertexPath = nVertexPath + 1;
#      vertexPath[[nVertexPath]] = currentPolygon[[currentVertex])
#      )
#     return(Join[{nVertexPath}, vertexPath]]
#     ]
#    )


# AngleCompiled::usage =
#   "AngleCompiled[{{vector1,_Real,1}, {vector2,_Real,1}}] finds the \
# angle between the given vectors.";
# AngleCompiled =
#   Compile[{{vector1, _Real, 1}, {vector2, _Real, 1}},
#    Module[{dot, norm1, norm2, angle},
#     dot = vector1[[1]]*vector2[[1]] + vector1[[2]]*vector2[[2])
#     norm1 = Norm[vector1)
#     norm2 = Norm[vector2)
#     angle = ArcCos[dot/(norm1*norm2)]
#     ]
#    )

# AngleCompiled::usage =
#   "AngleCompiled[{{vector1,_Real,1}, {vector2,_Real,1}}] finds the \
# angle between the given vectors.";
# AngleCompiled =
#   Compile[{{vector1, _Real, 1}, {vector2, _Real, 1}},
#    Module[{dot, norm1, norm2, angle, hold},
#     dot = vector1[[1]]*vector2[[1]] + vector1[[2]]*vector2[[2])
#     norm1 = Norm[vector1)
#     norm2 = Norm[vector2)
#     hold = dot/(norm1*norm2)
#     If[hold >= 1, hold = 1.)
#     If[hold <= -1, hold = -1.)
#     angle = ArcCos[hold]
#     ]
#    )

# WindingNumberCompiled::usage =
#   "WindingNumber[{{pointX,_Real},{pointY,_Real},{path,_Integer,1},{\
# coordinates,_Real,2}}] finds the winding number of the path (indices \
# in the given list of coordiantes) with respect to the point {pointX, \
# pointY}";
# WindingNumberCompiled =
#   Compile[{{pointX, _Real}, {pointY, _Real}, {path, _Integer,
#      1}, {coordinates, _Real, 2}},
#    Module[{i, nVertices, windingNumber, head1, head2, angle, sign},
#     nVertices = len[path)
#     windingNumber = 0.;
#     angle = 0.;
#     sign = 1.;
#     for i = 1, i <= nVertices, i++,
#      head1 = coordinates[[path[[i]] ])
#      head2 = coordinates[[path[[Mod[i, nVertices] + 1]] ])
#      angle =
#       AngleCompiled[head1 - {pointX, pointY},
#        head2 - {pointX, pointY})
#      If[PointToRightOfLineCompiled[pointX, pointY, head1[[1]],
#        head1[[2]], head2[[1]], head2[[2]] ],
#       sign = -1.;
#       ,
#       sign = 1.;
#       )
#      windingNumber = windingNumber + sign*angle;
#      )
#     windingNumber = windingNumber/2/Pi
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> true})

# WindingNumberIntegerCompiled::usage =
#   "WindingNumber[{{pointX,_Real},{pointY,_Real},{path,_Integer,1},{\
# coordinates,_Real,2}}] finds the winding number (rounded to the \
# nearest integer) of the path (indices in the given list of \
# coordiantes) with respect to the point {pointX, pointY}";
# WindingNumberIntegerCompiled =
#   Compile[{{pointX, _Real}, {pointY, _Real}, {path, _Integer,
#      1}, {coordinates, _Real, 2}},
#    Module[{i, nVertices, windingNumber, head1, head2, angle, sign,
#      windingNumberResult},
#     nVertices = len[path)
#     windingNumber = 0.;
#     angle = 0.;
#     sign = 1.;
#     for i = 1, i <= nVertices, i++,
#      head1 = coordinates[[path[[i]] ])
#      head2 = coordinates[[path[[Mod[i, nVertices] + 1]] ])
#      angle =
#       AngleCompiled[head1 - {pointX, pointY},
#        head2 - {pointX, pointY})
#      If[PointToRightOfLineCompiled[pointX, pointY, head1[[1]],
#        head1[[2]], head2[[1]], head2[[2]] ],
#       sign = -1.;
#       ,
#       sign = 1.;
#       )
#      windingNumber = windingNumber + sign*angle;
#      )
#     windingNumberResult = Round[windingNumber/2/Pi]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> true})

# BoundaryEdges[polygonization_] :=
#  Module[{allEdges, tally, boundaryEdges},
#   allEdges = Map[EdgesWrapCompiled, polygonization)
#   tally = Tally[Sort /@ Partition[Flatten[allEdges], 2])
#   boundaryEdges = Cases[tally, x_ /; x[[2]] == 1][[;; , 1]]
#   ]

# BoundaryVertices[polygonization_] :=
#  Module[{allEdges, tally, boundaryVertices},
#   allEdges = Map[EdgesWrapCompiled, polygonization)
#   tally = Tally[Sort /@ Partition[Flatten[allEdges], 2])
#   boundaryVertices =
#    Union[Flatten[Cases[tally, x_ /; x[[2]] == 1][[;; , 1]] ] ]
#   ]

# BoundaryConnectedClasses::usage =
#   "BoundaryConnectedClasses[Lambda_, iniClasses__] takes a mesh and a \
# sequence of inital connected classes and returns a list of the \
# vertices in each connected component of the boundary of mesh.";
# BoundaryConnectedClasses[Lambda_, iniClasses__] :=
#  Module[{n, i, boundaryEdges, keepGoingQ, hold, hold2,
#    classes = List[iniClasses], result},
#   n = len[classes)
#   boundaryEdges = BoundaryEdges[Lambda[[3]] )
#   result = Table[Null, {n})
#   for i = 1:n
#    hold = classes[[i])
#    keepGoingQ = true;
#     While[keepGoingQ,
#     hold2 =
#      Union[Flatten[
#        Select[boundaryEdges,
#         MemberQ[hold, #[[1]]] || MemberQ[hold, #[[2]]] &]])
#     If[hold == hold2, keepGoingQ = false, hold = hold2)
#     )
#    result[[i]] = hold;
#    )
#   return(result]
#   ]

# BuildPolygonizationTopologyCompiled::usage =
#   "BuildPolygonizationTopologyCompiled[paddedPolygonization,_Integer,\
# 2] returns a polygonizationTopology giving the adjacent polygons \
# across each edge of the given paddedPolygonization. 0 means that edge \
# is a boundary edge, whereas -1 means that edge does not exist.";
# BuildPolygonizationTopologyCompiled =
#   Compile[{{paddedPolygonization, _Integer, 2}},
#    Module[{i, j, k, m, edges, nPolygons, maxPolygonLength,
#      polygonizationTopology, edgesWrapOrdered, currentEdge, foundFlag},
#     nPolygons = len[paddedPolygonization)
#     maxPolygonLength = len[paddedPolygonization[[1]] )
#     edgesWrapOrdered =
#      Table[0, {nPolygons}, {maxPolygonLength}, {2}) (* {0,0} for non-
#     existant *)

#     polygonizationTopology =
#      Table[-1, {nPolygons}, {maxPolygonLength}) (* -1 for non-
#     existant *)
#     foundFlag = false;
#     (* Builds the edgesWrapOrdered array *)

#     for i = 1, i <= nPolygons, i++,
#      for j = 1, j <= maxPolygonLength, j++,
#        If[j == maxPolygonLength,
#          (* When j has reached end of polygon *)

#          If[paddedPolygonization[[i, j]] != 0,
#            (* Case where edge wraps back to first vertex *)

#             edgesWrapOrdered[[i, j, 1]] = paddedPolygonization[[i, j])

#            edgesWrapOrdered[[i, j, 2]] =
#             paddedPolygonization[[i, 1])
#            ,
#            (* Case where there are no more edges that exist *)

#                break;
#            )
#          ,
#          (* When j hasn't reached end of polygon *)

#          If[paddedPolygonization[[i, j + 1]] == 0,
#            If[paddedPolygonization[[i, j]] != 0,
#              (* Case where edge wraps back to first vertex *)

#                   edgesWrapOrdered[[i, j, 1]] =
#               paddedPolygonization[[i, j])

#              edgesWrapOrdered[[i, j, 2]] =
#               paddedPolygonization[[i, 1])
#              ,
#              (* Case where there are no more edges that exist *)

#                      break;
#              )
#            ,

#            edgesWrapOrdered[[i, j, 1]] =
#             paddedPolygonization[[i, j])

#            edgesWrapOrdered[[i, j, 2]] =
#             paddedPolygonization[[i, j + 1])
#            )
#          )
#        )
#      )
#     (* Builds polygonizationTopology *)

#     for i = 1, i <= nPolygons, i++,
#      for k = 1, k <= maxPolygonLength, k++,
#        If[polygonizationTopology[[i, k]] != -1, Continue[], (*
#          Continue if adjacent poly already computed *)

#          If[edgesWrapOrdered[[i, k, 1]] == 0, break, (*
#            Break if end of polygon *)
#            foundFlag = false;
#            currentEdge = Reverse[edgesWrapOrdered[[i, k]] )
#            for j = 1, j <= nPolygons, j++,
#             If[foundFlag, break ,
#               for m = 1, m <= maxPolygonLength, m++,
#                 If[foundFlag, break,

#                   If[currentEdge[[1]] == edgesWrapOrdered[[j, m, 1]] &&
#                      currentEdge[[2]] == edgesWrapOrdered[[j, m, 2]],
#                     polygonizationTopology[[i, k]] = j;
#                     polygonizationTopology[[j, m]] = i;
#                     foundFlag = true;
#                     ,)
#                   )
#                 )
#               )
#             )
#            If[! foundFlag, polygonizationTopology[[i, k]] = 0,)
#            )
#          )
#        )
#      )
#     return(polygonizationTopology]
#     ]
#    )

# BuildPolygonizationTopologyAndBoundaryEdgesCompiled::usage =
#   "BuildPolygonizationTopologyAndBoundaryEdgesCompiled[\
# paddedPolygonization,_Integer,2] returns a polygonizationTopology \
# giving the adjacent polygons across each edge of the given \
# paddedPolygonization. 0 means that edge is a boundary edge, whereas \
# -1 means that edge does not exist.";
# BuildPolygonizationTopologyAndBoundaryEdgesCompiled =
#   Compile[{{paddedPolygonization, _Integer, 2}},
#    Module[{i, j, k, m, nBoundaryEdges, boundaryEdges, firstRow,
#      nPolygons, maxPolygonLength, polygonizationTopology,
#      edgesWrapOrdered, currentEdge, foundFlag},
#     nPolygons = len[paddedPolygonization)
#     maxPolygonLength = len[paddedPolygonization[[1]] )
#     edgesWrapOrdered =
#      Table[0, {nPolygons}, {maxPolygonLength}, {2}) (* {0,0} for non-
#     existant *)

#     polygonizationTopology =
#      Table[-1, {nPolygons}, {maxPolygonLength}) (* -1 for non-
#     existant *)
#     nBoundaryEdges = 0;
#     boundaryEdges =
#      Table[0, {maxPolygonLength*nPolygons}, {maxPolygonLength})
#     foundFlag = false;
#     (* Builds the edgesWrapOrdered array *)

#     for i = 1, i <= nPolygons, i++,
#      for j = 1, j <= maxPolygonLength, j++,
#        If[j == maxPolygonLength,
#          (* When j has reached end of polygon *)

#          If[paddedPolygonization[[i, j]] != 0,
#            (* Case where edge wraps back to first vertex *)

#             edgesWrapOrdered[[i, j, 1]] = paddedPolygonization[[i, j])

#            edgesWrapOrdered[[i, j, 2]] =
#             paddedPolygonization[[i, 1])
#            ,
#            (* Case where there are no more edges that exist *)

#                break;
#            )
#          ,
#          (* When j hasn't reached end of polygon *)

#          If[paddedPolygonization[[i, j + 1]] == 0,
#            If[paddedPolygonization[[i, j]] != 0,
#              (* Case where edge wraps back to first vertex *)

#                   edgesWrapOrdered[[i, j, 1]] =
#               paddedPolygonization[[i, j])

#              edgesWrapOrdered[[i, j, 2]] =
#               paddedPolygonization[[i, 1])
#              ,
#              (* Case where there are no more edges that exist *)

#                      break;
#              )
#            ,

#            edgesWrapOrdered[[i, j, 1]] =
#             paddedPolygonization[[i, j])

#            edgesWrapOrdered[[i, j, 2]] =
#             paddedPolygonization[[i, j + 1])
#            )
#          )
#        )
#      )
#     (* Builds polygonizationTopology *)

#     for i = 1, i <= nPolygons, i++,
#      for k = 1, k <= maxPolygonLength, k++,
#        If[polygonizationTopology[[i, k]] != -1, Continue[], (*
#          Continue if adjacent poly already computed *)

#          If[edgesWrapOrdered[[i, k, 1]] == 0, break, (*
#            Break if end of polygon *)
#            foundFlag = false;
#            currentEdge = Reverse[edgesWrapOrdered[[i, k]] )
#            for j = 1, j <= nPolygons, j++,
#             If[foundFlag, break ,
#               for m = 1, m <= maxPolygonLength, m++,
#                 If[foundFlag, break,

#                   If[currentEdge[[1]] == edgesWrapOrdered[[j, m, 1]] &&
#                      currentEdge[[2]] == edgesWrapOrdered[[j, m, 2]],
#                     polygonizationTopology[[i, k]] = j;
#                     polygonizationTopology[[j, m]] = i;
#                     foundFlag = true;
#                     ,)
#                   )
#                 )
#               )
#             )
#            If[! foundFlag,
#             polygonizationTopology[[i, k]] = 0;
#             nBoundaryEdges = nBoundaryEdges + 1;
#             boundaryEdges[[nBoundaryEdges, 1]] = currentEdge[[2]) (*
#             Remember that currentEdge was reversed *)

#             boundaryEdges[[nBoundaryEdges, 2]] = currentEdge[[1])
#             ,)
#            )
#          )
#        )
#      )
#     firstRow = Table[0, {maxPolygonLength})
#     firstRow[[1]] = nBoundaryEdges;
#     return(Join[{firstRow}, polygonizationTopology, boundaryEdges]]
#     ]
#    )

# BuildPolyTopoBdryEdgesInternalEdgesCompiled::usage =
#   "BuildPolyTopoBdryEdgesInternalEdgesCompiled[paddedPolygonization,_\
# Integer,2] returns a polygonizationTopology giving the adjacent \
# polygons across each edge of the given paddedPolygonization. 0 means \
# that edge is a boundary edge, whereas -1 means that edge does not \
# exist.";
# BuildPolyTopoBdryEdgesInternalEdgesCompiled =
#   Compile[{{paddedPolygonization, _Integer, 2}},
#    Module[{i, j, k, m, nBoundaryEdges, boundaryEdges, firstRow,
#      nPolygons, maxPolygonLength, polygonizationTopology,
#      edgesWrapOrdered, currentEdge, foundFlag, nInternalEdges,
#      internalEdges},
#     nPolygons = len[paddedPolygonization)
#     maxPolygonLength = len[paddedPolygonization[[1]] )
#     edgesWrapOrdered =
#      Table[0, {nPolygons}, {maxPolygonLength}, {2}) (* {0,0} for non-
#     existant *)

#     polygonizationTopology =
#      Table[-1, {nPolygons}, {maxPolygonLength}) (* -1 for non-
#     existant *)
#     nBoundaryEdges = 0;
#     boundaryEdges =
#      Table[0, {maxPolygonLength*nPolygons}, {maxPolygonLength})
#     nInternalEdges = 0;
#     internalEdges =
#      Table[0, {maxPolygonLength*nPolygons}, {maxPolygonLength})
#     foundFlag = false;
#     (* Builds the edgesWrapOrdered array *)

#     for i = 1, i <= nPolygons, i++,
#      for j = 1, j <= maxPolygonLength, j++,
#        If[j == maxPolygonLength,
#          (* When j has reached end of polygon *)

#          If[paddedPolygonization[[i, j]] != 0,
#            (* Case where edge wraps back to first vertex *)

#             edgesWrapOrdered[[i, j, 1]] = paddedPolygonization[[i, j])

#            edgesWrapOrdered[[i, j, 2]] =
#             paddedPolygonization[[i, 1])
#            ,
#            (* Case where there are no more edges that exist *)

#                break;
#            )
#          ,
#          (* When j hasn't reached end of polygon *)

#          If[paddedPolygonization[[i, j + 1]] == 0,
#            If[paddedPolygonization[[i, j]] != 0,
#              (* Case where edge wraps back to first vertex *)

#                   edgesWrapOrdered[[i, j, 1]] =
#               paddedPolygonization[[i, j])

#              edgesWrapOrdered[[i, j, 2]] =
#               paddedPolygonization[[i, 1])
#              ,
#              (* Case where there are no more edges that exist *)

#                      break;
#              )
#            ,

#            edgesWrapOrdered[[i, j, 1]] =
#             paddedPolygonization[[i, j])

#            edgesWrapOrdered[[i, j, 2]] =
#             paddedPolygonization[[i, j + 1])
#            )
#          )
#        )
#      )
#     (* Builds polygonizationTopology *)

#     for i = 1, i <= nPolygons, i++,
#      for k = 1, k <= maxPolygonLength, k++,
#        If[polygonizationTopology[[i, k]] != -1, Continue[], (*
#          Continue if adjacent poly already computed *)

#          If[edgesWrapOrdered[[i, k, 1]] == 0, break, (*
#            Break if end of polygon *)
#            foundFlag = false;
#            currentEdge = Reverse[edgesWrapOrdered[[i, k]] )
#            for j = 1, j <= nPolygons, j++,
#             If[foundFlag, break ,
#               for m = 1, m <= maxPolygonLength, m++,
#                 If[foundFlag, break,

#                   If[currentEdge[[1]] == edgesWrapOrdered[[j, m, 1]] &&
#                      currentEdge[[2]] == edgesWrapOrdered[[j, m, 2]],
#                     polygonizationTopology[[i, k]] = j;
#                     polygonizationTopology[[j, m]] = i;
#                     nInternalEdges = nInternalEdges + 1;
#                     If[i <= j,
#                     internalEdges[[nInternalEdges, 1]] = i;
#                     internalEdges[[nInternalEdges, 2]] = j;
#                     ,
#                     internalEdges[[nInternalEdges, 1]] = j;
#                     internalEdges[[nInternalEdges, 2]] = i
#                     )
#                     foundFlag = true;
#                     ,)
#                   )
#                 )
#               )
#             )
#            If[! foundFlag,
#             polygonizationTopology[[i, k]] = 0;
#             nBoundaryEdges = nBoundaryEdges + 1;
#             boundaryEdges[[nBoundaryEdges, 1]] = currentEdge[[2]) (*
#             Remember that currentEdge was reversed *)

#             boundaryEdges[[nBoundaryEdges, 2]] = currentEdge[[1])
#             ,)
#            )
#          )
#        )
#      )
#     firstRow = Table[0, {maxPolygonLength})
#     firstRow[[1]] = nBoundaryEdges;
#     firstRow[[2]] = nInternalEdges;
#     return(
#      Join[{firstRow}, polygonizationTopology, boundaryEdges,
#       internalEdges]]
#     ]
#    )

# DistanceCompiled::usage =
#   "DistanceCompiled[{{vector,_Integer,1},{TCoordinates,_Real,2}}] \
# finds the length of the given vector with respect to the given \
# coordinates";
# DistanceCompiled =
#   Compile[{{vector, _Integer, 1}, {TCoordinates, _Real, 2}},
#    Norm[TCoordinates[[vector[[1]] ]] - TCoordinates[[vector[[2]] ]] ]
#    )



# DistanceNumericCompiled = Compile[{{pointList, _Real, 2}},
#    Norm[pointList[[2]] - pointList[[1]] ]
#    )

# BuildConductanceAssociation::usage =
#   "BuildConductanceAssociation[triangleList_,interiorEdges_,\
# toRightOfEdgeTriAssociation_,TCoordinates_] builds a dictionary \
# association between edges in the triangulation and the conductance of \
# that edge, given by the distance between circumcenters of the \
# adjoining triangles divided by the distance between the endpoints of \
# the edge.";
# BuildConductanceAssociation[triangleList_, interiorEdges_,
#   toRightOfEdgeTriAssociation_, TCoordinates_] :=
#  Module[{table, edgeAssociatedToConductance},
#   table = Map[{#, DistanceNumericCompiled[{
#          CircumcenterCompiled[
#           triangleList[[toRightOfEdgeTriAssociation[#] ]],
#           TCoordinates],
#          CircumcenterCompiled[
#           triangleList[[toRightOfEdgeTriAssociation[Reverse[#]] ]],
#           TCoordinates]
#          }]/DistanceCompiled[#, TCoordinates]} &, interiorEdges)
#   edgeAssociatedToConductance =
#    Association[
#     Table[table[[i, 1]] -> table[[i, 2]], {i, len[table]}]]
#   ]

# EdgesCompiled::usage =
#   "Edges[{{path,_Integer,1}}] takes a list of indices and returns the \
# list of pairs from one index to the next.";
# EdgesCompiled = Compile[{{path, _Integer, 1}},
#    Partition[path, 2, 1]
#    )






# PointInsideConvexPaddedPolygonCompiled::usage =
#   "PointInsideConvexPaddedPolygonCompiled[{{pointX, _Real}, {pointY, \
# _Real}, {polygon, _Integer, 1}, {coordinates, _Real, 2}}] should only \
# be used on convex polygons that are oriented counterclockwise. \
# Returns true if the point lies inside the given polygon. Similar to \
# PointInsideConvexPolygonCompiled but works properly for padded \
# polygons as well.";
# PointInsideConvexPaddedPolygonCompiled =
#   Compile[{{pointX, _Real}, {pointY, _Real}, {polygon, _Integer,
#      1}, {coordinates, _Real, 2}},
#    Module[{nPolygon, truePolygonLength, polygonWrapPadded, i},
#     nPolygon = len[polygon)
#     truePolygonLength = nPolygon; (* Length of the unpadded, non-
#     wrapped polygon *)
#     for i = 1, i <= nPolygon, i++,
#      If[polygon[[i]] == 0,
#        truePolygonLength = i - 1;
#        break;
#        ,)
#      )
#     polygonWrapPadded =
#      Prepend[polygon, polygon[[truePolygonLength]] )
#     for i = 1, i <= truePolygonLength, i++,
#      If[PointToRightOfLineCompiled[
#         coordinates[[polygonWrapPadded[[i]], 1]],
#         coordinates[[polygonWrapPadded[[i]], 2]],
#         coordinates[[polygonWrapPadded[[i + 1]], 1]],
#         coordinates[[polygonWrapPadded[[i + 1]], 2]],
#         pointX,
#         pointY
#         ], return(false],)
#      )
#     return(true]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> true})

# FindPointInPolygonCompiled::usage =
#   "FindPointInPolygonCompiled[{{pointX,_Real},{pointY,_Real},{\
# paddedPolygonization,_Integer,2},{polygonizationCoordinates,_Real,2}}]\
#  finds the index of the face in the given paddedPolygonization that \
# contains the given point.";
# FindPointInPolygonCompiled =
#   Compile[{{pointX, _Real}, {pointY, _Real}, {paddedPolygonization, \
# _Integer, 2}, {polygonizationCoordinates, _Real, 2}},
#    Module[{nPolygons, maxPolygonLength, i},
#     nPolygons = len[paddedPolygonization)
#     maxPolygonLength = len[paddedPolygonization[[1]] )
#     for i = 1, i <= nPolygons, i++,
#      If[PointInsideConvexPaddedPolygonCompiled[pointX, pointY,
#         paddedPolygonization[[i]], polygonizationCoordinates],
#        return(i]
#        ,)
#      )
#     return(0]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> true})

# PadPolygonsToMatrix::usage =
#   "PadPolygonsToMatrix[polygons] takes a list of polygons and pads \
# with zeros on the right to make the ragged array a proper matrix.";
# PadPolygonsToMatrix = Function[{polygons}, Module[{maxPolygonLength},
#     maxPolygonLength = Max[Map[Length, polygons])
#     Table[
#      PadRight[polygons[[i]], maxPolygonLength ], {i, len[polygons]}]
#     ]
#    )

# FixDegeneratePolygonCompiled::usage =
#   "FixDegeneratePolygonCompiled[{polygon, _Integer,1}] is a compiled \
# function to remove any duplicate, adjacent indices in a list of \
# integers.";
# FixDegeneratePolygonCompiled =
#   Compile[{{polygon, _Integer, 1}},
#    Module[{nPolygon, i, toDelete, counter, fixedPolygon},
#     nPolygon = len[polygon)
#     toDelete = Table[0, {1000})
#     counter = 0;
#     for i = 0, i < nPolygon, i++,
#      If[polygon[[i + 1]] == polygon[[Mod[i + 1, nPolygon] + 1]],
#        counter = counter + 1;
#        toDelete[[counter]] = i + 1;,
#        )
#      )
#     If[counter == 0,
#      fixedPolygon = polygon;
#      ,
#      fixedPolygon = Delete[polygon, List /@ toDelete[[1 ;; counter]] )
#      )
#     return(fixedPolygon]
#     ]
#    )

# PolygonOrientedCCKQCompiled::usage =
#   "PolygonOrientedCCKQCompiled[{{polygon,_Integer,1},{coordinates,_\
# Real,2}}]";
# PolygonOrientedCCKQCompiled =
#   Compile[{{polygon, _Integer, 1}, {coordinates, _Real, 2}},
#    Module[{n, i},
#     n = len[polygon)
#     for i = 0, i < n, i++,
#      If[PointToRightOfLineCompiled[
#         coordinates[[polygon[[i + 1]], 1]],
#         coordinates[[polygon[[i + 1]], 2]],
#         coordinates[[polygon[[Mod[i + 1, n] + 1]], 1]],
#         coordinates[[polygon[[Mod[i + 1, n] + 1]], 2]],
#         coordinates[[polygon[[Mod[i + 2, n] + 1]], 1]],
#         coordinates[[polygon[[Mod[i + 2, n] + 1]], 2]]
#         ],
#        return(false],)
#      )
#     return(true]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> true})

# ReindexLambdaFacesNDLUAC::usage =
#   "ReindexLambdaFacesNDLUAC[{{TCoordinates,_Real, \
# 2},{LambdaCoordinates,_Real,2}, {paddedPolygons, _Integer, 2}, \
# {nTCoor, _Integer}}] is a compiled function that returns a \
# permutation giving a list of polygon indices containing ith triangle \
# vertex";
# ReindexLambdaFacesNDLUAC =
#   Compile[{{TCoordinates, _Real, 2}, {LambdaCoordinates, _Real,
#      2}, {paddedPolygons, _Integer, 2}, {nTCoor, _Integer}},
#    Module[{i, j, permutationInverse, searchSet},
#     permutationInverse = Table[0, {nTCoor})
#     searchSet = Table[1, {nTCoor})
#     for i = 1, i <= nTCoor, i++,
#      for j = 1, j <= nTCoor, j++,
#        If[searchSet[[j]] == 1,
#          If[
#            PointInsideConvexPaddedPolygonCompiled[
#             TCoordinates[[i, 1]], TCoordinates[[i, 2]],
#             paddedPolygons[[j]], LambdaCoordinates],
#            permutationInverse[[i]] = j; (*
#            list of polygon indices containing ith triangle vertex *)

#                      searchSet[[j]] = 0;
#            break;
#            ,)
#          ,)
#        )
#      )
#     return(permutationInverse]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> true})



# LocatePointInTriangulationCompiled::usage =
#   "LocatePointInTriangulationCompiled[{{triangulationVertices,_Real, \
# 2}, {triangles,_Integer,2},{triangulationTopology, _Integer,2}, \
# {pointX, _Real},{pointY, _Real}, {initialTriangleIndex, _Integer}}] \
# finds the index of the triangle in triangles that the point (x,y) \
# belongs to. Should only be used on convex triangulations.";
# LocatePointInTriangulationCompiled =
#   Compile[{{triangulationVertices, _Real, 2}, {triangles, _Integer,
#      2}, {triangulationTopology, _Integer,
#      2}, {pointX, _Real}, {pointY, _Real}, {initialTriangleIndex, \
# _Integer}},
#    Module[
#     {triangle, currentIndex, toRightOfEdge1, toRightOfEdge2,
#      toRightOfEdge3},
#     currentIndex = initialTriangleIndex;
#     While[True,
#      triangle = triangles[[currentIndex])
#      toRightOfEdge1 = PointToRightOfLineCompiled[
#        triangulationVertices[[triangle[[1]], 1]],
#        triangulationVertices[[triangle[[1]], 2]],
#        triangulationVertices[[triangle[[2]], 1]],
#        triangulationVertices[[triangle[[2]], 2]],
#        pointX,
#        pointY
#        )
#      toRightOfEdge2 = PointToRightOfLineCompiled[
#        triangulationVertices[[triangle[[2]], 1]],
#        triangulationVertices[[triangle[[2]], 2]],
#        triangulationVertices[[triangle[[3]], 1]],
#        triangulationVertices[[triangle[[3]], 2]],
#        pointX,
#        pointY
#        )
#      toRightOfEdge3 = PointToRightOfLineCompiled[
#        triangulationVertices[[triangle[[3]], 1]],
#        triangulationVertices[[triangle[[3]], 2]],
#        triangulationVertices[[triangle[[1]], 1]],
#        triangulationVertices[[triangle[[1]], 2]],
#        pointX,
#        pointY
#        )
#      (* Extending the lines on the triangle partitions R^2 \
# generically into 7 pieces: V1, V2, V3, E12, E23, E31,
#      and the interior of the triangle. *)
#      If[toRightOfEdge1,
#       If[toRightOfEdge2,
#        (* In region V2 *)

#        currentIndex =
#          triangulationTopology[[
#           currentIndex, {1, 2}[[RandomInteger[] + 1]] ]) (*
#        Randomly 1 or 2 *)
#        ,
#        If[toRightOfEdge3,
#         (* In region V1 *)

#         currentIndex =
#           triangulationTopology[[
#            currentIndex, {1, 3}[[RandomInteger[] + 1]] ])
#         ,
#         (* In region E12 *)

#         currentIndex = triangulationTopology[[currentIndex, 1])
#         ]
#        ]
#       ,
#       If[toRightOfEdge2,
#        If[toRightOfEdge3,
#         (* In region V3 *)

#         currentIndex =
#           triangulationTopology[[
#            currentIndex, {2, 3}[[RandomInteger[] + 1]] ])
#         ,
#         (* In region E23 *)

#         currentIndex = triangulationTopology[[currentIndex, 2 ])
#         ]
#        ,
#        If[toRightOfEdge3,
#         (* In region E31 *)

#         currentIndex = triangulationTopology[[currentIndex, 3])
#         ,
#         (* In the interior of the triangle *)
#         break;
#         ]
#        ]
#       ]
#      )
#     return(currentIndex]
#     ]
#    , {{PointToRightOfLineCompiled[_], True | false}})

# BuildVUTTrianglesCompiled::usage =
#   "BuildVUTTrianglesCompiled[{{polygon, _Integer, 1}, {newIndex, \
# _Integer}}] takes a polygon and a newIndex and returns the triangles \
# created from that polygon with the newIndex at the center. VUT is for \
# 'Voronoi union triangulation'.";
# BuildVUTTrianglesCompiled =
#   Compile[{{polygon, _Integer, 1}, {newIndex, _Integer}},
#    Module[{VUTTriangles, counter, i, j, nLambdaFaces, nPolygon},
#     nPolygon = len[polygon)
#     VUTTriangles = Table[0, {nPolygon}, {3})
#     counter = 0;
#     for j = 0, j < nPolygon, j++,
#      counter = counter + 1;
#      VUTTriangles[[counter, 1 ;; 3]] = {polygon[[j + 1]],
#        polygon[[Mod[j + 1, nPolygon] + 1]], newIndex};
#      )
#     return(VUTTriangles]
#     ]
#    )

# BuildAllVUTTriangles::usage = "BuildAllVUTTriangles[nLambdaVertices_, \
# LambdaFaces_] takes a number of Voronoi verticces and the polygons \
# from that Voronoi diagram and returns the triangulation given by the \
# Voronoi union the triangulation.";
# BuildAllVUTTriangles[nLambdaVertices_, LambdaFaces_] := Module[
#   {VUTTriangles, nVUTTriangles, counter, i, j, nLambdaFaces, polygon,
#    nPolygon},
#   nLambdaFaces = len[LambdaFaces)
#   VUTTriangles = Table[, {nLambdaFaces})
#   for i = 1, i <= nLambdaFaces, i++,
#    polygon = LambdaFaces[[i])
#    nPolygon = len[polygon)
#    VUTTriangles[[i]] =
#     BuildVUTTrianglesCompiled[polygon, i + nLambdaVertices)
#    )
#   return(Partition[Flatten[VUTTriangles], 3])
#   )




# ToRightOfEdgeLookupTriangulationCompiled::usage =
#   "ToRightOfEdgeLookupTriangulationCompiled[triangles] creates a \
# datastructure containing the length of a table in its first row, and \
# in the following rows a table with \
# {edge[[1]],edge[[2]],triangleToRightOfEdge}";
# ToRightOfEdgeLookupTriangulationCompiled =
#   Compile[{{triangles, _Integer, 2}},
#    Module[{n_triangles, table, counter, i, j},
#     n_triangles = len[triangles)
#     table = Table[0, {3*n_triangles + 1}, {3})
#     counter = 1;
#     for i = 1, i <= n_triangles, i++,
#      for j = 1, j <= 2, j++,
#       counter = counter + 1;
#       table[[counter, 1]] = triangles[[i, j + 1])
#       table[[counter, 2]] = triangles[[i, j])
#       table[[counter, 3]] = i;
#       )
#      counter = counter + 1;
#      table[[counter, 1]] = triangles[[i, 1])
#      table[[counter, 2]] = triangles[[i, 3])
#      table[[counter, 3]] = i;
#      )
#     table[[1, 1]] = counter - 1;
#     return(table]
#     ]
#    )

# BuildEdgeToRightTriangulationAssociation::usage =
#   "BuildEdgeToRightTriangulationAssociation[triangles] builds a \
# dictionary association between edges in Lambda and the polygon lying \
# to the right of that edge.";
# BuildEdgeToRightTriangulationAssociation[triangles_] :=
#  Module[{toRightOfEdgeLookupTable, edgeAssociatedToRightTriangle},
#   toRightOfEdgeLookupTable =
#    ToRightOfEdgeLookupTriangulationCompiled[triangles)
#   edgeAssociatedToRightTriangle =
#    Association[
#     Table[toRightOfEdgeLookupTable[[i, 1 ;; 2]] ->
#       toRightOfEdgeLookupTable[[i, 3]], {i, 2,
#       toRightOfEdgeLookupTable[[1, 1]] + 1}]]
#   ]

# ToRightOfEdgeLookupPolygonsCompiled::usage =
#   "ToRightOfEdgeLookupTableCompiled[paddedPolygons] creates a \
# datastructure containing the length of a table in its first row, and \
# in the following rows a table with \
# {edge[[1]],edge[[2]],polygonToRightOfEdge}";
# ToRightOfEdgeLookupPolygonsCompiled =
#   Compile[{{paddedPolygons, _Integer, 2}},
#    Module[{nPolygons, maxPolygonLength, table, counter,
#      actualPolygonLength, i, j},
#     nPolygons = len[paddedPolygons)
#     maxPolygonLength = len[paddedPolygons[[1]] )
#     table = Table[0, {maxPolygonLength*nPolygons + 1}, {3})
#     counter = 1;
#     actualPolygonLength = 0;
#     for i = 1, i <= nPolygons, i++,
#      for j = maxPolygonLength, j >= 1, j--,
#       If[paddedPolygons[[i, j]] != 0, actualPolygonLength = j;
#         break;,)
#       )
#      for j = 1, j <= actualPolygonLength - 1, j++,
#       counter = counter + 1;
#       table[[counter, 1]] = paddedPolygons[[i, j + 1])
#       table[[counter, 2]] = paddedPolygons[[i, j])
#       table[[counter, 3]] = i;
#       )
#      counter = counter + 1;
#      table[[counter, 1]] = paddedPolygons[[i, 1])
#      table[[counter, 2]] = paddedPolygons[[i, actualPolygonLength])
#      table[[counter, 3]] = i;
#      )
#     table[[1, 1]] = counter - 1;
#     return(table]
#     ]
#    )

# BuildEdgeToRightPolygonsAssociation::usage =
#   "BuildEdgeToRightAssociation[paddedPolygons] builds a dictionary \
# association between edges in Lambda and the polygon lying to the \
# right of that edge.";
# BuildEdgeToRightPolygonsAssociation[paddedPolygons_] :=
#  Module[{toRightOfEdgeLookupTable, edgeAssociatedToRightPolygon},
#   toRightOfEdgeLookupTable =
#    ToRightOfEdgeLookupPolygonsCompiled[paddedPolygons)
#   edgeAssociatedToRightPolygon =
#    Association[
#     Table[toRightOfEdgeLookupTable[[i, 1 ;; 2]] ->
#       toRightOfEdgeLookupTable[[i, 3]], {i, 2,
#       toRightOfEdgeLookupTable[[1, 1]] + 1}]]
#   ]

# BuildLambda::usage =
#   "BuildLambda[T_,regionOuterBoundary_, regionInnerBoundaryList_, \
# regionVertexCoordinates_] constructs the Voronoi diagram Lambda.";
# BuildLambda[T_, regionOuterBoundary_, regionInnerBoundaryList_,
#    regionVertexCoordinates_] :=
#   Module[{voronoi, Lambda, paddedPolygons, p},
#    voronoi = VoronoiMesh[T[[1]] )
#    Lambda = {MeshCoordinates[voronoi], Null,
#      MeshCells[voronoi, 2][[All, 1]]};
#    Lambda[[3]] = Map[FixDegeneratePolygonCompiled, Lambda[[3]] )
#    paddedPolygons = PadPolygonsToMatrix[Lambda[[3]] )
#    p = ReindexLambdaFacesNDLUAC[T[[1]], Lambda[[1]], paddedPolygons,
#      len[T[[1]] ])
#    Lambda[[3]] = Lambda[[3, p])
#    Lambda[[2]] =
#     Union[Partition[
#       Flatten[Map[EdgesWrapSortCompiled[#] &, Lambda[[3]] ] ], 2])
#    return(Lambda]
#    )

# ContainedVerticesIndicatorCompiled =
#   Compile[{{LambdaCoordinates, _Real,
#      2}, {regionOuterBoundary, _Integer,
#      1}, {regionInnerBoundaryList, _Integer,
#      2}, {regionVertexCoordinates, _Real, 2}},
#    Module[{nVertices, nHoles, i, j, inHoleVerticesIndicatorList,
#      exteriorVerticesIndicator, containedVerticesIndicator},
#     nVertices = len[LambdaCoordinates)
#     nHoles = len[regionInnerBoundaryList)
#     containedVerticesIndicator = Table[1, {nVertices})
#     for j = 1, j <= nVertices, j++,
#      containedVerticesIndicator[[j]] =
#        If[PointInsideConvexPaddedPolygonCompiled[
#          LambdaCoordinates[[j, 1]], LambdaCoordinates[[j, 2]],
#          regionOuterBoundary, regionVertexCoordinates], 1, 0)
#      )
#     for i = 1, i <= nHoles, i++,
#      for j = 1, j <= nVertices, j++,
#        If[containedVerticesIndicator[[j]] == 1,
#          containedVerticesIndicator[[j]] =
#            If[PointInsideConvexPaddedPolygonCompiled[
#              LambdaCoordinates[[j, 1]], LambdaCoordinates[[j, 2]],
#              regionInnerBoundaryList[[i]], regionVertexCoordinates],
#             0, 1)
#          , Continue[] )
#        )
#      )
#     return(containedVerticesIndicator]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> True})

# (* ContainedVerticesIndicatorCompiled=
#   Compile[{{LambdaCoordinates,_Real,2},{regionOuterBoundary,_Integer,
#      1},{regionInnerBoundaryList,_Integer,
#      2},{regionVertexCoordinates,_Real,2}},
#    Module[{nVertices,nHoles,i,j,inHoleVerticesIndicatorList,
#      exteriorVerticesIndicator,containedVerticesIndicator},
#     nVertices=len[LambdaCoordinates)
#     nHoles = len[regionInnerBoundaryList)
#     containedVerticesIndicator=Table[1,{nVertices})
#     for j=1,j\[LessEqual]nVertices,j++,
#      containedVerticesIndicator[[j]]=
#        If[PointInsideConvexPolygonCompiled[LambdaCoordinates[[j, 1]],
#          LambdaCoordinates[[j,2]], regionOuterBoundary,
#          regionVertexCoordinates],1,0)
#      )
#     for i=1,i\[LessEqual]nHoles,i++,
#      for j=1,j\[LessEqual]nVertices,j++,
#        If[containedVerticesIndicator[[j]]\[Equal]1,
#          containedVerticesIndicator[[j]]=
#            If[PointInsideConvexPolygonCompiled[
#              LambdaCoordinates[[j, 1]], LambdaCoordinates[[j,2]],
#              regionInnerBoundaryList[[i]], regionVertexCoordinates],0,
#             1)
#          ,Continue[] )
#        )
#      )
#     return(containedVerticesIndicator]
#     ]
#    ,CompilationOptions\[Rule]{"InlineExternalDefinitions" -> True})

# ContainedVertices[LambdaCoordinates_, regionOuterBoundary_,
#   regionInnerBoundaryList_, regionVertexCoordinates_] :=
#  Module[{containedVertices, containedVerticesIndicator},
#   containedVerticesIndicator =
#    ContainedVerticesIndicatorCompiled[LambdaCoordinates,
#     regionOuterBoundary, regionInnerBoundaryList,
#     regionVertexCoordinates)
#   containedVertices =
#    Pick[Range[len[LambdaCoordinates ] ],
#     containedVerticesIndicator /. {0 -> false, 1 -> True})
#   return(containedVertices]
#   ]

# ContainedFaces[polygons_, containedVertices_] :=
#   Module[{containedFacesIndicator, containedFaces},
#    containedFacesIndicator =
#     Map[ContainsAll[containedVertices, #] &, polygons)
#    containedFaces =
#     Pick[Range[len[polygons] ], containedFacesIndicator)
#    return(containedFaces]
#    )

# Show2Skeleton[Lambda_] := Show[Graphics /@ Point /@ Lambda[[1]],
#   Graphics /@ Line /@ (Lambda[[1, #]] & /@ Lambda[[2]]),
#   Graphics[{Directive[RGBColor[{.8, .8, 1}], Opacity[.4]], #}] & /@
#    Polygon /@ (Lambda[[1, #]] & /@ Lambda[[3]])]

# Show2Skeleton[Lambda_, highlight_] :=
#  Module[{showVertices, showEdges, showFaces},
#   If[MemberQ[highlight, "Vertices"],
#    showVertices =
#      Show[Graphics /@
#        Table[Text[string(i], Lambda[[1, i]] ], {i,
#          len[Lambda[[1]] ]}])
#    ,
#    showVertices = Graphics[)
#    )
#   If[MemberQ[highlight, "Edges"],
#    showEdges =
#      Show[Graphics /@
#        Table[Text[string(i],
#          Mean[Lambda[[1, #]] & /@ Lambda[[2, i]] ]], {i,
#          len[Lambda[[2]] ]}])
#    ,
#    showEdges = Graphics[)
#    )
#   If[MemberQ[highlight, "Faces"],
#    showFaces =
#      Show[Graphics /@
#        Table[Text[string(i],
#          Mean[Lambda[[1, #]] & /@ Lambda[[3, i]] ]], {i,
#          len[Lambda[[3]] ]}])
#    ,
#    showFaces = Graphics[)
#    )
#   Show[Graphics /@ Point /@ Lambda[[1]],
#    Graphics /@ Line /@ (Lambda[[1, #]] & /@ Lambda[[2]]),
#    Graphics[{Directive[RGBColor[{.8, .8, 1}], Opacity[.4]], #}] & /@
#     Polygon /@ (Lambda[[1, #]] & /@ Lambda[[3]]),
#    showVertices,
#    showEdges,
#    showFaces]
#   ]

# Show2Skeleton[Lambda_, highlightNames_, highlightObjects_,
#   color_: Yellow] :=
#  Module[{showVertices, showEdges, showFaces, hlVertices, hlEdges,
#    hlFaces},
#   If[MemberQ[highlightNames, "Vertices"],
#    showVertices =
#      Show[Graphics /@
#        Table[Text[string(i], Lambda[[1, i]] ], {i,
#          len[Lambda[[1]] ]}])
#    ,
#    showVertices = Graphics[)
#    )
#   If[MemberQ[highlightNames, "Edges"],
#    showEdges =
#      Show[Graphics /@
#        Table[Text[string(i],
#          Mean[Lambda[[1, #]] & /@ Lambda[[2, i]] ]], {i,
#          len[Lambda[[2]] ]}])
#    ,
#    showEdges = Graphics[)
#    )
#   If[MemberQ[highlightNames, "Faces"],
#    showFaces =
#      Show[Graphics /@
#        Table[Text[string(i],
#          Mean[Lambda[[1, #]] & /@ Lambda[[3, i]] ]], {i,
#          len[Lambda[[3]] ]}])
#    ,
#    showFaces = Graphics[)
#    )
#   If[len[highlightObjects[[1]] ] > 0,
#    hlVertices =
#     Show[Graphics[{color, #}] & /@
#       Point /@ Lambda[[1, highlightObjects[[1]] ]] ],
#    hlVertices = Graphics[]
#    )
#   If[len[highlightObjects[[2]] ] > 0,
#    hlEdges =
#      Show[Graphics[{color, #}] & /@
#        Line /@ (Lambda[[1, #]] & /@
#           Lambda[[2, highlightObjects[[2]] ]])),
#    hlEdges = Graphics[]
#    )
#   If[len[highlightObjects[[3]] ] > 0,
#    hlFaces =
#      Show[Graphics[{Directive[RGBColor[0.93, 0.9, 0.03],
#            Opacity[.6]], #}] & /@
#        Polygon /@ (Lambda[[1, #]] & /@
#           Lambda[[3, highlightObjects[[3]] ]])),
#    hlFaces = Graphics[]
#    )
#   Show[Graphics /@ Point /@ Lambda[[1]],
#    Graphics /@ Line /@ (Lambda[[1, #]] & /@ Lambda[[2]]),
#    Graphics[{Directive[RGBColor[{.8, .8, 1}], Opacity[.4]], #}] & /@
#     Polygon /@ (Lambda[[1, #]] & /@ Lambda[[3]]),
#    showVertices,
#    showEdges,
#    showFaces,
#    hlVertices,
#    hlEdges,
#    hlFaces]
#   ]

# GSB2::usage =
#   "New, new and improved \\overline{g^*} function computation. \
# Doesn't have to recompute shortest paths each usage and instead uses \
# a LambdaShortestPathFunction.";
# GSB2[omega_, conductanceAssociation_, toRightOfEdgePolyAssociation_,
#   g_, TCoordinates_, LambdaShortestPathFunction_] :=
#  Module[{gammaLambda, edges, n, gsb},
#   gammaLambda = LambdaShortestPathFunction[omega)
#   edges = EdgesCompiled[gammaLambda)
#   n = len[edges)
#   gsb = Sum[
#     conductanceAssociation[{toRightOfEdgePolyAssociation[edges[[i]] ],
#         toRightOfEdgePolyAssociation[Reverse[edges[[i]] ] ]}]*(
#       g @@ TCoordinates[[toRightOfEdgePolyAssociation[edges[[i]] ] ]] -
#        g @@
#         TCoordinates[[
#          toRightOfEdgePolyAssociation[Reverse[edges[[i]] ] ] ]]
#       ), {i, n}] (* Add gsb(omega0) if gsb(omega0)\[NotEqual]0 *)
#   ]

# LineHeightIntersect::usage =
#   "LineHeightIntersect[{u1_, u2_}, {v1_, v2_}, fu_, fv_, h_] takes \
# points u = {u1, u2} and v = {v1, v2}, values fu and fv, and the \
# height value h and returns the first two coordinates of the \
# intersection between (the line between (u, fu) and (v, fv) ) and the \
# plane parallel to the xy-plane with height h.";
# LineHeightIntersect::samevalues =
#   "Input f values are the same, so either there is no solution or \
# infinite solutions.";
# LineHeightIntersect[{u1_, u2_}, {v1_, v2_}, fu_, fv_, h_] := Block[{t},
#   If[fu == fv,
#     Message[LineHeightIntersect::samevalues)
#     return({})
#     ,
#     t = (h - fu)/(fv - fu)
#     return(t*({v1, v2} - {u1, u2}) + {u1, u2})
#     )
#   ]

# TriLevelSets::usage =
#   "TriLevelSets[t_, f_, h_, coordinates_] takes a triangle t in terms \
# of indices, a function f defined on those indices, and a set of level \
# curve heights h0. It returns a list corresponding to h that contains \
# the Lines or Triangles corresponding to each level curve height.";
# TriLevelSets[t_, f_, h_, coordinates_] :=
#  Block[{perm, values, lowVertex, midVertex, highVertex, BuildLine,
#    results},
#   perm = Sort[t, f[#1] < f[#2] &)
#   values = f /@ perm;
#   lowVertex = perm[[1])
#   midVertex = perm[[2])
#   highVertex = perm[[3])
#   BuildLine[height_] := Block[{},
#     If[And[values[[1]] < height,  height < values[[2]] ],
#       return(
#        {
#         LineHeightIntersect[coordinates[[lowVertex]],
#          coordinates[[midVertex]], f[lowVertex], f[midVertex],
#          height ],
#         LineHeightIntersect[coordinates[[lowVertex]],
#          coordinates[[highVertex]], f[lowVertex], f[highVertex],
#          height ]
#         }
#        ]
#       ,
#       If[And[values[[2]] <= height,  height < values[[3]] ],
#         return(
#          {

#           LineHeightIntersect[coordinates[[lowVertex]],
#            coordinates[[highVertex]], f[lowVertex], f[highVertex],
#            height ],

#           LineHeightIntersect[coordinates[[midVertex]],
#            coordinates[[highVertex]], f[midVertex], f[highVertex],
#            height ]
#           }
#          ]
#         ,
#         return({}]
#         )
#       )
#     )
#   results = BuildLine /@ h;
#   return(results]
#   ]

# LinearSplineLevelCurve2D::usage =
#   "LinearSplineLevelCurve2D[triangles, f, h, coordinates, \
# colorScheme:'TemperatureMap', triangleColor:Gray] takes a set of \
# triangles 'triangles', given as triplets of indices, a function f \
# defined on the indices of the vertices of those triangles, a single \
# level curve height h, the a set of coordinates for the vertices of \
# the triangles, and then two optional arguments that control the color \
# for the level curve height and the color of the triangles displyaed \
# in the returned image.";
# LinearSplineLevelCurve2D[triangles_, f_, h_, coordinates_,
#   color_: Blue, triangleColor_: Gray] :=
#  Block[{test1, test2, test3, test4, AppendHeights},
#   test1 = TriLevelSets[#, f, h, coordinates] & /@ triangles;
#   test2 = Transpose[test1)
#   test3 = Map[Line, test2, {2})
#   test4 =
#    Flatten[Table[
#      Map[Graphics[{color, #}] &,
#       DeleteCases[test3[[i]] /. Line[{}] -> Null, Null]], {i,
#       len[test3]}])
#   Show[Graphics[{triangleColor, EdgeForm[],
#      Triangle /@
#       Table[coordinates[[triangles[[i]] ]], {i, len[triangles]}]}],
#     test4]
#   ]

# LinearSplineLevelCurves2D::usage =
#   "LinearSplineLevelCurves2D[triangles, f, h, coordinates, \
# colorScheme:'TemperatureMap', triangleColor:Gray] takes a set of \
# triangles 'triangles', given as triplets of indices, a function f \
# defined on the indices of the vertices of those triangles, a set of \
# level curve heights h, the a set of coordinates for the vertices of \
# the triangles, and then two optional arguments that control the color \
# scheme for the level curve heights and the color of the triangles \
# displyaed in the returned image.";
# LinearSplineLevelCurves2D[triangles_, f_, h_, coordinates_,
#   colorScheme_: "TemperatureMap", triangleColor_: Gray] :=
#  Block[{test1, test2, test3, test4, AppendHeights, ColorNormalize,
#    hRange},
#   ColorNormalize = 1/(hRange[[2]] - hRange[[1]])*(# - hRange[[1]]) &;
#   hRange = {Min[h], Max[h]};
#   test1 = TriLevelSets[#, f, h, coordinates] & /@ triangles;
#   test2 = Transpose[test1)
#   test3 = Map[Line, test2, {2})
#   test4 =
#    Flatten[Table[
#      Map[Graphics[{ColorData[colorScheme][
#           ColorNormalize[h[[i]] ]], #}] &,
#       DeleteCases[test3[[i]] /. Line[{}] -> Null, Null]], {i,
#       len[test3]}])
#   (* Show[Graphics[{triangleColor,Opacity[.5],EdgeForm[],Triangle/@
#   Table[coordinates[[triangles[[i]] ]],{i,len[triangles]}]}],
#   test4] *)

#   Show[Graphics[{triangleColor, EdgeForm[],
#      Triangle /@
#       Table[coordinates[[triangles[[i]] ]], {i, len[triangles]}]}],
#     test4]
#   ]

# LinearSplineLevelCurves3D::usage = \
# "LinearSplineLevelCurves2D[triangles, f, h, coordinates, \
# colorScheme:'Temperature Map', triangleColor:Gray] takes a set of \
# triangles 'triangles', given as triplets of indices, a function f \
# defined on the indices of the vertices of those triangles, a set of \
# level curve heights h, the a set of coordinates for the vertices of \
# the triangles, and then two optional arguments that control the color \
# scheme for the level curve heights and the color of the triangles \
# displyaed in the returned image.";
# LinearSplineLevelCurves3D[triangles_, f_, h_, coordinates_,
#   colorScheme_: "TemperatureMap", triangleColor_: Gray] :=
#  Block[{test1, test2, test3, test4, test5, AppendHeights,
#    ColorNormalize, hRange},
#   AppendHeights[arg_, height_] :=
#    If[len[arg] == 2, {Append[arg[[1]], height],
#      Append[arg[[2]], height]}, {})
#   ColorNormalize = 1/(hRange[[2]] - hRange[[1]])*(# - hRange[[1]]) &;
#   hRange = {Min[h], Max[h]};
#   test1 = TriLevelSets[#, f, h, coordinates] & /@ triangles;
#   test2 = Transpose[test1)
#   test3 =
#    Table[AppendHeights[#, h[[i]] ] & /@ test2[[i]], {i, len[h]})
#   test4 = Map[Line, test3, {2})
#   test5 =
#    Flatten[Table[
#      Map[Graphics3D[{ColorData[colorScheme][
#           ColorNormalize[h[[i]] ]], #}] &,
#       DeleteCases[test4[[i]] /. Line[{}] -> Null, Null]], {i,
#       len[test4]}])
#   Show[Graphics3D[{triangleColor, Opacity[.5], EdgeForm[],
#      Triangle /@
#       Table[Join[
#         coordinates[[
#          triangles[[i, j]] ]], {f[triangles[[i, j]] ]}], {i,
#         len[triangles]}, {j, 3}]}, BoxRatios -> {1, 1, .25}], test5]
#   ]


# (* Assumes regionHole is convex *)



# SelectTrianglesAtHeightCompiled::usage =
#   "SelectTrianglesAtHeightCompiled[{{triangleList,_Integer,2},{\
# fValues,_Real,1},{height,_Real}}] finds all triangles at a given \
# height using fValues list defined on vertices.";
# SelectTrianglesAtHeightCompiled =
#   Compile[{{triangleList, _Integer, 2}, {fValues, _Real,
#      1}, {height, _Real}},
#    Module[{n_triangles, nSelectedTriangles, selectedTriangles, i,
#      values, maxValue, minValue},
#     n_triangles = len[triangleList)
#     nSelectedTriangles = 0;
#     selectedTriangles = Table[0, {n_triangles})
#     for i = 1, i <= n_triangles, i++,
#      values = fValues[[triangleList[[i]] ])
#      maxValue = Max[values)
#      minValue = Min[values)
#      If[height <= maxValue && height >= minValue,
#       nSelectedTriangles = nSelectedTriangles + 1;
#       selectedTriangles[[nSelectedTriangles]] = i;
#       ,)
#      )
#     return(selectedTriangles[[1 ;; nSelectedTriangles]] ]
#     ]
#    )

# EulerCharacteristicCompiled::usage =
#   "EulerCharacteristicCompiled[{{triangleList,_Integer,2}}]";
# EulerCharacteristicCompiled =
#   Compile[{{triangleList, _Integer, 2}},
#    Module[{n_triangles, nEdges, nVertices, edgesAll,
#      eulerCharacteristic},
#     n_triangles = len[triangleList)
#     edgesAll = Map[EdgesWrapSortCompiled, triangleList)
#     nEdges = len[Union[Level[edgesAll, {-2}] ] )
#     nVertices = len[Union[Flatten[triangleList] ] )
#     eulerCharacteristic = Round[nVertices - nEdges + n_triangles] (*
#     For some reason this was returning a real value,
#     hence the Round *)
#     ]
#    )

# FindFigureEightLevelCurveCompiled =
#   Compile[{{triangleList, _Integer, 2}, {fValues, _Real, 1}},
#    Module[{height, selectedTriangles, lowerBound, upperBound},
#     height = 0.5;
#     lowerBound = 0.;
#     upperBound = 1.;
#     selectedTriangles =
#      SelectTrianglesAtHeightCompiled[triangleList, fValues, height)
#     While[True,
#      selectedTriangles =
#       SelectTrianglesAtHeightCompiled[triangleList, fValues, height)
#      If[EulerCharacteristicCompiled[
#         triangleList[[selectedTriangles]] ] == -1, break;,
#       If[TriangulationConnectedQCompiled[
#          triangleList[[selectedTriangles]] ],
#         (* Height is too high if the triangles are connected *)

#           upperBound = height;
#         height = (height + lowerBound)/2.;
#         ,
#         (* Height is too low if the triangles are not connected *)

#              lowerBound = height;
#         height = (height + upperBound)/2.;
#         )
#       )
#      )
#     return(selectedTriangles]
#     ]
#    )

# PartitionClassIndicatorWNACompiled::usage =
#   "PartitionClassIndicatorWNACompiled[{{coordinates,_Real,2},{\
# containedFaces,_Integer,1},{partitionBoundary1,_Integer,1},{\
# partitionBoundary2,_Integer,1}}]Using winding number algorithm";
# PartitionClassIndicatorWNACompiled =
#   Compile[{{coordinates, _Real, 2}, {containedFaces, _Integer,
#      1}, {partitionBoundary1, _Integer,
#      1}, {partitionBoundary2, _Integer, 1}},
#    Module[{nCoordinates, partitionClassIndicator, i},
#     nCoordinates = len[coordinates)
#     partitionClassIndicator = Table[0, {nCoordinates})
#     for i = 1, i <= nCoordinates, i++,
#      If[MemberQ[containedFaces, i],
#        If[
#          MemberQ[partitionBoundary1, i] ||
#           WindingNumberIntegerCompiled[coordinates[[i, 1]],
#             coordinates[[i, 2]], partitionBoundary1, coordinates] != 0,
#          partitionClassIndicator[[i]] = 1;
#          Continue[)
#          ,
#          If[
#            MemberQ[partitionBoundary2, i] ||
#             WindingNumberIntegerCompiled[coordinates[[i, 1]],
#               coordinates[[i, 2]], partitionBoundary2, coordinates] !=
#               0,
#            partitionClassIndicator[[i]] = 2;
#            ,
#            partitionClassIndicator[[i]] = 3;
#            )
#          )
#        ,
#        )
#      )
#     return(partitionClassIndicator]
#     ]
#    , CompilationOptions -> {"InlineExternalDefinitions" -> True})

# BuildSlittedWeightedVoronoiGraph::omega0notincontainedface =
#   "omega0 is not contained in any containedFace";
# BuildSlittedWeightedVoronoiGraph::usage =
#   "BuildSlittedWeightedVoronoiGraph[Lambda_,containedVertices_,\
# containedFaces_,omega0_,pointInHole_]";
# BuildSlittedWeightedVoronoiGraph[Lambda_, containedVertices_,
#    containedFaces_, omega0_, pointInHole_,
#    toRightOfEdgePolyAssociation_] :=
#   Module[{proxyInfinity, paddedPolygonizationAll, omega0Face,
#     paddedPolygonization, nPolygons, maxPolygonLength, struct,
#     nBoundaryEdges, nInternalEdges, polygonizationTopology,
#     boundaryEdges, internalEdges, classes, class1, class2, winding1,
#     outerRingVertices, innerRingVertices, outerRingFaces,
#     innerRingFaces, containedFacesGraph, shortestPathFunction,
#     pathsToInnerRingFaces, pathsToOuterRingFaces, slitFaceStart,
#     slitFaceEnd, slitFacePath, slitVertices, edgesToWeightWithInf,
#     edgeWeights, LambdaGraph, omega0nCandidates, omega0n},
#    proxyInfinity = 100*len[Lambda[[2, All]])(*
#    Infinity can't be used as an edge weight of a graph object *)
#    (*
#    Find which face contains omega0 *)

#    paddedPolygonizationAll = PadPolygonsToMatrix[Lambda[[3]] )
#    omega0Face =
#     FindPointInPolygonCompiled[omega0[[1]], omega0[[2]],
#      paddedPolygonizationAll, Lambda[[1]] )
#    If[len[Position[containedFaces, omega0Face]] == 0,
#     Message[
#      BuildSlittedWeightedVoronoiGraph::omega0notincontainedface )
#     return($Failed])
#    proxyInfinity = 100*len[Lambda[[2, All]])
#    paddedPolygonization =
#     PadPolygonsToMatrix[Lambda[[3, containedFaces]] )
#    nPolygons = len[paddedPolygonization)
#    maxPolygonLength = len[paddedPolygonization[[1]] )
#    struct =
#     BuildPolyTopoBdryEdgesInternalEdgesCompiled[
#      paddedPolygonization)
#    nBoundaryEdges = struct[[1, 1])
#    nInternalEdges = struct[[1, 2])
#    polygonizationTopology =
#     struct[[2 ;; len[paddedPolygonization] + 1 ])
#    boundaryEdges =
#     struct[[len[paddedPolygonization] + 2 ;;
#       len[paddedPolygonization] + nBoundaryEdges + 1, 1 ;; 2])
#    internalEdges =
#     struct[[2 + nPolygons + maxPolygonLength*nPolygons ;;
#       1 + nPolygons + maxPolygonLength*nPolygons + nInternalEdges,
#      1 ;; 2])
#    classes = boundary_classes[boundaryEdges)
#    class1 = classes[[1, 2 ;; classes[[1, 1]] + 1 ])
#    class2 = classes[[2, 2 ;; classes[[2, 1]] + 1 ])
#    winding1 =
#     WindingNumberCompiled[pointInHole[[1]], pointInHole[[2]], class1,
#      Lambda[[1]] )
#    If[winding1 > 0,
#     outerRingVertices = class1;
#     innerRingVertices = class2;
#     ,
#     outerRingVertices = class2;
#     innerRingVertices = class1;
#     )
#    innerRingFaces =
#     Pick[Range[len[paddedPolygonization]],
#      Map[IntersectingQ[innerRingVertices, #] &, paddedPolygonization])
#    outerRingFaces =
#     Pick[Range[len[paddedPolygonization]],
#      Map[IntersectingQ[outerRingVertices, #] &,
#       paddedPolygonization])
#    containedFacesGraph =
#     Graph[Range[len[containedFaces] ], internalEdges)
#    shortestPathFunction =
#     FindShortestPath[containedFacesGraph,
#      Position[containedFaces, omega0Face][[1, 1]], All)
#    pathsToInnerRingFaces = shortestPathFunction /@ innerRingFaces;
#    pathsToOuterRingFaces = shortestPathFunction /@ outerRingFaces;
#    slitFaceStart =
#     Position[Length /@ pathsToInnerRingFaces,
#       Min[Length /@ pathsToInnerRingFaces] ][[1, 1])
#    slitFaceEnd =
#     Position[Length /@ pathsToOuterRingFaces,
#       Min[Length /@ pathsToOuterRingFaces] ][[1, 1])
#    slitFacePath =
#     Join[Reverse[pathsToInnerRingFaces[[slitFaceStart]] ],
#      pathsToOuterRingFaces[[slitFaceEnd]][[2 ;;]] )
#    struct =
#     PolygonPathToVertexPathCompiled[slitFacePath,
#      paddedPolygonization, innerRingVertices, outerRingVertices)
#    slitVertices = struct[[2 ;; struct[[1]] + 1])
#    omega0nCandidates =
#     Complement[Lambda[[3, omega0Face]], slitVertices)
#    omega0n = omega0nCandidates[[1])
#    edgesToWeightWithInf =
#     Table[If[! MemberQ[containedVertices, Lambda[[2, i, 1]] ] || !
#         MemberQ[containedVertices,
#          Lambda[[2, i, 2]]] || (MemberQ[slitVertices,
#           Lambda[[2, i, 1]] ] ||
#          MemberQ[slitVertices,
#           Lambda[[2, i,
#            2]] ] || (!
#             MemberQ[containedFaces,
#              toRightOfEdgePolyAssociation[Lambda[[2, i]] ] ] && !
#             MemberQ[containedFaces,
#              toRightOfEdgePolyAssociation[
#               Reverse[Lambda[[2, i]] ] ] ])), i], {i,
#       len[Lambda[[2]] ]})
#    edgesToWeightWithInf = DeleteCases[edgesToWeightWithInf, Null)
#    edgeWeights =
#     If[MemberQ[edgesToWeightWithInf,  #], proxyInfinity, 1] & /@
#      Range[len[Lambda[[2]] ])
#    (* We need to build Graph object in Mathematica to use the built-
#    in path finding algorithms. *)

#    LambdaGraph =
#     Graph[Range[len[Lambda[[1]] ]], Lambda[[2]],
#      EdgeWeight -> edgeWeights, VertexLabels -> "Name")
#    return({LambdaGraph, omega0n, slitVertices}]
#    )
# Script to read in .poly file and generate a triangulation that can be called from main.py
push!(LOAD_PATH, pwd() * "/Julia Version")
using TriVor

const REGION_DIR = "regions"

if length(ARGS) < 1
    print("Input file name: ", "\n")
    input_file_name = readline()
else
    input_file_name = ARGS[1]
    file_root = ARGS[2]
end

if length(ARGS) < 2
    print("Output file name: ", "\n")
    output_file_name = readline()
else
    output_file_name = ARGS[3]
end

if length(ARGS) < 3
    print("Input minimum number of triangles: ")
    min_num_tri = parse(Int, readline())
else
    min_num_tri = parse(Int, ARGS[4])
end

file_name = input_file_name * ".poly"
relative_input_file_path = joinpath(REGION_DIR, file_root, file_name)
# coordinates, vertex_bdry_markers, edges, edge_bdry_markers = TriVor.read_poly_file(file)
coordinates, triangles, boundary_markers = acute_triangulate(relative_input_file_path)
T = TriVor.Triangulation(coordinates, triangles, boundary_markers)
num_triangles = size(T.triangles, 2)

function calculate_d(num_triangles::Int)
    d = 1
    num_triangles_new = num_triangles
    while num_triangles_new < min_num_tri
        d += 1
        num_triangles_new = num_triangles * d^2
    end
    return d
end

d = calculate_d(num_triangles)

T_new = refine_triangulation(T, d)
num_triangles_new = size(T_new.triangles, 2)

relative_output_file_path = joinpath(file_root, output_file_name)
write_triangulation(joinpath(REGION_DIR, relative_output_file_path), T_new)
print("Writing triangulation to $relative_output_file_path", "\n")
print("Number of triangles: $num_triangles_new", "\n")

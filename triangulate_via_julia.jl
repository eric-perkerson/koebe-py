# Script to read in .poly file and generate a triangulation that can be called from main.py
push!(LOAD_PATH, pwd() * "/Julia Version")
using TriVor

const REGION_DIR = "regions"

if length(ARGS) < 1
    print("Input file name: ", "\n")
    input_file_name_or_stem = readline()
else
    input_file_name_or_stem = ARGS[1]
end

if length(ARGS) < 2
    print("Output file name: ", "\n")
    output_file_name = readline()
else
    output_file_name = ARGS[2]
end

if length(ARGS) < 3
    print("Input minimum number of triangles: ")
    min_num_tri = parse(Int, readline())
else
    min_num_tri = parse(Int, ARGS[3])
end

# file_stem = split(input_file_name_or_stem, ".")[1]
file_stem = join(split(input_file_name_or_stem, ".")[1:end-1], ".")
file_path = joinpath(REGION_DIR, file_stem)
if file_path[end-4:end] == ".poly"
    file_stem = file_path[1:end-5]
else
    file_stem = file_path
end
file_name = file_stem * ".poly"

# coordinates, vertex_bdry_markers, edges, edge_bdry_markers = TriVor.read_poly_file(file)
coordinates, triangles, boundary_markers = acute_triangulate(file_name)
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
print("output_file_name = $output_file_name", "\n")
write_triangulation(joinpath(REGION_DIR, output_file_name), T_new)
print("Writing triangulation to $output_file_name", "\n")
print("Number of triangles: $num_triangles", "\n")

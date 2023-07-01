"""Combine a .node and .ele file into one .poly file"""

from triangulation import Triangulation
from region import Region
from pathlib import Path


def write_poly(Region, Triangulation, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f'{Triangulation.num_vertices} 2 0 1\n')
        for vertex in range(Triangulation.num_vertices):
            f.write(f'{vertex + 1} {Triangulation.vertices[vertex, 0]} {Triangulation.vertices[vertex, 1]} {Triangulation.boundary_markers[vertex]}\n')
        num_edges = len(Triangulation.triangulation_edges_unique)
        f.write(f'{num_edges} 0\n')
        for edge in range(num_edges):
            f.write(f'{edge + 1} {Triangulation.triangulation_edges_unique[edge, 0]} {Triangulation.triangulation_edges_unique[edge, 1]}\n')
        f.write(f'{len(Region.points_in_holes)}\n')
        for hole in range(len(Region.points_in_holes)):
            f.write(f'{hole} {Region.points_in_holes[hole, 0]} {Region.points_in_holes[hole, 1]}\n')


if __name__ == '__main__':
    file_stem = 'test'
    path = Path(f'regions/{file_stem}/{file_stem}.poly')
    T = Triangulation.read(path)
    R = Region.read_poly(path)
    write_poly(R, T, f'regions/{file_stem}/{file_stem}.combined.poly')

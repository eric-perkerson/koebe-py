from triangulation import Triangulation
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import subprocess
from region import Region
#from region import Region
import pyvista


#  To run the code with no coordinates, but with a file_stem,
#  the folder with the same name must
# contain a file named file_stem.poly
#file_stem = '3_fold_sym'

#file_stem = 'No_3_fold_sym'  

#file_stem = "3_holes_along_line"
#file_stem = "non_concentric_annulus"
#file_stem = "non_concentric_annulus_nonconvex"

#file_stem = "complete_3_fold_sym"

#file_stem = 'No_3_fold_sym' 

#file_stem = "test_example_5"

#file_stem = "test3" Last one I tried with success

#file_stem = "No_3_fold_sym_at_all" #Last one I tried with success

file_stem = "vertex18"
####  The following part, till the next line of comments,
####  works directly with coordinates given as below.


# domain = Region.region_from_components(
#    [
#     [(2.0, 0.0),
#     (1.0000000000000002, 1.7320508075688772),
#     (-0.9999999999999996, 1.7320508075688776),
#     (-2.0, 0),
#     (-1.0000000000000009, -1.7320508075688767),
#     (1.0, -1.7320508075688772)
#     ],
#     [(1.0, 2.4492935982947065e-17),
#     (1.0999999999999999, 0.17320508075688773),
#     (1.3, 0.17320508075688776),
#     (1.4, 0.0),
#     (1.3, -0.1732050807568877),
#     (1.1, -0.1732050807568878)
#     ],
#     [(-1.2086554390135442, 0.45922011883810776),
#     (-1.1586554390135442, 0.5458226592165516),
#     (-1.058655439013544, 0.5458226592165516),
#     (-1.008655439013544, 0.45922011883810776),
#     (-1.058655439013544, 0.3726175784596639),
#     (-1.1586554390135442, 0.37261757845966387)
#     ],
#     [(-0.2774790660436858, -0.9749279121818235),
#     (-0.027479066043685857, -0.5419152102896043),
#     (0.47252093395631417, -0.5419152102896043),
#     (0.7225209339563142, -0.9749279121818236),
#     (0.47252093395631434, -1.407940614074043),
#     (-0.027479066043685496, -1.4079406140740431)
#     ]
#    ]
# )






# # domain = Region.region_from_components(
# #     [
# #         [
# #             (2.0, 0.0),
# #             (1.0000000000000002, 1.7320508075688772),
# #             (-0.9999999999999996, 1.7320508075688776),
# #             (-2.0, 0.0),
# #             (-1.0000000000000009, -1.7320508075688767),
# #             (1.0, -1.7320508075688772)
# #         ],
# #         [
# #             (0.9000000000000001, 0.0),
# #             (1.0, 0.17320508075688773),
# #             (1.2000000000000002, 0.17320508075688776),
# #             (1.3, 0.0),
# #             (1.2000000000000002, -0.1732050807568877),
# #             (1.0000000000000002, -0.1732050807568878)
# #         ],
# #         [
# #             (-0.7499999999999998, 0.9526279441628828),
# #             (-0.6499999999999999, 1.1258330249197706),
# #             (-0.44999999999999984, 1.1258330249197706),
# #             (-0.3499999999999998, 0.9526279441628828),
# #             (-0.44999999999999973, 0.7794228634059951),
# #             (-0.6499999999999997, 0.779422863405995)
# #         ],
# #         [
# #             (-0.7500000000000004, -0.9526279441628823),
# #             (-0.6500000000000006, -0.7794228634059945),
# #             (-0.4500000000000005, -0.7794228634059945),
# #             (-0.3500000000000005, -0.9526279441628823),
# #             (-0.4500000000000004, -1.12583302491977),
# #             (-0.6500000000000004, -1.12583302491977)
# #         ]
# #     ]
# # )


# with open(f"regions/{file_stem}/{file_stem}.poly", 'w', encoding='utf-8') as f:
#     domain.write(f)

#############################################################
#############################################################
subprocess.run([
    #    'julia',
        'julia',
        'triangulate_via_julia.jl',
        file_stem,
        file_stem,
        "1000"        # Change back to 1000 after vertex18 has run
    ])

t = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')
t.write(f'regions/{file_stem}/{file_stem}.output.poly')

subprocess.run([
        'python',
        'mesh_conversion/mesh_conversion.py',
        '-p',
        f'regions/{file_stem}/{file_stem}.output.poly',
        '-n',
        f'regions/{file_stem}/{file_stem}.node',
        '-e',
        f'regions/{file_stem}/{file_stem}.ele',
    ])

subprocess.run([
    'python',
    'mesh_conversion/fenicsx_solver.py',
    file_stem,
])

##################################
# why not read .output.poly?
#################################
tri = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')

print(tri.singular_heights)
print(tri.singular_vertices)

values= tri.pde_values[:-1]
plt.scatter(
    tri.vertices[:, 0],
    tri.vertices[:, 1],
    #c=tri.pde_values
    c=values
)
plt.savefig(f'regions/{file_stem}/{file_stem}.scatter.png')
plt.show()


#change how_singular_level_curves=True to False in case of an annulus
tri.show(
    f'regions/{file_stem}/{file_stem}.png',
    show_level_curves=True,
    show_singular_level_curves=False,
    dpi=500,
    num_level_curves=700,
    line_width=0.75
       )
plt.show()




#### From older version
# path = Path(f'regions/{file_stem}/{file_stem}')
# tri = Triangulation.read(path)

#from region import Region
# domain = Region.region_from_components(
#     [
#         [
#             (2.0, 0.0),
#             (1.7320508075688774, 0.9999999999999999),
#             (1.0000000000000002, 1.7320508075688772),
#             (0.0, 2.0),
#             (-0.9999999999999996, 1.7320508075688776),
#             (-1.7320508075688774, 0.9999999999999999),
#             (-2.0, 0.0),
#             (-1.7320508075688776, -0.9999999999999996),
#             (-1.0000000000000009, -1.7320508075688767),
#             (0.0, -2.0),
#             (1.0, -1.7320508075688772),
#             (1.7320508075688767, -1.0000000000000009)
#         ],
#         [
#             (-0.1, 0.0),
#             (-0.08660254037844388, 0.049999999999999996),
#             (-0.05000000000000002, 0.08660254037844387),
#             (0.0, 0.1),
#             (0.049999999999999996, 0.08660254037844388),
#             (0.08660254037844385, 0.050000000000000024),
#             (0.1, 0.0),
#             (0.08660254037844388, -0.04999999999999999),
#             (0.05000000000000004, -0.08660254037844385),
#             (0.0, -0.1),
#             (-0.04999999999999994, -0.0866025403784439),
#             (-0.08660254037844385, -0.05000000000000004)
#         ],
#         [ 
#             (0.40000000000000013, 0.8660254037844386),
#             (0.4133974596215562, 0.9160254037844386),
#             (0.45000000000000007, 0.9526279441628824),
#             (0.5000000000000001, 0.9660254037844386),
#             (0.5500000000000002, 0.9526279441628824),
#             (0.586602540378444, 0.9160254037844386),
#             (0.6000000000000001, 0.8660254037844386),
#             (0.586602540378444, 0.8160254037844386),
#             (0.5500000000000002, 0.7794228634059948),
#             (0.5000000000000001, 0.7660254037844386),
#             (0.4500000000000002, 0.7794228634059946),
#             (0.4133974596215563, 0.8160254037844386)
#         ],
#         [
#             (-0.6000000000000004, -0.8660254037844384),
#             (-0.5866025403784443, -0.8160254037844383),
#             (-0.5500000000000005, -0.7794228634059945),
#             (-0.5000000000000004, -0.7660254037844384),
#             (-0.45000000000000046, -0.7794228634059945),
#             (-0.4133974596215566, -0.8160254037844383),
#             (-0.40000000000000047, -0.8660254037844384),
#             (-0.41339745962155655, -0.9160254037844384),
#             (-0.4500000000000004, -0.9526279441628822),
#             (-0.5000000000000004, -0.9660254037844384),
#             (-0.5500000000000004, -0.9526279441628823),
#             (-0.5866025403784443, -0.9160254037844384)
#         ]    
#     ]
# )

# with open(f"regions/{file_stem}/{file_stem}.poly", 'w', encoding='utf-8') as f:
#     domain.write(f)


# =An example with 3-fold symmetry
# from region import Region
# domain = Region.region_from_components(
#     [
#         [
#             (2.0, 0.0),
#             (1.0000000000000002, 1.7320508075688772),
#             (-0.9999999999999996, 1.7320508075688776),
#             (-2.0, 2.4492935982947064e-16),
#             (-1.0000000000000009, -1.7320508075688767),
#             (1.0, -1.7320508075688772)
#         ],
#         [
#             (0.9000000000000001, 2.4492935982947065e-17),
#             (1.0, 0.17320508075688773),
#             (1.2000000000000002, 0.17320508075688776),
#             (1.3, 0.0),
#             (1.2000000000000002, -0.1732050807568877),
#             (1.0000000000000002, -0.1732050807568878)
#         ],
#         [
#             (-0.7499999999999998, 0.9526279441628828),
#             (-0.6499999999999999, 1.1258330249197706),
#             (-0.44999999999999984, 1.1258330249197706),
#             (-0.3499999999999998, 0.9526279441628828),
#             (-0.44999999999999973, 0.7794228634059951),
#             (-0.6499999999999997, 0.779422863405995)
#         ],
#         [
#             (-0.7500000000000004, -0.9526279441628823),
#             (-0.6500000000000006, -0.7794228634059945),
#             (-0.4500000000000005, -0.7794228634059945),
#             (-0.3500000000000005, -0.9526279441628823),
#             (-0.4500000000000004, -1.12583302491977),
#             (-0.6500000000000004, -1.12583302491977)
#         ]
#     ]
# )
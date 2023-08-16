import subprocess

# for i in range(5):
#     # command = f'python mesh_conversion/mesh_conversion.py -p regions/3_fold_sym.output.poly -n regions/3_fold_sym.node -e regions/3_fold_sym.ele'
#     subprocess.run([
#         'python',
#         'mesh_conversion/mesh_conversion.py',
#         '-p',
#         f'regions/3_fold_sym/3_fold_sym.output.poly',
#         '-n',
#         f'regions/3_fold_sym/3_fold_sym.node',
#         '-e',
#         f'regions/3_fold_sym/3_fold_sym.ele',
#     ])
#     break


subprocess.run([
        'python',
        'mesh_conversion/mesh_conversion.py',
        '-p',
        f'regions/3_fold_sym/3_fold_sym.output.poly',
        '-n',
        f'regions/3_fold_sym/3_fold_sym.node',
        '-e',
        f'regions/3_fold_sym/3_fold_sym.ele',
    ])

#python mesh_conversion/mesh_conversion.py -p regions/test_example_0/test_example_0.output.poly -n regions/test_example_0/test_example_0.node -e regions/test_example_0/test_example_0.ele

#regions/test_example_0/test_example_0.output.poly

#1.12.2
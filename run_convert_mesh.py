import subprocess

for i in range(5):
    # command = f'python mesh_conversion/mesh_conversion.py -p regions/test_example_{i}.output.poly -n regions/test_example_{i}.node -e regions/test_example_{i}.ele'
    subprocess.run([
        'python',
        'mesh_conversion/mesh_conversion.py',
        '-p',
        f'regions/test_example_{i}.output.poly',
        '-n',
        f'regions/test_example_{i}.node',
        '-e',
        f'regions/test_example_{i}.ele',
    ])
    break


#python mesh_conversion/mesh_conversion.py -p regions/test_example_0/test_example_0.output.poly -n regions/test_example_0/test_example_0.node -e regions/test_example_0/test_example_0.ele

#regions/test_example_0/test_example_0.output.poly

#1.12.2
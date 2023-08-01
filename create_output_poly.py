from triangulation import Triangulation

for i in range(5):
    t = Triangulation.read(f'regions/test_example_{i}/test_example_{i}.poly')
    t.write(f'regions/test_example_{i}/test_example_{i}.output.poly')

    print("finish running")

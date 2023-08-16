from triangulation import Triangulation

# for i in range(5):
#     t = Triangulation.read(f'regions/3_fold_sym/3_fold_sym.poly')
#     t.write(f'regions/3_fold_sym/3_fold_sym.output.poly')

#     print("finish running")



t = Triangulation.read(f'regions/3_fold_sym/3_fold_sym.poly')
t.write(f'regions/3_fold_sym/3_fold_sym.output.poly')

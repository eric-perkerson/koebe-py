conda deactivate
conda env remove -n ct
conda env create -f ct_env.yml
conda activate ct
conda install -c conda-forge fenics-dolfinx mpich
conda install meshio 
pip install gmsh

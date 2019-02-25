# Pearu's conda notes

## Building a numba-feedstock and uploading to anaconda pearu channel

The following instructions demonstrate how to rebuild a conda-forge package from your own branch. Don't use this example in production!
```
# Prerequisities
conda install conda-build -c conda-forge
conda install anaconda-client -c conda-forge
conda activate base
conda config --set anaconda_upload no
# Register in anaconda, see https://docs.anaconda.com/anaconda-cloud/user-guide/howto/#use-the-anaconda-client-cli

#
git clone https://github.com/Quansight/numba-feedstock.git
git checkout v0.42
conda build reciepe/
# remember the generated tar.bz2 file path
anaconda login
anaconda upload /home/pearu/miniconda3/conda-bld/linux-64/numba-0.42.1-py37hf484d3e_0.tar.bz2

# Testing
conda env create -n test-env1
conda activate test-env1
conda install -c pearu numba
conda list  # the output:
...
numba                     0.42.1           py37hf484d3e_0    pearu
...
```

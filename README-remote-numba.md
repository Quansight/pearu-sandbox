# Setting up remote numba dependencies in a conda environment

Remote numba project requires that llvmdev has been built against all stable targets. 
Atm, llvmdev-feedstock is built against host target
(see [llvmdev-feedstock PR 59](https://github.com/conda-forge/llvmdev-feedstock/pull/59) for the state)
and for remote numba project many conda packages must rebuilt. The building instructions will be given below.
However, if the output of
```
llvm-config --targets-built
```
reports all the targets you need, the rebuilding llvmdev and related packages will be unnecessary.

## Rebuilding llvmdev and related packages

```
conda activate base
conda install make cmake conda-build -c conda-forge
git clone https://github.com/conda-forge/llvmdev-feedstock.git
git clone https://github.com/conda-forge/clangdev-feedstock.git
git clone https://github.com/numba/llvmlite.git
git clone https://github.com/numba/numba.git

# Edit llvmdev-feedstock/recipe/build.sh by removing `-DLLVM_TARGETS_TO_BUILD=host`
# and increment the build number in `llvmdev-feedstock/recipe/meta.yaml`
conda build llvmdev-feedstock/recipe   # takes about 191m (user)

# Increment the build number in clangdev-feedstock/recipe/meta.yaml
conda build clangdev-feedstock/recipe  # takes about 84m (user)

cd llvmlite
git checkout f008359  # 0.27.1 release
cd ..
GIT_DESCRIBE_NUMBER=2000 conda build llvmlite/conda-recipes/llvmlite/

GIT_DESCRIBE_NUMBER=2000 conda build numba/buildscripts/condarecipe.local/
```

### Installing the built packages
```
conda install -n remote-numba-dev llvmdev=7.0.1 clangdev=7.0.1 llvmlite=0.27.1 numba -c local
```

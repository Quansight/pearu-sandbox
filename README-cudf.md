# Introduction

Recently, arrow CUDA support has been improved:

+ [ARROW-1424](https://github.com/apache/arrow/pull/2536) - add cuda support to pyarrow [MERGED]
+ [ARROW-3451](https://github.com/apache/arrow/pull/2732) - support numba.cuda and pyarrow.cuda interoperability [MERGED]
+ [ARROW-3624](https://github.com/apache/arrow/pull/2844) - support zero-sized device buffers and device-to-device copies [APPROVED]

The above makes it possbile to use arrow CudaBuffer for 
managing device memory within cudf (former pygdf) so 
that one can still use numba.jit decorated functions in cudf.
 
In the following we describe how to switch between cuda backends (numba.cuda or pyarrow.cuda)
within cudf. Since not all required features are available in arrow release yet
(currently the latest arrow version is 0.11.1),
we'll need to use arrow source from its repository.

# Setting up conda environment
```
conda create -n cudf-arrow python>=3.6 pytest cmake setuptools numpy cffi \
  numba>=0.40 pandas cython -c conda-forge
conda activate cudf-arrow
```

# Getting cudf branch with arrow support and preparing the build environment

```
git clone git@github.com:Quansight/pygdf.git
cd pygdf
git submodule update --init --recursive
git checkout cudf-arrow

export ARROW_GITHUB=https://github.com/Quansight/arrow
export PARQUET_ARROW_VERSION=cuda-buffer  # change to master when PR 2844 is merged

# Adjust if needed:
#export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-9.2/nvvm/libdevice
export NUMBAPRO_NVVM=/usr/local/cuda-9.2/nvvm/lib64/libnvvm.so
```

# Building libgdf

```
mkdir build-libgdf
cd build-libgdf
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../libgdf/
make -j4
make test
make pytest # expect test_arrow_availability to fail
```
It is expected that `test_arrow_availability` will fail because pyarrow is not installed. 
If pyarrow is installed, you should remove it because pyarrow needs to be built with 
CUDA support enabled, see below.

Install libgdf for cudf:
```
make install
python setup.py develop
```

# Building pyarrow with CUDA support

```
export ARROW_HOME=`pwd`/CMakeFiles/thirdparty/arrow-download/arrow-prefix/src/arrow-install/usr/local/
cd CMakeFiles/thirdparty/arrow-download/arrow-prefix/src/arrow/python/
python setup.py build_ext --with-cuda develop
cd -
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARROW_HOME/lib
```

# Building cudf

```
cd ..
python setup.py develop
```

# Testing cudf and libgdf

```
py.test -sv cudf
cd build-libgdf
make pytest  # now all tests should pass
```

Testing results:
```
cudf in master:
# 4 failed, 1020 passed, 18 skipped, 2 xfailed, 108 warnings in 117.85 seconds


```


# Prerequisites

[Building/testing libgdf with the latest arrow version](README-libgdf.md)

Assuming that the conda environment that has been used for installing libgdf (libgdf-arrow011, for instance) has been activated.

# Building pyarrow with CUDA support

This requires that libgdf build uses ` -DARROW_BUILD_SHARED=ON -DARROW_HDFS=ON -DARROW_GPU=ON -DARROW_COMPUTE=ON -DARROW_PYTHON=ON`.
```
cd libgdf/build/
cd CMakeFiles/thirdparty/arrow-download/arrow-prefix/src/arrow/python/
ARROW_HOME=../../arrow-install/usr/local/ python setup.py build_ext --with-cuda develop
cd -
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/CMakeFiles/thirdparty/arrow-download/arrow-prefix/src/arrow-install/usr/local/lib
cd ../pygdf
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-9.2/nvvm/libdevice
export NUMBAPRO_NVVM=/usr/local/cuda-9.2/nvvm/lib64/libnvvm.so
python setup.py develop
py.test -sv . # most tests should pass ok

```

# Building/testing pygdf

```
cd pygdf/
python setup.py develop
# required by numba:
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-9.2/nvvm/libdevice
export NUMBAPRO_NVVM=/usr/local/cuda-9.2/nvvm/lib64/libnvvm.so

py.test -sv .
```

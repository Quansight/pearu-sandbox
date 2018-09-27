# Prerequisites

[Building/testing libgdf with the latest arrow version](README-libgdf.md)

Assuming that the conda environment that has been used for installing libgdf (libgdf-arrow011, for instance) has been activated.

# Building pyarrow with CUDA support

This requires that libgdf build uses `-DARROW_GPU -DARROW_PYTHON`.
```
cd libgdf/build/
cd CMakeFiles/thirdparty/arrow-download/arrow-prefix/src/arrow/python/
python setup.py develop --with-cuda
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

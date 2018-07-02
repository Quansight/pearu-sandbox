# Prerequisites

[Building/testing libgdf with arrow version 0.9.0](README-libgdf.md)

Assuming that libgdf_dev090 conda environment has been activated.

# Building/testing pygdf

```
conda install pandas
cd pygdf/
python setup.py develop
# required by numba:
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-9.2/nvvm/libdevice
export NUMBAPRO_NVVM=/usr/local/cuda-9.2/nvvm/lib64/libnvvm.so

py.test -sv .
```

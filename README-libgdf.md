# Introduction

libgdf was developed with arrow 0.7.1. In arrow >= 0.8 the ipc code changed in a way that broke libgdf.
In pearu-sandbox/libgdf the corresponding code is disabled along with test_ipc. This alone allows one to
succesfully build and test libgdf using the recent version of arrow (currently it is 0.9.0).

# Building libgdf against arrow 0.9.0 and Python 3.6

```
conda env create --name libgdf_dev-arrow090 --file libgdf/conda_environments/dev_py36.yml
conda activate libgdf_dev-arrow090
conda update arrow-cpp # otherwise expect './libgdf.so: undefined symbol: _ZN5arrow3ipc8internal4json10JsonWriter6FinishEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE'
mkdir build-libgdf
cd build-libgdf
cmake -DARROW_METADATA_VERSION=4 ../libgdf/
make
make pytest
```

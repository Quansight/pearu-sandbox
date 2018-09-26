# Introduction

# Building libgdf against arrow-master and Python 3.7

```
conda create -n libgdf-arrow011 python=3.7 pytest cmake setuptools numpy cffi -c conda-forge
conda activate libgdf-arrow011
cd git/libgdf
git submodule update --init --recursive
mkdir build
cd build
export PARQUET_ARROW_VERSION=master
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make
make pytest
make install
python setup.py install
```

# Building libgdf against arrow-0.7.1 and Python 3.6

libgdf was developed with arrow 0.7.1. In arrow >= 0.8 the ipc code changed in a way that broke libgdf.
In pearu-sandbox/libgdf the corresponding code is disabled along with test_ipc. This alone allows one to
succesfully build and test libgdf using the recent version of arrow (currently it is 0.9.0).


```
conda env create --name libgdf_dev-arrow071 --file libgdf/conda_environments/dev_py36.yml
conda activate libgdf_dev-arrow071
conda install arrow-cpp=0.7.1 pyarrow=0.7.1 -c conda-forge # downgrades arrow
cd build-libgdf/
rm -rf . # clean up
cd ../libgdf/thirdparty/cub && git checkout b165e1f && cd -
cd ../libgdf/thirdparty/moderngpu && git checkout c1fd31d && cd -
cmake -DARROW_METADATA_VERSION=3 -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../libgdf/
make
make pytest
make install
python setup.py install
```


# Building libgdf against arrow 0.9.0 and Python 3.6

```
conda env create --name libgdf_dev-arrow090 --file libgdf/conda_environments/dev_py36.yml
conda activate libgdf_dev-arrow090
conda update arrow-cpp # otherwise expect './libgdf.so: undefined symbol: _ZN5arrow3ipc8internal4json10JsonWriter6FinishEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE'
mkdir build-libgdf
cd build-libgdf
cmake -DARROW_METADATA_VERSION=4 -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../libgdf/
make
make pytest

# install for pygdf
make install
python setup.py install
```

# Next

[Building/testing pygdf](README-pygdf.md)

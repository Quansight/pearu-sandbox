
# Developing arrow on Ubuntu 18.04 with CUDA

```
conda create -n pyarrow-dev
conda install python numpy six setuptools cython pandas pytest \
      cmake flatbuffers rapidjson boost-cpp thrift-cpp snappy zlib \
      gflags brotli jemalloc lz4-c zstd \
      double-conversion glog autoconf hypothesis numba \
      clangdev=6 flake8 gtest \
      -c conda-forge
cd git/Quansight
git clone https://github.com/quansight/arrow.git
```

```
conda activate pyarrow-dev
cd arrow
export ARROW_BUILD_TYPE=release
export ARROW_BUILD_TOOLCHAIN=$CONDA_PREFIX
export ARROW_HOME=$CONDA_PREFIX
export PARQUET_HOME=$CONDA_PREFIX
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-9.2/nvvm/libdevice
export NUMBAPRO_NVVM=/usr/local/cuda-9.2/nvvm/lib64/libnvvm.so

cmake -DCMAKE_BUILD_TYPE=$ARROW_BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
      -DARROW_PARQUET=off  -DARROW_PYTHON=on  \
      -DARROW_PLASMA=off -DARROW_BUILD_TESTS=OFF \
      -DARROW_CUDA=on \
      -DCLANG_FORMAT_BIN=`which clang-format` \
      ..
make -j3
make install
make format # after changing cpp/ files
cd ../../python
python setup.py build_ext --build-type=$ARROW_BUILD_TYPE --with-cuda develop
py.test -sv pyarrow/

```

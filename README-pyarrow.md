
# Developing arrow on Ubuntu 18.04 with CUDA

```
conda create -n pyarrow-dev
conda install python numpy six setuptools cython pandas pytest \
      cmake flatbuffers rapidjson boost-cpp thrift-cpp snappy zlib \
      gflags brotli jemalloc lz4-c zstd \
      double-conversion glog autoconf \
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

cmake -DCMAKE_BUILD_TYPE=$ARROW_BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
      -DARROW_PARQUET=off  -DARROW_PYTHON=on  \
      -DARROW_PLASMA=off -DARROW_BUILD_TESTS=OFF \
      -DARROW_CUDA=on ..
make -j3
```

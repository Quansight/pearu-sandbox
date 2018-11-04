
# Building mapd-core in Ubuntu 16.04 and 18.04 while using conda dependencies

## Prepare conda environment with mapd-core dependencies

```
conda create -n omnisci-dev python>=3.6 pytest cmake setuptools numpy numba>=0.40 \
  clangdev=6 llvmdev=6 arrow-cpp>=0.11 boost-cpp=1.67 boost=1.67 go gperftools gdal \
  thrift-cpp=0.11.0 thrift=0.11.0 gflags glog libarchive maven bisonpp flex \
  gxx_linux-64 doxygen -c conda-forge
conda activate omnisci-dev
# Ubuntu 16.04:
conda install gxx_linux-64 # provides g++ 7.2
# Ubuntu 18.04: it already has g++ 7.3
# even when using g++ for compilation, clangdev dependency is still required
```

## Check out mapd-core and prepare the build directory

```
cd git
git clone https://github.com/Quansight/mapd-core
mkdir build-mapd-core
cd build-mapd-core
```

## Run cmake

```
export PREFIX=$CONDA_PREFIX
export CXXFLAGS="-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0"
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"

cmake \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_BUILD_TYPE=debug \
      -DENABLE_AWS_S3=off \
      -DENABLE_FOLLY=off \
      -DENABLE_JAVA_REMOTE_DEBUG=off \
      -DMAPD_IMMERSE_DOWNLOAD=off \
      -DMAPD_DOCS_DOWNLOAD=off \
      -DPREFER_STATIC_LIBS=off \
      -DENABLE_CUDA=off \
  ../mapd-core
```

## Compile and build

```
make -j4
```

## Testing

```
# in another terminal run:
bin/initdb data
bin/mapd_server
#
make sanity_tests
# here 6 out of 14 tests fail.., both Ubuntu 16.04 and 18.04:
The following tests FAILED:
	  2 - UpdelStorageTest (Failed)
	  3 - ImportTest (SEGFAULT)
	  4 - AlterColumnTest (Failed)
	  6 - ExecuteTest (SEGFAULT)
	 13 - TopKTest (Failed)
	 18 - CtasTest (Failed)
```

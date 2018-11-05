
# Building mapd-core while using conda dependencies

Testing on the following platforms (all have GPU card Quadro P2000):

1. Ubuntu 16.04
2. KVM client Ubuntu 16.04
3. KVM client Ubuntu 18.04
4. KVM client Centos 7

## Prepare conda environment with mapd-core dependencies

```
conda create -n omnisci-dev python>=3.6 pytest cmake setuptools numpy numba>=0.40 \
  clangdev=6 llvmdev=6 arrow-cpp>=0.11 boost-cpp=1.67 boost=1.67 go gperftools gdal \
  thrift-cpp=0.11.0 thrift=0.11.0 gflags glog libarchive maven bisonpp flex \
  doxygen -c conda-forge
conda activate omnisci-dev
# Ubuntu 16.04: it has g++ 5.4
conda install gxx_linux-64 -c conda-forge # provides g++ 7.2
# Ubuntu 18.04: it already has g++ 7.3

# Centos 7.0: it has g++ 4.8.5
conda install zlib gxx_linux-64 -c conda-forge

# even when using g++ for compilation, clangdev dependency is still required
```

## Check out mapd-core and prepare the build directory

```
cd git
git clone https://github.com/Quansight/mapd-core
mkdir mapd-core/build && cd mapd-core/build   # note: build directory must be inside mapd-core for tests
```

## Run cmake

```
export PREFIX=$CONDA_PREFIX
export CXXFLAGS="-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0"
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"

# Centos 7.0:
export CMAKE_COMPILERS="" #"-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
export ZLIB_ROOT=$PREFIX
export CXXFLAGS="$CXXFLAGS -msse4.1"

# Ubuntu 16.04, 18.04:
export CMAKE_COMPILERS=""

#
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
      $CMAKE_COMPILERS \
  ..
```

Setting `-DENABLE_CUDA=on` fails on Ubuntu 16.04 (not finding librt) but works on Ubuntu 18.04.

## Compile and build

```
make -j4
```

## Testing

```
mkdir tmp && bin/initdb tmp
make sanity_tests
# here 6 out of 14 tests fail in: Ubuntu 16.04 (CPU), 18.04 (CPU), 18.04 (GPU), Centos 7.0 (CPU)
The following tests FAILED:
	  2 - UpdelStorageTest (Failed)
	  3 - ImportTest (Failed)
	  4 - AlterColumnTest (Failed)
	  6 - ExecuteTest (SEGFAULT)
	 13 - TopKTest (Failed)
	 18 - CtasTest (Failed)
```

Test logs show that many of the above failing tests acctually pass OK but there seems to be issues with logging, for instance:
```
18: pure virtual method called
18: terminate called without an active exception
18: E1105 19:14:28.051285  8062 QueryRunner.cpp:111] Interrupt signal (6) received.
18: WARNING: Logging before InitGoogleLogging() is written to STDERR
18: I1105 19:14:28.056200  8062 Calcite.cpp:447] Destroy Calcite Class
18: I1105 19:14:28.056221  8062 Calcite.cpp:449] End of Calcite Destructor 
14/14 Test #18: CtasTest .........................***Failed  280.83 sec
```

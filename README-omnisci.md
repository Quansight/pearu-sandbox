
# Building mapd-core while using conda dependencies: CXX11ABI packages

## Setup

```
cd git
git clone https://github.com/Quansight/mapd-core
# Install system dependencies:
sudo apt-get install libc6-dev gcc g++
# Setup conda environment:
wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/omnisci-dev-ubuntu18.yaml
conda env create --file=omnisci-dev-ubuntu18.yaml -n omnisci-dev-cpu
conda activate omnisci-dev-cpu
# Build mapd-core:
cd git/mapd-core/
```

### KVM client Ubuntu 18.04 - CUDA disabled

```
export PREFIX=$CONDA_PREFIX
mkdir build-cpu && cd build-cpu
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=debug \
  -DENABLE_AWS_S3=off -DENABLE_FOLLY=off -DENABLE_JAVA_REMOTE_DEBUG=off \
  -DMAPD_IMMERSE_DOWNLOAD=off -DMAPD_DOCS_DOWNLOAD=off -DPREFER_STATIC_LIBS=off \
  -DENABLE_CUDA=off -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DENABLE_PROFILER=off -DMAPD_EDITION=OS ..
```
### KVM client Ubuntu 18.04 - CUDA enabled

Make sure that the output of `llvm-config --targets-built` contains `NVPTX`. If not, see below how to rebuild `llvmdev` and `clangdev`.

```
mkdir build-cuda && cd build-cuda
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=debug \
  -DENABLE_AWS_S3=off -DENABLE_FOLLY=off -DENABLE_JAVA_REMOTE_DEBUG=off \
  -DMAPD_IMMERSE_DOWNLOAD=off -DMAPD_DOCS_DOWNLOAD=off -DPREFER_STATIC_LIBS=off \
  -DENABLE_CUDA=on -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DENABLE_PROFILER=off -DMAPD_EDITION=OS ..
```

## Building and testing mapd-core

```
make -j `nproc`
# Run sanity tests:
mkdir tmp && bin/initdb tmp
make sanity_tests
# Basic usage:
mkdir data && bin/initdb data
# run bin/mapd_server in another console
bash ../insert_sample_data # select table flights_2008_10k
bin/mapdql -p HyperInteractive
```

## Building llvmdev and clangdev with NVPTX target support

If [llvmdev-feedstock PR 59](https://github.com/conda-forge/llvmdev-feedstock/pull/59) has not been merged then rebuild `llvmdev` and `clangdev` conda packages as follows:

```
conda activate base
conda install make cmake conda-build -c conda-forge
git clone https://github.com/conda-forge/llvmdev-feedstock.git
git clone https://github.com/conda-forge/clangdev-feedstock.git

# Edit llvmdev-feedstock/recipe/build.sh by removing `-DLLVM_TARGETS_TO_BUILD=host`
# and increment build number in llvmdev-feedstock/recipe/meta.yaml
conda build llvmdev-feedstock/recipe   # takes about 191m (user)

# Increment build number in clangdev-feedstock/recipe/meta.yaml
conda build clangdev-feedstock/recipe  # takes about 84m (user)
```
To install the build llvmdev and clangdev, run
```
conda install -n omnisci-dev-cuda llvmdev=7.0.1 clangdev=7.0.1 -c local
```

# Building mapd-core while using conda dependencies: pre-CXX11ABI packages

Testing on the following platforms (all have GPU card Quadro P2000):

1. Ubuntu 16.04
2. KVM client Ubuntu 16.04
3. KVM client Ubuntu 18.04
4. KVM client Centos 7
5. Darwin 18.0 

## Prepare conda environment with mapd-core dependencies

```
conda create -n omnisci-dev python>=3.6 pytest cmake setuptools numpy numba>=0.40 \
  clangdev=6 llvmdev=6 arrow-cpp>=0.11 boost-cpp=1.67 boost=1.67 go gperftools gdal \
  thrift-cpp=0.11.0 thrift=0.11.0 gflags glog libarchive maven bisonpp flex \
  cython pyarrow>=0.11 parquet-cpp blosc \
  doxygen -c conda-forge
conda activate omnisci-dev
# Ubuntu 16.04: it has g++ 5.4
conda install gxx_linux-64 -c conda-forge # provides g++ 7.2

# Ubuntu 18.04: it already has g++ 7.3 and using conda gxx_linux-64 will lead
# to build failures (not finding librt)

# Centos 7.0: it has g++ 4.8.5
conda install zlib gxx_linux-64 -c conda-forge

# Darwin 18.0
conda install openjdk=8 -c conda-forge

# even when using g++ for compilation, clangdev dependency is still required
```

If GPU enabled mapd server is required then skip installing `clangdev=6 llvmdev=6`, see below.

## Check out mapd-core and prepare the build directory

```
cd git
git clone https://github.com/Quansight/mapd-core
mkdir mapd-core/build && cd mapd-core/build   # note: build directory must be inside mapd-core for tests
```

## Run cmake

```
# Centos 7.0:
export PREFIX=$CONDA_PREFIX
export CMAKE_COMPILERS="" #"-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
export CXXFLAGS="-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0"
# with the latest conda use _GLIBCXX_USE_CXX11_ABI default 1
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"
export ZLIB_ROOT=$PREFIX
export CXXFLAGS="$CXXFLAGS -msse4.1"

# Ubuntu 16.04:
export PREFIX=$CONDA_PREFIX
export CXXFLAGS="-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0"
# with the latest conda use _GLIBCXX_USE_CXX11_ABI default 1
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"
export CMAKE_COMPILERS=""

# Ubuntu 18.04:
export PREFIX=$CONDA_PREFIX
export CMAKE_COMPILERS=""
export CXXFLAGS="-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0"
# with the latest conda use _GLIBCXX_USE_CXX11_ABI default 1
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"

# Darwin 18.0:
export PREFIX=$CONDA_PREFIX
export CMAKE_COMPILERS="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
export CXXFLAGS="-std=c++14"
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"
export ZLIB_ROOT=$PREFIX
export LibArchive_ROOT=$PREFIX

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
# make sure that cmake finds all required packages from conda environment
```

Setting `-DENABLE_CUDA=on` fails on Ubuntu 16.04 (not finding `librt`) but works on Ubuntu 18.04.

Ubuntu 18.04, using `-DPREFER_STATIC_LIBS=on` requires `LDFLAGS="-ldl"`

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

## Using mapd-core via pymapd

```
conda install -c conda-forge pymapd

mkdir data && bin/initdb data
# in another terminal, run the server: bin/mapd_server
bash ../insert_sample_data # select table flights_2008_10k
```
In Python, execute
```
from pymapd import connect
con = connect(user="mapd", password= "HyperInteractive", host="localhost", dbname="mapd")
c = con.cursor()
c.execute("SELECT * FROM flights_2008_10k LIMIT 100")
print(list(c)[0])
```

## GPU enabled mapd-core

GPU enabled mapd-core requires
1. using `-DENABLE_CUDA=on` in mapd-core cmake configuration
2. llvm that supports `NVPTX` target.

At the moment of writing this, conda-forge provided llvmdev does not include the `NVPTX` target support 
and accessing mapd server will result with the following failure:
```
F0106 16:50:27.459915 18721 NativeCodegen.cpp:675] No available targets are compatible with this triple.
*** Check failure stack trace: ***
```

To verify whether a llvm installation has the required target support, run
```
llvm-config --targets-built
```
If the output contains `NVPTX`, then llvm is good. Otherwise, one must obtain llvm that supports `NVPTX` target.

I chosed to rebuild conda-forge [llvmdev-feedstock](https://github.com/conda-forge/llvmdev-feedstock) and [clangdev-feedstock](https://github.com/conda-forge/clangdev-feedstock) locally:

```
conda activate base
git clone https://github.com/conda-forge/llvmdev-feedstock
git clone https://github.com/conda-forge/clangdev-feedstock.git

export CXXFLAGS="-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0"

# Edit llvmdev-feedstock/recipe/build.sh to include `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"` cmake argument
# and increment build number in llvmdev-feedstock/recipe/meta.yaml
conda build llvmdev-feedstock/recipe   # takes about 96m (user)

# Edit clangdev-feedstock/recipe/meta.yaml to include the following two lines:
{% set version = "7.0.1" %}
{% set sha256 = "a45b62dde5d7d5fdcdfa876b0af92f164d434b06e9e89b5d0b1cbc65dfe3f418" %}

conda build clangdev-feedstock/recipe  # takes about 84m (user)
```
To install the build llvmdev and clangdev, run
```
conda install -n omnisci-dev --use-local llvmdev=7.0.1 clangdev=7.0.1
```
Now continue with mapd-core building (see above) with the following cmake command:
```
cmake \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_BUILD_TYPE=debug \
      -DENABLE_AWS_S3=off \
      -DENABLE_FOLLY=off \
      -DENABLE_JAVA_REMOTE_DEBUG=off \
      -DMAPD_IMMERSE_DOWNLOAD=off \
      -DMAPD_DOCS_DOWNLOAD=off \
      -DPREFER_STATIC_LIBS=off \
      -DENABLE_CUDA=on \
      $CMAKE_COMPILERS \
      -DMAPD_EDITION=CE \
  ..
```

## Possible issues and solutions

### mapd-core build fail
```
[ 88%] Linking CXX executable bin/mapd_server

QueryEngine/libQueryEngine.a(NativeCodegen.cpp.o): In function `Executor::initializeNVPTXBackend() const':
/home/pearu/git/omnisci/mapd-core-internal/QueryEngine/NativeCodegen.cpp:673: undefined reference to `llvm::TargetRegistry::lookupTarget(std::string const&, std::string&)'
collect2: error: ld returned 1 exit status
```
This problem appears when linking together libraries using different C++ ABI verions.

### mapd-core build fails
```
[ERROR] COMPILATION ERROR : 
[ERROR] /Users/user/dev/pearu/git/mapd-core/java/thrift/src/gen/com/mapd/thrift/server/TRenderParseResult.java:[10,18] package javax.annotation does not exist
...
```
As a solution, use `openjdk=8` when installing conda packages.

### mapd-core build fails
```
[ 55%] Linking CXX executable bin/initdb
Undefined symbols for architecture x86_64:
  "_archive_read_free", referenced from:
      Archive::~Archive() in libCsvImport.a(Importer.cpp.o)
  "_archive_read_support_filter_bzip2", referenced from:
      Archive::Archive(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, bool) in libCsvImport.a(Importer.cpp.o)
...
```
As a solution, use `export LibArchive_ROOT=$PREFIX` prior `cmake`.

### conda CC env

```
$ echo $CC
/use/the/cgo/conda/package/instead
```
Solution:
```
export CC=
export CXX=
```


### mapd-core build fail

```
[ 53%] Generating ../libjwt.a, ../libjwt.h
can't load package: package main: build constraints exclude all Go files in /home/pearu/git/omnisci/mapd-core-internal/Licensing
Licensing/CMakeFiles/Licensing.dir/build.make:61: recipe for target 'libjwt.a' failed
make[2]: *** [libjwt.a] Error 1
```
Solution?:
```
conda install -c conda-forge go-cgo  # ???
CGO_ENABLED=1 CC=clang CGO_LDFLAGS= CGO_CFLAGS= CGO_CPPFLAGS=  make
```


### mapd-code build fail

```
In file included from /home/pearu/git/Quansight/mapd-core/QueryEngine/DecodersImpl.h:28:0,
                 from /home/pearu/git/Quansight/mapd-core/QueryEngine/RuntimeFunctions.cpp:37:
/home/pearu/git/Quansight/mapd-core/QueryEngine/DecodersImpl.h: In function 'int64_t fixed_width_int_decode_noinline(const int8_t*, int32_t, int64_t)':
/home/pearu/git/Quansight/mapd-core/QueryEngine/DecodersImpl.h:31:8: error: inlining failed in call to always_inline 'int64_t fixed_width_int_decode(const int8_t*, int32_t, int64_t)': function body can be overwritten at link time
 SUFFIX(fixed_width_int_decode)(const int8_t* byte_stream,
        ^
/home/pearu/git/Quansight/mapd-core/QueryEngine/../Shared/funcannotations.h:63:22: note: in definition of macro 'SUFFIX'
 #define SUFFIX(name) name
                      ^~~~
In file included from /home/pearu/git/Quansight/mapd-core/QueryEngine/RuntimeFunctions.cpp:37:0:
/home/pearu/git/Quansight/mapd-core/QueryEngine/DecodersImpl.h:86:69: note: called from here
   return SUFFIX(fixed_width_int_decode)(byte_stream, byte_width, pos);
```

Solution:
```
export CXXFLAGS="$CXXFLAGS -msse4.1"  # ?
```

### mapd-code build fail

```
[ 58%] Linking CXX executable bin/mapd_server
CMakeFiles/mapd_server.dir/MapDServer.cpp.o: In function `boost::system::system_category()':
/home/pearu/miniconda3/envs/omnisci-conda/include/boost/system/error_code.hpp:472: undefined reference to `boost::system::detail::system_category_instance'
```
Solution:
```
export CXXFLAGS="$CXXFLAGS -DBOOST_ERROR_CODE_HEADER_ONLY"
```
and rerun cmake and make.

### Using mapd-core version 4.5.0 or older and /usr/bin/java is missing

This leads to a crash when running `initdb tmp`, for instance.

Solution:
```
sed -i 's/\/usr\/bin\/java/'`which java|sed 's/\//\\\\\//g'`'/g' Calcite/Calcite.cpp
```
and rebuild. Or grap mapd-core from its git repo.

### cmake warning

```
CMake Warning at /home/pearu/miniconda3/envs/omnisci-gpu-dev/share/cmake-3.14/Modules/FindCUDA.cmake:893 (message):
  Expecting to find librt for libcudart_static, but didn't find it.
```
Solution:
```
ln -s /usr/lib/x86_64-linux-gnu/librt.a $CONDA_PREFIX/lib/
```

### cmake failure

```
CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
Please set them or make sure they are set and tested correctly in the CMake files:
CUDA_CUDA_LIBRARY (ADVANCED)
```
Solution:
```
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $CONDA_PREFIX/lib/
```

### mapd-core build failure

```
 44%] Building CXX object Tests/CMakeFiles/StringDictionaryTest.dir/StringDictionaryTest.cpp.o
/home/pearu/miniconda3/envs/omnisci-gpu-dev/bin/ld: /home/pearu/miniconda3/envs/omnisci-gpu-dev/bin/../x86_64-conda_cos6-linux-gnu/sysroot/lib/librt.so.1: undefined reference to `__vdso_clock_gettime@GLIBC_PRIVATE'
clang-7: error: linker command failed with exit code 1 (use -v to see invocation)
```
Solution: ?

### mapd-core build failure

```
[ 46%] Linking CXX executable ../bin/omnisci_sd_server
/home/pearu/miniconda3/envs/omnisci-gpu-dev/bin/ld: ../Shared/libShared.a(File.cpp.o): in function `boost::system::error_code::error_code()':
/home/pearu/miniconda3/envs/omnisci-gpu-dev/include/boost/system/error_code.hpp:636: undefined reference to `boost::system::detail::system_category_instance'
/home/pearu/miniconda3/envs/omnisci-gpu-dev/bin/ld: /home/pearu/miniconda3/envs/omnisci-gpu-dev/include/boost/system/error_code.hpp:636: undefined reference to `boost::system::detail::system_category_instance'
/home/pearu/miniconda3/envs/omnisci-gpu-dev/bin/ld: /home/pearu/miniconda3/envs/omnisci-gpu-dev/include/boost/system/error_code.hpp:636: undefined reference to `boost::system::detail::system_category_instance'
/home/pearu/miniconda3/envs/omnisci-gpu-dev/bin/ld: /home/pearu/miniconda3/envs/omnisci-gpu-dev/bin/../x86_64-conda_cos6-linux-gnu/sysroot/lib/librt.so.1: undefined reference to `__vdso_clock_gettime@GLIBC_PRIVATE'
clang-7: error: linker command failed with exit code 1 (use -v to see invocation)
```
Solution: ?

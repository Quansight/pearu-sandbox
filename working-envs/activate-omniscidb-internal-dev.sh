#
# Prepare pytorch development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of omniscidb-cuda-dev or omniscidb-cpu-dev conda environment
#
# Author: Pearu Peterson
# Created: November 2019
#

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=release -DMAPD_EDITION=EE -DMAPD_DOCS_DOWNLOAD=off -DENABLE_AWS_S3=off -DENABLE_FOLLY=off -DENABLE_JAVA_REMOTE_DEBUG=off -DENABLE_PROFILER=off -DPREFER_STATIC_LIBS=off"
export CMAKE_OPTIONS_CUDA_EXTRA=""
export CMAKE_OPTIONS_NOCUDA_EXTRA="-DENABLE_CUDA=off"

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    
    # . /usr/local/cuda-10.1.243/env.sh
    test -f /usr/local/cuda-10.2.89/env.sh && source /usr/local/cuda-10.2.89/env.sh || source /usr/local/cuda-10.1.243/env.sh
 
    export CMAKE_OPTIONS_CUDA_EXTRA="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DENABLE_CUDA=on"
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/omniscidb-dev.yaml
    # conda env create  --file=omniscidb-dev.yaml -n omniscidb-cuda-dev
    #
    # conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/omniscidb-dev.yaml -n omniscidb-cuda-dev
    #
    # conda install -y -n omniscidb-cuda-dev -c conda-forge nvcc_linux-64
    USE_ENV="${USE_ENV:-omniscidb-cuda-dev}"
    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda $USE_ENV
    else
        conda activate $USE_ENV
    fi
    export CXXFLAGS="$CXXFLAGS -I$CUDA_HOME/include"
    export CPPFLAGS="$CPPFLAGS -I$CUDA_HOME/include"
    export CFLAGS="$CFLAGS -I$CUDA_HOME/include"
    export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"

else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/omniscidb-dev.yaml
    # conda env create  --file=omniscidb-dev.yaml -n omniscidb-cpu-dev
    USE_ENV="${USE_ENV:-omniscidb-cpu-dev}"

    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda $USE_ENV
    else
        conda activate $USE_ENV
    fi
fi


export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot

export CXXFLAGS="`echo $CXXFLAGS | sed 's/-fPIC//'`"
export CXXFLAGS="$CXXFLAGS -DBOOST_ERROR_CODE_HEADER_ONLY"
export CXXFLAGS="$CXXFLAGS -D__STDC_FORMAT_MACROS"
export CXXFLAGS="$CXXFLAGS -Dsecure_getenv=getenv"

#export CC=$CONDA_PREFIX/bin/clang
#export CXX=$CONDA_PREFIX/bin/clang++

export CMAKE_CC=$CC
export CMAKE_CXX=$CXX

export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_C_COMPILER=$CMAKE_CC -DCMAKE_CXX_COMPILER=$CMAKE_CXX"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_SYSROOT=$CONDA_BUILD_SYSROOT"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_TESTS=on"
export CMAKE_OPTIONS_NOCUDA="$CMAKE_OPTIONS $CMAKE_OPTIONS_NOCUDA_EXTRA"
export CMAKE_OPTIONS_CUDA="$CMAKE_OPTIONS $CMAKE_OPTIONS_CUDA_EXTRA"

# resolves `fatal error: boost/regex.hpp: No such file or directory`
echo -e "#!/bin/sh\n${CUDA_HOME}/bin/nvcc -ccbin $CC -v \$@" > $PWD/nvcc
chmod +x $PWD/nvcc
export PATH=$PWD:$PATH

# resolves UdfTest fatal error: 'cstdint' file not found
test -f nvcc-boost-include-dirs.patch || wget https://raw.githubusercontent.com/conda-forge/omniscidb-cuda-feedstock/master/recipe/recipe/nvcc-boost-include-dirs.patch
test -f get_cxx_include_path.sh || wget https://raw.githubusercontent.com/conda-forge/omniscidb-cuda-feedstock/master/recipe/recipe/get_cxx_include_path.sh
. get_cxx_include_path.sh
export CPLUS_INCLUDE_PATH=$(get_cxx_include_path)

echo -e "Local branches:\n"
git branch

function h () {
cat << EndOfMessage

To select conda environment, define:

  export USE_ENV=omniscidb-cuda-dev

for instance, before sourcing this script.

To apply patches, run:

  patch -p1 < nvcc-boost-include-dirs.patch  [apply for omniscidb 5.0]

To configure, run:

  mkdir -p build-nocuda && cd build-nocuda
  cmake -Wno-dev \$CMAKE_OPTIONS_NOCUDA ..

  mkdir -p build && cd build
  cmake -Wno-dev \$CMAKE_OPTIONS_CUDA ..

To build, run:

  make -j $NCORES

To test, run:

  mkdir tmp && bin/initdb tmp
  make sanity_tests

To serve, run:

  mkdir data && bin/initdb data
  bin/omnisci_server --enable-runtime-udf --enable-table-functions

EndOfMessage

}

h

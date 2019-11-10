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

export CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=release -DMAPD_DOCS_DOWNLOAD=off -DENABLE_AWS_S3=off -DENABLE_FOLLY=off -DENABLE_JAVA_REMOTE_DEBUG=off -DENABLE_PROFILER=off -DPREFER_STATIC_LIBS=off"

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    . /usr/local/cuda-10.1.243/env.sh

    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/omniscidb-dev.yaml
    # conda env create  --file=omniscidb-dev.yaml -n omniscidb-cuda-dev
    #
    # conda env create  --file=git/Quansight/pearu-sandbox/conda-envs/omniscidb-dev.yaml -n omniscidb-cuda-dev
    #
    # conda install -y -n omniscidb-cuda-dev -c conda-forge nvcc_linux-64

    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda omniscidb-cuda-dev
    else
        conda activate omniscidb-cuda-dev
    fi
    #export CXXFLAGS="$CXXFLAGS -L$CUDA_HOME/lib64"
    export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"

    export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DENABLE_CUDA=on"
else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/omniscidb-dev.yaml
    # conda env create  --file=omniscidb-dev.yaml -n omniscidb-cpu-dev
    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda omniscidb-cpu-dev
    else
        conda activate omniscidb-cpu-dev
    fi
    export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_CUDA=off"
fi

export CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
export NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export LDFLAGS="`echo $LDFLAGS | sed 's/-Wl,--as-needed//'`"
export LDFLAGS="$LDFLAGS -lresolv -pthread -lrt"
export CXXFLAGS="`echo $CXXFLAGS | sed 's/-fPIC//'`"
export CXXFLAGS="$CXXFLAGS -DBOOST_ERROR_CODE_HEADER_ONLY"
export CFLAGS="`echo $CFLAGS | sed 's/-fPIC//'`"
export CC=${CONDA_PREFIX}/bin/clang
export CXX=${CONDA_PREFIX}/bin/clang++
export EXTRA_CMAKE_OPTIONS="$EXTRA_CMAKE_OPTIONS -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX}"
export CMAKE_OPTIONS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX $CMAKE_OPTIONS"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_TESTS=off"

GCCVERSION=$(basename $(dirname $($GXX -print-libgcc-file-name)))
GXXINCLUDEDIR=$CONDA_PREFIX/$HOST/include/c++/$GCCVERSION
export CXXFLAGS="$CXXFLAGS -I$GXXINCLUDEDIR -I$GXXINCLUDEDIR/$HOST -I$GXXINCLUDEDIR/backward -I$GXXINCLUDEDIR/include"
#export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/$HOST/include/c++/$GCCVERSION:$CONDA_PREFIX/lib/gcc/$HOST/$GCCVERSION/include

sed -i 's!ARGS -std=c++14!ARGS -std=c++14 -I'$GXXINCLUDEDIR' -I'$GXXINCLUDEDIR/$HOST'!g' QueryEngine/CMakeLists.txt
sed -i 's!arg_vector\[3\] = {arg0, arg1!arg_vector\[4\] = {arg0, arg1, "-extra-arg=-I'$GXXINCLUDEDIR' -I'$GXXINCLUDEDIR/$HOST'"!g' QueryEngine/UDFCompiler.cpp


echo -e "Local branches:\n"
git branch

cat << EndOfMessage

To configure, run:

  mkdir -p build && cd build

  cmake -Wno-dev \$CMAKE_OPTIONS ..

To build, run:

  make -j $NCORES

To test, run:

  make sanity_tests

EndOfMessage



let overlay = self: super: {
      arrow-cpp = super.callPackage ./arrow-cpp-0.13.0.nix { };
    };

    pkgs = import (builtins.fetchTarball {
      url = "https://github.com/NixOS/nixpkgs-channels/archive/42f0be81ae05a8fe6d6e8e7f1c28652e7746e046.tar.gz";
      sha256 = "1rxb5kmghkzazqcv4d8yczdiv2srs4r7apx4idc276lcikm0hdmf";
    }) { overlays = [ overlay ]; };

    pythonPackages = pkgs.python3Packages;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake
    go # includes cgo
    maven
    boost
    llvmPackages_8.llvm # 5, 6, 7, 8, 9, 10 are options
    llvmPackages_8.clang # again 5, 6, 7, 8, 9, 10 are options
    snappy
    gflags
    glog
    libarchive
    # libkml   # missing in Nix
    libpng
    libiconv
    c-blosc
    gdal
    arrow-cpp
    thrift
    ncurses
    flex
    # bisonpp   # missing in Nix
    openssl
    openjdk8
    xz
    bzip2
    zlib
    rdkafka
    curl
    double-conversion
    # when using omniscidb-internal repo
    openldap
    cudatoolkit
  ];
  shellHook = ''

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=release -DMAPD_EDITION=EE -DMAPD_DOCS_DOWNLOAD=off -DENABLE_AWS_S3=off -DENABLE_FOLLY=off -DENABLE_JAVA_REMOTE_DEBUG=off -DENABLE_PROFILER=off -DPREFER_STATIC_LIBS=off"

if [[ -x "$(command -v nvidia-smi)" ]]
then
    #. /usr/local/cuda-10.2.89/env.sh


    if [[ -z "$CUDA_HOME" ]]
    then
      CUDA_GDB_EXECUTABLE=$(which cuda-gdb || exit 0)
      if [[ -n "$CUDA_GDB_EXECUTABLE" ]]
      then
        export CUDA_HOME=$(dirname $(dirname $CUDA_GDB_EXECUTABLE))
      else
        echo "Cannot determine CUDA_HOME: cuda-gdb not in PATH"
        return 1
      fi
    fi
    export CMAKE_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib/stubs/
    export CXXFLAGS="$CXXFLAGS -I$CUDA_HOME/include"
    export CPPFLAGS="$CPPFLAGS -I$CUDA_HOME/include"
    export CFLAGS="$CFLAGS -I$CUDA_HOME/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath-link,$CUDA_HOME/lib64 -L$CUDA_HOME/lib64"
    export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DENABLE_CUDA=on"
else
    export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_CUDA=off"
fi

#export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot

export CXXFLAGS="`echo $CXXFLAGS | sed 's/-fPIC//'`"
export CXXFLAGS="$CXXFLAGS -DBOOST_ERROR_CODE_HEADER_ONLY"
export CXXFLAGS="$CXXFLAGS -D__STDC_FORMAT_MACROS"
export CXXFLAGS="$CXXFLAGS -Dsecure_getenv=getenv"

#export CC=$CONDA_PREFIX/bin/clang
#export CXX=$CONDA_PREFIX/bin/clang++

export CMAKE_CC=$CC
export CMAKE_CXX=$CXX

export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_C_COMPILER=$CMAKE_CC -DCMAKE_CXX_COMPILER=$CMAKE_CXX"
#export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
#export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX"
#export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_SYSROOT=$CONDA_BUILD_SYSROOT"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_TESTS=on"
#export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"

cd ~/git/omnisci-work/omniscidb-internal/build
pwd

function h () {
cat << EndOfMessage

To configure, run:

  mkdir -p build && cd build

  cmake -Wno-dev \$CMAKE_OPTIONS ..

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

  '';
}

#
# Prepare pytorch development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of ~/git/Quansight/pytorch
#   Existence of pytorch-cuda-dev or pytorch-dev conda environment
#
# Author: Pearu Peterson
# Created: November 2019
#

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export USE_XNNPACK=0
export USE_MKLDNN=0

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader

    export USE_CUDA=${USE_CUDA:-1}
    if [[ "$USE_CUDA" = "0" ]]
    then
        echo "CUDA DISABLED"
    else
        CUDA_VERSION=${CUDA_VERSION:-10.1.243}
        . /usr/local/cuda-${CUDA_VERSION}/env.sh
    fi
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/pytorch-cuda-dev.yaml
    # conda env create  --file=pytorch-cuda-dev.yaml -n pytorch-cuda-dev

    Environment=pytorch${Python-}-cuda-dev
    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda $Environment
    else
        conda activate $Environment
    fi

    if [[ "$USE_CUDA" = "1" ]]
    then
        # LDFLAGS, CXXFLAGS, etc must be set after activating the conda environment
        export CXXFLAGS="$CXXFLAGS -L$CUDA_HOME/lib64"  # ???

        export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"
    fi
    # fixes mkl linking error:
    export CFLAGS="$CFLAGS -L$CONDA_PREFIX/lib"

    #export NCCL_ROOT=${CUDA_HOME}
    #export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${CUDA_HOME}/pkgconfig/

    export USE_NCCL=0
    # See https://github.com/NVIDIA/nccl/issues/244
    # https://github.com/pytorch/pytorch/issues/35363
    if [[ "" && ! -f third_party/nccl/nccl/issue244.patch ]]
    then
        cat > third_party/nccl/nccl/issue244.patch <<EOF
diff --git a/src/include/socket.h b/src/include/socket.h
index 68ce235..b4f09b9 100644
--- a/src/include/socket.h
+++ b/src/include/socket.h
@@ -327,7 +327,11 @@ static ncclResult_t createListenSocket(int *fd, union socketAddress *localAddr)
   if (socketToPort(&localAddr->sa)) {
     // Port is forced by env. Make sure we get the port.
     int opt = 1;
+#if defined(SO_REUSEPORT)
     SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
+#else
+    SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
+#endif
   }
 
   // localAddr port should be 0 (Any port)
EOF
        patch --verbose third_party/nccl/nccl/src/include/socket.h third_party/nccl/nccl/issue244.patch
    fi

    if [[ "" && ! -f torch/nccl_python.patch ]]
    then
        cat > torch/nccl_python.patch  <<EOF
diff --git a/torch/CMakeLists.txt b/torch/CMakeLists.txt
index 6167ceb1d9..aeb275d0d7 100644
--- a/torch/CMakeLists.txt
+++ b/torch/CMakeLists.txt
@@ -249,7 +249,9 @@ endif()
 
 if (USE_NCCL)
     list(APPEND TORCH_PYTHON_SRCS
-      \${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
+      \${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp
+      \${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp
+      )
     list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
     list(APPEND TORCH_PYTHON_LINK_LIBRARIES __caffe2_nccl)
 endif()
EOF
        patch --verbose torch/CMakeLists.txt torch/nccl_python.patch
    fi
else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/pytorch-dev.yaml
    # conda env create  --file=pytorch-dev.yaml -n pytorch-dev
    Environment=pytorch${Python-}-dev
    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda $Environment
    else
        conda activate $Environment
    fi
    export USE_CUDA=0
    export USE_NCCL=0
fi


# https://github.com/pytorch/cpuinfo/issues/36
if [[ ! -f third_party/cpuinfo/issue36.patch ]]
then
    cat > third_party/cpuinfo/issue36.patch <<EOF
diff --git a/src/api.c b/src/api.c
index 0cc5d4e..5903edf 100644
--- a/src/api.c
+++ b/src/api.c
@@ -10,6 +10,7 @@
 
	#include <unistd.h>
	#include <sys/syscall.h>
+	#include <asm-generic/unistd.h>
 #endif
 
 bool cpuinfo_is_initialized = false;
EOF
    patch --verbose third_party/cpuinfo/src/api.c third_party/cpuinfo/issue36.patch
fi

export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXFLAGS="`echo $CXXFLAGS | sed 's/-std=c++17/-std=c++14/'`"
export CXXFLAGS="$CXXFLAGS -L$CONDA_PREFIX/lib"  # ???
export CXXFLAGS="$CXXFLAGS -D__STDC_FORMAT_MACROS"

# Failure:
# FAILED: nccl_external-prefix/src/nccl_external-stamp/nccl_external-build nccl/lib/libnccl_static.a
# ...
# Generating rules
# > /home/pearu/git/Quansight/pytorch/build/nccl/obj/collectives/device/Makefile.rules
# In file included from include/core.h:14:0,
#                  from bootstrap.cc:8:
# include/socket.h: In function 'ncclResult_t createListenSocket(int*, socketAddress*)':
# include/socket.h:329:60: error: 'SO_REUSEPORT' was not declared in this scope
#      SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
# Fix:
# export USE_NCCL=0

export MAX_JOBS=$NCORES



if [[ "" && ! -n "$(type -t layout_conda)" ]]; then
    cd ~/git/Quansight/pytorch${Python-}
fi

echo -e "Local branches:\n"
git branch

cat << EndOfMessage

To update, run:

  git pull --rebase
  git submodule sync --recursive
  git submodule update --init --recursive

To clean, run:

  git clean -xdf
  git submodule foreach --recursive git clean -xfd

To build, run:

  python setup.py develop

To test, run:

  pytest -sv test/test_torch.py -k ...
  python test/run_test.py

To disable CUDA build, set:

  deactivate the environment
  export USE_CUDA=0  [currently USE_CUDA=${USE_CUDA}]
  reactivate the environment

EndOfMessage

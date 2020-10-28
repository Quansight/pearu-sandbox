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
# Updated: May 1 2020 for katex and USE_XNNPACK=1
#

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export USE_XNNPACK=${USE_XNNPACK:1}
export USE_MKLDNN=${USE_MKLDNN:-0}

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    USE_ENV=${USE_ENV:-pytorch${Python-}-cuda-dev}

    if [[ "$CONDA_DEFAULT_ENV" = "$USE_ENV" ]]
    then
        echo "deactivating $USE_ENV"
        conda deactivate
    fi

    export USE_CUDA=${USE_CUDA:-1}
    if [[ "$USE_CUDA" = "0" ]]
    then
        echo "CUDA DISABLED"
    else
        # when using cuda version different from 10.1, say 10.2, then run
        #   conda install -c conda-forge nvcc_linux-64=10.2 magma-cuda102
        CUDA_VERSION=${CUDA_VERSION:-11.0.3}
        . /usr/local/cuda-${CUDA_VERSION}/env.sh
    fi

    if [[ $CONDA_ENV_LIST = *"$USE_ENV"* ]]
    then
        if [[ -n "$(type -t layout_conda)" ]]; then
            layout_conda $USE_ENV
        else
            conda activate $USE_ENV
        fi
    else
        echo "conda environment does not exist. To create $USE_ENV, run:"
        echo "conda env create --file=~/git/Quansight/pearu-sandbox/conda-envs/pytorch-cuda-dev.yaml -n $USE_ENV"
        exit 1
    fi

    # Don't set *FLAGS before activating the conda environment.

    if [[ "$USE_CUDA" = "1" ]]
    then
        # fixes FAILED: lib/libc10_cuda.so ... ld: cannot find -lcudart
        export CXXFLAGS="$CXXFLAGS -L$CUDA_HOME/lib64"

        #export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"
        export LDFLAGS="${LDFLAGS} -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"
    fi

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
    USE_ENV=${USE_ENV:-pytorch${Python-}-dev}

    if [[ $CONDA_ENV_LIST = *"$USE_ENV"* ]]
    then
        if [[ "$CONDA_DEFAULT_ENV" = "$USE_ENV" ]]
        then
            echo "deactivating $USE_ENV"
            conda deactivate
        fi
        if [[ -n "$(type -t layout_conda)" ]]; then
            layout_conda $USE_ENV
        else
            conda activate $USE_ENV
        fi
    else
        echo "conda environment does not exist. To create $USE_ENV, run:"
        echo "conda env create --file=~/git/Quansight/pearu-sandbox/conda-envs/pytorch-dev.yaml -n $USE_ENV"
        exit 1
    fi
    # Don't set *FLAGS before activating the conda environment.

    export USE_CUDA=0
    export USE_NCCL=0
fi

# fixes mkl linking error:
export CFLAGS="$CFLAGS -L$CONDA_PREFIX/lib"

export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export CXXFLAGS="`echo $CXXFLAGS | sed 's/-std=c++17/-std=c++14/'`"
# fixes Linking CXX shared library lib/libtorch_cpu.so ... ld: cannot find -lmkl_intel_lp64
export CXXFLAGS="$CXXFLAGS -L$CONDA_PREFIX/lib"
# fixes FAILED: caffe2/torch/CMakeFiles/torch_python.dir/csrc/DataLoader.cpp.o ... error: expected ')' before 'PRId64'
export CXXFLAGS="$CXXFLAGS -D__STDC_FORMAT_MACROS"

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
  git submodule update -f --init --recursive

To clean, run:
  git clean -xddf
  git submodule foreach --recursive git clean -xfdd

To build, run:
  export PYTHONPATH=`pwd`  [optional for some pytorch versions]
  python setup.py develop

To test, run:
  pytest -sv test/test_torch.py -k ...
  python test/run_test.py

To disable CUDA build, set:
  conda deactivate
  export USE_CUDA=0  [currently USE_CUDA=${USE_CUDA}]
  <source the activate-pytorch-dev.sh script>

To enable CUDA version, say 10.2, run
  conda install -c conda-forge -c pytorch nvcc_linux-64=10.2 magma-cuda102
  conda deactivate
  export CUDA_VERSION=10.2.89  [currently CUDA_VERSION=${CUDA_VERSION}]
  <source the activate-pytorch-dev.sh script>
  <clean & re-build>

To enable MKL-DNN build, run

  conda deactivate
  export USE_MKLDNN=1  [currently USE_MKLDNN=${USE_MKLDNN}]
  <source the activate-pytorch-dev.sh script>

EndOfMessage

if [[ -x "$(command -v katex)" ]]
then
    cat << EndOfMessage
Found katex, you can build documentation using:
  python setup.py develop
  cd docs
  make html
EndOfMessage
else
    cat << EndOfMessage
katex not found, you cannot build documentation
To install katex, run:"
  conda install -c conda-forge yarn nodejs matplotlib
  yarn global add katex --prefix \$CONDA_PREFIX
  python -m pip install -r docs/requirements.txt
  conda deactivate
  <source the activate-pytorch-dev.sh script>
EndOfMessage
fi

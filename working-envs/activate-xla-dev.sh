#
# Prepare xla/jax/flax development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of jax-cuda-dev or pytorch-dev conda environment
#
# Author: Pearu Peterson
# Created: December 2023

if [[ -x "$(command -v lscpu)" ]]
then
    CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
    NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
    export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`
else
    export NCORES=8
fi
export PYTHONWARNINGS=${PYTHONWARNINGS:ignore}

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )
export JAX_BUILD_OPTIONS="--editable"

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    USE_ENV=${USE_ENV:-xla-cuda-dev}

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
        #   conda install -c conda-forge nvcc_linux-64=10.2
        CUDA_VERSION=${CUDA_VERSION:-12.3.2}
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
        echo "mamba env create --file=~/git/Quansight/pearu-sandbox/conda-envs/xla-cuda-dev.yaml -n $USE_ENV"
        exit 1
    fi

    # Don't set *FLAGS before activating the conda environment.

    if [[ "$USE_CUDA" = "1" ]]
    then
        # fixes FAILED: lib/libc10_cuda.so ... ld: cannot find -lcudart
        #export CXXFLAGS="$CXXFLAGS -L$CUDA_HOME/lib64"
        #export LDFLAGS="${LDFLAGS} -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"
        export JAX_BUILD_OPTIONS="${JAX_BUILD_OPTIONS} --enable_cuda --cuda_path ${CUDA_PATH}"
    fi

    #export NCCL_ROOT=${CUDA_HOME}
    #export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${CUDA_HOME}/pkgconfig/

else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/jax-dev.yaml
    # conda env create  --file=jax-dev.yaml -n jax-dev
    USE_ENV=${USE_ENV:-xla-dev}

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
        echo "mamba env create --file=~/git/Quansight/pearu-sandbox/conda-envs/xla-dev.yaml -n $USE_ENV"
        exit 1
    fi
    # Don't set *FLAGS before activating the conda environment.

fi

# use clang
export CC=`which clang`
export CXX=`which clang++`
export USE_BAZEL_VERSION=6.5.0

# fixes mkl linking error:
export CFLAGS="$CFLAGS -L$CONDA_PREFIX/lib"

export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# fixes: CUDA backend failed to initialize: Unable to load cuPTI
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# PyTorch uses C++14
#export CXXFLAGS="`echo $CXXFLAGS | sed 's/-std=c++17/-std=c++14/'`"
# fixes Linking CXX shared library lib/libtorch_cpu.so ... ld: cannot find -lmkl_intel_lp64
#export CXXFLAGS="$CXXFLAGS -L$CONDA_PREFIX/lib"
# fixes FAILED: caffe2/torch/CMakeFiles/torch_python.dir/csrc/DataLoader.cpp.o ... error: expected ')' before 'PRId64'
#export CXXFLAGS="$CXXFLAGS -D__STDC_FORMAT_MACROS"
# fixes FAILED: test_api/CMakeFiles/test_api.dir/dataloader.cpp.o ...c++ stl_algobase.h:431:30: error: argument 1 null where non-null expected [-Werror=nonnull]
# see also gh-77646
#export CXXFLAGS="$CXXFLAGS -Wno-error=nonnull"
#export MAX_JOBS=$NCORES

export TF_CUDA_PATHS=$CUDA_HOME

if [[ "$(git rev-parse --is-inside-work-tree 2>&1)" = "true" ]]
then
    echo -e "Local branches:\n"
    git branch
else
    cat << EndOfMessage
Not inside a git repository.

EndOfMessage
fi

# https://github.com/pytorch/pytorch/wiki/clang-format
# export PATH=`pwd`/tools/linter:$PATH

# test -d .clang-tidy-bin || mkdir .clang-tidy-bin
# test -f .clang-tidy-bin/clang-tidy || ln -s `which clang-tidy` .clang-tidy-bin/

cat << EndOfMessage

Install bazel:
  # download bazelisk from https://github.com/bazelbuild/bazelisk/releases
  # move bazelisk-linux-amd64 to your PATH and apply chmod +x
  export USE_BAZEL_VERSION=<desired version> (currently, $(bazel --version))
  bazel --version

To setup, run:
  git remote add upstream https://github.com/openxla/xla.git

To update, run:
  git fetch upstream
  git rebase upstream/main

To clean, run:
  git clean -xddf

To build, run:

  #export PYTHONPATH=`pwd`  [optional]
  #python build/build.py --bazel_options=--override_repository=xla=$(realpath ../xla) ${JAX_BUILD_OPTIONS}
  #pip install -e /home/pearu/git/pearu/jax/dist  [run once]

To test, run:
  ./configure.py --backend=CUDA --cuda_compiler=CLANG
  bazel build --test_output=all --spawn_strategy=sandboxed //xla/tests:complex_unary_op_test
  bazel-out/k8-opt/bin/xla/tests/complex_unary_op_test_gpu

To disable CUDA build, set:
  conda deactivate
  export USE_CUDA=0  [currently USE_CUDA=${USE_CUDA}]
  <source the activate-jax-dev.sh script>

To enable CUDA version, say 10.2, run
  conda install -c conda-forge nvcc_linux-64=10.2
  conda deactivate
  export CUDA_VERSION=10.2.89  [currently CUDA_VERSION=${CUDA_VERSION}]
  <source the activate-jax-dev.sh script>
  <clean & re-build>

To prepare commits:

  wget https://google.github.io/styleguide/pylintrc
  ylint --rcfile pylintrc xla/client/lib/generate_math_impl.py

  #mypy --config=pyproject.toml --show-error-codes jax
  #ruff jax
  # make clang-tidy CHANGED_ONLY=--changed-only

See also:
  #https://jax.readthedocs.io/en/latest/developer.html

EndOfMessage

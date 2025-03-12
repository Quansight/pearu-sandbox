#
# Prepare stablehlo development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of stablehlo-dev conda environment
#
# Author: Pearu Peterson
# Created: February 2024

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export PYTHONWARNINGS=${PYTHONWARNINGS:ignore}

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    USE_ENV=${USE_ENV:-stablehlo-dev}

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
        CUDA_VERSION=${CUDA_VERSION:-12.1.0}
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
        echo "mamba env create --file=~/git/Quansight/pearu-sandbox/conda-envs/stablehlo-dev.yaml -n $USE_ENV"
        exit 1
    fi

    # Don't set *FLAGS before activating the conda environment.

    #if [[ "$USE_CUDA" = "1" ]]
    #then
        # fixes FAILED: lib/libc10_cuda.so ... ld: cannot find -lcudart
        #export CXXFLAGS="$CXXFLAGS -L$CUDA_HOME/lib64"
        #export LDFLAGS="${LDFLAGS} -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"
        #export JAX_BUILD_OPTIONS="${JAX_BUILD_OPTIONS} --enable_cuda --cuda_path ${CUDA_PATH}"
    #fi

    #export NCCL_ROOT=${CUDA_HOME}
    #export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${CUDA_HOME}/pkgconfig/

else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/jax-dev.yaml
    # conda env create  --file=jax-dev.yaml -n jax-dev
    USE_ENV=${USE_ENV:-stablehlo-dev}

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
        echo "mamba env create --file=~/git/Quansight/pearu-sandbox/conda-envs/stablehlo-dev.yaml -n $USE_ENV"
        exit 1
    fi
    # Don't set *FLAGS before activating the conda environment.

fi

# fixes mkl linking error:
#export CFLAGS="$CFLAGS -L$CONDA_PREFIX/lib"

export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# fixes: CUDA backend failed to initialize: Unable to load cuPTI
# export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# workaround https://github.com/llvm/llvm-project/issues/76515
export CXXFLAGS="$CXXFLAGS -Wno-deprecated-declarations"

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

[[ "$(uname)" != "Darwin" ]] && LLVM_ENABLE_LLD="ON" || LLVM_ENABLE_LLD="OFF"

export STABLEHLO_CMAKE_OPTIONS="-GNinja \
  -DSTABLEHLO_ENABLE_LLD=${LLVM_ENABLE_LLD} \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DSTABLEHLO_ENABLE_SPLIT_DWARF=ON \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DMLIR_DIR=${PWD}/llvm-build/lib/cmake/mlir"

#   -DSTABLEHLO_ENABLE_SANITIZER=address

export LLVM_SYMBOLIZER_PATH=`which llvm-symbolizer-10`

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

To setup, run:
  git remote add upstream https://github.com/openxla/stablehlo.git
  cd stablehlo
  git clone https://github.com/llvm/llvm-project.git

To update, run:
  git fetch upstream
  git rebase upstream/main

  # llvm_version: $(test -f llvm-project/stablehlo_llvm_version.txt && cat llvm-project/stablehlo_llvm_version.txt || echo "N/A")
  (cd llvm-project && git fetch && git checkout $(cat build_tools/llvm_version.txt) && echo $(cat build_tools/llvm_version.txt) > stablehlo_llvm_version.txt)
  MLIR_ENABLE_BINDINGS_PYTHON=ON build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

To clean, run:
  git clean -xddf

To build, run:
  mkdir -p build && cd build
  cmake .. \$STABLEHLO_CMAKE_OPTIONS
  cmake --build . -j $NCORES

  ./build_tools/github_actions/ci_build_cmake.sh "`pwd`/llvm-build" "`pwd`/build"

  # to workaround not found zlib.h:
  bazel build --lockfile_mode=error //... --@llvm_zlib//:llvm_enable_zlib=false
  bazel test //... --@llvm_zlib//:llvm_enable_zlib=false

To test, run:
  ninja check-stablehlo-tests
  PYTHONPATH=`pwd`/python_packages/stablehlo/ python

  cd build
  bin/stablehlo-translate --interpret -split-input-file ../stablehlo/tests/interpret/exponential_minus_one.mlir
  bin/stablehlo-opt --chlo-legalize-to-stablehlo ../stablehlo/tests/math/asin_complex64.mlir | bin/stablehlo-translate --interpret
  for t in \$(ls ../stablehlo/tests/math/*.mlir); do bin/stablehlo-opt --chlo-legalize-to-stablehlo \$t | bin/stablehlo-translate --interpret ; done

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
  yapf --style='{based_on_style: google, indent_width: 2}' build_tools/math/generate_tests.py -i
  yapf --style='{based_on_style: google, indent_width: 2}' build_tools/math/generate_ChloDecompositionPatternsMath.py -i
  clang-format --style Google -i /path/to/cpp-file
  ./build_tools/github_actions/lint_whitespace_checks.sh -f

See also:
  TBD

EndOfMessage

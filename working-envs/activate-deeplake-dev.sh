#
# Prepare deeplake development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of pytorch-cuda-dev or pytorch-dev conda environment
#
# Author: Pearu Peterson
# Created: June 2023
#

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    USE_ENV=${USE_ENV:-deeplake-dev}

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
        CUDA_VERSION=${CUDA_VERSION:-11.5.0}
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
        echo "conda env create --file=~/git/Quansight/pearu-sandbox/conda-envs/deeplake-dev.yaml -n $USE_ENV"
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

    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/deeplake-dev.yaml
    # conda env create  --file=deeplake-dev.yaml -n deeplake-dev
    USE_ENV=${USE_ENV:-deeplake-dev}

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
        echo "conda env create --file=~/git/Quansight/pearu-sandbox/conda-envs/deeplake-dev.yaml -n $USE_ENV"
        exit 1
    fi
    # Don't set *FLAGS before activating the conda environment.

    export USE_CUDA=0
fi

export MAX_JOBS=$NCORES


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

test -d .clang-tidy-bin || mkdir .clang-tidy-bin
test -f .clang-tidy-bin/clang-tidy || ln -s `which clang-tidy` .clang-tidy-bin/

cat << EndOfMessage

To update, run:
  git pull --rebase

To clean, run:
  git clean -xddf

To build, run:
  export PYTHONPATH=`pwd`  [optional for some pytorch versions]
  python setup.py develop

To test, run:

To disable CUDA build, set:
  conda deactivate
  export USE_CUDA=0  [currently USE_CUDA=${USE_CUDA}]
  <source the activate-deeplake-dev.sh script>

To enable CUDA version, say 10.2, run
  conda deactivate
  export CUDA_VERSION=10.2.89  [currently CUDA_VERSION=${CUDA_VERSION}]
  <source the activate-deeplake-dev.sh script>
  <clean & re-build>

EndOfMessage

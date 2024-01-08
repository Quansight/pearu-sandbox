#
# Prepare pydlpack development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of pydlpack-dev conda environment
#
# Author: Pearu Peterson
# Created: December 2023

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export PYTHONWARNINGS=${PYTHONWARNINGS:ignore}

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )
CONDA_ENV_YAML=pydlpack-dev.yaml

export PYTHONPATH=`pwd`

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    USE_ENV=${USE_ENV:-pydlpack${Python-}-dev}

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
        echo "mamba env create --file=~/git/Quansight/pearu-sandbox/conda-envs/${CONDA_ENV_YAML} -n $USE_ENV"
        exit 1
    fi

    # Don't set *FLAGS before activating the conda environment.

    if [[ "$USE_CUDA" = "1" ]]
    then
        echo "using cuda"
    fi

    #export NCCL_ROOT=${CUDA_HOME}
    #export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${CUDA_HOME}/pkgconfig/

else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/jax-dev.yaml
    # conda env create  --file=jax-dev.yaml -n jax-dev
    USE_ENV=${USE_ENV:-pyyaml${Python-}-dev}

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
        echo "mamba env create --file=~/git/Quansight/pearu-sandbox/conda-envs/${CONDA_ENV_YAML} -n $USE_ENV"
        exit 1
    fi
    # Don't set *FLAGS before activating the conda environment.

fi

if [[ "$(git rev-parse --is-inside-work-tree 2>&1)" = "true" ]]
then
    echo -e "Local branches:\n"
    git branch
else
    cat << EndOfMessage
Not inside a git repository.

EndOfMessage
fi


cat << EndOfMessage

To update, run:
  git pull --rebase

To clean, run:
  git clean -xddf

To build, run:
  export PYTHONPATH=`pwd`  [optional]
  python build/build.py
  pip install -e /home/pearu/git/peary/pydlpack/dist  [run once]

To test, run:
  pytest -n auto tests

To disable CUDA build, set:
  conda deactivate
  export USE_CUDA=0  [currently USE_CUDA=${USE_CUDA}]
  <source the $0 script>

To enable CUDA version, say 10.2, run
  conda install -c conda-forge nvcc_linux-64=10.2
  conda deactivate
  export CUDA_VERSION=10.2.89  [currently CUDA_VERSION=${CUDA_VERSION}]
  <source the $0 script>
  <clean & re-build>

To prepare commits:
  TBD

EndOfMessage

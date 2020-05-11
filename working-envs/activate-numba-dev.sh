#
# Prepare numba development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of ~/git/Quansight/numba
#   Existence of numba-cuda-dev numba-dev conda environment
#
# Author: Pearu Peterson
# Created: December 2019
#

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`
CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )

if [[ -x "$(command -v nvidia-smi)" ]]
then
    USE_ENV=${USE_ENV:-numba-cuda-dev}
    ENV_DEV_YAML=numba-cuda-dev.yaml
    export USE_CUDA=${USE_CUDA:-1}
else
    USE_ENV=${USE_ENV:-numba-dev}
    ENV_DEV_YAML=numba-dev.yaml
    export USE_CUDA=0
fi

if [[ "$USE_CUDA" = "1" ]]
then
    # when using cuda version different from 10.1, say 10.2, then run
    #   conda install -c conda-forge nvcc_linux-64=10.2
    CUDA_VERSION=${CUDA_VERSION:-10.1.243}
    . /usr/local/cuda-${CUDA_VERSION}/env.sh
fi

if [[ "$CONDA_DEFAULT_ENV" = "$USE_ENV" ]]
then
    echo "deactivating $USE_ENV"
    conda deactivate
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
    echo "conda env create --file=~/git/Quansight/pearu-sandbox/conda-envs/$ENV_DEV_YAML -n $USE_ENV"
    exit 1
fi

export PYTEST_ADDOPTS="-sv --assert=plain"
export NUMBA_DEVELOPER_MODE=1

echo -e "Local branches:\n"
git branch

cat << EndOfMessage

To update, run:

  git pull --rebase

To build, run:

  python setup.py develop

To test, run:

  pytest -sv numba/tests

To disable CUDA environment, set:
  conda deactivate
  export USE_CUDA=0  [currently USE_CUDA=${USE_CUDA}]
  <source the activate-pytorch-dev.sh script>

To select CUDA version, say 10.2.89, set
  export CUDA_VERSION=10.2.89  [currently CUDA_VERSION=${CUDA_VERSION}]

EndOfMessage

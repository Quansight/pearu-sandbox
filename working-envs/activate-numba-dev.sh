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

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    . /usr/local/cuda-10.1.243/env.sh

    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/numba-cuda-dev.yaml
    # conda env create  --file=numba-cuda-dev.yaml -n numba-cuda-dev
    Environment=numba-cuda-dev

    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda $Environment
    else
        conda activate $Environment
    fi
else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/numba-dev.yaml
    # conda env create  --file=numba-dev.yaml -n numba-dev
    Environment=numba-dev
    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda $Environment
    else
        conda activate $Environment
    fi
fi

echo -e "Local branches:\n"
git branch

cat << EndOfMessage

To update, run:

  git pull --rebase

To build, run:

  python setup.py develop

To test, run:

  pytest -sv numba/tests

EndOfMessage

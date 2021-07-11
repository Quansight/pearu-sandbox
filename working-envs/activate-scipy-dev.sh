#
# Prepare scipy development environment
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of scipy-dev conda environment
#
# Author: Pearu Peterson
# Created: January 2021
#

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`
CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )

USE_ENV=${USE_ENV:-scipy-dev}
ENV_DEV_YAML=scipy-dev.yaml

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

echo -e "Local branches:\n"
git branch

cat << EndOfMessage

To update, run:

  git pull --rebase

To build, run:

  python setup.py develop

To test, run:

  pytest -sv scipy/tests

EndOfMessage

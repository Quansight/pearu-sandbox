CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export USE_ENV=${USE_ENV:-gcs-dev}
if hash conda; then
    CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )
else
    CONDA_ENV_LIST=$USE_ENV
fi

echo "USE_ENV=$USE_ENV"
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
    echo "conda environment $USE_ENV does not exist. To create $USE_ENV, run:"
    echo "conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/gcs-dev.yaml -n $USE_ENV"
    exit 1
fi

echo -e "Local branches:\n"
git branch

function h () {

    cat << EndOfMessage

To test, run:

  pytest -sv -n $NCORES . -x -r s

EndOfMessage

}

h

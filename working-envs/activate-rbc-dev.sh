
#  wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/rbc-dev.yaml
# conda env create  --file=rbc-dev.yaml -n rbc-dev

export USE_ENV=${USE_ENV:-rbc-dev}
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
    echo "conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/rbc-dev.yaml -n $USE_ENV"
    exit 1
fi


export OMNISCIDB_DEV_LABEL=${OMNISCIDB_DEV_LABEL:-master}

echo -e "Local branches:\n"
git branch

function h () {

    cat << EndOfMessage

To develop, run:

  python setup.py develop

To test, run:

  pytest -sv rbc -x -r s

or

  OMNISCIDB_DEV_LABEL=<label> pytest -sv rbc -x -r s

where <label> (currently OMNISCIDB_DEV_LABEL=${OMNISCIDB_DEV_LABEL}) can be

  master        - assumes testing against omniscidb master
  docker-dev    - assumes testing against omniscidb dev docker image
  <undefined>   - assumes testing against some omniscidb release version
  <branch name> - assumes testing against omniscidb branch

To use different conda environment, run:
  conda deactivate
  export USE_ENV=<env name> (currently USE_ENV=${USE_ENV})
  <source activate-rbc-dev.sh>

EndOfMessage

}

h

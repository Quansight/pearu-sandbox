
#  wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/rbc-dev.yaml
# conda env create  --file=rbc-dev.yaml -n rbc-dev

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )
USE_ENV="${USE_ENV:-rbc-dev}"

if [[ $CONDA_ENV_LIST = *"$USE_ENV"* ]]
then
    if [[ "$CONDA_DEFAULT_ENV" = "$USE_ENV" ]]
    then
        echo "deactivating $USE_ENV"
        conda deactivate
    fi
    if [[ -n "$(type -t layout_conda)" ]]; then
        layout_conda rbc-dev
    else
        conda activate rbc-dev
    fi
else
    echo "conda environment does not exist. To create $USE_ENV, run:"
    echo "conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/rbc-dev.yaml -n $USE_ENV"
    exit 1
fi

echo -e "Local branches:\n"
git branch

function h () {

    cat << EndOfMessage

To develop, run:

  python setup.py develop

To test, run:

  pytest -sv rbc -x -r s

EndOfMessage

}

h

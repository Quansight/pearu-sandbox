
#  wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/adventofcode.yaml
# conda env create  --file=adventofcode.yaml -n adventofcode

export USE_ENV=${USE_ENV:-adventofcode}
if hash conda; then
    CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )
else
    CONDA_ENV_LIST=$USE_ENV
fi

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
    echo "conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/adventofcode.yaml -n $USE_ENV"
    exit 1
fi

echo -e "Local branches:\n"
git branch

function h () {

    cat << EndOfMessage

To develop, run:

  cd $USERNAME
  python dayXY.py

EndOfMessage

}

h

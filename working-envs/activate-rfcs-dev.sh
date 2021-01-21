
#  wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/rfcs-dev.yaml
# conda env create  --file=rfcs-dev.yaml -n rfcs-dev

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )
USE_ENV="${USE_ENV:-rfcs-dev}"

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
    echo "conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/rfcs-dev.yaml -n $USE_ENV"
    exit 1
fi

echo -e "Local branches:\n"
git branch

function h () {

    cat << EndOfMessage

To develop, run:

  python ~/git/Quansight/pearu-sandbox/latex_in_markdown/watch_latex_md.py --html --git /path/to/markdown/documents/ 

EndOfMessage

}

h

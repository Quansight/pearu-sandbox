CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export USE_ENV=${USE_ENV:-markdown_api_docs-dev}
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
    echo "conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/markdown_api_docs-dev.yaml -n $USE_ENV"
    exit 1
fi

echo -e "Local branches:\n"
git branch

function h () {

    cat << EndOfMessage

To test, run:

  pytest -sv -n $NCORES . -x -r s

To release:
- make sure that main branch is greem
- https://github.com/pearu/markdown_api_docs/releases
  -> Draft a new release
     -> Create new tag: vX.Y.Z
     -> Title: Release X.Y
     -> Generate release notes
     -> Publish release
- https://github.com/pearu/markdown_api_docs/actions
  -> check that package is succesfully uploaded to PyPi
- https://github.com/conda-forge/markdown_api_docs-feedstock
  -> check that PR is auto-created and landed

EndOfMessage

}

h

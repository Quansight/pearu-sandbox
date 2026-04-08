#  wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/litt-dev.yaml
# conda env create  --file=litt-dev.yaml -n litt-dev

export USE_ENV=${USE_ENV:-litt-dev}
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
    echo "conda env create  --file=~/git/openteams-ai/pearu-sandbox/conda-envs/litt-dev.yaml -n $USE_ENV"
    exit 1
fi

export LITT_TEST_BASE_FOLDER=$HOME/git/LifeMRM/LITT_project/litt_test_data/
export PYTHONPATH=$HOME/git/LifeMRM/LITT_project/litt:$PYTHONPATH

echo -e "Local branches:\n"
git branch

function h () {

    cat << EndOfMessage

To develop, run:

  # python setup.py develop

To test, run:

  pytest -sv litt/system/tests/regression/test_performance.py -x
  # 1 passed, 1 skipped, 147 warnings, 42 subtests passed in 1115.40s (0:18:35)
  # 1 passed, 1 skipped, 147 warnings, 42 subtests passed in 1240.19s (0:20:40)

  pytest -sv litt/system/tests/regression/test_regression.py
  # 6 failed, 8 passed, 74 warnings, 4 subtests passed in 2029.74s (0:33:49)
  # 6 failed, 8 passed, 74 warnings, 4 subtests passed in 2017.20s (0:33:37)

  PYTHONPATH=. python ./litt/run_model.py

To use different conda environment, run:
  conda deactivate
  export USE_ENV=<env name> (currently USE_ENV=${USE_ENV})
  <source activate-litt-dev.sh>

EndOfMessage

}

h


#  wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/rbc-dev.yaml
# conda env create  --file=rbc-dev.yaml -n rbc-dev

if [[ -n "$(type -t layout_conda)" ]]; then
    layout_conda rbc-dev
else
    conda activate rbc-dev
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

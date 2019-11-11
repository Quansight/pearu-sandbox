#
# Support direnv and conda cooperation
#
# Usage:
#   1. Add the following line to your .bashrc before conda initialization block:
#        export PRECONDA_PS1=$PS1
#   2. Append the following line to your .bashrc:
#        source /path/to/bashrc_direnv.sh
#
# Author: Pearu Peterson
# Created: November 2019

eval "$(direnv hook bash)"
export DIRENV_LOG_FORMAT=
show_conda_env() {
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        if [[ -n "$DIRENV_DIR" ]]; then
            echo ":($(basename $CONDA_DEFAULT_ENV))"
        else
            echo "($(basename $CONDA_DEFAULT_ENV))"
        fi
    fi
}
export -f show_conda_env
export CUSTOM_PS1="\[\033[01;35m\]"'$(show_conda_env)'"\[\033[00m\]"$PRECONDA_PS1

PS1="${CUSTOM_PS1:-default PS1}"

save_function() {
    local ORIG_FUNC=$(declare -f $1)
    local NEWNAME_FUNC="$2${ORIG_FUNC#$1}"
    eval "$NEWNAME_FUNC"
}

if [[ "$(type -t conda)" == "function" ]]; then
    save_function conda BASH_RC_conda
    conda() {
        local BASH_RC_ENV_BEFORE=$(mktemp /tmp/bashrc-script.XXXXXX)
        local BASH_RC_ENV_AFTER=$(mktemp /tmp/bashrc-script.XXXXXX)
        env | sort > $BASH_RC_ENV_BEFORE
        BASH_RC_conda "$@"
        env | sort > $BASH_RC_ENV_AFTER
        local BASH_RC_DIFF=`diff $BASH_RC_ENV_BEFORE $BASH_RC_ENV_AFTER`
        rm -rf $BASH_RC_ENV_BEFORE $BASH_RC_ENV_AFTER
        case "$1" in
            install|update|upgrade|remove|uninstall)
                PS1="${CUSTOM_PS1:-default PS1}"  # restore PS1
            ;;
            *)
            ;;
        esac
        if [[ -n "$BASH_RC_DIFF" ]]; then
            echo -e "conda command changed the environment:\n<START DIFF>\n$BASH_RC_DIFF\n<END DIFF>"
            if [[ -n "$DIRENV_DIR" ]]; then
                echo "resetting the environment"
                direnv allow
            fi
        fi

    }
fi

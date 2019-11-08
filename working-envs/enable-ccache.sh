#
# Enable ccache for compilers in the current conda environment
#
# Usage:
#
#  source <this file>
#
# Assumptions:
#   the target conda environment has been activated
#   ccache has been installed to ccache conda environment:
#     conda create -n ccache
#     conda activate ccache
#     conda install -y -c conda-forge ccache
#     ccache -M 50Gi
#     conda deactivate
#
# Author: Pearu Peterson
# Created: November 2019

if [[ -z "${ENABLE_CCACHE_SH+x}" ]]
then
    export ENABLE_CCACHE_SH=1
    CCACHE=$(dirname $CONDA_PREFIX)/ccache/bin/ccache
    if [[ -x $CCACHE ]]
    then
        CCACHE_TARGET=$CONDA_PREFIX/ccache
        
        export CUDA_NVCC_EXECUTABLE=$CCACHE_TARGET/cuda/nvcc
        
        if [[ ! -d $CCACHE_TARGET ]]
        then
            mkdir -p $CCACHE_TARGET/lib
            mkdir -p $CCACHE_TARGET/cuda
            test -x ${GCC:-x} && ln -svf $CCACHE $CCACHE_TARGET/lib/$(basename $GCC)
            test -x ${GXX:-x} && ln -svf $CCACHE $CCACHE_TARGET/lib/$(basename $GXX)
            test -x ${CC:-x} && ln -svf $CCACHE $CCACHE_TARGET/lib/$(basename $CC)
            test -x ${CXX:-x} && ln -svf $CCACHE $CCACHE_TARGET/lib/$(basename $CXX)
            if [[ -x "$(command -v nvcc)" ]]
            then
                NVCC_ORIG="$(command -v nvcc)"
                ln -svf $CCACHE $CUDA_NVCC_EXECUTABLE
            fi
        fi

        export PATH=$CCACHE_TARGET/lib:$PATH
        export GCC=$CCACHE_TARGET/lib/$(basename $GCC)
        export GXX=$CCACHE_TARGET/lib/$(basename $GXX)
        export CC=$CCACHE_TARGET/lib/$(basename $CC)
        export CXX=$CCACHE_TARGET/lib/$(basename $CXX)
    else
        echo "Failed to enable ccache for $CONDA_PREFIX: $CCACHE is not executable"
        return 1
    fi

    echo "To explore ccache, use"
    echo "  \$CCACHE -s"
else
    echo "ccache has already been enabled, do nothing"
fi

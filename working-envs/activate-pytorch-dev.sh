#
# Prepare pytorch development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of ~/git/Quansight/pytorch
#   Existence of pytorch-cuda-dev or pytorch-dev conda environment
#
# Author: Pearu Peterson
# Created: November 2019
#

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader
    . /usr/local/cuda-10.1.243/env.sh

    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/pytorch-cuda-dev.yaml
    # conda env create  --file=pytorch-cuda-dev.yaml -n pytorch-cuda-dev
    #conda activate pytorch-cuda-dev
    if [[ -n "$(type -t layout_conda)" ]]; then
        time layout_conda pytorch-cuda-dev
    else
        conda activate pytorch-cuda-dev
    fi
    export USE_CUDA=1
    export CXXFLAGS="$CXXFLAGS -L$CUDA_HOME/lib64"
    export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"
else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/pytorch-dev.yaml
    # conda env create  --file=pytorch-dev.yaml -n pytorch-dev
    if [[ -n "$(type -t layout_conda)" ]]; then
        time layout_conda pytorch-dev
    else
        conda activate pytorch-dev
    fi
    export USE_CUDA=0
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXFLAGS="`echo $CXXFLAGS | sed 's/-std=c++17/-std=c++11/'`"
export CXXFLAGS="$CXXFLAGS -L$CONDA_PREFIX/lib"

# Failure:
# FAILED: nccl_external-prefix/src/nccl_external-stamp/nccl_external-build nccl/lib/libnccl_static.a
# ...
# Generating rules
# > /home/pearu/git/Quansight/pytorch/build/nccl/obj/collectives/device/Makefile.rules
# In file included from include/core.h:14:0,
#                  from bootstrap.cc:8:
# include/socket.h: In function 'ncclResult_t createListenSocket(int*, socketAddress*)':
# include/socket.h:329:60: error: 'SO_REUSEPORT' was not declared in this scope
#      SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
# Fix:
export USE_NCCL=0

export CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
export NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`
export MAX_JOBS=$NCORES

if [[ ! -n "$(type -t layout_conda)" ]]; then
    cd ~/git/Quansight/pytorch
fi

echo -e "Local branches:\n"
git branch
echo
echo -e "To update, run:\n"
echo "  git pull --rebase"
echo "  git submodule sync --recursive"
echo "  git submodule update --init --recursive"
echo -e "\nTo build, run:\n"
echo "  python setup.py develop"
echo -e "\nTo test, run:\n"
echo "  pytest -sv test/test_torch.py -k ..."
echo

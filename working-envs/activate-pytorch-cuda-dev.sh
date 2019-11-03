#
# Prepare pytorch development environment, CUDA enabled
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of ~/git/Quansight/pytorch
#   Existence of pytorch-cuda-dev conda environment
#
# Author: Pearu Peterson
# Created: November 2019
#

# wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
# read set_cuda_env.sh reader
. /usr/local/cuda-10.1.243/env.sh

# wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/pytorch-cuda-dev.yaml
# conda env create  --file=pytorch-cuda-dev.yaml -n pytorch-cuda-dev
conda activate pytorch-cuda-dev

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXFLAGS="`echo $CXXFLAGS | sed 's/-std=c++17/-std=c++11/'`"
export CXXFLAGS="$CXXFLAGS -L$CONDA_PREFIX/lib -L$CUDA_HOME/lib64"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"

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

export USE_CUDA=1

# Using more jobs than 24 would be wasting of qgpu RAM
export MAX_JOBS=24

cd ~/git/Quansight/pytorch

echo -e "To update, run:\n"
echo "git pull --rebase"
echo "git submodule sync --recursive"
echo "git submodule update --init --recursive"
echo -e "\nTo build, run:\n"
echo "python setup.py develop"
echo -e "\nTo test, run:\n"
echo "pytest -sv test/test_torch.py -k ..."

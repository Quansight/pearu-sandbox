
CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

echo "NCORES=$NCORES"
echo "CUDA_HOME=$CUDA_HOME"

conda activate $CONDA_ENV
export USE_CUDA=1
# LDFLAGS, CXXFLAGS, etc must be set after activating the conda environment
export CXXFLAGS="$CXXFLAGS -L$CUDA_HOME/lib64"  # ???
export CFLAGS="$CFLAGS -L$CONDA_PREFIX/lib"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"
export USE_NCCL=0

export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXFLAGS="`echo $CXXFLAGS | sed 's/-std=c++17/-std=c++14/'`"
export CXXFLAGS="$CXXFLAGS -L$CONDA_PREFIX/lib"  # ???
export MAX_JOBS=$NCORES

cd pytorch
python setup.py develop
python -c 'import torch; print("CUDA is available:", torch.cuda.is_available())'

cd -
conda deactivate


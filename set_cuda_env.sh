#
# Set CUDA environment.
#
# Installation:
#   ln -s /path/to/cuda_set_env.sh /path/to/cuda/home/env.sh
# Usage:
#   source <this file>
#
# Author: Pearu Peterson
# Created: October 2019
#

#echo "This is ${BASH_SOURCE[0]}"

# Restore previous environment
if [ ! -z "${CUDA_ENV_SH_CUDA_HOME_BACKUP+x}" ]
then
  # Restore previous environment:
  export CUDA_HOME=${CUDA_ENV_SH_CUDA_HOME_BACKUP}
  export PATH=${CUDA_ENV_SH_PATH_BACKUP}
  export LD_LIBRARY_PATH=${CUDA_ENV_SH_LD_LIBRARY_PATH_BACKUP}
  export CUDAToolkit_ROOT=${CUDA_ENV_SH_CUDAToolkit_ROOT_BACKUP}
fi

# Backup current environment
export CUDA_ENV_SH_CUDA_HOME_BACKUP=${CUDA_HOME}
export CUDA_ENV_SH_PATH_BACKUP=${PATH}
export CUDA_ENV_SH_LD_LIBRARY_PATH_BACKUP=${LD_LIBRARY_PATH}
export CUDA_ENV_SH_CUDAToolkit_ROOT_BACKUP=${CUDAToolkit_ROOT}

# Set new environment
export CUDA_HOME=$(dirname ${BASH_SOURCE[0]})
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Required for cmake-3.17 or newer:
export CUDAToolkit_ROOT=${CUDA_HOME}


# Show new environment
#echo "CUDA_HOME=${CUDA_HOME}"
#echo "PATH=${PATH}"
#echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
#echo "CUDAToolkit_ROOT=${CUDAToolkit_ROOT}"

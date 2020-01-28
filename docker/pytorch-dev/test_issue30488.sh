#!/bin/bash
echo "CONDA_ENV=$CONDA_ENV"
conda activate $CONDA_ENV
cd pytorch
pwd
python -c 'import torch; print("CUDA is available:", torch.cuda.is_available())'
# pytest -sv test/test_torch.py -k test_lu_cuda
# pytest -sv test/test_torch.py -k test_det_logdet_slogdet_batched_cuda_float64

python torch/utils/collect_env.py

conda list | grep torch

git clean -xdf
python setup.py develop --cmake-only

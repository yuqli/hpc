#!/usr/bin/env python
# srun --mem=20000 -t 5:00:00 --pty -c 20 /bin/bash
module purge
module load anaconda3/5.3.0
# conda create --name pytorch-mpi python=3.6
source activate pytorch-mpi
module load cuda/9.2.88
module load openmpi/intel/2.0.3
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
#conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
#conda install -c mingfeima mkldnn
#conda install -c pytorch magma-cuda92

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

python setup.py clean
python setup.py install

cd ~
python -c 'import torch; print(torch.__version__)'
mpirun -np 4 python -c 'import torch.distributed as dist; dist.init_process_group(backend="mpi"), print("hello", dist.get_rank())'

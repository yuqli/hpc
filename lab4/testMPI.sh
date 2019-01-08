#!/usr/bin/env bash
module load cuda/9.2.88
module load openmpi/intel/2.0.3
pirun -np 4 /home/am9031/anaconda/bin/python -c 'import torch.distributed as dist; dist.init_process_group(backend="mpi"), print("hello", dist.get_rank())'

#!/bin/bash

#SBATCH --job-name=lab4e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/out_%j
#SBATCH --error=out/err_%j
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

use_cuda='false'
workers=8
opt='sgd'
num_proc=1 # number of processors, must be equal to #SBATCH --nodes
out_file=log-$num_proc-$use_cuda-$workers-$opt

module load openmpi/intel/2.0.3
module load cuda/9.2.88

mpirun -np $num_proc /home/am9031/anaconda/bin/python lab4_singlenode.py --use_cuda $use_cuda --workers $workers --opt $opt | tee $out_file

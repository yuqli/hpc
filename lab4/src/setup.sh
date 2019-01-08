#!/usr/bin/env python
module load anaconda3/5.3.0
source activate pytorch_env
module load cuda/9.2.88
module load openmpi/intel/2.0.3
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

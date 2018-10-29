#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --time=5:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=visionExperiment
#SBATCH --mail-type=END
#SBATCH --mail-user=yl5090@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:2

module purge
module load anaconda3/5.3.0 
source activate nlp 
python lab2.py --device gpu --path /scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon --workers 24 --optimizer adadelta 
conda deactivate

#!/bin/sh

#SBATCH --job-name="exp01"
#SBATCH --time=96:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=10312
#SBATCH --tasks-per-node=4
#SBATCH --output=checkpoints/exp01/output_%j.out
#SBATCH --error=checkpoints/exp01/error_%j.out

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python exp/run_tcga_crc-hpc.py
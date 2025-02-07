#!/bin/sh

#SBATCH --job-name="cond-conch"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --tasks-per-node=4
#SBATCH --output=cond-conch-output_%j.out

export PYTHONFAULTHANDLER=1

srun python -m exp.run_tcga_crc

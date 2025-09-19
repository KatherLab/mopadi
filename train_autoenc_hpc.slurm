#!/bin/sh

#SBATCH --job-name="mopadi-autoenc"
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=13437
#SBATCH --tasks-per-node=4
#SBATCH --output=mopadi-output_%j.out

export PYTHONFAULTHANDLER=1

srun mopadi autoenc --config conf.yaml

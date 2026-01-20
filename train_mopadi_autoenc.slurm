#!/bin/sh

#SBATCH --job-name="mopadi-autoenc"
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=13437
#SBATCH --tasks-per-node=4
#SBATCH --output=outs/mopadi_autoenc_output_%j.out

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd ~/mopadi
source .venv/bin/activate

srun mopadi autoenc --config conf.yaml

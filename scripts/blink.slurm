#!/bin/bash

#SBATCH --job-name=blink
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoints/%u/duck/logs/blink-%j.log
#SBATCH --error=/checkpoints/%u/duck/logs/blink-%j.log
#SBATCH --partition=learnai4p
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8 
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=5-00:00:00

export PYTHONPATH=.:$PYTHONPATH
## export NCCL_SOCKET_IFNAME=ens32

module purge

srun python duck/main.py run_name=blink_ens32 trainer=1host ablations.blink=True data.batch_size=32

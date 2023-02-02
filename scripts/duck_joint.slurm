#!/bin/bash

#SBATCH --job-name=duck_joint
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoints/%u/duck/logs/duck_joint_%j.log
#SBATCH --error=/checkpoints/%u/duck/logs/duck_joint_%j.log
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

srun python duck/main.py run_name=duck_joint trainer=1host data.batch_size=32 ablations.joint_ent_rel_encoding=True

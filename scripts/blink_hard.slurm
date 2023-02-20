#!/bin/bash

#SBATCH --job-name=blink_hard
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoints/%u/duck/logs/blink_hard_%j.log
#SBATCH --error=/checkpoints/%u/duck/logs/blink_hard_%j.log
#SBATCH --partition=learnai4p
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8 
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=72:00:00

export PYTHONPATH=.:$PYTHONPATH
## export NCCL_SOCKET_IFNAME=ens32

module purge

srun python duck/main.py run_name=blink_hard \
     trainer=1host data.batch_size=16 \
     ablations.blink=True \
     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/blink_epoch\=0_val_f1\=0.692.ckpt" \
     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_blink_val_f1\=0.692.json" &
wait

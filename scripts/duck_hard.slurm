#!/bin/bash

#SBATCH --job-name=duck_hard
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoints/%u/duck/logs/duck_hard_%j.log
#SBATCH --error=/checkpoints/%u/duck/logs/duck_hard_%j.log
#SBATCH --partition=learnai4p
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8 
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=72:00:00

export PYTHONPATH=.:$PYTHONPATH
## export NCCL_SOCKET_IFNAME=ens32

module purge

## srun python duck/main.py run_name=duck_hard \
##      trainer=1host data.batch_size=16 \
##      duck.boxes.parametrization=spherical \
##      "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_spherical_mention_epoch\=0_val_f1\=0.752.ckpt" \
##      "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_spherical_val_f1\=0.752.json" &
## wait

## srun python duck/main.py run_name=duck_hard_roberta \
##     trainer=1host data.batch_size=16 \
##      duck.boxes.parametrization=spherical \
##      "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_roberta_epoch\=0_val_f1\=0.755.ckpt" \
##      "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_roberta_val_f1\=0.755.json" &
## wait

srun python duck/main.py run_name=duck_hard_roberta \
    trainer=1host data.batch_size=16 \
    duck.boxes.parametrization=spherical \
    "resume_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_hard_roberta_epoch\=0_val_f1\=0.864_last.ckpt" \
    "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_roberta_val_f1\=0.755.json" &
wait

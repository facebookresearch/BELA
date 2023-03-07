#!/bin/bash

#SBATCH --job-name=duck_hard
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoints/%u/duck/logs/%j.log
#SBATCH --error=/checkpoints/%u/duck/logs/%j.log
#SBATCH --partition=learnai4p
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8 
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=72:00:00

export PYTHONPATH=.:$PYTHONPATH
## export NCCL_SOCKET_IFNAME=ens32

module purge


# srun python duck/main.py run_name=duck_hard \
#     trainer=1host \
#     duck.boxes.parametrization=spherical \
#     duck.duck_loss.margin_pos=0 \
#     duck.duck_loss.margin_neg=2 \
#     duck.duck_loss.sampling_temperature=0.5 \
#     ablations.no_priors=false \
#     data.batch_size=8 \
#     optim.lr=1.0e-5 \
#     lr_scheduler=null \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.653_129668_last.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.653_129668.json" 

# srun python duck/main.py run_name=duck_hard \
#     trainer=1host \
#     duck.boxes.parametrization=spherical \
#     duck.duck_loss.margin_pos=0 \
#     duck.duck_loss.margin_neg=2 \
#     duck.duck_loss.sampling_temperature=0.5 \
#     ablations.no_priors=true \
#     data.batch_size=8 \
#     optim.lr=1.0e-5 \
#     lr_scheduler=null \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.668_129669_last.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.668_129669.json"


# srun python duck/main.py run_name=duck_hard \
#     trainer=1host \
#     duck.boxes.parametrization=spherical \
#     duck.duck_loss.margin_pos=1 \
#     duck.duck_loss.margin_neg=1 \
#     duck.duck_loss.sampling_temperature=1.0 \
#     ablations.no_priors=true \
#     data.batch_size=8 \
#     optim.lr=1.0e-5 \
#     lr_scheduler=null \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.693_129670_last.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.693_129670.json" 

# srun python duck/main.py run_name=duck_hard \
#     trainer=1host \
#     duck.boxes.parametrization=spherical \
#     duck.duck_loss.margin_pos=1 \
#     duck.duck_loss.margin_neg=1 \
#     duck.duck_loss.sampling_temperature=1.0 \
#     ablations.no_priors=false \
#     data.batch_size=8 \
#     optim.lr=1.0e-5 \
#     lr_scheduler=null \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.655_129671_last.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.655_129671.json" 


# srun python duck/main.py run_name=duck_hard \
#     trainer=1host \
#     duck.boxes.parametrization=spherical \
#     duck.duck_loss.margin_pos=2 \
#     duck.duck_loss.margin_neg=2 \
#     duck.duck_loss.sampling_temperature=0.5 \
#     ablations.no_priors=false \
#     data.batch_size=8 \
#     optim.lr=1.0e-5 \
#     lr_scheduler=null \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.541_129672_last.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.541_129672.json" 

# srun python duck/main.py run_name=duck_hard \
#     trainer=1host \
#     duck.boxes.parametrization=spherical \
#     duck.duck_loss.margin_pos=2 \
#     duck.duck_loss.margin_neg=2 \
#     duck.duck_loss.sampling_temperature=0.5 \
#     ablations.no_priors=true \
#     data.batch_size=8 \
#     optim.lr=1.0e-5 \
#     lr_scheduler=null \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.652_129673_last.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.652_129673.json" 


# srun python duck/main.py run_name=duck_hard \
#     trainer=1host \
#     duck.boxes.parametrization=spherical \
#     duck.duck_loss.margin_pos=0 \
#     duck.duck_loss.margin_neg=1 \
#     duck.duck_loss.sampling_temperature=0.5 \
#     ablations.no_priors=true \
#     data.batch_size=8 \
#     optim.lr=1.0e-5 \
#     lr_scheduler=null \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.667_129675_last.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.667_129675.json" 


srun python duck/main.py run_name=duck_hard \
    trainer=1host \
    duck.boxes.parametrization=spherical \
    duck.duck_loss.margin_pos=0 \
    duck.duck_loss.margin_neg=1 \
    duck.duck_loss.sampling_temperature=0.5 \
    ablations.no_priors=false \
    data.batch_size=8 \
    optim.lr=1.0e-5 \
    lr_scheduler=null \
    "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.644_129676_last.ckpt" \
    "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.644_129676.json" \

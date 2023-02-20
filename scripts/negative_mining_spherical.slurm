#!/bin/bash

#SBATCH --job-name=neg_spherical
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoints/%u/duck/logs/negatives_spherical_%j.log
#SBATCH --error=/checkpoints/%u/duck/logs/negatives_spherical_%j.log
#SBATCH --partition=learnai4p
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=12:00:00

export PYTHONPATH=.:$PYTHONPATH
## export NCCL_SOCKET_IFNAME=ens32

module purge

## srun python duck/preprocessing/negative_mining.py \
##     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck_spherical_mention_epoch\=0_val_f1\=0.752.ckpt" \
##     "output_path=/fsx/matzeni/data/duck/negatives_spherical_val_f1\=0.752.json" \
##     "output_index_path=/fsx/matzeni/data/duck/embeddings_spherical_val_f1\=0.752.pt"

srun python duck/preprocessing/negative_mining.py \
    "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_spherical_mention_epoch\=0_val_f1\=0.745.ckpt" \
    "output_path=/fsx/matzeni/data/duck/negatives_spherical_val_f1\=0.745.json" \
    "output_index_path=/fsx/matzeni/data/duck/embeddings_spherical_val_f1\=0.745.pt"
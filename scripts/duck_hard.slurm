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

# srun python duck/main.py run_name=duck_hard trainer=1host \
#     duck.duck_loss.sampling_temperature=1.0 \
#     duck.duck_loss._target_=duck.task.duck_loss.DuckNegativeSamplingLoss \
#     data.relation_threshold=5 \
#     duck.entity_priors=false \
#     data.batch_size=8 \
#     data.num_priors_per_mention=0 \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.747_129895.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.747_129895.json"

# srun python duck/main.py run_name=duck_hard trainer=1host \
#     duck.duck_loss.sampling_temperature=0.5 \
#     duck.duck_loss._target_=duck.task.duck_loss.DuckNegativeSamplingLossWithTemperature \
#     data.relation_threshold=5 \
#     data.batch_size=8 \
#     data.num_priors_per_mention=0 \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.743_129918.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.743_129918.json"


# srun python duck/main.py run_name=duck_hard trainer=1host \
#     duck.duck_loss.sampling_temperature=0.5 \
#     duck.duck_loss._target_=duck.task.duck_loss.DuckNegativeSamplingLossWithTemperature \
#     data.relation_threshold=5 \
#     data.batch_size=8 \
#     data.num_priors_per_mention=0 \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.749_129919.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.749_129919.json"


srun python duck/main.py run_name=duck_hard_cartesian trainer=1host \
    data.relation_threshold=0 \
    data.batch_size=8 \
    data.num_neighbors_per_entity=5 \
    duck.boxes.parametrization=relu \
    data.transform.max_mention_len=128 \
    data.num_priors_per_mention=0 \
    trainer.max_epochs=1 \
    trainer.min_epochs=1 \
    duck.duck_loss.sampling_temperature=1.0 \
    "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck_cartesian_epoch\=1_val_f1\=0.654_130181_last.ckpt" \
    "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_hard_val_f1\=0.840_130006.json"



# srun python duck/main.py run_name=duck_hard trainer=1host \
#     duck.duck_loss.sampling_temperature=0.5 \
#     duck.duck_loss._target_=duck.task.duck_loss.DuckNegativeSamplingLossWithTemperature \
#     data.relation_threshold=5 \
#     data.batch_size=8 \
#     data.num_priors_per_mention=0 \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_epoch\=0_val_f1\=0.762_130010.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_val_f1\=0.762_130010.json"


# srun python duck/main.py run_name=duck_hard_bert trainer=1host \
#     data.relation_threshold=0 \
#     data.batch_size=4 \
#     data.num_neighbors_per_entity=5 \
#     data.transform.max_mention_len=512 \
#     data.num_priors_per_mention=0 \
#     trainer.max_epochs=1 \
#     trainer.min_epochs=1 \
#     +trainer.accumulate_grad_batches=4 \
#     duck.duck_loss.sampling_temperature=1.0 \
#     "ckpt_path=/checkpoints/matzeni/duck/checkpoints/duck/duck_hard_bert_epoch\=0_val_f1\=0.860_130071.ckpt" \
#     "data.neighbors_path=/fsx/matzeni/data/duck/negatives/negatives_duck_hard_bert_val_f1\=0.860_130071.json"


# Bi-encoder Entity Linking Architecture

## Install requirements

Create the environment from the environment.yml file:

```
conda env create -f environment.yml
```

## Running tests

```
PYTHONPATH=.:$PYTHONPATH python -m unittest
```

## Pretrain disambiguation model on wikipedia data

The default path to the data is `/fsx/movb/data/matcha`. You need to modify config with appropriate paramters in `bela/conf/joint_el_disambiguation_only.yaml`. This is Hydra config in YAML format. Please read more about [hydra here](https://hydra.cc/) and [yaml here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html). To run training (you should be on machine with GPU):

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el_disambiguation_only
```

## Train model on entity linking data

Config is stored in `bela/conf/joint_el.yaml`. To run training (you should be on machine with GPU):

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el_disambiguation_only
```

## Train model using SLURM

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py -m --config-name joint_el_disambiguation_only trainer=slurm trainer.num_nodes=1 trainer.gpus=8
```

## Start interactive SLURM session
```
srun --gres=gpu:1 --partition=a100 --time=3:00:00 --pty /bin/bash -l
```

Saves to output to:
/data/home/kassner/BELA/multirun/*

## run tensorboard
```
tensorboard --logdir ./ --port 6017
```  

## run jupyterlab
```
jupyter-lab --ip=0.0.0.0 --port=8888
ssh cluster_id -L 8844:a100-st-p4d24xlarge-35:8888
http://127.0.0.1:8844/
```
# Bi-encoder Entity Linking Architecture

## Install requirements

First you need to install pytorch with cuda11 support:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

Then you can install other packages:
```
pip install -r requirements.txt
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
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el
```

## Train model using SLURM

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py -m --config-name joint_el_disambiguation_only trainer=slurm trainer.num_nodes=1 trainer.gpus=8
```

## Data

# Pre-training

# Joint Training

1. Wikipedia:
   1. t2
   2. t1
   3. Number of novel entities:
   4. Counts novel entities:
2. Wikinews:
   1. t2
   2. t1
   3. Number of novel entities:
   4. Counts novel entities:
3. BBC:
   1. t2: 5.101
   2. t1: 6.775 
   3. Number of novel entities:
   4. Counts novel entities:


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
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el_disambiguation_only
```

## Train model using SLURM

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py -m --config-name joint_el_disambiguation_only trainer=slurm trainer.num_nodes=1 trainer.gpus=8
```

## Start interactive SLURM session
```
srun --gres=gpu:1 --partition=a100 --time=3:00:00 --pty /bin/bash -l --memory
```
## run tensorboard
tensorboard --logdir ./ --port 6017  

## run jupyterlab
jupyter-lab --ip=0.0.0.0 --port=8888
ssh cluster_id -L 8844:a100-st-p4d24xlarge-35:8888
http://127.0.0.1:8844/

## BLINK encodings
PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/transfer_blink_format_t1.py --base_wikidata /fsx/kassner/wikidata/
PYTHONPATH=.:$PYTHONPATH python blink/main_dense.py --test_mentions /fsx/kassner/OSCAR/subset/cnn_bbc_novel_blink_format.jsonl  --fast 

## Mention Detection

PYTHONPATH=.:$PYTHONPATH python bela/main.py -m --config-name joint_el_eval_md trainer=slurm trainer.num_nodes=1 trainer.gpus=8

## Cluster

PYTHONPATH=.:$PYTHONPATH python bela/scripts/cluster_greedy_nn.py --input ../BLINK/output/cnn_bbc_novel_blink_format_mention_embeddings.tsv --output output_clustering/



/checkpoints/kassner/hydra_outputs/main/2021-11-02-091125
bert
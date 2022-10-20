# Bi-encoder Entity Linking Architecture

## Install requirmenets

First you need to install pytorch with cuda11 support:

```
conda install "pytorch=1.12.1=aws*" cudatoolkit=11.6 torchvision torchaudio \
--override-channels \
-c https://aws-pytorch.s3.us-west-2.amazonaws.com \
-c pytorch \
-c nvidia \
-c conda-forge
```

Then you can install other packages:
```
pip install -r requirements.txt
```

## Runing tests

```
PYTHONPATH=.:$PYTHONPATH python -m unittest
```

## Bela
### Pretrain disambiguation model on wikipedia data

The default path to the data is `/fsx/movb/data/matcha`. You need to modify config with appropriate paramters in `bela/conf/joint_el_disambiguation_only.yaml`. This is Hydra config in YAML format. Please read more about [hydra here](https://hydra.cc/) and [yaml here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html). To run training (you should be on machine with GPU):

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el_disambiguation_only
```

### Train model on entity linkind data

Config is stored in `bela/conf/joint_el.yaml`. To run training (you should be on machine with GPU):

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el
```

### Train model using SLUMR

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py -m --config-name joint_el_disambiguation_only trainer=slurm trainer.num_nodes=1 trainer.gpus=8
```

## MBlink (Multilingual Blink)
### Train on in-batch negatives (english, xlmr-base)

```
PYTHONPATH=.:$PYTHONPATH python mblink/main.py --config-name blink_xlmr trainer.gpus=8
```

Available configs:
* blink_xlmr - English, xlmr-base
* blink_xlmr_large - English, xlmr-large

### Train model using SLUMR

```
PYTHONPATH=.:$PYTHONPATH python mblink/main.py -m --
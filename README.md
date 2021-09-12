# Bi-encoder Entity Linking Architecture

## Install requirmenets

```
pip install -r requirements.txt
```

## Pretrain disambiguation model on wikipedia data

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el_disambiguation_only
```

## Train model on entity linkind data

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el
```

## Train model using SLUMR

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py -m --config-name joint_el_disambiguation_only trainer=slurm trainer.num_nodes=1 trainer.gpus=8
```

## Runing tests

```
PYTHONPATH=.:$PYTHONPATH python -m unittest
```

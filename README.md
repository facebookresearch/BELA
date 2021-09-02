# Bi-encoder Entity Linking Architecture

## Pretrain disambiguation model on wikipedia data (locally)

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el_disambiguation_only
```

## Train model on entity linkind data (locally)

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el
```

## Train model on 8 GPUs

```
PYTHONPATH=.:$PYTHONPATH python bela/main.py --config-name joint_el trainer.gpus=8
```

## Runing tests

```
PYTHONPATH=.:$PYTHONPATH python -m unittest
```

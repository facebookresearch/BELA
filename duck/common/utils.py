import logging
from pathlib import Path
import random
from tkinter import E
from typing import Dict, List, Set
import torch
import numpy as np
from omegaconf import OmegaConf
import sys
import pickle
import json


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    
    Args:
        dtype: torch dtype of supertype float

    Returns:
        float: Tiny value

    Raises:
        TypeError: Given non-float or unknown type
    """

    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")

    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def make_reproducible(seed, ngpus=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if ngpus > 0:
        torch.cuda.manual_seed_all(seed)


def dict_to_yaml(config, path='config.yaml'):
    config = OmegaConf.create(config)
    yaml = OmegaConf.to_yaml(config)

    with open(path, 'w') as f:
        f.write(yaml)



def get_logger(output_dir=None):
    if output_dir != None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('DUCK')
    logger.setLevel(10)
    return logger


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l.strip()) for l in f.readlines()]


def most_frequent_relations(
    ent_to_rel: Dict[str, List[str]],
    k: int
) -> List[str]:
    all_properties = {p for pids in ent_to_rel.values() for p in pids}
    rel_to_ent = {p: set() for p in all_properties}
    for e, rels in ent_to_rel.items():
        for p in rels:
            rel_to_ent[p].add(e)
    rel_to_num_ents = {
        p: len(ents)
        for p, ents in sorted(
            rel_to_ent.items(), key=lambda x: len(x[1]), reverse=True
        )
    }
    return list(rel_to_num_ents.keys())[:k]


def device(device_id=None, gpu=None):
    if device_id is not None:
        return torch.device(str(device_id))
    if gpu is None and torch.cuda.is_available():
        return torch.device("cuda")
    if gpu:
        return torch.device("cuda")
    return torch.device('cpu')
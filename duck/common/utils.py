import logging
from optparse import OptionParser
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
import copy


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


def make_reproducible(seed=42, ngpus=1):
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


def activation_function(name: str):
    name = name.lower().strip()
    activations = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "softmax": torch.softmax
    }
    if name in activations:
        return activations[name]
    options =  ", ".join(activations.keys())
    raise ValueError(f"Unsupported activation {name}. The available options are: {options}")


def device(device_id=None, gpu=None):
    if device_id is not None:
        return torch.device(str(device_id))
    if gpu is None and torch.cuda.is_available():
        return torch.device("cuda")
    if gpu:
        return torch.device("cuda")
    return torch.device('cpu')


def generate_mask_list(values):
    if len(values) == 0:
        return []
    if isinstance(values[0], torch.Tensor):
        return [torch.ones_like(v).bool() for v in values]
    if not isinstance(values[0], list):
        return [True] * len(values)
    return [generate_mask_list(v) for v in values]


def max_depth_of_nested_list(lst):
    if not isinstance(lst, list):
        return 0
    if len(lst) == 0:
        return 1
    depths = [
        max_depth_of_nested_list(item)
        for item in lst
        if isinstance(item, list)
    ]
    max_depth = max(depths) if len(depths) > 0 else 0
    return 1 + max_depth


def empty_list_of_depth(n):
    result = []
    for i in range(n - 1):
        result = [result]
    return result


def equalize_depth(nested_list, depth=None):
    if depth is None:
        depth = max_depth_of_nested_list(nested_list)
    if depth == 0 or not isinstance(nested_list, list):
        return
    if len(nested_list) == 0 and depth > 1:
        item = empty_list_of_depth(depth - 1)
        nested_list.append(item)
    for item in nested_list:
        if isinstance(item, list):
            equalize_depth(item, depth - 1) 


def _list_to_tensor(values, pad_value, dtype=None):
    if isinstance(values, torch.Tensor):
        return values
    if len(values) == 0:
        return torch.tensor([], dtype=dtype)
    if isinstance(values[0], torch.Tensor):
        return pad_tensors(values, pad_value=pad_value)
    if not isinstance(values[0], list):
        return torch.tensor(values, dtype=dtype)
    return _list_to_tensor([
        _list_to_tensor(v, pad_value=pad_value) for v in values
    ], pad_value=pad_value)


def list_to_tensor(values, pad_value, dtype=None):
    values = list(copy.deepcopy(values))
    equalize_depth(values)
    mask = generate_mask_list(values)
    tensor = _list_to_tensor(values, pad_value=pad_value, dtype=dtype)
    mask = _list_to_tensor(mask, pad_value=False)
    return tensor, mask


def pad_tensors(tensors, pad_value):
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.full(padded_dim, pad_value)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, :size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, :size[0], :size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
        elif len(size) == 4:
            padded_tensor[i, :size[0], :size[1], :size[2], :size[3]] = tensor
        else:
            raise ValueError('Padding is supported for up to 4D tensors')
    return padded_tensor

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
import math
from einops import rearrange, repeat


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


def seed_prg(seed=42, ngpus=1):
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
    if name is None:
        return None
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


def any_none(values):
    return any(
        v is None or (
            isinstance(v, list) and any_none(v)
        )
        for v in values
    )
    

def _list_to_tensor(values, pad_value, dtype=None, size=None):
    if isinstance(values, torch.Tensor):
        return values
    if len(values) == 0:
        return torch.tensor([], dtype=dtype)
    if isinstance(values[0], torch.Tensor):
        return pad_tensors(values, pad_value=pad_value, size=size)
    if not isinstance(values[0], list):
        return torch.tensor(values, dtype=dtype)
    return _list_to_tensor([
        _list_to_tensor(v, pad_value=pad_value, size=size) for v in values
    ], pad_value=pad_value, size=size)


def list_to_tensor(values, pad_value, dtype=None, size=None):
    values = list(copy.deepcopy(values))
    equalize_depth(values)
    mask = generate_mask_list(values)
    tensor = _list_to_tensor(values, pad_value=pad_value, dtype=dtype, size=size)
    mask = _list_to_tensor(mask, pad_value=False, size=size)
    return tensor, mask


def pad_tensors(tensors, pad_value, size=None):
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        if size is not None:
            dim_size = size
            if isinstance(size, tuple):
                dim_size = size[dim] or 0
            max_dim = max(max_dim, dim_size)
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


def logsubexp(x, y, eps=1e-7):
    return x + torch.log1p(-torch.exp(y - x) + eps)


def log1mexp(x: torch.Tensor, split_point=None,
             exp_zero_eps=1e-7, clamp=True) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).
    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).
    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
    """
    split_point = split_point if split_point is not None else math.log(0.5)
    if clamp:
        x = x.clamp(max=0.0)
    logexpm1_switch = x > split_point
    result = torch.zeros_like(x)
    logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
    result[logexpm1_switch] = logexpm1.detach() + (
        logexpm1_bw - logexpm1_bw.detach())
    #Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
    result[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]).clamp(0, 1))

    return result


# def logexpm1(x: torch.Tensor, split_point=None,
#              exp_zero_eps=1e-7, clamp=True) -> torch.Tensor:
#     """
#     Computes log(exp(x) - 1).
#     Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).
#     = log1p(-exp(x)) * log(-1) when x <= log(1/2)
#     or
#     = log(expm1(x)) when log(1/2) < x <= 0
#     """
#     split_point = split_point if split_point is not None else math.log(0.5)
#     if clamp:
#         x = x.clamp(max=0.0)
#     logexpm1_switch = x > split_point
#     result = torch.zeros_like(x)
#     logexpm1 = (torch.expm1(x[logexpm1_switch]).log().clamp_min(1e-38))
#     # hack the backward pass
#     # if expm1(x) gets very close to zero, then the grad log() will produce inf
#     # and inf*0 = nan. Hence clip the grad so that it does not produce inf
#     logexpm1_bw = torch.log(torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
#     result[logexpm1_switch] = logexpm1.detach() + (
#         logexpm1_bw - logexpm1_bw.detach())
#     #Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
#     result[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]).clamp(0, 1)) * torch.log(-torch.ones_like(x))

#     return result


def logexpm1(x: torch.Tensor):
    """
    Numerically stable implementation of the inverse softplusL log(exp(x) - 1)
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    also see: https://github.com/JuliaStats/LogExpFunctions.jl/blob/59b8c0984359a7093561049b5e268ea979bedfa6/src/basicfuns.jl#L231
    """
    result = torch.zeros_like(x)
    eps = 1e-5
    # zone0 = (x.abs() <= eps)
    zone1 = (x <= 18.) # & (x.abs() > eps)
    zone2 = (x > 18.) & (x < 33.3) 
    zone3 = (x >= 33.3)
    result[zone1] = torch.log(torch.relu(torch.expm1(x[zone1])) + eps)
    # result[zone1] = torch.log(torch.expm1(x[zone1]))
    result[zone2] = x[zone2] - torch.exp(-(x[zone2]))
    result[zone3] = torch.exp(-x[zone3])
    return result


def mean_over_batches(batch_values, prefix=None, suffix=None):
    """
    :param batch_values: a list of dictionaries containing metrics for each batch
    :param prefix: prefix to add to the name of each metric
    :return: a dictionary containing mean values over batches for each metric
    """
    metrics = batch_values[0].keys()
    prefix = "" if prefix is None else f"{prefix}_"
    suffix = "" if suffix is None else f"_{suffix}"
    return {
        f"{prefix}{metric}{suffix}": torch.tensor([x[metric] for x in batch_values]).float().mean()
        for metric in metrics
    }

def prefix_suffix_keys(dictionary, prefix=None, suffix=None, separator=""):
    prefix = "" if prefix is None else f"{prefix}{separator}"
    suffix = "" if suffix is None else f"{separator}{suffix}"
    return {
        f"{prefix}{key}{suffix}": value for key, value in dictionary.items()
    }

def metric_dict_to_string(kv_map, separator="\t"):
    lines = [f"{key}: {value:.3f}" for key, value in kv_map.items()]
    return separator.join(lines)

def tensor_set_intersection(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts > 1]

def tensor_set_difference(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts == 1]

def cartesian_to_spherical(cartesian):
    eps = 1e-6
    n = cartesian.size(-1)
    x = repeat(cartesian, "... n1 -> ... n1 n2", n2=n)
    mask = torch.tril(torch.ones(n, n, device=cartesian.device)) == 1
    x = x.masked_fill(mask == 0, float(0.0))
    x = torch.sqrt(torch.sum(x ** 2, dim=-2)) + eps
    angle = torch.acos(cartesian[..., :-1] / x[..., :-1])
    neg_mask = cartesian[..., -1] < 0
    angle[neg_mask, -1] = 2 * torch.pi - angle[neg_mask, -1]
    radius = torch.linalg.vector_norm(cartesian, ord=2, dim=-1)
    return radius, angle
    


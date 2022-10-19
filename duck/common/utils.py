import logging
from pathlib import Path
import random
from tkinter import E
import torch
import numpy as np
from omegaconf import OmegaConf
import sys

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
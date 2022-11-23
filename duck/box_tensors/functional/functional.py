from typing import (
    List,
    Sequence,
    Tuple,
    Union,
    Any,
    Optional,
    Type,
    TypeVar
)
import logging
import torch
from torch import Tensor
from einops import rearrange, repeat

from duck.box_tensors.box_tensor import BoxTensor, TBoxTensor


def stack_box(
    boxes: Sequence[BoxTensor],
    dim: int = 0
):
    left = torch.stack([box.left for box in boxes], dim=dim)
    right = torch.stack([box.right for box in boxes], dim=dim)
    return BoxTensor((left, right))


def cat_box(
    boxes: Sequence[BoxTensor],
    dim: int = 0
):
    left = torch.cat([box.left for box in boxes], dim=dim)
    right = torch.cat([box.right for box in boxes], dim=dim)
    return BoxTensor((left, right))


def rearrange_box(object: Union[BoxTensor, Sequence[BoxTensor]], expression: str, **kwargs) -> BoxTensor:
    box = object
    if isinstance(object, Sequence):
        box = stack_box(object)
    return box.rearrange(expression, **kwargs)


def repeat_box(object: Union[BoxTensor, Sequence[BoxTensor]], expression: str, **kwargs) -> BoxTensor:
    box = object
    if isinstance(object, Sequence):
        box = stack_box(object)
    return box.repeat(expression, **kwargs)
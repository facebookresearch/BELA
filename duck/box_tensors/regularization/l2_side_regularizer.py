from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from duck.box_tensors import BoxTensor
from duck.box_tensors.regularization.regularizer import BoxRegularizer


def l2_side_regularizer(
    box_tensor: BoxTensor, log_scale: bool = False
) -> torch.Tensor:
    left = box_tensor.left  # (..., box_dim)
    right = box_tensor.right  # (..., box_dim)
    eps = 1e-13

    if not log_scale:
        return (right - left) ** 2
    else:
        return torch.log(torch.abs(right - left) + eps)


class L2BoxSideRegularizer(BoxRegularizer):
    def __init__(
        self,
        weight: float,
        log_scale: bool = False,
        reduction: str = 'mean',
        min_threshold: float = None,
        max_threshold: float = None
    ) -> None:
        super().__init__(
            weight,
            log_scale=log_scale,
            reduction=reduction,
        )
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def _forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        regularization = l2_side_regularizer(box_tensor, log_scale=self.log_scale)
        mask = torch.zeros_like(box_tensor.left).bool()
        if self.min_threshold is not None:
            mask |= box_tensor.left > self.min_threshold
        if self.max_threshold is not None:
            mask |= box_tensor.right < self.max_threshold
        regularization[mask] = 0.0
        return regularization

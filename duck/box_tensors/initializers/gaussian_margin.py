from typing import List, Tuple
import numpy as np
import torch
from duck.box_tensors import BoxTensor
from duck.box_tensors.initializers.abstract_initializer import BoxInitializer

def gaussian_margin_boxes(
    dimensions: int,
    num_boxes: int,
    minimum: float = 0.0,
    maximum: float = 1.0,
    mean: float = 0.0,
    stddev: float = 1e-2,
    margin: float = 0.0,
    cut=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8

    if stddev <= 0:
        raise ValueError(f"Standard deviation should be >_ 0 but got {stddev}")

    if mean < 0:
        raise ValueError(
            ValueError(f"Mean should be >_ 0 but got {stddev}")
        )

    if maximum < minimum:
        raise ValueError(f"Expected: maximum {maximum} >= minimum {minimum}")
    
    delta_max = np.random.normal(mean, stddev, size=(num_boxes, dimensions))
    delta_min = np.random.normal(mean, stddev, size=(num_boxes, dimensions))
    if cut:
        delta_min = np.abs(delta_min)
        delta_max = np.abs(delta_max)

    left = minimum + margin + delta_min + eps
    right = maximum - margin - delta_max - eps

    return torch.tensor(left), torch.tensor(right)


class GaussianMarginBoxInitializer(BoxInitializer):
    def __init__(
        self,
        dimensions: int,
        num_boxes: int,
        minimum: float = 0.0,
        maximum: float = 1.0,
        mean: float = 0.0,
        stddev: float = 1e-3,
        cut=True
    ) -> None:
        self.dimensions = dimensions
        self.num_boxes = num_boxes
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.stddev = stddev
        self.cut = cut

    def __call__(self, t: torch.Tensor) -> None: 
        left, right = gaussian_margin_boxes(
            self.dimensions,
            self.num_boxes,
            self.minimum,
            self.maximum,
            self.mean,
            self.stddev,
            self.cut
        )
        with torch.no_grad():
            data = torch.cat((left, right), -1)
            t.copy_(data)

    

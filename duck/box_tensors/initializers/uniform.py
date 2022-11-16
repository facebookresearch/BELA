from typing import List, Tuple
import numpy as np
import torch
from duck.box_tensors import BoxTensor
from duck.box_tensors.initializers.abstract_initializer import BoxInitializer


def uniform_boxes(
    dimensions: int,
    num_boxes: int,
    minimum: float = 0.0,
    maximum: float = 1.0,
    delta_min: float = 0.01,
    delta_max: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8

    if delta_min <= 0:
        raise ValueError(f"Delta min should be > 0 but is {delta_min}")

    if (delta_max - delta_min) <= 0:
        raise ValueError(
            f"Expected: delta_max {delta_max}  > delta_min {delta_min} "
        )

    if delta_max > (maximum - minimum):
        raise ValueError(
            f"Expected: delta_max {delta_max} <= (max-min) {maximum - minimum}"
        )

    if maximum <= minimum:
        raise ValueError(f"Expected: maximum {maximum} > minimum {minimum}")
    centers = np.random.uniform(
        minimum + delta_max / 2.0 + eps,
        maximum - delta_max / 2.0 - eps,
        size=(num_boxes, dimensions),
    )

    deltas = np.random.uniform(
        delta_min, delta_max - eps, size=(num_boxes, dimensions)
    )
    left = centers - deltas / 2.0 + eps
    right = centers + deltas / 2.0 - eps
    assert (left >= minimum).all()
    assert (right <= maximum).all()

    return torch.tensor(left), torch.tensor(right)


class UniformBoxInitializer(BoxInitializer):
    def __init__(
        self,
        dimensions: int,
        num_boxes: int,
        minimum: float = 0.0,
        maximum: float = 1.0,
        delta_min: float = 0.01,
        delta_max: float = 0.5,
    ) -> None:
        self.dimensions = dimensions
        self.num_boxes = num_boxes
        self.minimum = minimum
        self.maximum = maximum
        self.delta_min = delta_min
        self.delta_max = delta_max

    def __call__(self, t: torch.Tensor) -> None: 
        left, right = uniform_boxes(
            self.dimensions,
            self.num_boxes,
            self.minimum,
            self.maximum,
            self.delta_min,
            self.delta_max,
        )
        with torch.no_grad():
            data = torch.cat((left, right), -1)
            t.copy_(data)

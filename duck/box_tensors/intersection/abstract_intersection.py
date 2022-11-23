from typing import Optional
import torch

from duck.box_tensors import BoxTensor
from duck.box_tensors.functional import stack_box


class AbstractIntersection(torch.nn.Module):
    """Abstract class for the intersection of boxes"""

    def __init__(
        self,
        dim: int = 0
    ):
        super(AbstractIntersection, self).__init__()
        self.dim = dim

    def forward(
        self,
        *args
    ) -> BoxTensor:
        assert len(args) > 0, "Expected at least one box tensor"
        box = args[0]
        if len(args) > 1:
            assert self.dim == 0, \
                "A sequence of tensors was given, but dim is not 0"
            box = stack_box(args)
        return self._forward(box)

    def _forward(self, box: BoxTensor) -> BoxTensor:
        raise NotImplementedError

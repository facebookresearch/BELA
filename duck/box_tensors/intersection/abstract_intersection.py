import torch

from duck.box_tensors import BoxTensor


class AbstractIntersection(torch.nn.Module):
    """Abstract class for the intersection of two boxes"""

    def forward(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        # broadcast if necessary
        if len(left.box_shape) >= len(right.box_shape):
            right.broadcast(left.box_shape)
        else:
            left.broadcast(right.box_shape)

        return self._forward(left, right)

    def _forward(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        raise NotImplementedError

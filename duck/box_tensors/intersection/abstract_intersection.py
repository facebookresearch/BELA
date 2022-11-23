from typing import Optional
import torch

from duck.box_tensors import BoxTensor


class AbstractIntersection(torch.nn.Module):
    """Abstract class for the intersection of boxes"""

    def forward(self,
        box1: BoxTensor,
        box2: Optional[BoxTensor] = None,
        dim: int = 0
    ) -> BoxTensor:
        box = box1
        if box2 is not None:
            # broadcast if necessary
            if len(box1.box_shape) >= len(box2.box_shape):
                box2.broadcast(box1.box_shape)
            else:
                box1.broadcast(box2.box_shape)
            
            box = box1.stack(box2, dim=dim)

        return self._forward(box, dim=dim)

    def _forward(self, box: BoxTensor, dim=0) -> BoxTensor:
        raise NotImplementedError

    def forward_old(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        # broadcast if necessary
        if len(left.box_shape) >= len(right.box_shape):
            right.broadcast(left.box_shape)
        else:
            left.broadcast(right.box_shape)

        return self._forward_old(left, right)

    def _forward_old(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        raise NotImplementedError

import torch

from duck.box_tensors.intersection.abstract_intersection import AbstractIntersection
from duck.box_tensors.box_tensor import BoxTensor, TBoxTensor


def hard_intersection(
    box: TBoxTensor,
    dim: int = 0
):
    left = torch.max(box.left, dim=dim).values
    right = torch.min(box.right, dim=dim).values
    
    return BoxTensor.from_corners(left, right)


class HardIntersection(AbstractIntersection):
    """Hard intersection operation as a Layer/Module"""
    def _forward(self, box: TBoxTensor):
        """Returns the intersection of t1 and t2.

        Args:
            t1: First operand of the intersection
            t2: Second operand of the intersection

        Returns:
            The BoxTensor obtained by performing the hard intersection of t1 and t2
        """
        return hard_intersection(box, self.dim)

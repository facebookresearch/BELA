import torch

from duck.box_tensors.intersection.abstract_intersection import AbstractIntersection
from duck.box_tensors.box_tensor import BoxTensor, TBoxTensor


def hard_intersection_old(t1: TBoxTensor, t2: TBoxTensor) -> TBoxTensor:
    """Hard Intersection of two boxes.

    Args:
        t1: BoxTensor representing the left operand
        t2: BoxTensor representing the right operand

    Returns:
         The BoxTensor obtained by interection of t1 and t2
    """
    left = torch.max(t1.left, t2.left)
    right = torch.min(t1.right, t2.right)

    return t1.from_corners(left, right)


def hard_intersection(
    box: TBoxTensor,
    dim: int = 0
):
    left = torch.max(box.left, dim=dim).values
    right = torch.min(box.right, dim=dim).values
    
    return BoxTensor.from_corners(left, right)


class HardIntersection(AbstractIntersection):
    """Hard intersection operation as a Layer/Module"""


    def _forward(self, box: TBoxTensor, dim=0):
        return hard_intersection(box, dim)

    def _forward_old(self, t1: BoxTensor, t2: BoxTensor) -> BoxTensor:
        """Returns the intersection of t1 and t2.

        Args:
            t1: First operand of the intersection
            t2: Second operand of the intersection

        Returns:
            The BoxTensor obtained by performing the hard intersection of t1 and t2
        """

        return hard_intersection_old(t1, t2)

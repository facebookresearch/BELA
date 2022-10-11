import torch

from bela.duck.box_tensors.intersection.abstract_intersection import AbstractIntersection
from bela.duck.box_tensors.box_tensor import TBoxTensor


def hard_intersection(t1: TBoxTensor, t2: TBoxTensor) -> TBoxTensor:
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


class HardIntersection(AbstractIntersection):
    """Hard intersection operation as a Layer/Module"""

    def _forward(self, t1: TBoxTensor, t2: TBoxTensor) -> TBoxTensor:
        """Returns the intersection of t1 and t2.

        Args:
            t1: First operand of the intersection
            t2: Second operand of the intersection

        Returns:
            The BoxTensor obtained by performing the hard intersection of t1 and t2
        """

        return hard_intersection(t1, t2)

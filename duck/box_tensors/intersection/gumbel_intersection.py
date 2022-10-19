from typing import Tuple, Optional

import torch

from duck.box_tensors.box_tensor import TBoxTensor
from duck.box_tensors.intersection.abstract_intersection import AbstractIntersection


def _compute_logaddexp_with_clipping_and_separate_forward(
    t1: TBoxTensor, t2: TBoxTensor, intersection_temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_data = torch.stack((t1.left, -t1.right), -2)
    t2_data = torch.stack((t2.left, -t2.right), -2)
    lse = torch.logaddexp(
        t1_data / intersection_temperature, t2_data / intersection_temperature
    )

    left = intersection_temperature * lse[..., 0, :]
    right = -intersection_temperature * lse[..., 1, :]

    left_value = torch.max(left, torch.max(t1.left, t2.left))
    right_value = torch.min(right, torch.min(t1.right, t2.right))

    left_final = (left - left.detach()) + left_value.detach()
    right_final = (right - right.detach()) + right_value.detach()

    return left_final, right_final


def _compute_logaddexp_with_clipping(
    t1: TBoxTensor, t2: TBoxTensor, intersection_temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_data = torch.stack((t1.left, -t1.right), -2)
    t2_data = torch.stack((t2.left, -t2.right), -2)
    lse = torch.logaddexp(
        t1_data / intersection_temperature, t2_data / intersection_temperature
    )

    left = intersection_temperature * lse[..., 0, :]
    right = -intersection_temperature * lse[..., 1, :]

    left_value = torch.max(left, torch.max(t1.left, t2.left))
    right_value = torch.min(right, torch.min(t1.right, t2.right))

    return left_value, right_value


def _compute_logaddexp(
    t1: TBoxTensor, t2: TBoxTensor, intersection_temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_data = torch.stack((t1.left, -t1.right), -2)
    t2_data = torch.stack((t2.left, -t2.right), -2)
    lse = torch.logaddexp(
        t1_data / intersection_temperature, t2_data / intersection_temperature
    )

    left = intersection_temperature * lse[..., 0, :]
    right = -intersection_temperature * lse[..., 1, :]

    return left, right


def gumbel_intersection(
    box1: TBoxTensor,
    box2: TBoxTensor,
    intersection_temperature: float = 1.0,
    approximation_mode: Optional[str] = None,
) -> TBoxTensor:
    """Computes the Gumbel intersection of box1 and box2

    Performs the intersection operation as described in 
    `Improving Local Identifiability in Probabilistic Box Embeddings
    <https://arxiv.org/abs/2010.04831>`.

    Args:
        box1: BoxTensor representing the left operand of the intersection
        box2: BoxTensor representing the right operand of the intersection
        intersection_temperature: gumbel's beta parameter
        approximation_mode: Either hard clipping ('clipping') or hard clipping with separate value
            for forward and backward passes  ('clipping_forward') or
            `None` to not use any approximation.

    Returns:
        The BoxTensor obtained by Gumbel interaction of box1 and box2.
    """
    t1 = box1
    t2 = box2

    if intersection_temperature == 0:
        raise ValueError("intersection_temperature must be non-zero.")

    if approximation_mode is None:
        left, right = _compute_logaddexp(t1, t2, intersection_temperature)
    elif approximation_mode == "clipping":
        left, right = _compute_logaddexp_with_clipping(
            t1, t2, intersection_temperature
        )
    elif approximation_mode == "clipping_forward":
        left, right = _compute_logaddexp_with_clipping_and_separate_forward(
            t1, t2, intersection_temperature
        )
    else:
        raise ValueError(
            f"{approximation_mode} is not a valid approximation_mode."
        )

    return box1.from_corners(left, right)


class GumbelIntersection(AbstractIntersection):
    """Gumbel intersection operation as a Layer/Module.

    Performs the intersection operation as described in 
    `Improving Local Identifiability in Probabilistic Box Embeddings
    <https://arxiv.org/abs/2010.04831>`_ .
    """

    def __init__(
        self,
        intersection_temperature: float = 1.0,
        approximation_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            intersection_temperature: Gumbel's beta parameter
            approximation_mode: Either hard clipping ('clipping') or hard clipping with separate value
            for forward and backward passes  ('clipping_forward') or
            `None` to not use any approximation.
        """
        super().__init__() 
        self.intersection_temperature = intersection_temperature
        self.approximation_mode = approximation_mode

    def _forward(self, left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
        """Computes the gumbel intersection of left and right.

        Args:
            left: First operand of the intersection
            right: Second operand of the intersection

        Returns:
            The box obtained by Gumbel intersection of the two operands

        """

        return gumbel_intersection(
            left,
            right,
            intersection_temperature=self.intersection_temperature,
            approximation_mode=self.approximation_mode,
        )

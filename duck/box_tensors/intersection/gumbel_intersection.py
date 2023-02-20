from typing import Tuple, Optional

import torch

from duck.box_tensors.box_tensor import BoxTensor, TBoxTensor
from duck.box_tensors.intersection.abstract_intersection import AbstractIntersection


def _compute_logsumexp_with_clipping_and_separate_forward(
    box: TBoxTensor,
    dim: int = 0,
    intersection_temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    left, right = _compute_logsumexp(
        box,
        dim=dim,
        intersection_temperature=intersection_temperature
    )

    left_value = torch.max(left, torch.max(box.left, dim=dim).values)
    right_value = torch.min(right, torch.min(box.right, dim=dim).values)

    left_final = (left - left.detach()) + left_value.detach()
    right_final = (right - right.detach()) + right_value.detach()

    return left_final, right_final


def _compute_logsumexp_with_clipping(
    box: TBoxTensor,
    dim: int = 0,
    intersection_temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    left, right = _compute_logsumexp(
        box,
        dim=dim,
        intersection_temperature=intersection_temperature
    )

    left_value = torch.max(left, torch.max(box.left, dim=dim).values)
    right_value = torch.min(right, torch.min(box.right, dim=dim).values)

    return left_value, right_value


def _compute_logsumexp(
    box: TBoxTensor,
    dim: int = 0,
    intersection_temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    box_data = torch.stack((box.left, -box.right), dim=-1)
    if dim < 0:
        dim -= 1
    lse = torch.logsumexp(
        box_data / intersection_temperature, dim=dim
    )

    left = intersection_temperature * lse[..., 0]
    right = -intersection_temperature * lse[..., 1]

    return left, right


def gumbel_intersection(
    box: TBoxTensor,
    dim: int = 0,
    intersection_temperature: float = 1.0,
    approximation_mode: Optional[str] = None,
) -> TBoxTensor:
    """Computes the Gumbel intersection of box1 and box2

    Performs the intersection operation as described in 
    `Improving Local Identifiability in Probabilistic Box Embeddings
    <https://arxiv.org/abs/2010.04831>`.

    Args:
        box: BoxTensor containing the boxes to intersect
        dim: dimension along which to perform the intersection
        intersection_temperature: gumbel's beta parameter
        approximation_mode: Either hard clipping ('clipping') or hard clipping with separate value
            for forward and backward passes  ('clipping_forward') or
            `None` to not use any approximation.

    Returns:
        The BoxTensor obtained by Gumbel intersection of box along dimension dim 
    """
    if intersection_temperature == 0:
        raise ValueError("intersection_temperature must be non-zero.")

    if approximation_mode is None:
        left, right = _compute_logsumexp(box, dim, intersection_temperature)
    elif approximation_mode == "clipping":
        left, right = _compute_logsumexp_with_clipping(
            box, dim, intersection_temperature
        )
    elif approximation_mode == "clipping_forward":
        left, right = _compute_logsumexp_with_clipping_and_separate_forward(
            box, dim, intersection_temperature
        )
    else:
        raise ValueError(
            f"{approximation_mode} is not a valid approximation_mode."
        )

    return BoxTensor.from_corners(left, right)


class GumbelIntersection(AbstractIntersection):
    """
    Gumbel intersection operation as a Layer/Module.

    Performs the intersection operation as described in 
    `Improving Local Identifiability in Probabilistic Box Embeddings
    <https://arxiv.org/abs/2010.04831>`_ .
    """

    def __init__(
        self,
        dim: int = 0,
        intersection_temperature: float = 1.0,
        approximation_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            dim: dimension along which to perform the intersection
            intersection_temperature: Gumbel's beta parameter
            approximation_mode: Either hard clipping ('clipping') or hard clipping with separate value
            for forward and backward passes  ('clipping_forward') or
            `None` to not use any approximation.
        """
        super().__init__(dim=dim) 
        self.intersection_temperature = intersection_temperature
        self.approximation_mode = approximation_mode

    def _forward(self, box: TBoxTensor):
        """
        Computes the gumbel intersection of boxes in a box tensor along
        a given dimension.

        Args:
            box: box tensor containing the boxes to intersect

        Returns:
            The box obtained by Gumbel intersection of the input tensor along 
            the given dimension
        """
        return gumbel_intersection(
            box,
            dim=self.dim, 
            intersection_temperature=self.intersection_temperature,
            approximation_mode=self.approximation_mode
        )
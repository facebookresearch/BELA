import numpy as np
import torch
from torch.nn.functional import softplus

from duck.common import constant
from duck.common.utils import tiny_value_of_dtype
from duck.box_tensors.volume.abstract_volume import AbstractVolume
from duck.box_tensors import BoxTensor

eps = tiny_value_of_dtype(torch.float)
euler_gamma = constant.EULER_GAMMA


def bessel_volume_approx(
    box_tensor: BoxTensor,
    volume_temperature: float = 1.0,
    intersection_temperature: float = 1.0,
    log_scale: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Implementation of the volume of a box tensor using the Softplus 
    function as an approximation of the Bessel funtion
    (as described here: https://arxiv.org/abs/2010.04831).

    Args:
        box_tensor: input box
        volume_temperature: 1/volume_temperature is the beta parameter for the softplus
        intersection_temperature: the intersection_temperature parameter (same value used in intersection).
        log_scale: Whether the output should be in log scale or not.


    Returns:
        Tensor of shape (..., ) when the input box has shape (..., 2, dim)
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"Scale should be in (0,1] but is {scale}")

    if volume_temperature == 0:
        raise ValueError("volume_temperature must be non-zero")

    if log_scale:
        return torch.sum(
            torch.log(
                softplus(
                    box_tensor.right
                    - box_tensor.left
                    - 2 * euler_gamma * intersection_temperature,
                    beta=1 / volume_temperature,
                )
                + eps
            ),
            dim=-1,
        ) + float(
            np.log(scale)
        )

    return (
        torch.prod(
            softplus(
                box_tensor.right
                - box_tensor.left
                - 2 * euler_gamma * intersection_temperature,
                beta=1 / volume_temperature,
            ),
            dim=-1,
        )
        * scale
    )


class BesselApproxVolume(AbstractVolume):
    """
    Implementation of the volume of a box tensor using the Softplus 
    function as an approximation of the Bessel funtion
    (as described here: https://arxiv.org/abs/2010.04831).
    """

    def __init__(
        self,
        log_scale: bool = True,
        volume_temperature: float = 1.0,
        intersection_temperature: float = 1.0,
    ) -> None:
        """
        Args:
            log_scale: Where the output should be in log scale.
            volume_temperature: 1/volume_temperature is the beta parameter for the softplus
            intersection_temperature: the intersection_temperature parameter (same value used in intersection).
        """
        super().__init__(log_scale)
        self.volume_temperature = volume_temperature
        self.intersection_temperature = intersection_temperature

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Probabilistic softplus-based (instead of hard ReLU-based) volume.

        Args:
            box_tensor: the input boxes

        Returns:
            a tensor representing the volume of the input boxes
        """
        return bessel_volume_approx(
            box_tensor,
            volume_temperature=self.volume_temperature,
            intersection_temperature=self.intersection_temperature,
            log_scale=self.log_scale,
        )

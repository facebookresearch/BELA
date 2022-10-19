import torch

from duck.box_tensors.volume.abstract_volume import AbstractVolume
from duck.box_tensors.box_tensor import BoxTensor
from duck.common.utils import tiny_value_of_dtype
from torch.nn.functional import softplus
import numpy as np

eps = tiny_value_of_dtype(torch.float)


def soft_volume(
    box_tensor: BoxTensor,
    volume_temperature: float = 1.0,
    log_scale: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    """Volume of a box tensor based on softplus instead of ReLU/clamp

    Args:
        box_tensor: input box tensor
        volume_temperature: 1/volume_temperature is the beta parameter for the softplus
        log_scale: Whether the output should be in log scale or not.

    Returns:
        Tensor of shape (..., ) when the input box tensor has shape (..., 2, dim)
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    if volume_temperature == 0:
        raise ValueError("volume_temperature must be non-zero")

    if log_scale:
        return torch.sum(
            torch.log(
                softplus(
                    box_tensor.right - box_tensor.left, beta=1 / volume_temperature
                )
                + eps
            ),
            dim=-1,
        ) + float(
            np.log(scale)
        )  # need this eps so that the gradient of the log does not blow

    return (
        torch.prod(
            softplus(box_tensor.right - box_tensor.left, beta=1 / volume_temperature),
            dim=-1,
        )
        * scale
    )


class SoftVolume(AbstractVolume):
    """Softplus-based volume."""

    def __init__(
        self, log_scale: bool = True, volume_temperature: float = 1.0
    ) -> None:
        """
        Args:
            log_scale: Where the output should be in log scale.
            volume_temperature: 1/volume_temperature is the beta parameter for the softplus
        """
        super().__init__(log_scale)
        self.volume_temperature = volume_temperature

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: the input box tensor

        Returns:
            The volume of the input BoxTensor as a
            Tensor of shape (..., ) when the input box tensor has shape (..., 2, dim)
        """

        return soft_volume(
            box_tensor,
            volume_temperature=self.volume_temperature,
            log_scale=self.log_scale,
        )

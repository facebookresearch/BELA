import torch

from duck.common.utils import tiny_value_of_dtype
from duck.box_tensors.volume.abstract_volume import AbstractVolume
from duck.box_tensors import BoxTensor

eps = tiny_value_of_dtype(torch.float)


def hard_volume(box_tensor: BoxTensor, log_scale: bool = True) -> torch.Tensor:
    """
    Computes the hard volume of the input box tensor

    Args:
        box_tensor: the input box tensor
        log_scale: Whether the output should be in log scale or not.

    Returns:
        Tensor of shape (..., ) when box_tensor has shape (..., 2, dim)
    """

    if log_scale:
        return torch.sum(
            torch.log((box_tensor.right - box_tensor.left).clamp_min(eps)), dim=-1
        )

    return torch.prod((box_tensor.right - box_tensor.left).clamp_min(0), dim=-1)


class HardVolume(AbstractVolume):
    """Hard ReLU-based volume."""

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Computes the hard volume of the input box.

        Args:
            box_tensor: the input box

        Returns:
            The volume of the input BoxTensor as a
            Tensor of shape (..., ) when box_tensor has shape (..., 2, dim)

        """
        return hard_volume(box_tensor, self.log_scale)

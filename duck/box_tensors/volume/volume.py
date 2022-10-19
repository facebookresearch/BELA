import torch
from duck.box_tensors.volume.abstract_volume import AbstractVolume
from duck.box_tensors.volume.hard_volume import hard_volume
from duck.box_tensors.volume.bessel_volume import bessel_volume_approx
from duck.box_tensors.volume.soft_volume import soft_volume
from duck.box_tensors.box_tensor import BoxTensor


class Volume(AbstractVolume):
    """General implementation of the volume of a BoxTensor."""

    def __init__(
        self,
        log_scale: bool = True,
        volume_temperature: float = 0.0,
        intersection_temperature: float = 0.0
    ) -> None:
        """
        Args:
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.
            volume_temperature: if non-zero, uses softplus instead of ReLU/clamp
            intersection_temperature: if non-zero, uses softplus as approximation of Bessel function
        """
        super().__init__()  # type:ignore
        self.log_scale = log_scale
        self.volume_temperature = volume_temperature
        self.intersection_temperature = intersection_temperature

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Base implementation is hard (ReLU) volume.

        Args:
            box_tensor: Input box tensor

        Returns:
            The volume of the input BoxTensor as a
            Tensor of shape (..., ) when box_tensor has shape (..., 2, dim)
        """
        if self.volume_temperature == 0 and self.intersection_temperature == 0:
            return hard_volume(box_tensor, self.log_scale)
        elif self.intersection_temperature == 0:
            return soft_volume(
                box_tensor, self.volume_temperature, self.log_scale
            )
        else:
            return bessel_volume_approx(
                box_tensor,
                self.volume_temperature,
                self.intersection_temperature,
                self.log_scale,
            )

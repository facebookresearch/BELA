import torch
from duck.box_tensors.box_tensor import BoxTensor


class AbstractVolume(torch.nn.Module):
    """Base interface for the volume of a box"""

    def __init__(self, log_scale: bool = True) -> None:
        """
        Args:
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical use case.
        """
        super().__init__()
        self.log_scale = log_scale

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        raise NotImplementedError

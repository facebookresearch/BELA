from typing import List, Tuple, Union, Dict, Any, Optional
import torch

from duck.box_tensors import BoxTensor


class BoxRegularizer(torch.nn.Module):
    def __init__(
        self,
        weight: float,
        log_scale: bool = True,
        reduction: str = 'sum',
        **kwargs: Any,
    ) -> None:
        super().__init__() 
        self.weight = weight
        self.log_scale = log_scale
        self.reduction = reduction

    def forward(self, box_tensor: BoxTensor) -> Union[float, torch.Tensor]:
        regularization = self._forward(box_tensor)
        return self.weight * self._reduce(regularization)

    def _forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        raise NotImplementedError

    def _reduce(self, reg_unreduced: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return torch.sum(reg_unreduced)
        elif self.reduction == "mean":
            return torch.mean(reg_unreduced)
        else:
            raise ValueError(f"Unsupported reduction {self.reduction}")

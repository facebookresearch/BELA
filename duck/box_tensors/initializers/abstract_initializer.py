from typing import Any
import torch

class BoxInitializer(object):

    """A base class interface which will initialize a :class:`torch.Tensor` or :class:`torch.nn.Parameter`"""

    def __call__(self, t: torch.Tensor, **kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with attributes {self.__dict__}"

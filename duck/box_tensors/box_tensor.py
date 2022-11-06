from typing import (
    Tuple,
    Union,
    Any,
    Optional,
    Type,
    TypeVar
)
import logging
import torch
from torch import Tensor
from einops import rearrange, repeat


logger = logging.getLogger(__name__)

# see: https://realpython.com/python-type-checking/#type-hints-for-methods
# to know why we need to use TypeVar
TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


class BoxTensor(object):
    """Base class defining the interface for a Tensor representing a box."""

    def __init__(self, data: Union[Tensor, Tuple[Tensor, Tensor]]) -> None:
        """
        :param data: Tensor of shape (*, 2, dim) or tuple of tensors.
                The 0th dimension (resp. the 0th element of the tuple) is for the bottom left corner
                and 1st dimension (resp. the 1st element of the tuple) is for the top right corner of the box
        """
        self.data: Optional[Tensor] = None
        self._left: Optional[Tensor] = None
        self._right: Optional[Tensor] = None

        self.reinit(data)

        super().__init__()

    def reinit(self, data: Union[Tensor, Tuple[Tensor, Tensor]]) -> None:
        assert data is not None

        if self.data is not None:
            if isinstance(data, Tensor):
                if data.shape != self.data.shape:
                    raise ValueError(
                        f"Cannot reinitialize with different shape. New {data.shape}, old {self.data.shape}"
                    )

        if self._left is not None:
            if self._left.shape != data[0].shape:
                raise ValueError(
                    f"Cannot reinitialize with different shape. New left shape ={data[0].shape}, old ={self._left.shape}"
                )

        if self._right is not None:
            if self._right.shape != data[1].shape:
                raise ValueError(
                    f"Cannot reinitialize with different shape. New right shape ={data[1].shape}, old ={self._right.shape}"
                )

        if isinstance(data, Tensor):
            if _box_shape_ok(data):
                self.data = data
            else:
                raise ValueError(
                    _shape_error_str("data", "(...,2,num_dims)", data.shape)
                )
        else:
            self._left, self._right = data


    @property
    def left(self) -> Tensor:
        """
        Left coordinate as Tensor

        :return: Tensor: left corner
        """
        if self.data is not None:
            return self.data[..., 0, :]
        else:
            return self._left

    @property
    def right(self) -> Tensor:
        """Right coordinate as Tensor

        :return: Tensor: right corner
        """

        if self.data is not None:
            return self.data[..., 1, :]
        else:
            return self._right

    @property
    def center(self) -> Tensor:
        """
        Center coordinate as Tensor

        :return: Tensor: Center of the box
        """

        return (self.left + self.right) / 2
    
    @property
    def device(self):
        return self.left.device

    @classmethod
    def check_corner_validity(cls: Type[TBoxTensor], left: Tensor, right: Tensor) -> None:
        """
        Checks that the corners form a valid box and raises a ValueError if the corners are not valid.

        :param left: Lower left coordinate of shape (..., dim)
        :param right: Top right coordinate of shape (..., dim)
        """

        if not (right >= left).all().item():
            raise ValueError(f"Invalid box: right < left where\nright = {right}\nleft = {left}")

        if left.shape != right.shape:
            raise ValueError(
                "Shape of left and right should be same but is {} and {}".format(
                    left.shape, right.shape
                )
            )

    @property
    def box_shape(self) -> Tuple:
        """The shape of the box (Same as the shape of left, right and center).

        :return: the shape of the box (a tuple).
        """

        if self.data is not None:
            data_shape = list(self.data.shape)
            _ = data_shape.pop(-2)
            return tuple(data_shape)
        else:
            assert self._left.shape == self._right.shape

            return self._left.shape

    def rearrange(self, expression: str, **kwargs) -> TBoxTensor:
        left = rearrange(self.left, expression, **kwargs)
        right = rearrange(self.right, expression, **kwargs)
        return BoxTensor((left, right))
    
    def repeat(self, expression: str, **kwargs) -> TBoxTensor:
        left = repeat(self.left, expression, **kwargs)
        right = repeat(self.right, expression, **kwargs)
        return BoxTensor((left, right))

    @classmethod
    def from_corners(
            cls: Type[TBoxTensor], left: Tensor, right: Tensor
    ) -> TBoxTensor:
        """
        Creates a box for the given min-max coordinates (left, right).

        :param left: lower left
        :param right: top right
        :return: A BoxTensor

        """
        assert left.shape == right.shape, "left and right shape not matching"

        return cls((left, right))

    def __repr__(self) -> str:
        if self.data is not None:
            return (
                f"{self.__class__.__name__}(\n"
                f"{self.data.__repr__()}\n)" 
            )
        else:
            return (
                f"{self.__class__.__name__}(\n\tleft={self._left.__repr__()},"
                f"\n\tright={self._right.__repr__()}\n)"  # type:ignore
            )

    def __eq__(self, other: TBoxTensor) -> bool:  # type:ignore
        return torch.allclose(self.left, other.left) and torch.allclose(
            self.right, other.right
        )

    def __getitem__(self, indx: Any) -> "BoxTensor":
        """Creates a BoxTensor for the min-max coordinates at the given indexes

        Args:
            indx: Indexes of the required boxes

        Returns: a slice of the BoxTensor
        """
        z1 = self.left[indx]
        z2 = self.right[indx]

        return self.from_corners(z1, z2)
    
    def cat(self, other: TBoxTensor, dim=0):
        left = torch.cat([self.left, other.left], dim=dim)
        right = torch.cat([self.right, other.right], dim=dim)
        return BoxTensor((left, right))

    def broadcast(self, target_shape: Tuple) -> None:
        """
        Broadcasts the internal data member in-place such that left and right
        are tensors that can be automatically broadcasted to perform
        arithmetic operations with shape `target_shape`.

        Args:
            target_shape: Shape of the broadcast target
        """
        self_box_shape = self.box_shape

        if self_box_shape[-1] != target_shape[-1]:
            raise ValueError(
                f"Cannot broadcast box of box_shape {self_box_shape} to {target_shape}."
                "Last dimensions should match."
            )

        if len(self_box_shape) > len(target_shape):
            # see if we have 1s in the right places in the self.box_shape
            raise ValueError(
                f"Lenght of self.box_shape ({len(self_box_shape)})"
                f" should be <= length of target_shape ({len(self_box_shape)})"
            )

        elif len(self_box_shape) == len(target_shape):
            # they can be of the form (4,1,10) & (1,4,10)

            for s_d, t_d in zip(self_box_shape[:-1], target_shape[:-1]):

                if not ((s_d == t_d) or (s_d == 1) or (t_d == 1)):
                    raise ValueError(
                        f"Incompatible shapes {self_box_shape} and target {target_shape}"
                    )
        else:  # <= target_shape
            potential_final_shape = list(self_box_shape)
            dim_to_unsqueeze = []

            for dim in range(-2, -len(target_shape) - 1, -1):  # (-2, -3, ...)
                if (
                    dim + len(potential_final_shape) < 0
                ):  # self has more dims left
                    potential_final_shape.insert(dim + 1, 1)
                    # +1 because
                    # insert indexing in list
                    # works differently than unsqueeze
                    dim_to_unsqueeze.append(dim)

                    continue

                if potential_final_shape[dim] != target_shape[dim]:
                    potential_final_shape.insert(dim + 1, 1)
                    dim_to_unsqueeze.append(dim)

            # final check
            assert len(potential_final_shape) == len(target_shape)

            for p_d, t_d in zip(potential_final_shape, target_shape):
                if not (p_d in (1, t_d)):
                    raise ValueError(
                        f"Cannot make box_shape {self_box_shape} compatible to {target_shape}"
                    )

            if self.data is not None:
                for d in dim_to_unsqueeze:
                    self.data.unsqueeze_(
                        d - 1
                    )  # -1 because of extra 2 at dim -2
            else:
                for d in dim_to_unsqueeze:
                    self.left.unsqueeze_(d)  
                    self.right.unsqueeze_(d)
            assert self.box_shape == tuple(potential_final_shape)


def _box_shape_ok(t: Tensor) -> bool:
    if len(t.shape) < 2:
        return False
    else:
        if t.size(-2) != 2:
            return False

        return True


def _shape_error_str(
    tensor_name: str, expected_shape: Any, actual_shape: Tuple
) -> str:
    return "Shape of {} has to be {} but is {}".format(
        tensor_name, expected_shape, tuple(actual_shape)
    )

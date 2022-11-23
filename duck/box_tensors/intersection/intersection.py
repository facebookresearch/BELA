from typing import Optional

from duck.box_tensors.intersection.abstract_intersection import AbstractIntersection
from duck.box_tensors.intersection.gumbel_intersection import gumbel_intersection
from duck.box_tensors.intersection.hard_intersection import hard_intersection
from duck.box_tensors import BoxTensor


class Intersection(AbstractIntersection):
    """General implementation of the intersection operation for probabilistic box tensors"""

    def __init__(
        self,
        dim: int = 0,
        intersection_temperature: float = 0.0,
        approximation_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            dim: dimension along which the intersection is performed
            intersection_temperature: Gumbel's beta parameter: if non-zero performs 
                the Gumbel intersection, otherwise returns the hard intersection
            approximation_mode: Only for gumbel intersection:
                Either hard clipping ('clipping') or hard clipping with
                separate value for forward and backward passes  ('clipping_forward')
                or no clipping ('None')
        """
        super().__init__(dim=dim)
        self.intersection_temperature = intersection_temperature
        self.approximation_mode = approximation_mode

    def _forward(self, box: BoxTensor) -> BoxTensor:
        """Performs the intersection of t1 and t2.

        Args:
           box: BoxTensor containing the boxes to intersect

        Returns:
            BoxTensor obtained by the intersection of box along dimension dim
        """
        if self.intersection_temperature == 0:
            return hard_intersection(box, dim=self.dim)
        else:
            return gumbel_intersection(
                box,
                dim=self.dim,
                intersection_temperature=self.intersection_temperature,
                approximation_mode=self.approximation_mode,
            )

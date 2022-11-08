from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from duck.box_tensors.box_tensor import BoxTensor

from duck.box_tensors.initializers.abstract_initializer import BoxInitializer
from duck.box_tensors.initializers.uniform import UniformBoxInitializer
from einops import rearrange


class BoxEmbedding(torch.nn.Embedding):
    """Embedding which returns boxes instead of vectors"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        box_initializer: BoxInitializer = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim * 2,
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.box_initializer = box_initializer
        self.reinit()
    
    def reinit(self):
        if self.box_initializer is None:
            self.box_initializer = UniformBoxInitializer(
                dimensions=self.embedding_dim,
                num_boxes=int(self.weight.shape[0])
            )

        self.box_initializer(self.weight)

    def forward(self, inputs: torch.Tensor) -> BoxTensor:
        emb = super().forward(inputs)
        emb = rearrange(emb, "... (box d) -> ... box d", box=2)
        left = emb[..., 0, :]
        right = emb[..., 1, :]
        return BoxTensor((left, right))

    def all_boxes(self) -> BoxTensor:
        weights = rearrange(self.weight, "... (box d) -> ... box d", box=2)
        left = weights[..., 0, :]
        right = weights[..., 1, :]
        return BoxTensor((left, right)) 

    def get_bounding_box(self) -> BoxTensor:
        all_ = self.all_boxes()
        left = all_.left 
        right = all_.right
        left_min, _ = left.min(dim=0)
        right_max, _ = right.max(dim=0)

        return BoxTensor.from_corners(left_min, right_max)

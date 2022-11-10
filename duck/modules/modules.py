from typing import Optional, Any
import torch
from torch import nn, Tensor
from duck.box_tensors.box_tensor import BoxTensor

from duck.box_tensors.initializers.abstract_initializer import BoxInitializer
from duck.box_tensors.initializers.uniform import UniformBoxInitializer
from einops import rearrange

from duck.box_tensors import BoxTensor
from bela.models.hf_encoder import HFEncoder
from einops import rearrange

from duck.common.utils import activation_function


class BoxEmbedding(torch.nn.Embedding):
    """Embedding layer returning boxes instead of vectors"""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        box_initializer: BoxInitializer = None,
        universe_idx: Optional[int] = None,
        universe_min: float = 0.0,
        universe_max: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim * 2,
            padding_idx=universe_idx,
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.box_initializer = box_initializer
        self.universe_idx = universe_idx
        self.universe_min = universe_min
        self.universe_max = universe_max
        self.reinit()
    
    def reinit(self):
        if self.box_initializer is None:
            self.box_initializer = UniformBoxInitializer(
                dimensions=self.embedding_dim,
                num_boxes=int(self.weight.shape[0]),
                minimum = self.universe_min,
                maximum = self.universe_max
            )

        self.box_initializer(self.weight)
        self._fill_universe_idx()

    def _fill_universe_idx(self):
        if self.universe_idx is not None:
            with torch.no_grad():
                universe_left = torch.full((self.embedding_dim,), self.universe_min)
                universe_right = torch.full((self.embedding_dim,), self.universe_max)
                universe_data = torch.cat([universe_left, universe_right])
                self.weight[self.universe_idx].copy_(universe_data)

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



class EntityEncoder(nn.Module):
    """
    Module for learning a non-linear transformation of pre-trained entity embeddings
    (so that entities can be moved inside the box representing their type).
    
    This implementation relies on a 2-layer Multi-layer perceptron (MLP).
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1
    ):
        """
        Intializes the entity encoder.
        
        Args:
            dim: the dimensionality of the input and output tensor
            hidden_dim: the dimensionality of the hidden layer of the MLP
            dropout: the dropout rate. The default value is 0.1
        """
        super(EntityEncoder, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 2 * dim
        
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.dim)
        )
    
    def forward(
        self,
        entity: Tensor
    ) -> Tensor:
        """
        Args:
            entity: the representation of an entity as a Tensor of size (..., dim)
            
        Returns:
            A Tensor of the same size as the input entity representation
        """
        return self.ffn(entity)
    

class SetToBoxTransformer(nn.Module):
    """
    Module that encodes sets of pre-trained embeddings as boxes.
    This implementation relies on a Transformer encoder (without positional encodings)
    and a permutation invariant aggregation function (the mean of the encodings of the elements in the set).
    The set of embeddings is mapped to a Tensor representing the corners of the box which is used to
    instatiate a BoxTensor that encodes the whole set.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        attn_heads: int = 8,
        dropout: float = 0.1,
        margin: int = 0.5,
        activation: str = "softmax"
    ):
        """
        Initializes the encoder.
        
        Args:
            dim: the dimensionality of the input and output of the module
            hidden_dim: the dimensionality of the hidden layer of the feed-forward network in the transformer encoder.
                The default value is 2 * dim
            num_layers: the number of transformer encoder layers
            attn_heads: the number of attention heads
            dropout: the dropout rate
        """
        super(SetToBoxTransformer, self).__init__()
        self.dim = dim
        hidden_dim = hidden_dim or 2 * dim
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            dim, attn_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        norm = nn.LayerNorm(dim)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers, norm=norm)
        self.left_proj = nn.Linear(dim, dim)
        self.offset_proj = nn.Linear(dim, dim)
        self.margin = margin
        self.activation_name = activation
        self.activation = activation_function(activation)
    
    def forward(
        self,
        input_set: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> BoxTensor:
        """
        Args:
            input_set: set of embeddings as a Tensor of size (batch_size, l, dim),
            where l is the number of embeddings in the set, dim is the dimensionality of the model
            and batch_size is the batch_size
        
        Returns:
            BoxTensor of size (batch_size, dim)
        """
        attention_mask = attention_mask.bool()
        if attention_mask.dim() == 3:
            attention_mask = torch.any(attention_mask.bool(), dim=-1)
        
        pad = torch.zeros([input_set.size(0), 1, input_set.size(-1)], device=input_set.device)
        pad_mask = torch.full([attention_mask.size(0), 1], True, device=input_set.device).bool()
        input_set = torch.cat([pad, input_set], dim=1)
        attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
        x = self.transformer_encoder(input_set, src_key_padding_mask=~attention_mask)
        x[~attention_mask] = 0.
        x = torch.mean(x, dim=1)
        left = self.left_proj(x)
        offset = self.offset_proj(x)
        if self.activation_name == "softmax":
            offset = torch.softmax(torch.stack([left, offset]), dim=0)[-1]
        else:
            offset = self.activation(offset)
        right = left + offset + self.margin
        return BoxTensor((left, right))


class HFSetToBoxTransformer(nn.Module):
    def __init__(
        self,
        hf_transformer: HFEncoder,
        batched=True 
    ):
        super().__init__()
        self.hf_transformer = hf_transformer
        self.dim = hf_transformer.transformer.config.hidden_size
        self.out_proj = nn.Linear(self.dim, 2 * self.dim)
        self.batched = batched
        self.min_margin = 1

    def forward(
        self,
        input_ids, 
        attention_mask
    ):
        batch_size = input_ids.size(0)
        x = input_ids

        if self.batched:
            x = rearrange(input_ids, "b l n -> (b l) n")
            attention_mask = rearrange(attention_mask, "b l n -> (b l) n")
            x, _ = self.hf_transformer(x, attention_mask=attention_mask)
        else:
            length = x.size(1)
            encodings = [
                self.hf_transformer(x[:, i, :], attention_mask[:, i, :])[0].detach()
                for i in range(length)
            ]
            x = rearrange(encodings, "l b d -> (b l) d")
            attention_mask = rearrange(attention_mask, "b l n -> (b l) n")
        x[torch.all(~(attention_mask.bool()), dim=-1)] = 0.
        x = rearrange(x, "(b l) d -> b l d", b=batch_size)
        x = torch.mean(x, dim=1)
        x = self.out_proj(x)
        x = x.view(-1, 2, self.dim)
        left = x[:, 0, :]
        offset = torch.sigmoid(x[:, 1, :]) + self.min_margin
        right = left + offset
        return BoxTensor((left, right))

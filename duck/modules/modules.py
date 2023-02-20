from typing import Callable, Optional, Any
import torch
from torch import nn, Tensor
from duck.box_tensors.box_tensor import BoxTensor

from duck.box_tensors.initializers.abstract_initializer import BoxInitializer
from duck.box_tensors.initializers import GaussianMarginBoxInitializer
from duck.box_tensors.initializers.uniform import UniformBoxInitializer
from einops import rearrange

from duck.box_tensors import BoxTensor
from bela.models.hf_encoder import HFEncoder
from einops import rearrange
import math

from duck.common.utils import activation_function


class BoxEmbedding(torch.nn.Embedding):
    """Embedding layer returning boxes instead of vectors"""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        box_parametrizaton: str = "uniform",
        universe_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim * 2,
            padding_idx=universe_idx,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.parametrization = box_parametrizaton
        self.universe_idx = universe_idx
        self.universe_min = 0.0
        self.universe_max =  1.0
        self.box_constructor = self._set_box_constructor()
        self.box_initializer = self._set_box_initializer()
        if kwargs.get("_weight") is None:
            self.reinit()
    
    def _set_box_constructor(self):
        if self.parametrization == "uniform":
            return None
        if self.parametrization == "gaussian_margin":
            return None
        if self.parametrization == "sigmoid":
            return BoxTensor.sigmoid_constructor
        if self.parametrization == "softplus":
            return BoxTensor.softplus_constructor
        raise ValueError(f"Unsupported parametrization {self.parametrization}")

    def _set_box_initializer(self):
        if self.parametrization == "uniform":
            return UniformBoxInitializer(
                dimensions=self.embedding_dim,
                num_boxes=int(self.weight.shape[0]),
                minimum = self.universe_min,
                maximum = self.universe_max
            )
        if self.parametrization == "gaussian_margin":
            return GaussianMarginBoxInitializer(
                dimensions=self.embedding_dim,
                num_boxes=int(self.weight.shape[0]),
                minimum = self.universe_min,
                maximum = self.universe_max,
                stddev=0.01
            )
        if self.parametrization == "sigmoid":
            self.universe_min = +100
            self.universe_max = +self.universe_min
            return None
        if self.parametrization == "softplus":
            return
        raise ValueError(f"Unsupported parametrization {self.parametrization}")
        
    def reinit(self):
        with torch.no_grad():
            self.box_constructor = self._set_box_constructor()
            self.box_initializer = self._set_box_initializer()
            self.reset_parameters()
            if self.box_initializer is not None:
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
        if self.box_constructor is not None:
            return self.box_constructor(left, right)
        return BoxTensor((left, right))

    def all_boxes(self) -> BoxTensor:
        weights = rearrange(self.weight, "... (box d) -> ... box d", box=2)
        left = weights[..., 0, :]
        right = weights[..., 1, :]
        if self.box_constructor is not None:
            return self.box_constructor(left, right)
        return BoxTensor((left, right))

    def get_bounding_box(self) -> BoxTensor:
        all_ = self.all_boxes()
        left = all_.left 
        right = all_.right
        left_min, _ = left.min(dim=0)
        right_max, _ = right.max(dim=0)
        return BoxTensor.from_corners(left_min, right_max)
    
    def set_universe_to_bounding_box(self) -> None:
        if self.box_constructor is not None:
            return
        with torch.no_grad():
            weights = rearrange(self.weight, "... (box d) -> ... box d", box=2)
            left = weights[..., 0, :]
            right = weights[..., 1, :]
            left_min, _ = left.min(dim=0)
            right_max, _ = right.max(dim=0)
            universe_data = torch.cat([left_min, right_max])
            self.weight[self.universe_idx].copy_(universe_data)
    
    def to_origin(self) -> None:
        with torch.no_grad():
            centroid = self.centroid.detach()
            all_ = self.all_boxes()
            left = all_.left - centroid
            right = all_.right - centroid
            weights = torch.cat([left, right], dim=-1)
            freeze = not self.weight.requires_grad
            return self.from_pretrained(weights, freeze=freeze, universe_idx=self.universe_idx)

    @property
    def centroid(self):
        all_boxes = self.all_boxes()
        return all_boxes.center.mean()

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze: bool = True,
        box_parametrization: str = "uniform",
        universe_idx: Optional[int] = None,
        **kwargs: Any
    ):
        
        assert embeddings.dim() == 2, \
            "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        assert cols % 2 == 0, \
            "The embedding size is expected to be even for all box parametrizations"

        embedding = cls(
            num_embeddings=rows,
            embedding_dim=int(cols / 2),
            box_parametrizaton=box_parametrization,
            universe_idx=universe_idx,
            _weight=embeddings,
            **kwargs
        )
        with torch.no_grad():
            embedding.weight.copy_(embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding
    
    def freeze(self):
        self.weight.requires_grad = False
        return self


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
    

class TransformerSetEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        attn_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(TransformerSetEncoder, self).__init__()
        self.dim = dim
        hidden_dim = hidden_dim or 2 * dim
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            dim, attn_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        norm = nn.LayerNorm(dim)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers, norm=norm)
       
    def forward(
        self,
        input_set: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            input_set: set of embeddings as a Tensor of size (batch_size, l, dim),
            where l is the number of embeddings in the set, dim is the dimensionality of the model
            and batch_size is the batch_size
        
        Returns:
            Tensor of size (batch_size, dim)
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
        return torch.mean(x, dim=1)


class JointEntRelsEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        attn_heads: int = 8,
        dropout: float = 0.1,
        enable_nested_tensor: bool = False
    ):
        super(JointEntRelsEncoder, self).__init__()
        self.dim = dim
        hidden_dim = hidden_dim or 2 * dim
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            dim, attn_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        norm = nn.LayerNorm(dim)
        self.enable_nested_tensor = enable_nested_tensor
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers,
            norm=norm,
            ## remove for compatibility with previous versions of pytorch
            # enable_nested_tensor=self.enable_nested_tensor
        )
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            dim, attn_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers,
            norm=nn.LayerNorm(dim)
        )

    def forward(self, entities, rels, attention_mask):
        """
        entities: (bsz, dim)
        rels: (bsz, n, dim)
        """
        attention_mask = attention_mask.bool()
        if attention_mask.dim() == 3:
            attention_mask = torch.any(attention_mask.bool(), dim=-1)
        
        pad = torch.zeros([rels.size(0), 1, rels.size(-1)], device=rels.device)
        pad_mask = torch.full([attention_mask.size(0), 1], True, device=rels.device).bool()
        rels = torch.cat([pad, rels], dim=1)
        attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
        rels = self.transformer_encoder(rels, src_key_padding_mask=~attention_mask)
        if not self.enable_nested_tensor:
            rels[~attention_mask] = 0.
            result = self.transformer_decoder(entities.unsqueeze(1), rels, memory_key_padding_mask=~attention_mask)
        else:
            result = self.transformer_decoder(entities.unsqueeze(1), rels)
        return result.squeeze(1)

    
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


class EmbeddingToBox(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        box_parametrization: str = "softplus",
        padding_idx: Optional[int] = None,
        output_size: Optional[int] = None
    ):
        super(EmbeddingToBox, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        self.dim = embeddings.size(-1)
        self.output_size = output_size or self.dim
        self.hidden_dim = 2 * self.output_size
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.parametrization = box_parametrization
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.ffn(x)
        x = rearrange(x, "... (corners d) -> ... corners d", corners=2)
        v1 = x[..., 0, :]
        v2 = x[..., 1, :]
        return BoxTensor.construct(
            self.parametrization,
            v1, v2
        )

import torch
from torch import nn
from torch import Tensor

from duck.box_tensors import BoxTensor


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
    

class DuckTypeEncoder(nn.Module):
    """
    Module that encodes sets of pre-trained relation embeddings as boxes.
    This implementation relies on a Transformer encoder (without positional encodings)
    and a permutation invariant aggregation function (the mean of the encodings of the elements in the set).
    The set of relation embeddings is mapped to a Tensor representing the corners of the box which is used to
    instatiate a BoxTensor that encodes the whole set.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        attn_heads: int = 8,
        dropout: float = 0.1
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
        super(DuckTypeEncoder, self).__init__()
        self.dim = dim
        hidden_dim = hidden_dim or 2 * dim
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            dim, attn_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        norm = nn.LayerNorm(dim)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers, norm=norm)
        self.out_proj = nn.Linear(dim, 2 * dim)
    
    def forward(
        self,
        relation_set: Tensor
    ) -> BoxTensor:
        """
        Args:
            relation_set: set of relations as a Tensor of size (batch_size, l, dim),
            where l is the number of relations in the set, dim is the dimensionality of the model
            and batch_size is the batch_size
        
        Returns:
            BoxTensor of size (batch_size, dim)
        """
        x = self.transformer_encoder(relation_set)
        x = torch.mean(x, dim=1)
        x = self.out_proj(x)
        x = x.view(-1, 2, self.dim)
        return BoxTensor(x)

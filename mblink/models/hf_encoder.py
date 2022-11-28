#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

from transformers import AutoModel
from torch import nn


class HFEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "xlm-roberta-base",
        projection_dim: Optional[int] = None,
        output_dropout: Optional[float] = 0.0, 
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        self.embedding_dim = self.transformer.encoder.config.hidden_size
        self.project = nn.Identity()  # to make torchscript happy
        if projection_dim:
            self.project = nn.Sequential(
                nn.Linear(self.embedding_dim, projection_dim), nn.LayerNorm(projection_dim)
            )
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, input_ids, attention_mask=None):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_layer = output["last_hidden_state"]
        sentence_rep = self.project(last_layer[:, 0, :])
        return self.output_dropout(sentence_rep), last_layer

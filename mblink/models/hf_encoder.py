#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn
from transformers import AutoModel


class HFEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "xlm-roberta-base",
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask=None):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_layer = output["last_hidden_state"]
        sentence_rep = last_layer[:, 0, :]
        return sentence_rep, last_layer

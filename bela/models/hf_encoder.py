#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn

from transformers import AutoModel, AutoConfig


class HFEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "xlm-roberta-base",
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)

    def forward(self, tokens):
        output = self.transformer(**tokens, output_hidden_states=True)
        last_layer = output["last_hidden_state"]
        hidden_states = output["hidden_states"]
        sentence_rep = last_layer[:, 0, :]
        return sentence_rep, hidden_states

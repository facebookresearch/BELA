#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from transformers import AutoTokenizer


class HFTransform(nn.Module):
    def __init__(
        self,
        model_path: str = "bert-large-cased",
        max_seq_len: int = 512,
        add_special_tokens: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sep_token = self.tokenizer.sep_token
        self.max_seq_len = max_seq_len
        self.add_special_tokens = add_special_tokens

    def forward(self, texts):
        return self.tokenizer(
            texts,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=self.add_special_tokens,
        )["input_ids"]

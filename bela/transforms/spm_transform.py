#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import os
import torch.nn as nn
import sentencepiece as spm
from .sentencepiece_pb2 import SentencePieceText


class SPMTransform(nn.Module):
    def __init__(
        self,
        sp_model_path: Optional[str] = None,
        max_seq_len: int = 256,
        add_special_tokens: bool = True,
    ):
        super().__init__()
        sp_model_path = sp_model_path or os.path.join(os.path.dirname(__file__), "../data/sp_model")
        self.processor = spm.SentencePieceProcessor(sp_model_path)
        self.sep_token = '</s>'
        self.unk_token_id = 3
        self.max_seq_len = max_seq_len
        self.add_special_tokens = add_special_tokens

    def forward(self, texts):
        output = []
        for text in texts:
            spt = SentencePieceText()
            spt.ParseFromString(self.processor.encode_as_serialized_proto(text))
            current_offset = 0
            leading_whitespaces_count = 0
            for char in text:
                if char.isspace():
                    leading_whitespaces_count += 1
                else:
                    break

            token_ids_with_offsets = []
            if self.add_special_tokens:
                token_ids_with_offsets.append((0,0,0))
            for idx, piece in enumerate(spt.pieces):
                if piece.id != 0:
                    token_id = piece.id + 1
                else:
                    token_id = self.unk_token_id
                if idx == 0:
                    # if we process first token, append leading whitespacess count to the sp token length
                    token_ids_with_offsets.append((token_id, current_offset, current_offset + len(piece.surface) + leading_whitespaces_count))
                    current_offset += len(piece.surface) + leading_whitespaces_count
                else:
                    token_ids_with_offsets.append((token_id, current_offset, current_offset + len(piece.surface)))
                    current_offset += len(piece.surface)

                # take into account special tokens
                if idx == self.max_seq_len - 3:
                    break

            if self.add_special_tokens:
                token_ids_with_offsets.append((2,current_offset,0))

            output.append(token_ids_with_offsets)
        return output

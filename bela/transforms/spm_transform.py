#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple, Optional
import os
import torch.nn as nn
import sentencepiece as spm
from bela.transforms.sentencepiece_pb2 import SentencePieceText


def convert_text_to_spm_tokens_with_char_offsets(text: str, spm_processor: spm.SentencePieceProcessor) -> List[Tuple[int, int, int]]:
    spt = SentencePieceText()
    spt.ParseFromString(spm_processor.encode_as_serialized_proto(text))
    # Set offset to number of starting whitespaces (spm removes them)
    start_offset = len(text) - len(text.lstrip())
    token_ids_with_offsets = []
    for piece in spt.pieces:
        # NOTE: It seems we can't use piece.begin and piece.end because it sometimes break, e.g. with
        # `text = "   Martina Steuk (née Kämpfert; born 11 November 1959) is a German former track and field athlete "
        # At some point piece.begin will not correspond to the actual offset in the text
        end_offset = start_offset + len(piece.surface)
        assert text[start_offset:end_offset] == piece.surface, f"{text[start_offset:end_offset]} != {piece.surface}"
        token_ids_with_offsets.append((piece.id, start_offset, end_offset))
        start_offset = end_offset
    return token_ids_with_offsets


def convert_sp_to_char_offsets(
        text: str,
        sp_offset: int,
        sp_length: int,
        spm_processor,
    ) -> Tuple[int, int]:
    """Inefficient but simple way to convert sp offsets to char offsets."""
    token_ids_with_offsets = convert_text_to_spm_tokens_with_char_offsets(text, spm_processor)
    char_offset = token_ids_with_offsets[sp_offset][1]
    char_length = token_ids_with_offsets[sp_offset + sp_length - 1][2] - char_offset
    return char_offset, char_length


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
            token_ids_with_offsets = convert_text_to_spm_tokens_with_char_offsets(text, self.processor)
            # Add 1 to all token ids except for the unk token
            token_ids_with_offsets = [
                (token_id + 1 if token_id != 0 else self.unk_token_id, start_offset, end_offset)
                for token_id, start_offset, end_offset in token_ids_with_offsets
            ]
            # Limit the number of tokens to max_seq_len (accounting for special tokens)
            token_ids_with_offsets = token_ids_with_offsets[:self.max_seq_len - 2]
            if self.add_special_tokens:
                last_offset = token_ids_with_offsets[-1][2]
                token_ids_with_offsets = [(0, 0, 0)] + token_ids_with_offsets + [(2, last_offset, 0)]
            assert len(token_ids_with_offsets) <= self.max_seq_len
            output.append(token_ids_with_offsets)
        return output

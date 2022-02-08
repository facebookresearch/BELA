#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from bela.transforms.hf_transform import HFTransform


def pieces_to_texts(
    texts_pieces_token_ids: List[List[int]],
    texts: List[List[str]],
    texts_mention_offsets: List[List[int]],
    texts_mention_lengths: List[List[int]],
    # bos_idx: int,
    # eos_idx: int,
    max_seq_len: int = 512,
):
    """
    Function takes an array with SP tokenized word tokens and original texts
    and convert youda tokenized batch to SP tokenized batch. Mention offsets
    and lengths are also converted with respect to SP tokens.

    Inputs:
        1) texts_pieces_token_ids: List with sp tokens per text token
        2) texts: original yoda tokenized texts
        3) texts_mention_offsets: mention offsets in original texts
        4) texts_mention_lengths: mention lengths in original texts
        5) bos_idx: tokenizer bos index
        6) eos_idx: tokenizer eos index
        7) max_seq_len: tokenizer max sequence length

    Outputs:
        new_texts_token_ids: List[List[int]] - text batch with sp tokens
        new_seq_lengths: List[int] - sp tokenized texts lengths
        new_mention_offsets: List[List[int]] - converted mention offsets
        new_mention_lengths: List[List[int]] - converted mention lengths
    """
    new_texts_token_ids: List[List[int]] = []
    new_seq_lengths: List[int] = []
    new_mention_offsets: List[List[int]] = []
    new_mention_lengths: List[List[int]] = []
    tokens_mapping: List[List[Tuple[int, int]]] = []  # bs x idx x 2

    pieces_offset = 0
    for text, old_mention_offsets, old_mention_lengths in zip(
        texts,
        texts_mention_offsets,
        texts_mention_lengths,
    ):
        mapping: List[Tuple[int, int]] = []
        text_token_ids: List[int] = []
        mention_offsets: List[int] = []
        mention_lengths: List[int] = []

        for token_ids in texts_pieces_token_ids[
            pieces_offset : pieces_offset + len(text)
        ]:
            token_ids = token_ids[1:-1]
            current_pos = len(text_token_ids)
            mapping.append((current_pos, current_pos + len(token_ids)))
            text_token_ids.extend(token_ids)

        text_token_ids = text_token_ids[: max_seq_len - 1]
        # text_token_ids.append(eos_idx)
        for old_offset, old_length in zip(old_mention_offsets, old_mention_lengths):
            new_offset = mapping[old_offset][0]
            new_end = mapping[old_offset + old_length - 1][1]
            new_length = new_end - new_offset

            if new_end > max_seq_len - 1:
                break

            mention_offsets.append(new_offset)
            mention_lengths.append(new_length)

        new_texts_token_ids.append(text_token_ids)
        new_seq_lengths.append(len(text_token_ids))
        new_mention_offsets.append(mention_offsets)
        new_mention_lengths.append(mention_lengths)
        mapping = [(start, end) for start, end in mapping if end < max_seq_len]
        tokens_mapping.append(mapping)

        pieces_offset += len(text)

    return (
        new_texts_token_ids,
        new_seq_lengths,
        new_mention_offsets,
        new_mention_lengths,
        tokens_mapping,
    )


@torch.jit.script
def pad_tokens_mapping(
    tokens_mapping: List[List[Tuple[int, int]]]
) -> List[List[Tuple[int, int]]]:
    seq_lens: List[int] = []
    for seq in tokens_mapping:
        seq_lens.append(len(seq))
    pad_to_length = max(seq_lens)

    for mapping in tokens_mapping:
        padding = pad_to_length - len(mapping)
        if padding >= 0:
            for _ in range(padding):
                mapping.append((0, 1))
        else:
            for _ in range(-padding):
                mapping.pop()
    return tokens_mapping


@torch.jit.script
def pad_2d(
    batch: List[List[int]], seq_lens: List[int], pad_idx: int, max_len: int = -1
) -> List[List[int]]:
    pad_to_length = max(seq_lens)
    if max_len > 0:
        pad_to_length = min(pad_to_length, max_len)
    for sentence in batch:
        padding = pad_to_length - len(sentence)
        if padding >= 0:
            for _ in range(padding):
                sentence.append(pad_idx)
        else:
            for _ in range(-padding):
                sentence.pop()
    return batch


class JointELCollate(nn.Module):
    def __init__(
        self,
        pad_idx: int = 1,
        token_ids_column: str = "input_ids",
        seq_lens_column: str = "seq_lens",
        pad_mask_column: str = "attention_mask",
        mention_pad_idx: int = 0,
        mention_offsets_column: str = "mention_offsets",
        mention_lengths_column: str = "mention_lengths",
        mentions_seq_lengths_column: str = "mentions_seq_lengths",
        entities_column: str = "entities",
        tokens_mapping_column: str = "tokens_mapping",
    ):
        super().__init__()
        self._pad_idx = pad_idx
        self.token_ids_column = token_ids_column
        self.seq_lens_column = seq_lens_column
        self.pad_mask_column = pad_mask_column
        self._mention_pad_idx = mention_pad_idx
        self.mention_offsets_column = mention_offsets_column
        self.mention_lengths_column = mention_lengths_column
        self.entities_column = entities_column
        self.mentions_seq_lengths_column = mentions_seq_lengths_column
        self.tokens_mapping_column = tokens_mapping_column

    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        token_ids = batch[self.token_ids_column]
        assert torch.jit.isinstance(token_ids, List[List[int]])
        seq_lens = batch[self.seq_lens_column]
        assert torch.jit.isinstance(seq_lens, List[int])

        pad_token_ids = pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in token_ids],
            batch_first=True,
            padding_value=float(self._pad_idx),
        )
        pad_mask = torch.ne(pad_token_ids, self._pad_idx).to(dtype=torch.long)
        text_model_inputs: Dict[str, torch.Tensor] = {
            self.token_ids_column: pad_token_ids,
            self.pad_mask_column: pad_mask,
        }

        mention_offsets = batch[self.mention_offsets_column]
        torch.jit.isinstance(mention_offsets, List[List[int]])
        mention_lengths = batch[self.mention_lengths_column]
        torch.jit.isinstance(mention_lengths, List[List[int]])
        mentions_seq_lengths = batch[self.mentions_seq_lengths_column]
        torch.jit.isinstance(mentions_seq_lengths, List[int])
        entities = batch[self.entities_column]
        torch.jit.isinstance(entities, List[List[int]])
        tokens_mapping = batch[self.tokens_mapping_column]
        torch.jit.isinstance(tokens_mapping, List[List[Tuple[int, int]]])

        mentions_model_inputs = {}

        mentions_model_inputs[self.mention_offsets_column] = torch.tensor(
            pad_2d(
                mention_offsets,
                seq_lens=mentions_seq_lengths,
                pad_idx=self._mention_pad_idx,
            ),
            dtype=torch.long,
        )
        mentions_model_inputs[self.mention_lengths_column] = torch.tensor(
            pad_2d(
                mention_lengths,
                seq_lens=mentions_seq_lengths,
                pad_idx=self._mention_pad_idx,
            ),
            dtype=torch.long,
        )
        mentions_model_inputs[self.entities_column] = torch.tensor(
            pad_2d(
                entities,
                seq_lens=mentions_seq_lengths,
                pad_idx=self._mention_pad_idx,
            ),
            dtype=torch.long,
        )
        mentions_model_inputs[self.tokens_mapping_column] = torch.tensor(
            pad_tokens_mapping(tokens_mapping),
            dtype=torch.long,
        )
        return text_model_inputs, mentions_model_inputs


class JointELTransform(HFTransform):
    def __init__(
        self,
        model_path: str = "bert-large-cased",
        max_seq_len: int = 512,
        texts_column: str = "texts",
        mention_offsets_column: str = "mention_offsets",
        mention_lengths_column: str = "mention_lengths",
        mentions_seq_lengths_column: str = "mentions_seq_lengths",
        entities_column: str = "entities",
        token_ids_column: str = "input_ids",
        seq_lens_column: str = "seq_lens",
        pad_mask_column: str = "attention_mask",
        tokens_mapping_column: str = "tokens_mapping",
    ):
        super().__init__(model_path=model_path)
        # self.bos_idx = self.tokenizer.bos_token_id
        # self.eos_idx = self.tokenizer.eos_token_id
        self.max_seq_len = max_seq_len

        self.texts_column = texts_column
        self.token_ids_column = token_ids_column
        self.seq_lens_column = seq_lens_column
        self.pad_mask_column = pad_mask_column
        self.mention_offsets_column = mention_offsets_column
        self.mention_lengths_column = mention_lengths_column
        self.mentions_seq_lengths_column = mentions_seq_lengths_column
        self.entities_column = entities_column
        self.tokens_mapping_column = tokens_mapping_column

        self._collate = JointELCollate(
            pad_idx=self.tokenizer.pad_token_id,
            token_ids_column=token_ids_column,
            seq_lens_column=seq_lens_column,
            pad_mask_column=pad_mask_column,
            mention_offsets_column=mention_offsets_column,
            mention_lengths_column=mention_lengths_column,
            mentions_seq_lengths_column=mentions_seq_lengths_column,
            entities_column=entities_column,
            tokens_mapping_column=tokens_mapping_column,
        )

    def transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = batch[self.texts_column]
        torch.jit.isinstance(texts, List[List[str]])
        mention_offsets = batch[self.mention_offsets_column]
        torch.jit.isinstance(mention_offsets, List[List[int]])
        mention_lengths = batch[self.mention_lengths_column]
        torch.jit.isinstance(mention_lengths, List[List[int]])
        entities = batch[self.entities_column]
        torch.jit.isinstance(entities, List[List[int]])

        texts_pieces = [token for tokens in texts for token in tokens]
        texts_pieces_token_ids: List[List[int]] = super().forward(
            texts_pieces
        )

        (
            token_ids,
            seq_lens,
            mention_offsets,
            mention_lengths,
            tokens_mapping,
        ) = pieces_to_texts(
            texts_pieces_token_ids,
            texts,
            mention_offsets,
            mention_lengths,
            # bos_idx=self.bos_idx,
            # eos_idx=self.eos_idx,
            max_seq_len=self.max_seq_len,
        )
        entities = [
            text_entities[: len(text_mention_offsets)]
            for text_entities, text_mention_offsets in zip(entities, mention_offsets)
        ]
        mentions_seq_lens: List[int] = [
            len(text_mention_offsets) for text_mention_offsets in mention_offsets
        ]

        return {
            self.token_ids_column: token_ids,
            self.seq_lens_column: seq_lens,
            self.mention_offsets_column: mention_offsets,
            self.mention_lengths_column: mention_lengths,
            self.mentions_seq_lengths_column: mentions_seq_lens,
            self.entities_column: entities,
            self.tokens_mapping_column: tokens_mapping,
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._collate(self.transform(batch))

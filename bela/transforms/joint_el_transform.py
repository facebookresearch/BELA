#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from bela.transforms.hf_transform import HFTransform
from bela.transforms.spm_transform import SPMTransform


class ReadState(Enum):
    ReadAlphaNum = 1
    ReadSpace = 2
    ReadOther = 3


def insert_spaces(text: str) -> Tuple[str, List[int]]:
    """
    The raw string inputs are sometimes miss spaces between
    text pieces, like smiles could joint text:

    [smile]Some text.[smile] another text.

    This function modify text string to separate alphanumeric tokens
    from any other tokens to make models live easier. The above example
    will become:

    [smile] Some text . [smile] another text .
    """
    out_str: str = ""
    insertions: List[int] = []

    # In the beginning of the string we assume we just read some space
    state: ReadState = ReadState.ReadSpace
    for idx, char in enumerate(utf8_chars(text)):
        if state == ReadState.ReadSpace:
            if unicode_isspace(char):
                pass
            elif unicode_isalnum(char):
                state = ReadState.ReadAlphaNum
            else:
                state = ReadState.ReadOther
        elif state == ReadState.ReadAlphaNum:
            if unicode_isspace(char):
                state = ReadState.ReadSpace
            elif unicode_isalnum(char):
                pass
            else:
                out_str += " "
                insertions.append(idx)
                state = ReadState.ReadOther
        elif state == ReadState.ReadOther:
            if unicode_isspace(char):
                state = ReadState.ReadSpace
            elif unicode_isalnum(char):
                out_str += " "
                insertions.append(idx)
                state = ReadState.ReadAlphaNum
            else:
                pass
        out_str += char

    return out_str, insertions


def lower_bound(a: List[int], x: int) -> int:
    lo: int = 0
    hi: int = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def align_start(start: int, starts: List[int]) -> int:
    new_start: int = start
    if start not in starts:
        if len(starts) > 0:
            lb = lower_bound(starts, start)
            if lb == len(starts) or starts[lb] != start:
                new_start = starts[max(0, lb - 1)]
    return new_start


def align_end(end: int, ends: List[int]) -> int:
    new_end: int = end
    if end not in ends:
        if len(ends) > 0:
            lb = lower_bound(ends, end)
            if lb < len(ends):
                new_end = ends[lb]
    return new_end


def pieces_to_texts(
    texts_pieces_token_ids: List[List[int]],
    texts: List[List[str]],
    texts_mention_offsets: List[List[int]],
    texts_mention_lengths: List[List[int]],
    bos_idx: int,
    eos_idx: int,
    max_seq_len: int = 256,
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
    tokens_mapping: List[List[List[int]]] = []  # bs x idx x 2

    pieces_offset = 0
    for text, old_mention_offsets, old_mention_lengths in zip(
        texts,
        texts_mention_offsets,
        texts_mention_lengths,
    ):
        mapping: List[Tuple[int, int]] = []
        text_token_ids: List[int] = [bos_idx]
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
        text_token_ids.append(eos_idx)

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
        mapping = [[start, end] for start, end in mapping if end < max_seq_len]
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
def pad_tokens_mapping(tokens_mapping: List[List[List[int]]]) -> List[List[List[int]]]:
    seq_lens: List[int] = []
    for seq in tokens_mapping:
        seq_lens.append(len(seq))
    pad_to_length = max(seq_lens)

    for mapping in tokens_mapping:
        padding = pad_to_length - len(mapping)
        if padding >= 0:
            for _ in range(padding):
                mapping.append([0, 1])
        else:
            for _ in range(-padding):
                mapping.pop()
    return tokens_mapping


def pieces_to_texts(
    texts_pieces_token_ids: List[List[int]],
    texts: List[List[str]],
    texts_mention_offsets: List[List[int]],
    texts_mention_lengths: List[List[int]],
    bos_idx: int,
    eos_idx: int,
    max_seq_len: int = 256,
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
    tokens_mapping: List[List[List[int]]] = []  # bs x idx x 2

    pieces_offset = 0
    for text, old_mention_offsets, old_mention_lengths in zip(
        texts,
        texts_mention_offsets,
        texts_mention_lengths,
    ):
        mapping: List[Tuple[int, int]] = []
        text_token_ids: List[int] = [bos_idx]
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
        text_token_ids.append(eos_idx)

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
        mapping = [[start, end] for start, end in mapping if end < max_seq_len]
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


class JointELCollate(torch.nn.Module):
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
        sp_tokens_boundaries_column: str = "sp_tokens_boundaries",
        insertions_column: str = "insertions",
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
        self.sp_tokens_boundaries_column = sp_tokens_boundaries_column
        self.insertions_column = insertions_column

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        token_ids = batch[self.token_ids_column]
        assert torch.jit.isinstance(token_ids, List[List[int]])
        seq_lens = batch[self.seq_lens_column]
        assert torch.jit.isinstance(seq_lens, List[int])
        tokens_mapping = batch[self.tokens_mapping_column]
        assert torch.jit.isinstance(tokens_mapping, List[List[List[int]]])

        pad_token_ids = pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in token_ids],
            batch_first=True,
            padding_value=float(self._pad_idx),
        )
        pad_mask = torch.ne(pad_token_ids, self._pad_idx).to(dtype=torch.long)

        model_inputs: Dict[str, torch.Tensor] = {
            self.token_ids_column: pad_token_ids,
            self.pad_mask_column: pad_mask,
        }

        model_inputs[self.tokens_mapping_column] = torch.tensor(
            pad_tokens_mapping(tokens_mapping),
            dtype=torch.long,
        )

        if self.mention_offsets_column in batch:
            mention_offsets = batch[self.mention_offsets_column]
            assert torch.jit.isinstance(mention_offsets, List[List[int]])
            mention_lengths = batch[self.mention_lengths_column]
            assert torch.jit.isinstance(mention_lengths, List[List[int]])
            mentions_seq_lengths = batch[self.mentions_seq_lengths_column]
            assert torch.jit.isinstance(mentions_seq_lengths, List[int])
            entities = batch[self.entities_column]
            assert torch.jit.isinstance(entities, List[List[int]])

            model_inputs[self.mention_offsets_column] = torch.tensor(
                pad_2d(
                    mention_offsets,
                    seq_lens=mentions_seq_lengths,
                    pad_idx=self._mention_pad_idx,
                ),
                dtype=torch.long,
            )
            model_inputs[self.mention_lengths_column] = torch.tensor(
                pad_2d(
                    mention_lengths,
                    seq_lens=mentions_seq_lengths,
                    pad_idx=self._mention_pad_idx,
                ),
                dtype=torch.long,
            )
            model_inputs[self.entities_column] = torch.tensor(
                pad_2d(
                    entities,
                    seq_lens=mentions_seq_lengths,
                    pad_idx=self._mention_pad_idx,
                ),
                dtype=torch.long,
            )

        if self.sp_tokens_boundaries_column in batch:
            sp_tokens_boundaries = batch[self.sp_tokens_boundaries_column]
            assert torch.jit.isinstance(sp_tokens_boundaries, List[List[List[int]]])
            model_inputs[self.sp_tokens_boundaries_column] = torch.tensor(
                pad_tokens_mapping(sp_tokens_boundaries),
                dtype=torch.long,
            )

        if self.insertions_column in batch:
            insertions = batch[self.insertions_column]
            assert torch.jit.isinstance(insertions, List[List[int]])

            insertions_seq_lens: List[int] = []
            for seq in insertions:
                insertions_seq_lens.append(len(seq))
            model_inputs[self.insertions_column] = torch.tensor(
                pad_2d(
                    insertions,
                    seq_lens=insertions_seq_lens,
                    pad_idx=-1,
                ),
                dtype=torch.long,
            )

        return model_inputs



class JointELTransform(HFTransform):
    def __init__(
        self,
        model_path: str = "xlm-roberta-base",
        max_seq_len: int = 256,
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

        if 'xlm' in model_path:
            self.bos_idx = self.tokenizer.bos_token_id
            self.eos_idx = self.tokenizer.eos_token_id
        elif 'bert' in model_path:
            self.bos_idx = self.tokenizer.cls_token_id
            self.eos_idx = self.tokenizer.sep_token_id
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
            bos_idx=self.bos_idx,
            eos_idx=self.eos_idx,
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


class JointELXlmrRawTextTransform(SPMTransform):
    def __init__(
        self,
        sp_model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        max_seq_len: int = 256,
        insert_spaces: bool = False,
        mention_boundaries_on_word_boundaries: bool = False,
        align_mention_offsets_to_word_boundaries: bool = False,
        texts_column: str = "texts",
        mention_offsets_column: str = "mention_offsets",
        mention_lengths_column: str = "mention_lengths",
        mentions_seq_lengths_column: str = "mentions_seq_lengths",
        entities_column: str = "entities",
        token_ids_column: str = "input_ids",
        seq_lens_column: str = "seq_lens",
        pad_mask_column: str = "attention_mask",
        tokens_mapping_column: str = "tokens_mapping",
        sp_tokens_boundaries_column: str = "sp_tokens_boundaries",
        insertions_column: str = "insertions",
    ):
        super().__init__(
            sp_model_path=sp_model_path,
            max_seq_len=max_seq_len,
            add_special_tokens=False,
        )
        self.bos_idx = 0
        self.eos_idx = 2
        self.pad_idx = 1
        self.max_seq_len = max_seq_len
        self.insert_spaces = insert_spaces
        self.mention_boundaries_on_word_boundaries = (
            mention_boundaries_on_word_boundaries
        )
        self.align_mention_offsets_to_word_boundaries = (
            align_mention_offsets_to_word_boundaries
        )

        self.texts_column = texts_column
        self.mention_offsets_column = mention_offsets_column
        self.mention_lengths_column = mention_lengths_column
        self.mentions_seq_lengths_column = mentions_seq_lengths_column
        self.entities_column = entities_column
        self.token_ids_column = token_ids_column
        self.seq_lens_column = seq_lens_column
        self.tokens_mapping_column = tokens_mapping_column
        self.sp_tokens_boundaries_column = sp_tokens_boundaries_column
        self.insertions_column = insertions_column

        self._collate = JointELCollate(
            pad_idx=self.pad_idx,
            token_ids_column=token_ids_column,
            seq_lens_column=seq_lens_column,
            pad_mask_column=pad_mask_column,
            mention_offsets_column=mention_offsets_column,
            mention_lengths_column=mention_lengths_column,
            mentions_seq_lengths_column=mentions_seq_lengths_column,
            entities_column=entities_column,
            tokens_mapping_column=tokens_mapping_column,
            sp_tokens_boundaries_column=sp_tokens_boundaries_column,
        )

    def _calculate_alpha_num_boundaries(self, texts: List[str]):
        alpha_num_boundaries: List[List[List[int]]] = []
        for text in texts:
            example_alpha_num_boundaries: List[List[int]] = []
            cur_alpha_num_start: int = -1
            for idx, char in enumerate(text):
                if char.isalnum():
                    if cur_alpha_num_start == -1:
                        cur_alpha_num_start = idx
                else:
                    if cur_alpha_num_start != -1:
                        example_alpha_num_boundaries.append([cur_alpha_num_start, idx])
                        cur_alpha_num_start = -1

            if cur_alpha_num_start != -1:
                example_alpha_num_boundaries.append([cur_alpha_num_start, len(text)])

            alpha_num_boundaries.append(example_alpha_num_boundaries)

        return alpha_num_boundaries

    def _calculate_token_mapping(
        self,
        sp_token_ids: List[List[int]],
        sp_token_boundaries: List[List[List[int]]],
        word_boundaries: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        # Prepare list of possible mention start, ends pairs in terms of SP tokens.
        if self.mention_boundaries_on_word_boundaries:
            token_mapping: List[List[List[int]]] = []
            for ex_word_boundaries, ex_sp_token_boundaries in zip(
                word_boundaries, sp_token_boundaries
            ):
                ex_token_mapping: List[List[int]] = []
                sp_idx = 0
                for start, end in ex_word_boundaries:
                    while (
                        sp_idx < len(ex_sp_token_boundaries)
                        and start >= ex_sp_token_boundaries[sp_idx][1]
                    ):
                        sp_idx += 1
                    word_sp_start = sp_idx
                    word_sp_end = sp_idx
                    while (
                        word_sp_end < len(ex_sp_token_boundaries)
                        and end >= ex_sp_token_boundaries[word_sp_end][1]
                    ):
                        word_sp_end += 1

                    # check if end token <= max_seq_len - 2 (take into account EOS and BOS tokens)
                    if word_sp_end <= self.max_seq_len - 2:
                        # shift word_sp_start and word_sp_end by 1 taking into account EOS
                        ex_token_mapping.append([word_sp_start + 1, word_sp_end + 1])
                    else:
                        break
                token_mapping.append(ex_token_mapping)

            return token_mapping
        else:
            # Consider any SP token could be a start or end of the mention.
            return [
                [
                    [start, start + 1]
                    for start in range(  # start in range from 1 to maximum 255
                        1, min(len(example_sp_token_ids) - 2, self.max_seq_len - 2) + 1
                    )
                ]
                for example_sp_token_ids in sp_token_ids
            ]

    def _convert_mention_offsets(
        self,
        sp_token_boundaries: List[List[List[int]]],
        char_offsets: List[List[int]],
        char_lengths: List[List[int]],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        sp_offsets: List[List[int]] = []
        sp_lengths: List[List[int]] = []
        for example_char_offsets, example_char_lengths, example_token_boundaries in zip(
            char_offsets, char_lengths, sp_token_boundaries
        ):
            example_sp_offsets: List[int] = []
            example_sp_lengths: List[int] = []
            for offset, length in zip(example_char_offsets, example_char_lengths):
                token_idx = 0
                while (
                    token_idx < len(example_token_boundaries)
                    and example_token_boundaries[token_idx][0] <= offset
                ):
                    token_idx += 1
                if (
                    token_idx == len(example_token_boundaries)
                    or example_token_boundaries[token_idx][0] != offset
                ):
                    token_idx -= 1
                example_sp_offsets.append(token_idx)
                token_start_idx = token_idx
                while (
                    token_idx < len(example_token_boundaries)
                    and example_token_boundaries[token_idx][1] < offset + length
                ):
                    token_idx += 1
                example_sp_lengths.append(token_idx - token_start_idx + 1)

            # take into account BOS token and shift offsets by 1
            # also remove all pairs that go beyond max_seq_length - 1
            shifted_example_sp_offsets: List[int] = []
            for offset, length in zip(example_sp_offsets, example_sp_lengths):
                if 1 + offset + length <= self.max_seq_len - 1:
                    shifted_example_sp_offsets.append(offset + 1)
            example_sp_offsets = shifted_example_sp_offsets
            example_sp_lengths = example_sp_lengths[: len(example_sp_offsets)]
            sp_offsets.append(example_sp_offsets)
            sp_lengths.append(example_sp_lengths)

        return sp_offsets, sp_lengths

    def _adjust_mention_offsets_and_lengths(
        self,
        offsets: List[List[int]],
        lengths: List[List[int]],
        insertions: List[List[int]],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        new_offsets: List[List[int]] = []
        new_lengths: List[List[int]] = []

        for example_offsets, example_lengths, example_insertions in zip(
            offsets, lengths, insertions
        ):
            new_example_offsets: List[int] = []
            new_example_lengths: List[int] = []
            # assume that offsets, lengths sorted by offsets/lengths
            insertion_idx = 0
            current_shift = 0
            for offset, length in zip(example_offsets, example_lengths):
                while (
                    insertion_idx < len(example_insertions)
                    and example_insertions[insertion_idx] <= offset
                ):
                    current_shift += 1
                    insertion_idx += 1
                new_offset = offset + current_shift
                new_length = length
                length_insertion_idx = insertion_idx
                while (
                    length_insertion_idx < len(example_insertions)
                    and example_insertions[length_insertion_idx] < offset + length
                ):
                    new_length += 1
                    length_insertion_idx += 1
                new_example_offsets.append(new_offset)
                new_example_lengths.append(new_length)

            new_offsets.append(new_example_offsets)
            new_lengths.append(new_example_lengths)

        return new_offsets, new_lengths

    def _insert_spaces_to_texts(
        self, texts: List[str]
    ) -> Tuple[List[str], List[List[int]]]:
        all_texts: List[str] = []
        all_insertions: List[List[int]] = []

        for text in texts:
            out_text, insertions = insert_spaces(text)
            all_texts.append(out_text)
            all_insertions.append(insertions)

        return all_texts, all_insertions

    def _align_mention_offsets_to_word_boundaries(
        self,
        mention_offsets: List[List[int]],
        mention_lengths: List[List[int]],
        word_boundaries: List[List[List[int]]],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        In some training examples we can face situations where ground
        truth offsets point to the middle of the word, ex:
        ```
        Playlist in "#NuevaPlaylist ➡ Desempo"
        mente in "simplemente retirarte"
        ```
        we can align the offsets to the word boundaries, so in the examples
        above we will mark `NuevaPlaylist` and `simplemente` as mentions.
        """

        new_mention_offsets: List[List[int]] = []
        new_mention_lengths: List[List[int]] = []
        for ex_mention_offsets, ex_mention_length, ex_word_boundaries in zip(
            mention_offsets,
            mention_lengths,
            word_boundaries,
        ):
            starts: List[int] = []
            ends: List[int] = []
            for wb in ex_word_boundaries:
                starts.append(wb[0])
                ends.append(wb[1])
            ex_new_mention_offsets: List[int] = []
            ex_new_mention_lengths: List[int] = []
            for offset, length in zip(ex_mention_offsets, ex_mention_length):
                start = align_start(offset, starts)
                end = align_end(offset + length, ends)
                ex_new_mention_offsets.append(start)
                ex_new_mention_lengths.append(end - start)
            new_mention_offsets.append(ex_new_mention_offsets)
            new_mention_lengths.append(ex_new_mention_lengths)

        return new_mention_offsets, new_mention_lengths

    def transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = batch[self.texts_column]
        assert torch.jit.isinstance(texts, List[str])

        insertions: List[List[int]] = []
        if self.insert_spaces:
            texts, insertions = self._insert_spaces_to_texts(texts)

        word_boundaries = self._calculate_alpha_num_boundaries(texts)

        sp_tokens_with_indices: List[List[Tuple[int, int, int]]] = super().forward(texts)
        sp_token_ids: List[List[int]] = [
            [sp_token for sp_token, _, _ in tokens] for tokens in sp_tokens_with_indices
        ]
        # append bos and eos tokens
        sp_token_ids = [[self.bos_idx] + tokens + [self.eos_idx] for tokens in sp_token_ids]
        sp_token_boundaries: List[List[List[int]]] = [
            [[start, end] for _, start, end in tokens]
            for tokens in sp_tokens_with_indices
        ]
        seq_lens: List[int] = [
            len(example_token_ids) for example_token_ids in sp_token_ids
        ]

        tokens_mapping: List[List[List[int]]] = self._calculate_token_mapping(
            sp_token_ids,
            sp_token_boundaries,
            word_boundaries,
        )

        output: Dict[str, Any] = {
            self.token_ids_column: sp_token_ids,
            self.seq_lens_column: seq_lens,
            self.tokens_mapping_column: tokens_mapping,
            self.sp_tokens_boundaries_column: sp_token_boundaries,
        }
        if self.insert_spaces:
            output[self.insertions_column] = insertions

        if self.mention_offsets_column in batch:
            mention_offsets = batch[self.mention_offsets_column]
            assert torch.jit.isinstance(mention_offsets, List[List[int]])
            mention_lengths = batch[self.mention_lengths_column]
            assert torch.jit.isinstance(mention_lengths, List[List[int]])
            entities = batch[self.entities_column]
            assert torch.jit.isinstance(entities, List[List[int]])

            if self.insert_spaces:
                (
                    mention_offsets,
                    mention_lengths,
                ) = self._adjust_mention_offsets_and_lengths(
                    mention_offsets, mention_lengths, insertions
                )

            if self.align_mention_offsets_to_word_boundaries:
                (
                    mention_offsets,
                    mention_lengths,
                ) = self._align_mention_offsets_to_word_boundaries(
                    mention_offsets,
                    mention_lengths,
                    word_boundaries,
                )

            sp_offsets, sp_lengths = self._convert_mention_offsets(
                sp_token_boundaries,
                mention_offsets,
                mention_lengths,
            )

            entities: List[List[int]] = [
                example_entities[: len(example_mention_offsets)]
                for example_entities, example_mention_offsets in zip(
                    entities, sp_offsets
                )
            ]
            mentions_seq_lens: List[int] = [
                len(example_mention_offsets) for example_mention_offsets in sp_offsets
            ]

            output[self.mention_offsets_column] = sp_offsets
            output[self.mention_lengths_column] = sp_lengths
            output[self.mentions_seq_lengths_column] = mentions_seq_lens
            output[self.entities_column] = entities

        return output

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._collate(self.transform(batch))
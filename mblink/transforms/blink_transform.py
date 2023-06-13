# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from mblink.transforms.hf_transform import HFTransform

from mblink.utils.utils import (
    EntityCatalogueType,
    EntityCatalogue,
    MultilangEntityCatalogue,
    NegativesStrategy,
    order_entities,
)


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


def prepare_mention(
    context_left: List[int],
    mention_tokens: List[int],
    context_right: List[int],
    max_mention_length: int,
    mention_start_token: int,
    mention_end_token: int,
    bos_idx: int,
    eos_idx: int,
):
    context_left: List[int] = context_left[1:-1]
    mention_tokens: List[int] = mention_tokens[1:-1]
    context_right: List[int] = context_right[1:-1]

    mention_tokens = mention_tokens[: max_mention_length - 4]
    mention_tokens = [mention_start_token] + mention_tokens + [mention_end_token]

    left_quota = (max_mention_length - len(mention_tokens)) // 2 - 1
    right_quota = max_mention_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    empty_tokens: List[int] = []
    context_left = empty_tokens if left_quota == 0 else context_left[-left_quota:]
    context_right = context_right[:right_quota]

    context_left = [bos_idx] + context_left
    context_right = context_right + [eos_idx]

    context_tokens = context_left + mention_tokens + context_right

    return context_tokens


# class BlinkMentionRobertaTransform(HFTransform):
#     def __init__(
#         self,
#         mention_start_token: int = -2,
#         mention_end_token: int = -3,
#         model_path: Optional[str] = None,
#         max_seq_len: int = 64,
#     ):
#         super().__init__(
#             model_path=model_path,
#             max_seq_len=max_seq_len,
#         )
#         vocab_length = len(self.tokenizer.vocab)
#         self.bos_idx = self.tokenizer.bos_token_id
#         self.eos_idx = self.tokenizer.eos_token_id
#         self.mention_start_token = (vocab_length + mention_start_token) % vocab_length
#         self.mention_end_token = (vocab_length + mention_end_token) % vocab_length
#         self.max_mention_length = max_seq_len

#     def transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
#         left_context = batch["left_context"]
#         torch.jit.isinstance(left_context, List[str])
#         right_context = batch["right_context"]
#         torch.jit.isinstance(right_context, List[str])
#         mention = batch["mention"]
#         torch.jit.isinstance(mention, List[str])

#         left_token_ids: List[List[int]] = self.tokenizer(left_context)["input_ids"]
#         mention_token_ids: List[List[int]] = self.tokenizer(mention)["input_ids"]
#         right_token_ids: List[List[int]] = self.tokenizer(right_context)["input_ids"]

#         token_ids: List[List[int]] = []
#         attention_masks: List[List[int]] = []
#         seq_lens: List[int] = []
#         for lc_token_ids, m_token_ids, rc_token_ids, in zip(
#             left_token_ids,
#             mention_token_ids,
#             right_token_ids,
#         ):
#             sentence_token_ids = prepare_mention(
#                 lc_token_ids,
#                 m_token_ids,
#                 rc_token_ids,
#                 self.max_mention_length,
#                 self.mention_start_token,
#                 self.mention_end_token,
#                 self.bos_idx,
#                 self.eos_idx,
#             )
#             token_ids.append(sentence_token_ids)
#             attention_mask = [1] * len(sentence_token_ids)
#             attention_masks.append(attention_mask)
#             seq_lens.append(len(sentence_token_ids))

#         attention_masks = pad_2d(
#             attention_masks,
#             seq_lens,
#             pad_idx = 0,
#         )

#         return {
#             "input_ids": token_ids,
#             "attention_mask": attention_masks,
#         }


# class BlinkEntityPretokenizedTransform(HFTransform):
#     def __init__(
#         self,
#         model_path: Optional[str] = None,
#         max_seq_len: int = 64,
#     ):
#         super().__init__(
#             model_path=model_path,
#             max_seq_len=max_seq_len,
#         )
#         self.max_entity_length = max_seq_len

#     def transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
#         token_ids = batch["token_ids"]
#         torch.jit.isinstance(token_ids, List[List[int]])

#         result_token_ids: List[List[int]] = []
#         seq_lens: List[int] = []
#         attention_masks: List[List[int]] = []

#         for token_ids_per_sequence in token_ids:
#             if len(token_ids_per_sequence) > self.max_entity_length:
#                 eos_token = token_ids_per_sequence[-1]
#                 token_ids_per_sequence = token_ids_per_sequence[
#                     : self.max_entity_length
#                 ]
#                 token_ids_per_sequence[-1] = eos_token
#             result_token_ids.append(token_ids_per_sequence)
#             seq_len = len(token_ids_per_sequence)
#             attention_mask = [1] * len(token_ids_per_sequence)
#             attention_masks.append(attention_mask)
#             seq_lens.append(seq_len)

#         attention_masks = pad_2d(
#             attention_masks,
#             seq_lens,
#             pad_idx = 0,
#         )

#         return {
#             "input_ids": result_token_ids,
#             "attention_mask": attention_masks,
#         }


# class BlinkTransform(nn.Module):
#     def __init__(
#         self,
#         model_path: Optional[str] = None,
#         mention_start_token: int = -2,
#         mention_end_token: int = -3,
#         max_mention_len: int = 64,
#         max_entity_len: int = 64,
#     ):
#         super().__init__()
#         self.mention_transform = BlinkMentionRobertaTransform(
#             mention_start_token=mention_start_token,
#             mention_end_token=mention_end_token,
#             model_path=model_path,
#             max_seq_len=max_mention_len,
#         )
#         self.entity_transform = BlinkEntityPretokenizedTransform(
#             model_path=model_path,
#             max_seq_len=max_entity_len,
#         )

#     def forward(
#         self, batch: Dict[str, Any]
#     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         return self.mention_transform(batch), self.entity_transform(batch)

#     @property
#     def bos_idx(self):
#         return self.mention_transform.bos_idx

#     @property
#     def eos_idx(self):
#         return self.mention_transform.eos_idx


class BlinkTransform(HFTransform):
    def __init__(
        self,
        model_path: str = "bert-base-uncased",
        mention_start_token: int = 1,
        mention_end_token: int = 2,
        max_mention_len: int = 32,
        max_entity_len: int = 64,
        add_eos_bos_to_entity: bool = False,
    ):
        super().__init__(
            model_path=model_path,
        )
        vocab_length = self.tokenizer.vocab_size
        self.mention_start_token = (vocab_length + mention_start_token) % vocab_length
        self.mention_end_token = (vocab_length + mention_end_token) % vocab_length
        self.max_mention_len = max_mention_len
        self.max_entity_len = max_entity_len
        self.add_eos_bos_to_entity = add_eos_bos_to_entity

    def _transform_mention(
        self,
        left_context: List[str],
        mention: List[str],
        right_context: List[str],
    ) -> List[List[int]]:
        token_ids: List[List[int]] = []
        for sentence_lc, sentence_mention, sentence_rc, in zip(
            left_context,
            mention,
            right_context,
        ):
            lc_token_ids = self.tokenizer.encode(sentence_lc)
            mention_token_ids = self.tokenizer.encode(sentence_mention)
            rc_token_ids = self.tokenizer.encode(sentence_rc)

            sentence_token_ids = prepare_mention(
                lc_token_ids,
                mention_token_ids,
                rc_token_ids,
                self.max_mention_len,
                self.mention_start_token,
                self.mention_end_token,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
            )

            token_ids.append(sentence_token_ids)
        return token_ids

    def _transform_entity(
        self,
        entity_token_ids: List[List[int]],
    ) -> List[List[int]]:
        result_token_ids: List[List[int]] = []
        for token_ids in entity_token_ids:
            if self.add_eos_bos_to_entity:
                token_ids = [self.bos_idx] + token_ids + [self.eos_idx]
            if len(token_ids) > self.max_entity_len:
                token_ids = token_ids[: self.max_entity_len]
                token_ids[-1] = self.eos_idx
            result_token_ids.append(token_ids)
        return result_token_ids

    def _to_tensor(self, token_ids, attention_mask_pad_idx=0):
        seq_lens = [len(seq) for seq in token_ids]
        input_ids = pad_2d(
            token_ids,
            seq_lens,
            pad_idx = self.pad_token_id,
        )
        attention_mask = [[1]*seq_len for seq_len in seq_lens]
        attention_mask = pad_2d(
            attention_mask,
            seq_lens,
            pad_idx = attention_mask_pad_idx,
        )
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }

    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        left_context = batch["left_context"]
        torch.jit.isinstance(left_context, List[str])
        mention = batch["mention"]
        torch.jit.isinstance(mention, List[str])
        right_context = batch["right_context"]
        torch.jit.isinstance(right_context, List[str])
        entity_token_ids = batch["token_ids"]
        torch.jit.isinstance(entity_token_ids, List[List[int]])

        mention_token_ids = self._transform_mention(
            left_context,
            mention,
            right_context,
        )

        mention_tensors = self._to_tensor(
            mention_token_ids,
        )
        entity_token_ids = self._transform_entity(entity_token_ids)
        entity_tensors = self._to_tensor(
            entity_token_ids,
        )

        return (mention_tensors, entity_tensors)

    @property
    def bos_idx(self):
        return self.tokenizer.cls_token_id

    @property
    def eos_idx(self):
        return self.tokenizer.sep_token_id

import logging
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import os.path

from pytorch_lightning import LightningModule

from bela.datamodule.entity_encoder import embed


logger = logging.getLogger(__name__)


class ClassificationMetrics(NamedTuple):
    f1: float
    precision: float
    recall: float
    support: int
    tp: int
    fp: int
    fn: int
    # Bag-Of-Entities metrics: we consider targets and predictions as set
    # of entities instead of strong matching positions and entities.
    boe_f1: float
    boe_precision: float
    boe_recall: float
    boe_support: int
    boe_tp: int
    boe_fp: int
    boe_fn: int


class ClassificationHead(nn.Module):
    def __init__(
        self,
        ctxt_output_dim=1024,
    ):
        super(ClassificationHead, self).__init__()

        self.mlp = nn.Sequential(
            # [mention, candidate, mention - candidate, mention * candidate, md_score, dis_score]
            nn.Linear(4 * ctxt_output_dim + 2, ctxt_output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ctxt_output_dim, ctxt_output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ctxt_output_dim, 1),
        )

    def forward(self, mentions_repr, entities_repr, md_scores, dis_scores):
        features = [
            mentions_repr,
            entities_repr,
            mentions_repr - entities_repr,
            mentions_repr * entities_repr,
            md_scores,
            dis_scores,
        ]
        features = torch.cat(features, 1)
        return self.mlp(features)


class SaliencyClassificationHead(nn.Module):
    def __init__(
        self,
        ctxt_output_dim=1024,
    ):
        super(SaliencyClassificationHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(9 * ctxt_output_dim + 4, ctxt_output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ctxt_output_dim, ctxt_output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ctxt_output_dim, 1),
        )

    def forward(
        self, cls_tokens_repr, mentions_repr, entities_repr, md_scores, dis_scores
    ):
        cls_mention_dot_product = torch.sum(
            cls_tokens_repr * mentions_repr, 1
        ).unsqueeze(-1)
        cls_entity_dot_product = torch.sum(
            cls_tokens_repr * entities_repr, 1
        ).unsqueeze(-1)

        features = [
            cls_tokens_repr,
            mentions_repr,
            entities_repr,
            mentions_repr - entities_repr,
            mentions_repr * entities_repr,
            cls_tokens_repr - mentions_repr,
            cls_tokens_repr * mentions_repr,
            cls_tokens_repr - entities_repr,
            cls_tokens_repr * entities_repr,
            md_scores,
            dis_scores,
            cls_mention_dot_product,
            cls_entity_dot_product,
        ]
        features = torch.cat(features, 1)
        return self.mlp(features)


class SpanEncoder(nn.Module):
    def __init__(
        self,
        mention_aggregation="linear",
        ctxt_output_dim=1024,
        cand_output_dim=1024,
        dropout=0.1,
    ):
        super(SpanEncoder, self).__init__()

        if mention_aggregation == "linear":
            self.mention_mlp = nn.Linear(ctxt_output_dim * 2, cand_output_dim)
        # elif mention_aggregation == "mlp":
        #     self.mention_mlp = nn.Sequential(
        #         nn.Linear(ctxt_output_dim, ctxt_output_dim),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(ctxt_output_dim, cand_output_dim),
        #     )
        else:
            raise NotImplementedError()

    def forward(self, text_encodings, mention_offsets, mention_lengths):
        # print("text_encodings", text_encodings)
        # print("mention_offsets", mention_offsets)
        # print("mention_lengths", mention_lengths)

        # print("text_encodings shape", text_encodings.shape)
        # print("mention_offsets shape", mention_offsets.shape)
        # print("mention_lengths shape", mention_lengths.shape)
        idx = (
            torch.arange(mention_offsets.shape[0])
            .unsqueeze(1)
            .repeat(1, mention_offsets.shape[1])
        )
        mention_starts = text_encodings[idx, mention_offsets]
        mention_ends = text_encodings[
            idx,
            mention_lengths + mention_offsets - 1,
        ]
        mention_emb = torch.cat([mention_starts, mention_ends], dim=2)
        mention_encodings = self.mention_mlp(mention_emb)

        return mention_encodings


class MentionScoresHead(nn.Module):
    def __init__(
        self,
        encoder_output_dim=1024,
        max_mention_length=10,
    ):
        super(MentionScoresHead, self).__init__()
        self.max_mention_length = max_mention_length
        self.bound_classifier = nn.Linear(encoder_output_dim, 3)

    def forward(self, text_encodings, mask_ctxt, tokens_mapping):
        """
        Returns scores for *inclusive* mention boundaries
        """
        device = text_encodings.device
        # (bs, seqlen, 3)
        logits = self.bound_classifier(text_encodings)
        # (bs, seqlen, 1); (bs, seqlen, 1); (bs, seqlen, 1)
        # start_logprobs, end_logprobs, mention_logprobs = logits.split(1, dim=-1)
        start_logprobs = logits[:, :, 0].squeeze(-1)
        end_logprobs = logits[:, :, 1].squeeze(-1)
        mention_logprobs = logits[:, :, 2].squeeze(-1)

        # impossible to choose masked tokens as starts/ends of spans
        start_logprobs[mask_ctxt != 1] = float("-inf")
        end_logprobs[mask_ctxt != 1] = float("-inf")
        mention_logprobs[mask_ctxt != 1] = float("-inf")

        # take sum of log softmaxes:
        # log p(mention) = log p(start_pos && end_pos) = log p(start_pos) + log p(end_pos)
        # DIM: (bs, starts, ends)
        mention_scores = start_logprobs.unsqueeze(2) + end_logprobs.unsqueeze(1)

        # (bs, starts, ends)
        mention_cum_scores = torch.zeros(
            mention_scores.size(), dtype=mention_scores.dtype
        ).to(device)

        # add ends
        mention_logprobs_end_cumsum = torch.zeros(
            mask_ctxt.size(0), dtype=mention_scores.dtype
        ).to(device)
        for i in range(mask_ctxt.size(1)):
            mention_logprobs_end_cumsum += mention_logprobs[:, i]
            mention_cum_scores[:, :, i] += mention_logprobs_end_cumsum.unsqueeze(-1)

        # subtract starts
        mention_logprobs_start_cumsum = torch.zeros(
            mask_ctxt.size(0), dtype=mention_scores.dtype
        ).to(device)
        for i in range(mask_ctxt.size(1) - 1):
            mention_logprobs_start_cumsum += mention_logprobs[:, i]
            mention_cum_scores[
                :, (i + 1), :
            ] -= mention_logprobs_start_cumsum.unsqueeze(-1)

        # DIM: (bs, starts, ends)
        mention_scores += mention_cum_scores

        # DIM: (starts, ends, 2) -- tuples of [start_idx, end_idx]
        mention_bounds = torch.stack(
            [
                torch.arange(mention_scores.size(1))
                .unsqueeze(-1)
                .expand(mention_scores.size(1), mention_scores.size(2)),  # start idxs
                torch.arange(mention_scores.size(1))
                .unsqueeze(0)
                .expand(mention_scores.size(1), mention_scores.size(2)),  # end idxs
            ],
            dim=-1,
        ).to(device)
        # DIM: (starts, ends)
        mention_sizes = (
            mention_bounds[:, :, 1] - mention_bounds[:, :, 0] + 1
        )  # (+1 as ends are inclusive)

        # Remove invalids (startpos > endpos, endpos > seqlen) and renormalize
        # DIM: (bs, starts, ends)

        # valid mention starts mask
        select_indices = torch.cat(
            [
                torch.arange(tokens_mapping.shape[0])
                .unsqueeze(1)
                .repeat(1, tokens_mapping.shape[1])
                .unsqueeze(-1),
                tokens_mapping[:, :, 0].unsqueeze(-1).to(torch.device("cpu")),
            ],
            -1,
        ).flatten(0, 1)
        token_starts_mask = torch.zeros(mask_ctxt.size(), dtype=mask_ctxt.dtype)
        token_starts_mask[select_indices[:, 0], select_indices[:, 1]] = 1
        token_starts_mask[:, 0] = 0

        # valid mention ends mask
        select_indices = torch.cat(
            [
                torch.arange(tokens_mapping.shape[0])
                .unsqueeze(1)
                .repeat(1, tokens_mapping.shape[1])
                .unsqueeze(-1),
                (tokens_mapping[:, :, 1] - 1).unsqueeze(-1).to(torch.device("cpu")),
            ],
            -1,
        ).flatten(0, 1)
        token_ends_mask = torch.zeros(mask_ctxt.size(), dtype=mask_ctxt.dtype)
        token_ends_mask[select_indices[:, 0], select_indices[:, 1]] = 1
        token_ends_mask[:, 0] = 0

        # valid mention starts*ends mask
        valid_starts_ends_mask = torch.bmm(
            token_starts_mask.unsqueeze(2), token_ends_mask.unsqueeze(1)
        ).to(device)

        valid_mask = (
            (mention_sizes.unsqueeze(0) > 0)
            & torch.gt(mask_ctxt.unsqueeze(2), 0)
            & torch.gt(valid_starts_ends_mask, 0)
        )
        # DIM: (bs, starts, ends)
        # 0 is not a valid
        mention_scores[~valid_mask] = float("-inf")  # invalids have logprob=-inf (p=0)
        # DIM: (bs, starts * ends)
        mention_scores = mention_scores.view(mention_scores.size(0), -1)
        # DIM: (bs, starts * ends, 2)
        mention_bounds = mention_bounds.view(-1, 2)
        mention_bounds = mention_bounds.unsqueeze(0).expand(
            mention_scores.size(0), mention_scores.size(1), 2
        )

        if self.max_mention_length is not None:
            mention_scores, mention_bounds = self.filter_by_mention_size(
                mention_scores,
                mention_bounds,
            )

        return mention_scores, mention_bounds

    def batch_reshape_mask_left(
        self,
        input_t: torch.Tensor,
        selected: torch.Tensor,
        pad_idx: Union[int, float] = 0,
        left_align_mask: Optional[torch.Tensor] = None,
    ):
        """
        Left-aligns all ``selected" values in input_t, which is a batch of examples.
            - input_t: >=2D tensor (N, M, *)
            - selected: 2D torch.Bool tensor, 2 dims same size as first 2 dims of `input_t` (N, M)
            - pad_idx represents the padding to be used in the output
            - left_align_mask: if already precomputed, pass the alignment mask in
                (mask on the output, corresponding to `selected` on the input)
        Example:
            input_t  = [[1,2,3,4],[5,6,7,8]]
            selected = [[0,1,0,1],[1,1,0,1]]
            output   = [[2,4,0],[5,6,8]]
        """
        batch_num_selected = selected.sum(1)
        max_num_selected = batch_num_selected.max()

        # (bsz, 2)
        repeat_freqs = torch.stack(
            [batch_num_selected, max_num_selected - batch_num_selected], dim=-1
        )
        # (bsz x 2,)
        repeat_freqs = repeat_freqs.view(-1)

        if left_align_mask is None:
            # (bsz, 2)
            left_align_mask = (
                torch.zeros(input_t.size(0), 2).to(input_t.device).to(torch.bool)
            )
            left_align_mask[:, 0] = 1
            # (bsz x 2,): [1,0,1,0,...]
            left_align_mask = left_align_mask.view(-1)
            # (bsz x max_num_selected,): [1 xrepeat_freqs[0],0 x(M-repeat_freqs[0]),1 xrepeat_freqs[1],0 x(M-repeat_freqs[1]),...]
            left_align_mask = left_align_mask.repeat_interleave(repeat_freqs)
            # (bsz, max_num_selected)
            left_align_mask = left_align_mask.view(-1, max_num_selected)

        # reshape to (bsz, max_num_selected, *)
        input_reshape = (
            torch.empty(left_align_mask.size() + input_t.size()[2:])
            .to(input_t.device, input_t.dtype)
            .fill_(pad_idx)
        )
        input_reshape[left_align_mask] = input_t[selected]
        # (bsz, max_num_selected, *); (bsz, max_num_selected)
        return input_reshape, left_align_mask

    def prune_ctxt_mentions(
        self,
        mention_logits: torch.Tensor,
        mention_bounds: torch.Tensor,
        num_cand_mentions: int,
        threshold: float,
    ):
        """
            Prunes mentions based on mention scores/logits (by either
            `threshold` or `num_cand_mentions`, whichever yields less candidates)
        Inputs:
            mention_logits: torch.FloatTensor (bsz, num_total_mentions)
            mention_bounds: torch.IntTensor (bsz, num_total_mentions)
            num_cand_mentions: int
            threshold: float
        Returns:
            torch.FloatTensor(bsz, max_num_pred_mentions): top mention scores/logits
            torch.IntTensor(bsz, max_num_pred_mentions, 2): top mention boundaries
            torch.BoolTensor(bsz, max_num_pred_mentions): mask on top mentions
            torch.BoolTensor(bsz, total_possible_mentions): mask for reshaping from total possible mentions -> max # pred mentions
        """
        # (bsz, num_cand_mentions); (bsz, num_cand_mentions)
        num_cand_mentions = min(num_cand_mentions, mention_logits.shape[1])
        top_mention_logits, mention_pos = mention_logits.topk(
            num_cand_mentions, sorted=True
        )
        # (bsz, num_cand_mentions, 2)
        #   [:,:,0]: index of batch
        #   [:,:,1]: index into top mention in mention_bounds
        mention_pos = torch.stack(
            [
                torch.arange(mention_pos.size(0))
                .to(mention_pos.device)
                .unsqueeze(-1)
                .expand_as(mention_pos),
                mention_pos,
            ],
            dim=-1,
        )
        # (bsz, num_cand_mentions)
        top_mention_pos_mask = torch.sigmoid(top_mention_logits) > threshold

        # (total_possible_mentions, 2)
        #   tuples of [index of batch, index into mention_bounds] of what mentions to include
        mention_pos = mention_pos[
            top_mention_pos_mask
            | (
                # 2nd part of OR: if nothing is > threshold, use topK that are > -inf
                ((top_mention_pos_mask.sum(1) == 0).unsqueeze(-1))
                & (top_mention_logits > float("-inf"))
            )
        ]
        mention_pos = mention_pos.view(-1, 2)
        # (bsz, total_possible_mentions)
        #   mask of possible logits
        mention_pos_mask = torch.zeros(mention_logits.size(), dtype=torch.bool).to(
            mention_pos.device
        )
        mention_pos_mask[mention_pos[:, 0], mention_pos[:, 1]] = 1
        # (bsz, max_num_pred_mentions, 2)
        chosen_mention_bounds, chosen_mention_mask = self.batch_reshape_mask_left(
            mention_bounds, mention_pos_mask, pad_idx=0
        )
        # (bsz, max_num_pred_mentions)
        chosen_mention_logits, _ = self.batch_reshape_mask_left(
            mention_logits,
            mention_pos_mask,
            pad_idx=float("-inf"),
            left_align_mask=chosen_mention_mask,
        )
        return (
            chosen_mention_logits,
            chosen_mention_bounds,
            chosen_mention_mask,
            mention_pos_mask,
        )

    def filter_by_mention_size(
        self, mention_scores: torch.Tensor, mention_bounds: torch.Tensor
    ):
        """
        Filter all mentions > maximum mention length
        mention_scores: torch.FloatTensor (bsz, num_mentions)
        mention_bounds: torch.LongTensor (bsz, num_mentions, 2)
        """
        # (bsz, num_mentions)
        mention_bounds_mask = (
            mention_bounds[:, :, 1] - mention_bounds[:, :, 0] <= self.max_mention_length
        )
        # (bsz, num_filtered_mentions)
        mention_scores = mention_scores[mention_bounds_mask]
        mention_scores = mention_scores.view(mention_bounds_mask.size(0), -1)
        # (bsz, num_filtered_mentions, 2)
        mention_bounds = mention_bounds[mention_bounds_mask]
        mention_bounds = mention_bounds.view(mention_bounds_mask.size(0), -1, 2)
        return mention_scores, mention_bounds
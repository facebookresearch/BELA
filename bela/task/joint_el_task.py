#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import OrderedDict
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import faiss
import faiss.contrib.torch_utils
import hydra
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from bela.conf import (
    DataModuleConf,
    ModelConf,
    OptimConf,
    TransformConf,
)

import datetime
import os
import numpy as np


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
        ctxt_output_dim=768,
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
        ctxt_output_dim=768,
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
        ctxt_output_dim=768,
        cand_output_dim=768,
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
        encoder_output_dim=768,
        max_mention_length=10,
    ):
        super(MentionScoresHead, self).__init__()
        self.max_mention_length = max_mention_length
        self.bound_classifier = nn.Linear(encoder_output_dim, 3)

    def forward(self, text_encodings, mask_ctxt, tokens_mapping):
        """
        Retuns scores for *inclusive* mention boundaries
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


class JointELTask(LightningModule):
    def __init__(
        self,
        transform: TransformConf,
        model: ModelConf,
        datamodule: DataModuleConf,
        optim: OptimConf,
        embeddings_path: str,
        faiss_index_path: Optional[str] = None,
        n_retrieve_candidates: int = 10,
        eval_compure_recall_at: Tuple[int] = (1, 10, 100),
        warmup_steps: int = 0,
        load_from_checkpoint: Optional[str] = None,
        only_train_disambiguation: bool = False,
        train_el_classifier: bool = True,
        train_saliency: bool = True,
        md_threshold: float = 0.2,
        el_threshold: float = 0.4,
        saliency_threshold: float = 0.4,
        use_gpu_index: bool = False,
    ):
        super().__init__()

        # encoder setup
        self.encoder_conf = model
        self.optim_conf = optim

        self.embeddings_path = embeddings_path
        self.faiss_index_path = faiss_index_path

        self.n_retrieve_candidates = n_retrieve_candidates
        self.eval_compure_recall_at = eval_compure_recall_at

        self.warmup_steps = warmup_steps
        self.load_from_checkpoint = load_from_checkpoint

        self.disambiguation_loss = nn.CrossEntropyLoss()
        self.md_loss = nn.BCEWithLogitsLoss()
        self.el_loss = nn.BCEWithLogitsLoss()
        self.saliency_loss = nn.BCEWithLogitsLoss()
        self.only_train_disambiguation = only_train_disambiguation
        self.train_el_classifier = train_el_classifier
        self.train_saliency = train_saliency
        self.md_threshold = md_threshold
        self.el_threshold = el_threshold
        self.saliency_threshold = saliency_threshold
        self.use_gpu_index = use_gpu_index

    @staticmethod
    def _get_encoder_state(state, encoder_name):
        encoder_state = OrderedDict()
        for key, value in state["state_dict"].items():
            if key.startswith(encoder_name):
                encoder_state[key[len(encoder_name) + 1 :]] = value
        return encoder_state

    def setup_gpu_index(self):
        gpu_id = int(os.environ.get("LOCAL_RANK", "0"))

        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = gpu_id
        flat_config.useFloat16 = True

        res = faiss.StandardGpuResources()

        self.faiss_index = faiss.GpuIndexFlatIP(res, self.embedding_dim, flat_config)
        self.faiss_index.add(self.embeddings)

    def setup(self, stage: str):
        if stage == "test":
            return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False
        
        self.embeddings = torch.load(self.embeddings_path)
        self.embedding_dim = len(self.embeddings[0])
        self.embeddings.requires_grad = False

        self.encoder = hydra.utils.instantiate(
            self.encoder_conf,
        )

        self.project_encoder_op = nn.Identity()
        if self.encoder.embedding_dim != self.embedding_dim:
            self.project_encoder_op = nn.Sequential(
                nn.Linear(self.encoder.embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
            )

        self.span_encoder = SpanEncoder(
            ctxt_output_dim=self.embedding_dim,
            cand_output_dim=self.embedding_dim,
        )
        self.mention_encoder = MentionScoresHead(
            encoder_output_dim=self.embedding_dim,
        )
        self.el_encoder = ClassificationHead(
            ctxt_output_dim=self.embedding_dim,
        )
        if self.train_saliency:
            self.saliency_encoder = SaliencyClassificationHead(
                ctxt_output_dim=self.embedding_dim,
            )

        if self.load_from_checkpoint is not None:
            logger.info(f"Load encoders state from {self.load_from_checkpoint}")
            with open(self.load_from_checkpoint, "rb") as f:
                checkpoint = torch.load(f, map_location=torch.device("cpu"))

            encoder_state = self._get_encoder_state(checkpoint, "encoder")
            self.encoder.load_state_dict(encoder_state)

            span_encoder_state = self._get_encoder_state(checkpoint, "span_encoder")
            self.span_encoder.load_state_dict(span_encoder_state)

            project_encoder_op_state = self._get_encoder_state(
                checkpoint, "project_encoder_op"
            )
            if len(project_encoder_op_state) > 0:
                self.project_encoder_op.load_state_dict(project_encoder_op_state)

            mention_encoder_state = self._get_encoder_state(
                checkpoint, "mention_encoder"
            )
            if len(mention_encoder_state) > 0:
                self.mention_encoder.load_state_dict(mention_encoder_state)

            el_encoder_state = self._get_encoder_state(checkpoint, "el_encoder")
            if len(el_encoder_state) > 0:
                self.el_encoder.load_state_dict(el_encoder_state)

            saliency_encoder_state = self._get_encoder_state(
                checkpoint, "saliency_encoder"
            )
            if len(saliency_encoder_state) > 0 and self.train_saliency:
                self.saliency_encoder.load_state_dict(saliency_encoder_state)

        self.optimizer = hydra.utils.instantiate(self.optim_conf, self.parameters())

        if self.use_gpu_index:
            logger.info(f"Setup GPU index")
            self.setup_gpu_index()
            # self.embeddings = None
        else:
            logger.info(f"Setup CPU index")
            assert self.faiss_index_path is not None
            self.faiss_index = faiss.read_index(self.faiss_index_path)


    def sim_score(self, mentions_repr, entities_repr):
        # bs x emb_dim , bs x emb_dim
        scores = torch.sum(mentions_repr * entities_repr, 1)
        return scores

    def forward(
        self,
        text_inputs,
        attention_mask,
        mention_offsets,
        mention_lengths,
    ):
        # encode query and contexts
        _, last_layer = self.encoder(text_inputs, attention_mask)
        text_encodings = last_layer
        text_encodings = self.project_encoder_op(text_encodings)

        mentions_repr = self.span_encoder(
            text_encodings, mention_offsets, mention_lengths
        )
        return text_encodings, mentions_repr

    def configure_optimizers(self):
        return self.optimizer

    def _disambiguation_training_step(
        self, mentions_repr, mention_offsets, mention_lengths, entities_ids
    ):
        device = mentions_repr.get_device()

        # flat mentions and entities indices (mentions_num x embedding_dim)
        flat_mentions_repr = mentions_repr[mention_lengths != 0]
        flat_entities_ids = entities_ids[mention_lengths != 0]

        if flat_mentions_repr.shape[0] == 0:
            return None

        # obtain positive entities representations
        if self.use_gpu_index:
            entities_repr = torch.stack(
                [
                    self.faiss_index.reconstruct(flat_id)
                    for flat_id in flat_entities_ids.tolist()
                ]
            ).to(device)
        else:
            entities_repr = self.embeddings[flat_entities_ids.to("cpu")].to(device)

        # compute scores for positive entities
        pos_scores = self.sim_score(flat_mentions_repr, entities_repr)

        # retrieve candidates indices
        if self.use_gpu_index:
            (
                _,
                neg_cand_indices,
                neg_cand_repr,
            ) = self.faiss_index.search_and_reconstruct(
                flat_mentions_repr.detach().cpu().numpy().astype(np.float32),
                self.n_retrieve_candidates,
            )
            neg_cand_indices = torch.from_numpy(neg_cand_indices).to(device)
            neg_cand_repr = torch.from_numpy(neg_cand_repr).to(device)

        else:
            (
                _,
                neg_cand_indices,
            ) = self.faiss_index.search(
                flat_mentions_repr.detach().cpu().numpy().astype(np.float32),
                self.n_retrieve_candidates,
            )

            # get candidates embeddings
            neg_cand_repr = (
                self.embeddings[neg_cand_indices.flatten()]
                .reshape(
                    neg_cand_indices.shape[0],  # bs
                    neg_cand_indices.shape[1],  # n_retrieve_candidates
                    self.embeddings.shape[1],  # emb dim
                )
                .to(device)
            )

            neg_cand_indices = torch.from_numpy(neg_cand_indices).to(device)

        # compute scores (bs x n_retrieve_candidates)
        neg_cand_scores = torch.bmm(
            flat_mentions_repr.unsqueeze(1), neg_cand_repr.transpose(1, 2)
        ).squeeze(1)

        # zero score for the positive entities
        neg_cand_scores[
            neg_cand_indices.eq(
                flat_entities_ids.unsqueeze(1).repeat([1, self.n_retrieve_candidates])
            )
        ] = float("-inf")

        # append positive scores to neg scores (bs x (1 + n_retrieve_candidates))
        scores = torch.hstack([pos_scores.unsqueeze(1), neg_cand_scores])

        # cosntruct targets
        targets = torch.tensor([0] * neg_cand_scores.shape[0]).to(device)

        loss = self.disambiguation_loss(scores, targets)

        return loss

    def _md_training_step(
        self,
        text_encodings,
        text_pad_mask,
        gold_mention_offsets,
        gold_mention_lengths,
        entities_ids,
        tokens_mapping,
    ):
        device = text_encodings.get_device()

        mention_logits, mention_bounds = self.mention_encoder(
            text_encodings,
            text_pad_mask,
            tokens_mapping,
        )

        gold_mention_ends = gold_mention_offsets + gold_mention_lengths - 1
        gold_mention_bounds = torch.cat(
            [gold_mention_offsets.unsqueeze(-1), gold_mention_ends.unsqueeze(-1)], -1
        )
        gold_mention_bounds[gold_mention_lengths == 0] = -1

        gold_mention_pos_idx = (
            (
                mention_bounds.unsqueeze(1)
                - gold_mention_bounds.unsqueeze(
                    2
                )  # (bs, num_mentions, start_pos * end_pos, 2)
            )
            .abs()
            .sum(-1)
            == 0
        ).nonzero()

        # (bs, total_possible_spans)
        gold_mention_binary = torch.zeros(
            mention_logits.size(), dtype=mention_logits.dtype
        ).to(device)
        gold_mention_binary[gold_mention_pos_idx[:, 0], gold_mention_pos_idx[:, 2]] = 1

        # prune masked spans
        mask = mention_logits != float("-inf")
        masked_mention_logits = mention_logits[mask]
        masked_gold_mention_binary = gold_mention_binary[mask]

        return (
            self.md_loss(masked_mention_logits, masked_gold_mention_binary),
            mention_logits,
            mention_bounds,
        )

    def _el_training_step(
        self,
        text_encodings,
        mention_logits,
        mention_bounds,
        gold_mention_offsets,
        gold_mention_lengths,
        entities_ids,
        tokens_mapping,
    ):
        """
            Train "rejection" head.
        Inputs:
            text_encodings: last layer output of text encoder
            mention_logits: mention scores produced by mention detection head
            mention_bounds: mention bounds (start, end (inclusive)) by MD head
            gold_mention_offsets: ground truth mention offsets
            gold_mention_lengths: ground truth mention lengths
            entities_ids: entity ids for ground truth mentions
            tokens_mapping: sentencepiece to text token mapping
        Returns:
            el_loss: sum of entity linking loss over all predicted mentions
        """
        device = text_encodings.get_device()

        # get predicted mention_offsets and mention_bounds by MD model
        (
            chosen_mention_logits,
            chosen_mention_bounds,
            chosen_mention_mask,
            mention_pos_mask,
        ) = self.mention_encoder.prune_ctxt_mentions(
            mention_logits,
            mention_bounds,
            num_cand_mentions=50,
            threshold=self.md_threshold,
        )

        mention_offsets = chosen_mention_bounds[:, :, 0]
        mention_lengths = (
            chosen_mention_bounds[:, :, 1] - chosen_mention_bounds[:, :, 0] + 1
        )
        mention_lengths[mention_offsets == -1] = 0
        mention_offsets[mention_offsets == -1] = 0

        # get mention representations for predicted mentions
        mentions_repr = self.span_encoder(
            text_encodings, mention_offsets, mention_lengths
        )

        # flat mentions and entities indices (mentions_num x embedding_dim)
        flat_mentions_repr = mentions_repr[mention_lengths != 0]
        flat_mentions_scores = torch.sigmoid(
            chosen_mention_logits[mention_lengths != 0]
        )
        flat_mentions_repr = flat_mentions_repr[flat_mentions_scores > 0]

        # cand_scores, cand_indices = self.faiss_index.search(
        #     flat_mentions_repr.detach().cpu().numpy(), 1
        # )
        # cand_scores = torch.from_numpy(cand_scores)
        # cand_indices = torch.from_numpy(cand_indices)

        cand_scores, cand_indices = self.faiss_index.search(
            flat_mentions_repr.detach().cpu().numpy().astype(np.float32),
            1,
        )
        if self.use_gpu_index:
            cand_scores = torch.from_numpy(cand_scores)
            cand_indices = torch.from_numpy(cand_indices)

        # iterate over predicted and gold mentions to create targets for
        # predicted mentions
        targets = []
        for (
            e_mention_offsets,
            e_mention_lengths,
            e_gold_mention_offsets,
            e_gold_mention_lengths,
            e_entities,
        ) in zip(
            mention_offsets.detach().cpu().tolist(),
            mention_lengths.detach().cpu().tolist(),
            gold_mention_offsets.cpu().tolist(),
            gold_mention_lengths.cpu().tolist(),
            entities_ids.cpu().tolist(),
        ):
            e_gold_targets = {
                (offset, length): ent
                for offset, length, ent in zip(
                    e_gold_mention_offsets, e_gold_mention_lengths, e_entities
                )
            }
            e_targets = [
                e_gold_targets.get((offset, length), -1)
                for offset, length in zip(e_mention_offsets, e_mention_lengths)
            ]
            targets.append(e_targets)

        targets = torch.tensor(targets, device=device)
        flat_targets = targets[mention_lengths != 0][flat_mentions_scores > 0]
        md_scores = flat_mentions_scores[flat_mentions_scores > 0].unsqueeze(-1)
        # flat_entities_repr = self.embeddings[cand_indices.squeeze(1)].to(device)
        if self.use_gpu_index:
            flat_entities_repr = torch.stack(
                [
                    self.faiss_index.reconstruct(flat_id)
                    for flat_id in cand_indices.squeeze(1).tolist()
                ]
            ).to(device)
        else:
            flat_entities_repr = self.embeddings[cand_indices.squeeze(1)].to(device)

        cand_scores = cand_scores.to(device)
        cand_indices = cand_indices.to(device)

        predictions = self.el_encoder(
            flat_mentions_repr, flat_entities_repr, md_scores, cand_scores
        ).squeeze(1)

        binary_targets = (flat_targets == cand_indices.squeeze(1)).double()

        el_loss = self.el_loss(predictions, binary_targets)

        return el_loss

    def training_step(self, batch, batch_idx):
        """
        This receives queries, each with mutliple contexts.
        """

        text_inputs = batch["input_ids"]  # bs x mention_len
        text_pad_mask = batch["attention_mask"]
        gold_mention_offsets = batch["mention_offsets"]  # bs x max_mentions_num
        gold_mention_lengths = batch["mention_lengths"]  # bs x max_mentions_num
        entities_ids = batch["entities"]  # bs x max_mentions_num
        tokens_mapping = batch["tokens_mapping"]  # bs x max_tokens_in_input x 2

        # mention representations (bs x max_mentions_num x embedding_dim)
        text_encodings, mentions_repr = self(
            text_inputs, text_pad_mask, gold_mention_offsets, gold_mention_lengths
        )

        dis_loss = self._disambiguation_training_step(
            mentions_repr,
            gold_mention_offsets,
            gold_mention_lengths,
            entities_ids,
        )
        if dis_loss is not None:
            self.log("dis_loss", dis_loss, prog_bar=True)
        loss = dis_loss

        if not self.only_train_disambiguation:
            md_loss, mention_logits, mention_bounds = self._md_training_step(
                text_encodings,
                text_pad_mask,
                gold_mention_offsets,
                gold_mention_lengths,
                entities_ids,
                tokens_mapping,
            )
            self.log("md_loss", md_loss, prog_bar=True)
            if loss is not None:
                loss += md_loss
            else:
                loss = md_loss

            if self.train_el_classifier:
                el_loss = self._el_training_step(
                    text_encodings,
                    mention_logits,
                    mention_bounds,
                    gold_mention_offsets,
                    gold_mention_lengths,
                    entities_ids,
                    tokens_mapping,
                )
                self.log("el_loss", el_loss, prog_bar=True)
                loss += el_loss

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def _disambiguation_eval_step(
        self,
        mentions_repr,
        mention_offsets,
        mention_lengths,
        entities_ids,
    ):
        device = mentions_repr.device

        # flat mentions and entities indices (mentions_num x embedding_dim)
        flat_mentions_repr = mentions_repr[mention_lengths != 0]
        flat_entities_ids = entities_ids[mention_lengths != 0]

        # obtain positive entities representations
        # entities_repr = self.embeddings[flat_entities_ids.to("cpu")].to(device)
        if self.use_gpu_index:
            entities_repr = torch.stack(
                [
                    self.faiss_index.reconstruct(flat_id)
                    for flat_id in flat_entities_ids.tolist()
                ]
            ).to(device)
        else:
            entities_repr = self.embeddings[flat_entities_ids.to("cpu")].to(device)

        # compute scores for positive entities
        pos_scores = self.sim_score(flat_mentions_repr, entities_repr)

        # candidates to retrieve
        n_retrieve_candidates = max(self.eval_compure_recall_at)

        # retrieve negative candidates ids and scores
        neg_cand_scores, neg_cand_indices = self.faiss_index.search(
            flat_mentions_repr.detach().cpu().numpy().astype(np.float32),
            n_retrieve_candidates,
        )
        neg_cand_scores = torch.from_numpy(neg_cand_scores).to(device)
        neg_cand_indices = torch.from_numpy(neg_cand_indices).to(device)

        # zero score for the positive entities
        neg_cand_scores[
            neg_cand_indices.eq(
                flat_entities_ids.unsqueeze(1).repeat([1, n_retrieve_candidates])
            )
        ] = float("-inf")

        # append positive scores to neg scores
        scores = torch.hstack([pos_scores.unsqueeze(1), neg_cand_scores])

        # cosntruct targets
        targets = torch.tensor([0] * neg_cand_scores.shape[0]).to(device)

        loss = self.disambiguation_loss(scores, targets)

        # compute recall at (1, 10, 100)
        flat_entities_ids = flat_entities_ids.cpu().tolist()
        neg_cand_indices = neg_cand_indices.cpu().tolist()

        recalls = []
        for k in self.eval_compure_recall_at:
            recall = sum(
                entity_id in cand_entity_ids[:k]
                for entity_id, cand_entity_ids in zip(
                    flat_entities_ids, neg_cand_indices
                )
            )
            recalls.append(recall)

        return (
            recalls,
            len(flat_entities_ids),
            loss,
        )

    def _joint_eval_step(
        self,
        text_inputs,
        text_pad_mask,
        gold_mention_offsets,
        gold_mention_lengths,
        entities_ids,
        tokens_mapping,
    ):
        device = text_inputs.device

        # encode query and contexts
        _, last_layer = self.encoder(text_inputs)
        text_encodings = last_layer
        text_encodings = self.project_encoder_op(text_encodings)

        mention_logits, mention_bounds = self.mention_encoder(
            text_encodings, text_pad_mask, tokens_mapping
        )

        (
            chosen_mention_logits,
            chosen_mention_bounds,
            chosen_mention_mask,
            mention_pos_mask,
        ) = self.mention_encoder.prune_ctxt_mentions(
            mention_logits,
            mention_bounds,
            num_cand_mentions=50,
            threshold=self.md_threshold,
        )

        mention_offsets = chosen_mention_bounds[:, :, 0]
        mention_lengths = (
            chosen_mention_bounds[:, :, 1] - chosen_mention_bounds[:, :, 0] + 1
        )
        mention_lengths[mention_offsets == -1] = 0
        mention_offsets[mention_offsets == -1] = 0

        mentions_repr = self.span_encoder(
            text_encodings, mention_offsets, mention_lengths
        )

        # flat mentions and entities indices (mentions_num x embedding_dim)
        flat_mentions_repr = mentions_repr[mention_lengths != 0]

        mentions_scores = torch.sigmoid(chosen_mention_logits)
        # flat_mentions_repr = flat_mentions_repr[flat_mentions_scores > 0]

        # retrieve candidates top-1 ids and scores
        cand_scores, cand_indices = self.faiss_index.search(
            flat_mentions_repr.detach().cpu().numpy().astype(np.float32), 1
        )

        if self.train_el_classifier:
            # flat_entities_repr = self.embeddings[cand_indices.squeeze(1)].to(device)

            if self.use_gpu_index:
                flat_entities_repr = torch.stack(
                    [
                        self.faiss_index.reconstruct(flat_id)
                        for flat_id in cand_indices.squeeze(1).tolist()
                    ]
                ).to(device)
            else:
                flat_entities_repr = self.embeddings[cand_indices.squeeze(1)].to(device)

            flat_mentions_scores = mentions_scores[mention_lengths != 0].unsqueeze(-1)
            cand_scores = torch.from_numpy(cand_scores).to(device)
            el_scores = torch.sigmoid(
                self.el_encoder(
                    flat_mentions_repr,
                    flat_entities_repr,
                    flat_mentions_scores,
                    cand_scores,
                )
            ).squeeze(1)

        gold_mention_offsets = gold_mention_offsets.cpu().tolist()
        gold_mention_lengths = gold_mention_lengths.cpu().tolist()
        entities_ids = entities_ids.cpu().tolist()

        el_targets = []
        for offsets, lengths, example_ent_ids in zip(
            gold_mention_offsets,
            gold_mention_lengths,
            entities_ids,
        ):
            el_targets.append(
                {
                    (offset, length): ent_id
                    for offset, length, ent_id in zip(offsets, lengths, example_ent_ids)
                    if length != 0
                }
            )

        mention_offsets = mention_offsets.detach().cpu().tolist()
        mention_lengths = mention_lengths.detach().cpu().tolist()
        mentions_scores = mentions_scores.detach().cpu().tolist()

        el_predictions = []
        cand_idx = 0
        for offsets, lengths, md_scores in zip(
            mention_offsets, mention_lengths, mentions_scores
        ):
            example_predictions = {}
            for offset, length, md_score in zip(offsets, lengths, md_scores):
                if length != 0:
                    if md_score >= self.md_threshold:
                        if (
                            not self.train_el_classifier
                            or el_scores[cand_idx] >= self.el_threshold
                        ):
                            example_predictions[(offset, length)] = cand_indices[
                                cand_idx
                            ][0]
                    cand_idx += 1
            el_predictions.append(example_predictions)

        return el_targets, el_predictions

    def _eval_step(self, batch, batch_idx):
        
        text_inputs = batch["input_ids"]  # bs x mention_len
        text_pad_mask = batch["attention_mask"]
        mention_offsets = batch["mention_offsets"]  # bs x max_mentions_num
        mention_lengths = batch["mention_lengths"]  # bs x max_mentions_num
        entities_ids = batch["entities"]  # bs x max_mentions_num
        tokens_mapping = batch["tokens_mapping"]

        if self.only_train_disambiguation:
            text_encodings, mentions_repr = self(
                text_inputs, text_pad_mask, mention_offsets, mention_lengths
            )

            return self._disambiguation_eval_step(
                mentions_repr,
                mention_offsets,
                mention_lengths,
                entities_ids,
            )

        return self._joint_eval_step(
            text_inputs,
            text_pad_mask,
            mention_offsets,
            mention_lengths,
            entities_ids,
            tokens_mapping,
        )

    def _compute_disambiguation_metrics(self, outputs, log_prefix):
        total_recalls = [0] * len(self.eval_compure_recall_at)
        total_ent_count = 0
        total_loss = 0

        for recalls, count, loss in outputs:
            for idx in range(len(total_recalls)):
                total_recalls[idx] += recalls[idx]
            total_ent_count += count
            total_loss += loss

        metrics = {
            log_prefix + "_ent_count": total_ent_count,
            log_prefix + "_loss": total_loss,
        }

        for idx, recall_at in enumerate(self.eval_compure_recall_at):
            metrics[log_prefix + f"_recall_at_{recall_at}"] = (
                total_recalls[idx] / total_ent_count
            )

        return metrics

    @staticmethod
    def calculate_classification_metrics(targets, predictions):
        tp, fp, support = 0, 0, 0
        boe_tp, boe_fp, boe_support = 0, 0, 0
        for example_targets, example_predictions in zip(targets, predictions):
            for pos, ent in example_targets.items():
                support += 1
                if pos in example_predictions and example_predictions[pos] == ent:
                    tp += 1
            for pos, ent in example_predictions.items():
                if pos not in example_targets or example_targets[pos] != ent:
                    fp += 1

            example_targets_set = set(example_targets.values())
            example_predictions_set = set(example_predictions.values())
            for ent in example_targets_set:
                boe_support += 1
                if ent in example_predictions_set:
                    boe_tp += 1
            for ent in example_predictions_set:
                if ent not in example_targets_set:
                    boe_fp += 1

        def compute_f1_p_r(tp, fp, fn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            return f1, precision, recall

        fn = support - tp
        boe_fn = boe_support - boe_tp

        f1, precision, recall = compute_f1_p_r(tp, fp, fn)
        boe_f1, boe_precision, boe_recall = compute_f1_p_r(boe_tp, boe_fp, boe_fn)

        return ClassificationMetrics(
            f1=f1,
            precision=precision,
            recall=recall,
            support=support,
            tp=tp,
            fp=fp,
            fn=fn,
            boe_f1=boe_f1,
            boe_precision=boe_precision,
            boe_recall=boe_recall,
            boe_support=boe_support,
            boe_tp=boe_tp,
            boe_fp=boe_fp,
            boe_fn=boe_fn,
        )

    def _compute_el_metrics(self, outputs, log_prefix):
        el_targets = []
        el_predictions = []
        for (
            batch_el_targets,
            batch_el_predictions,
        ) in outputs:
            el_targets.extend(batch_el_targets)
            el_predictions.extend(batch_el_predictions)

        el_metrics = self.calculate_classification_metrics(el_targets, el_predictions)

        metrics = {
            log_prefix + "_f1": el_metrics.f1,
            log_prefix + "_precision": el_metrics.precision,
            log_prefix + "_recall": el_metrics.recall,
            log_prefix + "_support": el_metrics.support,
            log_prefix + "_tp": el_metrics.tp,
            log_prefix + "_fp": el_metrics.fp,
            log_prefix + "_fn": el_metrics.fn,
            log_prefix + "_boe_f1": el_metrics.boe_f1,
            log_prefix + "_boe_precision": el_metrics.boe_precision,
            log_prefix + "_boe_recall": el_metrics.boe_recall,
            log_prefix + "_boe_support": el_metrics.boe_support,
            log_prefix + "_boe_tp": el_metrics.boe_tp,
            log_prefix + "_boe_fp": el_metrics.boe_fp,
            log_prefix + "_boe_fn": el_metrics.boe_fn,
        }

        return metrics

    def _eval_epoch_end(self, outputs, log_prefix="valid"):
        if self.only_train_disambiguation:
            metrics = self._compute_disambiguation_metrics(outputs, log_prefix)
        else:
            metrics = self._compute_el_metrics(outputs, log_prefix)
        print("EVAL:")
        print(metrics)

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, valid_outputs):
        self._eval_epoch_end(valid_outputs)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        self._eval_epoch_end(test_outputs, "test")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        This hook will be called before loading state_dict from a checkpoint.
        setup("fit") will build the model before loading state_dict

        Args:
            checkpoint: A dictionary with variables from the checkpoint.
        """
        self.setup("fit")

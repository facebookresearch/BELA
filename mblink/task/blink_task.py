#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import OrderedDict
from typing import Optional

import hydra
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from mblink.conf import (
    DataModuleConf,
    ModelConf,
    OptimConf,
    TransformConf,
)


logger = logging.getLogger(__name__)


class InBatchTripletLoss(nn.Module):
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    # Blog post: https://omoindrot.github.io/triplet-loss
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Build the triplet loss over a matrix of computed scores
        For each mention distance to correct entity should be greater then distance to all
        other entities in batch by margin.
        Args:
            scores: n_mentions x n_entities matrix of distances between mentions and entities
            targets: vector of indices of correct entity for each mention (n_mentions)
        """
        one_hot_targets = torch.zeros(scores.shape).bool()
        one_hot_targets[torch.arange(targets.shape[0]), targets] = True

        pos_scores = scores[one_hot_targets].unsqueeze(1).repeat(1, scores.shape[1] - 1)
        neg_scores = scores[~one_hot_targets].reshape(
            scores.shape[0], scores.shape[1] - 1
        )

        loss = torch.relu(self.margin + neg_scores - pos_scores).mean()

        return loss


class InBatchMarginLoss(nn.Module):
    """
    Pushes positives scores above margin and negatives below 0.
    The loss calculated as max(0, maring - positive scores) +
    max(0, negative scores).
    """

    def __init__(self, margin: float = 100.0, pos_weight=1.0, use_mean=True):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.reduce_op = torch.mean if use_mean else torch.sum

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        one_hot_targets = torch.zeros(scores.shape).bool()
        one_hot_targets[torch.arange(targets.shape[0]), targets] = True

        pos_scores = scores[one_hot_targets]
        neg_scores = scores[~one_hot_targets]

        if self.pos_weight is None:
            pos_weight = scores.shape[1] - 1

        loss = self.reduce_op(
            pos_weight * torch.relu(self.margin - pos_scores)
        ) + self.reduce_op(torch.relu(neg_scores))

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module, second_weight=1.0):
        super().__init__()
        self.first = first
        self.second = second
        self.second_weight = second_weight

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.first(scores, targets) + self.second_weight * self.second(
            scores, targets
        )


class ElBiEncoderTask(LightningModule):
    def __init__(
        self,
        transform: TransformConf,
        model: ModelConf,
        datamodule: DataModuleConf,
        optim: OptimConf,
        in_batch_eval: bool = True,  # use only in batch contexts for validation
        warmup_steps: int = 0,
        filter_entities: bool = True,
        loss: str = "cross_entropy",
        triplet_loss_margin: float = 1.0,
        margin_loss_margin: float = 100.0,
        margin_loss_pos_weight: Optional[float] = None,
        margin_loss_weight: float = 1.0,
        margin_loss_mean: bool = True,
        load_from_checkpoint: Optional[str] = None,
    ):
        super().__init__()

        # encoder setup
        self.mention_encoder_conf = model
        self.entity_encoder_conf = model
        self.optim_conf = optim
        self.in_batch_eval = in_batch_eval
        self.warmup_steps = warmup_steps
        self.filter_entities = filter_entities
        self.load_from_checkpoint = load_from_checkpoint

        if loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "triplet":
            self.loss = InBatchTripletLoss(margin=triplet_loss_margin)
        elif loss == "margin":
            self.loss = CombinedLoss(
                nn.CrossEntropyLoss(),
                InBatchMarginLoss(
                    margin=margin_loss_margin,
                    pos_weight=margin_loss_pos_weight,
                    mean=margin_loss_mean,
                ),
                margin_loss_weight,
            )
        else:
            raise ValueError(f"Unsupported loss {loss}")

    @staticmethod
    def _get_encoder_state(state, encoder_name):
        encoder_state = OrderedDict()
        for key, value in state["state_dict"].items():
            if key.startswith(encoder_name):
                encoder_state[key[len(encoder_name) + 1 :]] = value
        return encoder_state

    def setup(self, stage: str):
        if stage == "test":
            return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False

        self.mention_encoder = hydra.utils.instantiate(
            self.mention_encoder_conf,
            _recursive_=False,
        )
        self.entity_encoder = hydra.utils.instantiate(
            self.entity_encoder_conf,
            _recursive_=False,
        )
        if self.load_from_checkpoint is not None:
            logger.info(f"Load encoders state from {self.load_from_checkpoint}")
            with open(self.load_from_checkpoint, "rb") as f:
                checkpoint = torch.load(f, map_location=torch.device("cpu"))
            entity_encoder_state = self._get_encoder_state(checkpoint, "entity_encoder")
            self.entity_encoder.load_state_dict(entity_encoder_state)
            mention_encoder_state = self._get_encoder_state(
                checkpoint, "mention_encoder"
            )
            self.mention_encoder.load_state_dict(mention_encoder_state)

        self.optimizer = hydra.utils.instantiate(
            self.optim_conf, self.parameters(), _recursive_=False
        )

    def sim_score(self, mentions_repr, entities_repr):
        scores = torch.matmul(mentions_repr, torch.transpose(entities_repr, 0, 1))
        return scores

    def forward(
        self,
        mentions_ids,
        entities_ids,
    ):
        # encode query and contexts
        mentions_repr = self.mention_encoder(mentions_ids)  # bs x d
        entities_repr = self.entity_encoder(entities_ids)  # bs x d
        return mentions_repr, entities_repr

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        """
        This receives queries, each with mutliple contexts.
        """

        mentions = batch["mentions"]  # bs x mention_len
        entities = batch["entities"]  # bs x entity len
        entity_ids = batch["entity_ids"]  # bs
        targets = batch["targets"]  # bs
        mask = batch["entity_tensor_mask"]  # bs

        mentions_repr, entities_repr = self(mentions, entities)
        if self.trainer._accelerator_connector.use_ddp:
            mentions_to_send = mentions_repr.detach()
            entities_to_send = entities_repr.detach()

            all_mentions_repr = self.all_gather(mentions_to_send)  # num_workers x bs
            all_entities_repr = self.all_gather(entities_to_send)
            all_targets = self.all_gather(targets)
            # we are not filtering duplicated entities now
            all_entity_ids = self.all_gather(entity_ids)
            all_mask = self.all_gather(mask)
            # offset = 0
            all_mentions_list = []
            all_entities_list = []
            all_entity_ids_list = []
            all_targets_list = []

            # Add current device representations first.
            # It is needed so we would not filter calculated on this
            # device representations.
            all_mentions_list.append(mentions_repr)
            entities_repr = entities_repr[mask.bool()]
            all_entities_list.append(entities_repr)
            all_entity_ids_list.append(entity_ids[mask.bool()].tolist())
            all_targets_list.append(targets)
            # offset += entities_repr.size(0)

            for i in range(all_targets.size(0)):
                if i != self.global_rank:
                    all_mentions_list.append(all_mentions_repr[i])
                    all_entities_list.append(all_entities_repr[i][all_mask[i].bool()])
                    all_entity_ids_list.append(
                        all_entity_ids[i][all_mask[i].bool()].tolist()
                    )
                    # all_targets[i] += offset
                    all_targets_list.append(all_targets[i])
                    # offset += all_entities_repr[i].size(0)

            mentions_repr = torch.cat(all_mentions_list, dim=0)  # total_ctx x dim
            # entities_repr = torch.cat(all_entities_list, dim=0)  # total_query x dim
            # targets = torch.cat(all_targets_list, dim=0)
            if self.filter_entities:
                entities_repr, targets = self._filter_entities_and_targets(
                    all_entities_list,
                    all_entity_ids_list,
                    all_targets_list,
                )
            else:
                entities_repr = torch.cat(all_entities_list, dim=0)
                targets = torch.cat(all_targets_list, dim=0)
            # entity_ids = torch.flatten(entity_ids)
        else:
            entities_repr = entities_repr[mask.bool()]

        scores = self.sim_score(mentions_repr, entities_repr)
        loss = self.loss(scores, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _filter_entities_and_targets(
        self, all_entities_list, all_entity_ids_list, all_targets_list
    ):
        filtered_entities_repr = []
        filtered_targets = []
        ent_indexes_map = {}

        for entities_repr, entity_ids, targets, in zip(
            all_entities_list,
            all_entity_ids_list,
            all_targets_list,
        ):
            for entity_repr, ent_id in zip(entities_repr, entity_ids):
                if ent_id not in ent_indexes_map:
                    ent_idx = len(ent_indexes_map)
                    ent_indexes_map[ent_id] = ent_idx
                    filtered_entities_repr.append(entity_repr)
            for target in targets.tolist():
                filtered_targets.append(ent_indexes_map[entity_ids[target]])

        filtered_entities_repr = torch.stack(filtered_entities_repr, dim=0)
        filtered_targets = torch.tensor(
            filtered_targets,
            dtype=torch.long,
            device=filtered_entities_repr.get_device(),
        )

        return filtered_entities_repr, filtered_targets

    def _eval_step(self, batch, batch_idx):
        mentions = batch["mentions"]  # bs x mention_len
        entities = batch["entities"]  # bs x entity len
        entity_ids = batch["entity_ids"]  # bs
        targets = batch["targets"]  # bs
        mask = batch["entity_tensor_mask"]  # bs

        mentions_repr, entities_repr = self(mentions, entities)
        entities_repr = entities_repr[mask.bool()]
        scores = self.sim_score(mentions_repr, entities_repr)  # bs x ctx_cnt
        loss = self.loss(scores, targets)

        return (
            self.compute_rank_metrics(scores, targets),
            mentions_repr,
            entities_repr,
            targets,
            entity_ids,
            loss,
        )

    def compute_rank_metrics(self, scores, target_labels):
        # Compute total un_normalized avg_ranks, mrr
        values, indices = torch.sort(scores, dim=1, descending=True)
        rank = 0
        mrr = 0.0
        for i, idx in enumerate(target_labels):
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item() + 1
            mrr += 1 / (gold_idx.item() + 1)
        return rank, mrr

    def _eval_epoch_end(self, outputs, log_prefix="valid"):
        total_avg_rank, total_ent_count, total_count = 0, 0, 0
        total_mrr = 0
        total_loss = 0
        if self.in_batch_eval:
            for metrics, mentions_repr, entities_repr, _, _, loss in outputs:
                rank, mrr = metrics
                total_avg_rank += rank
                total_mrr += mrr
                total_ent_count += entities_repr.size(0)
                total_count += mentions_repr.size(0)
                total_loss += loss
            total_ent_count = total_ent_count / len(outputs)
        else:
            # TODO: collect entities representations over all batches
            raise NotImplementedError("Only in-batch eval implementted!")
        metrics = {
            log_prefix + "_avg_rank": total_avg_rank / total_count,
            log_prefix + "_mrr": total_mrr / total_count,
            log_prefix + "_ent_count": total_ent_count,
            log_prefix + "_loss": total_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, valid_outputs):
        self._eval_epoch_end(valid_outputs)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        self._eval_epoch_end(test_outputs, "test")

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple, Union

import faiss  # @manual=//faiss/python:pyfaiss
import hydra
import numpy as np
import torch
import torch.nn as nn
import os.path

import torch.distributed as dist

from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
from bela.conf import (
    DataModuleConf,
    ModelConf,
    OptimConf,
    TransformConf,
)

from bela.datamodule.entity_encoder import embed, embed_cluster
from bela.task.joint_el_heads import (UnknownClassificationHead, ClassificationMetrics, MentionScoresHead, ClassificationHead, SaliencyClassificationHead, SpanEncoder)

logger = logging.getLogger(__name__)

class JointELTask(LightningModule):
    def __init__(
        self,
        transform: TransformConf,
        model: ModelConf,
        datamodule: DataModuleConf,
        optim: OptimConf,
        faiss_index_path: str,
        novel_entity_idx_path: Optional[str] = None,
        path_blink_model: Optional[str] = None,
        # save_mention_embeddings: Optional[bool] = False,
        embeddings_path: str = "",
        save_novel_md_path: str = None,
        n_retrieve_candidates: int = 10,
        eval_compute_recall_at: Tuple[int] = (1, 10, 100),
        warmup_steps: int = 0,
        load_from_checkpoint: Optional[str] = None,
        only_train_disambiguation: bool = False,
        train_el_classifier: bool = True,
        train_saliency: bool = True,
        train_unknown_classifier: bool = False,
        md_threshold: float = 0.2,
        # ue_threshold: float = 0.0,
        el_threshold: float = 0.0,
        saliency_threshold: float = 0.4,
        cluster_based: bool = False,
        mention_based: bool = False
    ):
        super().__init__()

        # encoder setup
        self.encoder_conf = model
        self.optim_conf = optim

        self.embeddings_path = embeddings_path
        self.faiss_index_path = faiss_index_path

        self.novel_entity_idx_path = novel_entity_idx_path
        self.save_novel_md_path = save_novel_md_path
        self.path_blink_model = path_blink_model
        self.description_based = True
        self.mention_based = False
        if cluster_based==True or mention_based==True:
            self.description_based = False   
        if mention_based==True:
            self.mention_based = True         

        self.n_retrieve_candidates = n_retrieve_candidates
        self.eval_compute_recall_at = eval_compute_recall_at

        self.warmup_steps = warmup_steps
        self.load_from_checkpoint = load_from_checkpoint

        self.disambiguation_loss = nn.CrossEntropyLoss()
        self.md_loss = nn.BCEWithLogitsLoss()
        self.el_loss = nn.BCEWithLogitsLoss()
        self.saliency_loss = nn.BCEWithLogitsLoss()
        self.only_train_disambiguation = only_train_disambiguation
        self.train_el_classifier = train_el_classifier
        self.train_saliency = train_saliency
        self.train_unknown_classifier = train_unknown_classifier
        self.md_threshold = md_threshold
        self.el_threshold = el_threshold
        self.saliency_threshold = saliency_threshold

    @staticmethod
    def _get_encoder_state(state, encoder_name):
        encoder_state = OrderedDict()
        for key, value in state["state_dict"].items():
            if key.startswith(encoder_name):
                encoder_state[key[len(encoder_name) + 1 :]] = value
        return encoder_state

    def setup(self, stage: str):
        #if stage == "test":
        #    return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False

        self.embeddings = torch.load(self.embeddings_path)
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        logger.info(f"Loaded faiss index from {self.faiss_index_path}")

        # embed novel entities and add to faiss index
        if self.novel_entity_idx_path is not None:
            # embed novel entities
            novel_base_path = ".".join(self.novel_entity_idx_path.split('.')[:-1])
            if self.description_based:
                novel_embedding_path = novel_base_path + "_descp.t7"
                updated_faiss_index = novel_base_path + "_updated_index_descp.faiss"
            elif self.mention_based:
                novel_embedding_path = novel_base_path + ".t7"
                updated_faiss_index = novel_base_path + "_updated_index_mention.faiss"
            else:   
                novel_embedding_path = novel_base_path + "_clust.t7"
                updated_faiss_index = novel_base_path + "_updated_index_clust.faiss"
            novel_base_path += ".jsonl"

            if not os.path.isfile(novel_embedding_path):
                logger.info(f"Embedding novel entities {novel_embedding_path}")
                if self.description_based:
                    entity_encodings = embed(self.path_blink_model, novel_base_path)
                else:
                    entity_encodings = embed_cluster(self.path_blink_model, novel_base_path)

                logger.info("Saved novel entity embeddings to %s", novel_embedding_path)
                torch.save(entity_encodings, novel_embedding_path)

            novel_entities_embeddings = torch.load(novel_embedding_path)
            logger.info(f"Loaded novel entity embeddings from {novel_embedding_path}")
            if not os.path.isfile(updated_faiss_index):

                '''phi = 0
                for i, item in enumerate(self.embeddings):
                    norms = (item ** 2).sum()
                    phi = max(phi, norms)'''
                # phi=406.7714
                # norms=169.0674
                buffer_size = 5000
                bs = int(buffer_size)
                num_indexed = 0
                n = len(novel_entities_embeddings)
                for i in range(0, n, bs):
                    vectors = novel_entities_embeddings[i: i + bs]
                    self.faiss_index.add(np.array(vectors))
                    num_indexed += bs
                    logger.info("data indexed %d", num_indexed)
                logger.info(f"Saved updated index to {updated_faiss_index}")
                faiss.write_index(self.faiss_index, updated_faiss_index)
            else:
                self.faiss_index = faiss.read_index(updated_faiss_index)
                logger.info(f"Loaded faiss index from {updated_faiss_index}")
            self.embeddings = torch.cat((self.embeddings, novel_entities_embeddings), 0)
            logger.info(f"Number of entities {len(self.embeddings)}")


        self.encoder = hydra.utils.instantiate(
            self.encoder_conf,
        )
        self.span_encoder = SpanEncoder()
        self.mention_encoder = MentionScoresHead()
        self.el_encoder = ClassificationHead()
        if self.train_saliency:
            self.saliency_encoder = SaliencyClassificationHead()
        if self.md_threshold:
            self.novel_el_predictions = []
            self.novel_el_buffer = 10
            self.buffer_md_idx = 0
        if self.train_unknown_classifier:
            self.unknown_classifier = UnknownClassificationHead()

        if self.load_from_checkpoint is not None:
            logger.info(f"Load encoders state from {self.load_from_checkpoint}")
            with open(self.load_from_checkpoint, "rb") as f:
                checkpoint = torch.load(f, map_location=torch.device("cpu"))

            encoder_state = self._get_encoder_state(checkpoint, "encoder")
            self.encoder.load_state_dict(encoder_state)

            span_encoder_state = self._get_encoder_state(checkpoint, "span_encoder")
            self.span_encoder.load_state_dict(span_encoder_state)

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

        # obtain positive entities representations
        entities_repr = self.embeddings[flat_entities_ids.to("cpu")].to(device)

        # compute scores for positive entities
        pos_scores = self.sim_score(flat_mentions_repr, entities_repr)


        # retrieve candidates indices
        query_vectors = flat_mentions_repr.detach().cpu().numpy()
        _, neg_cand_indices = self.faiss_index.search(
            query_vectors, self.n_retrieve_candidates
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
        salient_entities,
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
            saliency_loss: saliency loss if self.train_saliency is True else None
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

        query_vectors = flat_mentions_repr.detach().cpu().numpy()
        cand_scores, cand_indices = self.faiss_index.search(
            query_vectors, 1
        )
        cand_scores = torch.from_numpy(cand_scores)
        cand_indices = torch.from_numpy(cand_indices)

        # iterate over predicted and gold mentions to create targets for
        # predicted mentions
        targets = []
        saliency_targets = []
        for (
            e_mention_offsets,
            e_mention_lengths,
            e_gold_mention_offsets,
            e_gold_mention_lengths,
            e_entities,
            e_salient_entities,
        ) in zip(
            mention_offsets.detach().cpu().tolist(),
            mention_lengths.detach().cpu().tolist(),
            gold_mention_offsets.cpu().tolist(),
            gold_mention_lengths.cpu().tolist(),
            entities_ids.cpu().tolist(),
            salient_entities,
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

            if self.train_saliency:
                e_gold_saliency_targets = {
                    (offset, length): ent
                    for offset, length, ent in zip(
                        e_gold_mention_offsets, e_gold_mention_lengths, e_entities
                    )
                    if ent in e_salient_entities
                }
                e_saliency_targets = [
                    e_gold_saliency_targets.get((offset, length), -1)
                    for offset, length in zip(e_mention_offsets, e_mention_lengths)
                ]
                saliency_targets.append(e_saliency_targets)

        targets = torch.tensor(targets, device=device)
        flat_targets = targets[mention_lengths != 0][flat_mentions_scores > 0]
        md_scores = flat_mentions_scores[flat_mentions_scores > 0].unsqueeze(-1)
        flat_entities_repr = self.embeddings[cand_indices.squeeze(1)].to(device)
        cand_scores = cand_scores.to(device)
        cand_indices = cand_indices.to(device)

        predictions = self.el_encoder(
            flat_mentions_repr, flat_entities_repr, md_scores, cand_scores
        ).squeeze(1)

        binary_targets = (flat_targets == cand_indices.squeeze(1)).double()
        el_loss = self.el_loss(predictions, binary_targets)

        saliency_loss = None
        if self.train_saliency:
            saliency_targets = torch.tensor(saliency_targets, device=device)
            flat_saliency_targets = saliency_targets[mention_lengths != 0][
                flat_mentions_scores > 0
            ]

            cls_tokens_repr = text_encodings[:, 0, :]
            # repeat cls token for each mention
            flat_cls_tokens_repr = torch.repeat_interleave(
                cls_tokens_repr, (mention_lengths != 0).sum(axis=1), dim=0
            )
            # filter mentions with scores <= 0
            flat_cls_tokens_repr = flat_cls_tokens_repr[flat_mentions_scores > 0]

            saliency_predictions = self.saliency_encoder(
                flat_cls_tokens_repr,
                flat_mentions_repr,
                flat_entities_repr,
                md_scores,
                cand_scores,
            ).squeeze(1)

            binary_saliency_targets = (
                flat_saliency_targets == cand_indices.squeeze(1)
            ).double()
            saliency_loss = self.saliency_loss(
                saliency_predictions, binary_saliency_targets
            )

        return el_loss, saliency_loss

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
        salient_entities = batch["salient_entities"]  # bs

        # mention representations (bs x max_mentions_num x embedding_dim)
        text_encodings, mentions_repr = self(
            text_inputs, text_pad_mask, gold_mention_offsets, gold_mention_lengths
        )

        if self.train_unknown_classifier:
            loss = self._unknown_classification_training_step(
                mentions_repr,
                gold_mention_offsets,
                gold_mention_lengths,
                entities_ids,
                unknown_labels
            )
            self.log("train_loss", loss, prog_bar=True)
            return loss
            
        dis_loss = self._disambiguation_training_step(
            mentions_repr,
            gold_mention_offsets,
            gold_mention_lengths,
            entities_ids,
        )
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
            loss += md_loss

            if self.train_el_classifier:
                el_loss, saliency_loss = self._el_training_step(
                    text_encodings,
                    mention_logits,
                    mention_bounds,
                    gold_mention_offsets,
                    gold_mention_lengths,
                    entities_ids,
                    tokens_mapping,
                    salient_entities,
                )
                self.log("el_loss", el_loss, prog_bar=True)
                loss += el_loss

                if self.train_saliency:
                    self.log("sal_loss", saliency_loss, prog_bar=True)
                    loss += saliency_loss

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
        entities_repr = self.embeddings[flat_entities_ids.to("cpu")].to(device)

        # compute scores for positive entities
        pos_scores = self.sim_score(flat_mentions_repr, entities_repr)

        # candidates to retrieve
        n_retrieve_candidates = max(self.eval_compute_recall_at)

        # retrieve negative candidates ids and scores
        query_vectors = flat_mentions_repr.detach().cpu().numpy()
        neg_cand_scores, neg_cand_indices = self.faiss_index.search(
            query_vectors, n_retrieve_candidates
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
        for k in self.eval_compute_recall_at:
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
        salient_entities,
        batch_idx,
        data_example_ids
    ):
        device = text_inputs.device

        # encode query and contexts
        _, last_layer = self.encoder(text_inputs)
        text_encodings = last_layer

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
        query_vectors = flat_mentions_repr.detach().cpu().numpy()
        cand_scores, cand_indices = self.faiss_index.search(
            query_vectors, 1
        )

        if self.train_el_classifier:
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

            if self.train_saliency:
                cls_tokens_repr = text_encodings[:, 0, :]
                flat_cls_tokens_repr = torch.repeat_interleave(
                    cls_tokens_repr, (mention_lengths != 0).sum(axis=1), dim=0
                )

                saliency_scores = self.saliency_encoder(
                    flat_cls_tokens_repr,
                    flat_mentions_repr,
                    flat_entities_repr,
                    flat_mentions_scores,
                    cand_scores,
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

        saliency_targets = []
        if self.train_saliency:
            for example_targets, example_salient_entities in zip(
                el_targets, salient_entities
            ):
                saliency_targets.append(
                    {
                        pos: ent_id
                        for pos, ent_id in example_targets.items()
                        if ent_id in example_salient_entities
                    }
                )

        mention_offsets = mention_offsets.detach().cpu().tolist()
        mention_lengths = mention_lengths.detach().cpu().tolist()
        mentions_scores = mentions_scores.detach().cpu().tolist()

        el_predictions = []
        saliency_predictions = []
        cand_idx = 0
        for offsets, lengths, md_scores in zip(
            mention_offsets, mention_lengths, mentions_scores
        ):
            example_predictions = {}
            novel_predictions = {}
            example_saliency_predictions = {}
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
                            if self.save_novel_md_path:
                                novel_predictions[(offset, length, md_score, el_scores[cand_idx], cand_scores[cand_idx], flat_mentions_scores[cand_idx])] = cand_indices[
                                cand_idx
                            ][0]
                        if (
                            self.train_saliency
                            and saliency_scores[cand_idx] >= self.saliency_threshold
                        ):
                            example_saliency_predictions[
                                (offset, length)
                            ] = cand_indices[cand_idx][0]
                    cand_idx += 1
            el_predictions.append(example_predictions)
            novel_predictions = self.all_gather(novel_predictions)
            self.novel_el_predictions.append([data_example_ids, novel_predictions])
            saliency_predictions.append(example_saliency_predictions)


        return el_targets, el_predictions, saliency_targets, saliency_predictions

    def _joint_novel_step(
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
        query_vectors = flat_mentions_repr.detach().cpu().numpy()
        cand_scores, cand_indices = self.faiss_index.search(
            query_vectors, 1
        )

        if self.train_el_classifier:
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

        cand_idx = 0
        novel_predictions = []
        rank = self.global_rank
        for offsets, lengths, md_scores in zip(
            mention_offsets, mention_lengths, mentions_scores
        ):
            novel_prediction_example = []
            for offset, length, md_score in zip(offsets, lengths, md_scores):
                if length != 0:
                    if md_score >= self.md_threshold:
                        if (
                            not self.train_el_classifier
                            or el_scores[cand_idx] >= self.el_threshold
                        ):
                            # logging  detection features
                            # el_scores = Rejection score
                            # cand_scores = disambiguation score
                            # md_score and flat_mentions_scores = mention detection score
                            # flat_mentions_repr
                            # flat_entities_repr
                            novel_prediction_example.append([offset, length, float(el_scores[cand_idx]), float(cand_scores[cand_idx][0]), float(md_score), flat_mentions_repr[cand_idx], flat_entities_repr[cand_idx], int(cand_indices[
                            cand_idx])])
                    cand_idx += 1
            novel_predictions.append(novel_prediction_example)
        
        self.novel_el_predictions.append([rank, novel_predictions])


    def _eval_step(self, batch, batch_idx):
        text_inputs = batch["input_ids"]  # bs x mention_len
        text_pad_mask = batch["attention_mask"]
        mention_offsets = batch["mention_offsets"]  # bs x max_mentions_num
        mention_lengths = batch["mention_lengths"]  # bs x max_mentions_num
        entities_ids = batch["entities"]  # bs x max_mentions_num
        tokens_mapping = batch["tokens_mapping"]
        salient_entities = batch["salient_entities"]
        data_example_ids = batch["data_example_id"]

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
        
        if self.save_novel_md_path:

            return self._joint_novel_step(
            text_inputs,
            text_pad_mask,
            mention_offsets,
            mention_lengths,
            entities_ids,
            tokens_mapping,
        )
        return self._joint_eval_step(
            text_inputs,
            text_pad_mask,
            mention_offsets,
            mention_lengths,
            entities_ids,
            tokens_mapping,
            salient_entities,
            batch_idx,
            data_example_ids
        )

    def _compute_disambiguation_metrics(self, outputs, log_prefix):
        total_recalls = [0] * len(self.eval_compute_recall_at)
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

        for idx, recall_at in enumerate(self.eval_compute_recall_at):
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
        saliency_targets = []
        saliency_predictions = []

        for (
            batch_el_targets,
            batch_el_predictions,
            batch_saliency_targets,
            batch_saliency_predictions,
        ) in outputs:
            el_targets.extend(batch_el_targets)
            el_predictions.extend(batch_el_predictions)
            saliency_targets.extend(batch_saliency_targets)
            saliency_predictions.extend(batch_saliency_predictions)

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

        if self.train_saliency:
            saliency_metrics = self.calculate_classification_metrics(
                saliency_targets, saliency_predictions
            )

            metrics.update(
                {
                    log_prefix + "_e2e_f1": saliency_metrics.f1,
                    log_prefix + "_e2e_precision": saliency_metrics.precision,
                    log_prefix + "_e2e_recall": saliency_metrics.recall,
                    log_prefix + "_e2e_support": saliency_metrics.support,
                    log_prefix + "_e2e_tp": saliency_metrics.tp,
                    log_prefix + "_e2e_fp": saliency_metrics.fp,
                    log_prefix + "_e2e_fn": saliency_metrics.fn,
                    log_prefix + "_e2e_boe_f1": saliency_metrics.boe_f1,
                    log_prefix + "_e2e_boe_precision": saliency_metrics.boe_precision,
                    log_prefix + "_e2e_boe_recall": saliency_metrics.boe_recall,
                    log_prefix + "_e2e_boe_support": saliency_metrics.boe_support,
                    log_prefix + "_e2e_boe_tp": saliency_metrics.boe_tp,
                    log_prefix + "_e2e_boe_fp": saliency_metrics.boe_fp,
                    log_prefix + "_e2e_boe_fn": saliency_metrics.boe_fn,
                }
            )

        return metrics

    def _eval_epoch_end(self, outputs, log_prefix="valid"):
        if self.save_novel_md_path is not None:
            torch.save(self.novel_el_predictions, self.save_novel_md_path + "_" + f"{self.buffer_md_idx:02d}" + ".t7")
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
        if self.save_novel_md_path:
            if len(self.novel_el_predictions)>=self.novel_el_buffer: 
                torch.save(self.novel_el_predictions, self.save_novel_md_path + "_" + f"{self.buffer_md_idx:02d}" + ".t7")
                self.novel_el_predictions = []
                self.buffer_md_idx += 1
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        if self.save_novel_md_path:
            if len(self.novel_el_predictions)>=self.novel_el_buffer: 
                torch.save(self.novel_el_predictions, self.save_novel_md_path + "_" + f"{self.buffer_md_idx:02d}" + ".t7")
                self.novel_el_predictions = []
                self.buffer_md_idx += 1
        
        self._eval_epoch_end(test_outputs, "test")


#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple, Union

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

from bela.datamodule.entity_encoder import embed
from bela.task.joint_el_heads import (MentionScoresHead, SpanEncoder)

logger = logging.getLogger(__name__)

class ClusterTask(LightningModule):
    def __init__(
        self,
        transform: TransformConf,
        model: ModelConf,
        datamodule: DataModuleConf,
        optim: OptimConf,
        load_from_checkpoint: Optional[str] = None,
        save_embeddings_path: Optional[str] = None,
    ):
        super().__init__()

        # encoder setup
        self.encoder_conf = model
        self.load_from_checkpoint = load_from_checkpoint
        self.save_embeddings_path = save_embeddings_path
        self.all_tensors = []
        self.buffer = 500
        self.buffer_idx = 0

    @staticmethod
    def _get_encoder_state(state, encoder_name):
        encoder_state = OrderedDict()
        for key, value in state["state_dict"].items():
            if key.startswith(encoder_name):
                encoder_state[key[len(encoder_name) + 1 :]] = value
        return encoder_state

    def setup(self, stage: str):
        self.call_configure_sharded_model_hook = False

        self.encoder = hydra.utils.instantiate(
            self.encoder_conf,
        )
        self.span_encoder = SpanEncoder()
        self.mention_encoder = MentionScoresHead()

        if self.load_from_checkpoint is not None:
            logger.info(f"Load encoders state from {self.load_from_checkpoint}")
            with open(self.load_from_checkpoint, "rb") as f:
                checkpoint = torch.load(f, map_location=torch.device("cpu"))

            encoder_state = self._get_encoder_state(checkpoint, "encoder")
            self.encoder.load_state_dict(encoder_state)

            span_encoder_state = self._get_encoder_state(checkpoint, "span_encoder")
            self.span_encoder.load_state_dict(span_encoder_state)

    def _eval_step(self, batch, batch_idx):
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

        device = mentions_repr.get_device()

        # flat mentions and entities indices (mentions_num x embedding_dim)
        flat_mentions_repr = mentions_repr[gold_mention_lengths != 0]
        flat_entities_ids = entities_ids[gold_mention_lengths != 0]

        ent_mentions_repr = torch.cat([torch.unsqueeze(flat_entities_ids, 1), flat_mentions_repr], 1)
        self.all_tensors.append(ent_mentions_repr)

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

    def test_step(self, batch, batch_idx):
        if len(self.all_tensors)>=self.buffer:
            torch.save(self.all_tensors, self.save_embeddings_path + "_" + f"{self.buffer_idx:02d}" + ".t7")
            self.all_tensors = []
            self.buffer_idx += 1
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        torch.save(self.all_tensors, self.save_embeddings_path + "_" + f"{self.buffer_idx:02d}" + ".t7")

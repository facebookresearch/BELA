#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import mmap
from typing import List

import torch
from pytorch_lightning import LightningDataModule

from bela.transforms.joint_el_transform import JointELTransform

logger = logging.getLogger()


def get_seq_lengths(batch: List[List[int]]):
    return [len(example) for example in batch]


class EntityCatalogue:
    def __init__(self, idx_path):
        logger.info(f"Reading entity catalogue index {idx_path}")
        self.idx = {}
        with open(idx_path, "rt") as fd:
            for idx, line in enumerate(fd):
                ent_id = line.strip()
                self.idx[ent_id] = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, entity_id):
        ent_index = self.idx[entity_id]
        return ent_index

    def __contains__(self, entity_id):
        return entity_id in self.idx


class ElMatchaDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset for EL in Matcha format
    Each example in this dataset contains several mentions.
    We laso filter out mentions, that are not present in entity catalogue
    """

    def __init__(self, path, ent_catalogue):
        self.ent_catalogue = ent_catalogue

        self.file = open(path, mode="rt")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.offsets = []
        self.count = 0

        logger.info(f"Build mmap index for {path}")
        line = self.mm.readline()
        offset = 0
        while line:
            self.offsets.append(offset)
            self.count += 1
            offset = self.mm.tell()
            line = self.mm.readline()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        offset = self.offsets[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        example = json.loads(line)
        gt_entities = []
        for gt_entity in example["gt_entities"]:
            offset, length, entity, ent_type = gt_entity[:4]
            # if ent_type != "wiki":
            #     continue
            if entity in self.ent_catalogue:
                gt_entities.append((offset, length, self.ent_catalogue[entity]))

        gt_entities = sorted(gt_entities)

        # saliency data
        salient_entities = {}
        if "all_entities" in example and "gt_scores" in example:
            assert len(example["all_entities"]) == len(example["gt_scores"])
            salient_entities = {
                self.ent_catalogue[entity]
                for entity, score in zip(example["all_entities"], example["gt_scores"])
                if entity in self.ent_catalogue and score == 1
            }

        # blink predicts
        blink_predicts = None
        blink_scores = None
        if "blink_predicts" in example:
            blink_predicts = []
            blink_scores = []
            for predict, scores in zip(
                example["blink_predicts"], example["blink_scores"]
            ):
                candidates = []
                candidates_scores = []
                for candidate, score in zip(predict, scores):
                    if candidate in self.ent_catalogue:
                        candidates.append(self.ent_catalogue[candidate])
                        candidates_scores.append(score)
                blink_predicts.append(candidates)
                blink_scores.append(candidates_scores)

        # MD model predicts
        md_pred_offsets = example.get("md_pred_offsets")
        md_pred_lengths = example.get("md_pred_lengths")
        md_pred_scores = example.get("md_pred_scores")

        result = {
            "text": example["text"],
            "gt_entities": gt_entities,
            "salient_entities": salient_entities,
            "blink_predicts": blink_predicts,
            "blink_scores": blink_scores,
            "md_pred_offsets": md_pred_offsets,
            "md_pred_lengths": md_pred_lengths,
            "md_pred_scores": md_pred_scores,
        }

        return result


class JointELDataModule(LightningDataModule):
    """
    Read data from EL datatset and prepare mention/entity pairs tensors
    """

    def __init__(
        self,
        transform: JointELTransform,
        # Dataset args
        train_path: str,
        val_path: str,
        test_path: str,
        ent_catalogue_idx_path: str,
        batch_size: int = 2,
        drop_last: bool = False,  # drop last batch if len(dataset) not multiple of batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_workers = num_workers
        self.transform = transform

        self.ent_catalogue = EntityCatalogue(ent_catalogue_idx_path)

        self.datasets = {
            "train": ElMatchaDataset(
                train_path,
                self.ent_catalogue,
            ),
            "valid": ElMatchaDataset(val_path, self.ent_catalogue),
            "test": ElMatchaDataset(test_path, self.ent_catalogue),
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_train,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["valid"],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
        )

    def collate_eval(self, batch):
        return self.collate(batch, False)

    def collate_train(self, batch):
        return self.collate(batch, True)

    def collate(self, batch, is_train):
        """
        Input:
            batch: List[Example]

            Example fields:
               - "text": List[str] - post tokens
               - "gt_entities": List[Tuple[int, int, int]] - GT entities in text,
                    offset, length, entity id
               - "blink_predicts": List[List[int]] - list of entity ids for each MD prediction
               - "blink_scores": List[List[float]] - list of BLINK scores
               - "md_pred_offsets": List[int] - mention offsets predicted by MD
               - "md_pred_lengths": List[int] - mention lengths
               - "md_pred_scores": List[float] - MD scores
               - "salient_entities": Set[int] - salient GT entities ids
        """
        texts = []
        offsets = []
        lengths = []
        entities = []
        salient_entities = []

        for example in batch:
            texts.append(example["text"])
            example_offsets = []
            example_lengths = []
            example_entities = []
            for offset, length, entity_id in example["gt_entities"]:
                example_offsets.append(offset)
                example_lengths.append(length)
                example_entities.append(entity_id)
            offsets.append(example_offsets)
            lengths.append(example_lengths)
            entities.append(example_entities)

            salient_entities.append(example["salient_entities"])

        model_inputs = self.transform(
            {
                "texts": texts,
                "mention_offsets": offsets,
                "mention_lengths": lengths,
                "entities": entities,
            }
        )

        collate_output = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "mention_offsets": model_inputs["mention_offsets"],
            "mention_lengths": model_inputs["mention_lengths"],
            "entities": model_inputs["entities"],
            "tokens_mapping": model_inputs["tokens_mapping"],
            "salient_entities": salient_entities,
        }

        if "sp_tokens_boundaries" in model_inputs:
            collate_output["sp_tokens_boundaries"] = model_inputs[
                "sp_tokens_boundaries"
            ]

        return collate_output

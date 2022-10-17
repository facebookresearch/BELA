#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import mmap
from typing import List

import torch
from pytorch_lightning import LightningDataModule

from mblink.utils.utils import (
    EntityCatalogueType,
    EntityCatalogue,
    ElDatasetType,
    MultilangEntityCatalogue,
    NegativesStrategy,
    order_entities,
)

from mblink.transforms.blink_transform import BlinkTransform

logger = logging.getLogger()


class ElMatchaDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset for EL in Matcha format
    Each example in this dataset contains several mentions.
    We laso filter out mentions, that are not present in entity catalogue
    """

    def __init__(
        self, path, ent_catalogue, negatives=False, negatives_strategy="higher"
    ):
        self.ent_catalogue = ent_catalogue
        self.negatives = negatives
        self.negatives_strategy = NegativesStrategy(negatives_strategy)

        self.file = open(path, mode="r")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.offsets = []
        self.count = 0

        logger.info(f"Build mmap index for {path}")
        line = self.mm.readline()
        offset = 0
        while line:
            data = json.loads(line)
            for gt_ent_idx, gt_entity in enumerate(data["gt_entities"]):
                ent = gt_entity[2]
                if ent in self.ent_catalogue:
                    self.offsets.append((offset, gt_ent_idx))
                    self.count += 1
            offset = self.mm.tell()
            line = self.mm.readline()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        offset, gt_ent_idx = self.offsets[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        data = json.loads(line)
        offset, length, gt_entity = data["gt_entities"][gt_ent_idx][:3]
        entity_index, entity_tokens = self.ent_catalogue[gt_entity]

        result = {
            "context_left": " ".join(data["text"][:offset]),
            "mention": " ".join(data["text"][offset : offset + length]),
            "context_right": " ".join(data["text"][offset + length :]),
            "entity_id": gt_entity,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
        }

        if self.negatives:
            assert "gt_hard_negatives" in data

            neg_entities_ids = []
            neg_entities_indexes = []
            neg_entities_tokens = []
            for ent in data["gt_hard_negatives"][gt_ent_idx]:
                if (
                    ent == gt_entity
                    and self.negatives_strategy == NegativesStrategy.HIGHER
                ):
                    break
                entity_index, entity_tokens = self.ent_catalogue[ent]
                neg_entities_ids.append(ent)
                neg_entities_indexes.append(entity_index)
                neg_entities_tokens.append(entity_tokens)

            result["neg_entities_ids"] = neg_entities_ids
            result["neg_entities_indexes"] = neg_entities_indexes
            result["neg_entities_tokens"] = neg_entities_tokens

        return result


class ElBlinkDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset for EL in BLINK format
    Each example in this dataset contains one mention.
    We laso filter out mentions, that are not present in entity catalogue
    """

    def __init__(
        self, path, ent_catalogue, negatives=False, negatives_strategy="higher"
    ):
        self.ent_catalogue = ent_catalogue

        self.file = open(path, mode="r")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.offsets = []
        self.count = 0

        logger.info(f"Build mmap index for {path}")
        line = self.mm.readline()
        offset = 0
        while line:
            data = json.loads(line)
            if data["entity_id"] in self.ent_catalogue:
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
        data = json.loads(line)

        entity_id = data["entity_id"]
        entity_index, entity_tokens = self.ent_catalogue[entity_id]

        return {
            "context_left": data["context_left"],
            "mention": data["mention"],
            "context_right": data["context_right"],
            "entity_id": entity_id,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
        }


class ElBiEncoderDataModule(LightningDataModule):
    """
    Read data from EL datatset and prepare mention/entity pairs tensors
    """

    def __init__(
        self,
        transform: BlinkTransform,
        # Dataset args
        train_path: str,
        val_path: str,
        test_path: str,
        ent_catalogue_path: str,
        ent_catalogue_idx_path: str,
        dataset_type: str = "matcha",
        ent_catalogue_type: str = "simple",
        batch_size: int = 2,
        negatives: bool = False,
        negatives_strategy: str = "higher",
        max_negative_entities_in_batch: int = 0,
        drop_last: bool = False,  # drop last batch if len(dataset) not multiple of batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.negatives = negatives
        self.max_negative_entities_in_batch = max_negative_entities_in_batch
        self.drop_last = drop_last

        self.num_workers = num_workers
        self.transform = transform

        ent_catalogue_type = EntityCatalogueType(ent_catalogue_type)
        if ent_catalogue_type == EntityCatalogueType.SIMPLE:
            self.ent_catalogue = EntityCatalogue(
                ent_catalogue_path, ent_catalogue_idx_path
            )
        elif ent_catalogue_type == EntityCatalogueType.MULTI:
            self.ent_catalogue = MultilangEntityCatalogue(
                ent_catalogue_path, ent_catalogue_idx_path
            )
        else:
            raise NotImplementedError(
                f"Unknown ent_catalogue_type {ent_catalogue_type}"
            )

        dataset_type = ElDatasetType(dataset_type)
        if dataset_type == ElDatasetType.MATCHA:
            dataset_cls = ElMatchaDataset
        elif dataset_type == ElDatasetType.BLINK:
            dataset_cls = ElBlinkDataset
        else:
            raise NotImplementedError(f"Unknown dataset_type {dataset_type}")

        self.datasets = {
            "train": dataset_cls(
                train_path,
                self.ent_catalogue,
                negatives=negatives,
                negatives_strategy=negatives_strategy,
            ),
            "valid": dataset_cls(val_path, self.ent_catalogue),
            "test": dataset_cls(test_path, self.ent_catalogue),
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
        Prepare mention, entity tokens and target tensors
        """
        if self.negatives and is_train:
            (
                left_context,
                mention,
                right_context,
                _,
                entity_ids,
                entity_token_ids,
                _,
                neg_entities_ids,
                neg_entities_tokens,
            ) = zip(*[item.values() for item in batch])
        else:
            left_context, mention, right_context, _, entity_ids, entity_token_ids = zip(
                *[item.values() for item in batch]
            )
            neg_entities_ids = None
            neg_entities_tokens = None

        entity_token_ids, entity_ids, targets = order_entities(
            entity_token_ids,
            entity_ids,
            neg_entities_ids,
            neg_entities_tokens,
            self.max_negative_entities_in_batch,
        )
        pad_length = (
            len(batch) + self.max_negative_entities_in_batch - len(entity_token_ids)
        )
        entity_tensor_mask = [1] * len(entity_token_ids) + [0] * pad_length
        entity_token_ids += [
            [self.transform.bos_idx, self.transform.eos_idx]
        ] * pad_length
        entity_ids += [0] * pad_length

        mention_tensors, entity_tensors = self.transform(
            {
                "left_context": left_context,
                "mention": mention,
                "right_context": right_context,
                "token_ids": entity_token_ids,
            }
        )

        entity_ids = torch.tensor(entity_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        entity_tensor_mask = torch.tensor(entity_tensor_mask, dtype=torch.long)

        return {
            "mentions": mention_tensors,
            "entities": entity_tensors,
            "entity_ids": entity_ids,
            "targets": targets,
            "entity_tensor_mask": entity_tensor_mask,
        }

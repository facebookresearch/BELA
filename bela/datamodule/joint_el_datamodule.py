#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import mmap
from typing import List, Optional

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

    def __init__(
        self,
        path,
        ent_catalogue,
        use_raw_text,
        use_augmentation=False,
        augmentation_frequency=0.1,
    ):
        self.ent_catalogue = ent_catalogue
        self.use_raw_text = use_raw_text
        self.use_augmentation = use_augmentation
        self.augmentation_frequency = augmentation_frequency

        logger.info(f"Downloading file {path}")
        # TODO: Maybe we should lazily load the file to speed up datamodule instanciation (e.g. in model_eval.py)
        self.file = open(path, mode="r")
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

    def _add_char_offsets(self, tokens, gt_entities):
        offsets = []
        token_lengths = []
        current_pos = 0
        for token in tokens:
            offsets.append(current_pos)
            token_lengths.append(len(token))
            current_pos += len(token) + 1

        updated_gt_entities = []
        for gt_entity in gt_entities:
            offset, length, entity, ent_type = gt_entity[:4]
            char_offset = offsets[offset]
            char_length = (
                sum(token_lengths[offset + idx] for idx in range(length)) + length - 1
            )
            updated_gt_entities.append(
                (offset, length, entity, ent_type, char_offset, char_length)
            )

        return updated_gt_entities

    def __getitem__(self, index):
        offset = self.offsets[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        example = json.loads(line)
        gt_entities = []

        if self.use_raw_text and "original_text" not in example:
            example["gt_entities"] = self._add_char_offsets(
                example["text"], example["gt_entities"]
            )
            example["original_text"] = " ".join(example["text"])

        for gt_entity in example["gt_entities"]:
            if self.use_raw_text:
                _, _, entity, ent_type, offset, length = gt_entity[:6]
            else:
                offset, length, entity, ent_type = gt_entity[:4]
            if ent_type != "wiki":
                continue
            if entity in self.ent_catalogue:
                gt_entities.append((offset, length, self.ent_catalogue[entity]))

        gt_entities = sorted(gt_entities)

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
            "data_example_id": example.get("document_id") or example.get("data_example_id", ""),
            "text": example["original_text"] if self.use_raw_text else example["text"],
            "gt_entities": gt_entities,
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
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        drop_last: bool = False,  # drop last batch if len(dataset) not multiple of batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        use_raw_text: bool = True,
        use_augmentation: bool = False,
        augmentation_frequency: float = 0.1,
        shuffle: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.drop_last = drop_last

        self.num_workers = num_workers
        self.transform = transform

        self.ent_catalogue = EntityCatalogue(ent_catalogue_idx_path)

        self.shuffle = shuffle

        self.datasets = {
            "train": ElMatchaDataset(
                train_path,
                self.ent_catalogue,
                use_raw_text=use_raw_text,
                use_augmentation=use_augmentation,
                augmentation_frequency=augmentation_frequency,
            ) if train_path else None,
            "valid": ElMatchaDataset(
                val_path,
                self.ent_catalogue,
                use_raw_text=use_raw_text,
            ) if val_path else None,
            "test": ElMatchaDataset(
                test_path,
                self.ent_catalogue,
                use_raw_text=use_raw_text,
            ) if test_path else None,
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_train,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["valid"],
            shuffle=False,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
            drop_last=self.drop_last,
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
        """
        data_example_ids = []
        texts = []
        offsets = []
        lengths = []
        entities = []

        for example in batch:
            data_example_ids.append(example["data_example_id"])
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

        model_inputs = self.transform(
            {
                "texts": texts,
                "mention_offsets": offsets,
                "mention_lengths": lengths,
                "entities": entities,
            }
        )

        collate_output = {
            "data_example_ids": data_example_ids,
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "mention_offsets": model_inputs["mention_offsets"],
            "mention_lengths": model_inputs["mention_lengths"],
            "entities": model_inputs["entities"],
            "tokens_mapping": model_inputs["tokens_mapping"],
        }

        if "sp_tokens_boundaries" in model_inputs:
            collate_output["sp_tokens_boundaries"] = model_inputs[
                "sp_tokens_boundaries"
            ]

        return collate_output

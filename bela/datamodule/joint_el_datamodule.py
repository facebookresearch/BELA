#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import mmap
from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule

from bela.transforms.joint_el_transform import JointELTransform
from pytorch_lightning import seed_everything
seed_everything(1)

logger = logging.getLogger()


def get_seq_lengths(batch: List[List[int]]):
    return [len(example) for example in batch]


class EntityCatalogue:
    def __init__(self, idx_path, novel_entity_idx_path, reverse=False):
        logger.info(f"Reading entity catalogue index {idx_path}")
        self.idx = {}
        self.mapping = {}
        with open(idx_path, "rt") as fd:
            for idx, line in enumerate(fd):
                ent_id = line.strip()
                self.idx[ent_id] = idx
                self.mapping[ent_id] = [idx]

        logger.info(f"Reading novel entity catalogue index {novel_entity_idx_path}")
        if novel_entity_idx_path is not None:
            with open(novel_entity_idx_path, "r") as f:
                for line_ in f:
                    idx += 1
                    try:
                        line = json.loads(line_)
                        line = line["entity"]
                    except:
                        line = line_.strip()
                    if line not in self.idx:
                        self.idx[line] = idx
                        self.mapping[line] = [idx]
                    else:
                        self.mapping[line].append(idx)
                    
        logger.info(f"Number of entities {len(self.idx)}")
        if reverse:
            self.idx_referse = {}
            for ent in self.idx:
                self.idx_referse[self.idx[ent]] = ent
            for ent in self.mapping:
                for idx in self.mapping[ent]:
                    self.idx_referse[idx] = ent

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
    We also filter out mentions, that are not present in entity catalogue
    """

    def __init__(self, path, ent_catalogue, time_stamp, ent_subset):
        self.ent_catalogue = ent_catalogue
        self.debug_file = open("debug.jsonl", 'w')
        self.file = open(path, mode="rt")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.offsets = []
        self.count = 0
        self.time_stamp = time_stamp
        self.ent_subset = ent_subset

        logger.info(f"Build mmap index for {path}")
        line = self.mm.readline()
        line = json.loads(line)
        offset = 0
        num = 0
        if self.time_stamp is not None:
            year_ref, month_ref = self.time_stamp.split('-')
            year_ref = int(year_ref)
            month_ref = int(month_ref)
        while line:
            keep = True
            if self.time_stamp is not None:
                year, month = line['time_stamp'].split('_')
                year = int(year)
                month = int(month)
                if year < year_ref:
                    keep = False
                elif year == year_ref and month < month_ref:
                    keep = False
            if keep:
                num +=1
                self.offsets.append(offset)
                self.count += 1
            offset = self.mm.tell()
            line = self.mm.readline()
            try:
                line = json.loads(line)
            except:
                pass

    def __len__(self):
        return self.count

    def __getitem__(self, index):

        offset = self.offsets[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        example = json.loads(line)
        self.debug_file.write(json.dumps(example))
        self.debug_file.write("\n")
        gt_entities = []
        for gt_entity in example["gt_entities"]:
            offset, length, entity, ent_type = gt_entity[:4]
            if entity in self.ent_catalogue:
                #if "novel" not in ent_type:
                #    continue
                '''if self.ent_subset is not None:
                    if entity in self.ent_subset:
                        continue'''
                gt_entities.append((offset, length, self.ent_catalogue[entity], ent_type))

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
            "data_example_id": example["data_example_id"],
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
        ent_catalogue_idx_path: str,
        # Dataset args
        test_path: str,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        novel_entity_idx_path: Optional[str] = None,
        time_stamp: Optional[str] = None,
        ent_subset: Optional[str] = None,
        batch_size: int = 2,
        analyze: bool = False,
        classify_unknown: bool = False,
        drop_last: bool = False,  # drop last batch if len(dataset) not multiple of batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.analyze = analyze
        self.classify_unknown = classify_unknown

        self.num_workers = num_workers
        self.transform = transform

        self.ent_catalogue = EntityCatalogue(ent_catalogue_idx_path, novel_entity_idx_path)
        if ent_subset is not None:
            self.ent_subset = EntityCatalogue(ent_subset, None, True)
        else:
            self.ent_subset = ent_subset
        self.time_stamp = time_stamp

        if train_path is not None:
            self.datasets = {
                "train": ElMatchaDataset(
                    train_path,
                    self.ent_catalogue, self.time_stamp, self.ent_subset
                ),
                "valid": ElMatchaDataset(val_path, self.ent_catalogue, self.time_stamp, self.ent_subset),
                "test": ElMatchaDataset(test_path, self.ent_catalogue, self.time_stamp, self.ent_subset),
            }
        else:
            self.datasets = {
                    "test": ElMatchaDataset(test_path, self.ent_catalogue, self.time_stamp, self.ent_subset),
                }


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_train,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["valid"],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
            drop_last=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
            drop_last=True,
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
        lengths_debug = []
        entities = []
        salient_entities = []
        metadata = []
        data_example_ids = []
        unknown_labels = []

        for example in batch:
            texts.append(example["text"])
            data_example_ids.append(example["data_example_id"])
            example_offsets = []
            example_lengths = []
            example_entities = []
            example_metadata = []
            example_unknown_labels = []
            for offset, length, entity_id, info in example["gt_entities"]:
                example_offsets.append(offset)
                example_lengths.append(length)
                example_entities.append(entity_id)
                if self.analyze:
                    example_metadata.append(info)
                if self.classify_unknown:
                    if entity_id not in self.ent_subset.idx_referse:
                        example_unknown_labels.append(0.0)
                    else:
                        example_unknown_labels.append(1.0)
            offsets.append(example_offsets)
            lengths.append(example_lengths)
            lengths_debug.extend(example_offsets)
            entities.append(example_entities)
            unknown_labels.append(example_unknown_labels)
        
            if self.analyze:
                metadata.append(example_metadata)

            salient_entities.append(example["salient_entities"])
        text_tensors, mentions_tensors = self.transform(
            {
                "texts": texts,
                "mention_offsets": offsets,
                "mention_lengths": lengths,
                "entities": entities,
            }
        )
        new_unknown_labels = []
        if len(unknown_labels)>0:
            for length, unknown_label in zip(mentions_tensors["mention_lengths"], unknown_labels):
                new_unknown_labels.extend(unknown_label[0:len(length[length!=0])])

        if self.analyze:
            return {
            "input_ids": text_tensors["input_ids"],
            "attention_mask": text_tensors["attention_mask"],
            "mention_offsets": mentions_tensors["mention_offsets"],
            "mention_lengths": mentions_tensors["mention_lengths"],
            "entities": mentions_tensors["entities"],
            "tokens_mapping": mentions_tensors["tokens_mapping"],
            "salient_entities": salient_entities,
            "text": texts,
            "metadata": metadata,
            "data_example_id": data_example_ids
        }
        return {
            "input_ids": text_tensors["input_ids"],
            "attention_mask": text_tensors["attention_mask"],
            "mention_offsets": mentions_tensors["mention_offsets"],
            "mention_lengths": mentions_tensors["mention_lengths"],
            "entities": mentions_tensors["entities"],
            "tokens_mapping": mentions_tensors["tokens_mapping"],
            "salient_entities": salient_entities,
            "unknown_labels": new_unknown_labels,
            "data_example_id": data_example_ids
        }

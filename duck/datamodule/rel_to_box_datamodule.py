from typing import Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pytorch_lightning import LightningDataModule

from duck.common.utils import load_json
from omegaconf import open_dict
import numpy as np


logger = logging.getLogger()


class RelToBoxDataset(Dataset):
    def __init__(
        self,
        rel_probs: Dict[str, float],
        rel_index: Dict[str, int],
        only_conditional: bool = False,
        only_marginal: bool = False,
        target_prob_threshold: Optional[float] = None
    ) -> None:
        super().__init__()
        assert not (only_conditional and only_marginal)
        self.idx = rel_index
        self.only_conditional = only_conditional
        self.only_marginal = only_marginal
        self.target_prob_threshold = target_prob_threshold
        self.rel_probs = self._filter_rel_probs(rel_probs)
        self.keys = np.array(list(self.rel_probs.keys()))
        self.values = np.array(list(self.rel_probs.values()), dtype=np.float32)
    
    def _filter_rel_probs(self, rel_probs):
        result = {}
        for k, prob in rel_probs.items():
            rels = k.split("|")
            if len(rels) == 1 and self.only_conditional:
                continue
            if len(rels) > 1 and self.only_marginal:
                continue
            if rels[0] in self.idx and (len(rels) == 1 or rels[1] in self.idx):
                if self.target_prob_threshold is None or prob > self.target_prob_threshold:
                    result[k] = prob
        return result
        
    def __getitem__(self, index):
        key = self.keys[index]
        # target_probability = self.rel_probs[key]
        target_probability = self.values[index]
        rels = key.split("|")
        rel_ids = [self.idx[r] + 1 for r in rels]
        if len(rel_ids) == 1:
            rel_ids.append(0)

        assert len(rel_ids) == 2

        return {
            "rel_ids": rel_ids,
            "rel_labels": rels,
            "target_probability": target_probability
        }
    
    def __len__(self):
        return len(self.rel_probs)


class RelToBoxDataModule(LightningDataModule):
    def __init__(
        self,
        rel_probs_path: str,
        rel_catalogue_idx_path: str,
        batch_size: int = 2,
        target_prob_threshold: Optional[float] = None,
        only_conditional: bool = False,
        only_marginal: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        logger.info(f"Reading relation probabilities: {rel_probs_path}")
        self.rel_probs = load_json(rel_probs_path)
        logger.info(f"Reading relation catalogue index: {rel_catalogue_idx_path}")
        self.idx = {}
        with open(rel_catalogue_idx_path, "rt") as fd:
            for idx, line in enumerate(fd):
                rel = line.strip()
                self.idx[rel] = idx
    
        self.train_dataset = RelToBoxDataset(
            self.rel_probs,
            self.idx,
            target_prob_threshold=target_prob_threshold,
            only_conditional=only_conditional,
            only_marginal=only_marginal
        )
        self.val_dataset = RelToBoxDataset(
            self.rel_probs,
            self.idx,
            target_prob_threshold=target_prob_threshold,
            only_conditional=only_conditional,
            only_marginal=only_marginal
        )
        self.num_rels = len(self.idx)
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
    
    def collate(self, batch):
        # rel_ids, rel_labels, target_probability = zip(*[item.values() for item in batch])
        rel_ids = [r["rel_ids"] for r in batch]
        rel_labels = [r["rel_labels"] for r in batch]
        target_probability = [r["target_probability"] for r in batch]
        rel_ids = torch.tensor(list(rel_ids), dtype=torch.long)
        target_probability = torch.tensor(list(target_probability), dtype=torch.float32)
        return {
            "rel_ids": rel_ids,
            "rel_labels": rel_labels,
            "target_probability": target_probability
        }

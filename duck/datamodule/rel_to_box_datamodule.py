from typing import Dict
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pytorch_lightning import LightningDataModule

from duck.common.utils import load_json
from omegaconf import open_dict


logger = logging.getLogger()


class RelToBoxDataset(Dataset):
    def __init__(
        self,
        rel_probs: Dict[str, float],
        rel_index: Dict[str, int]
    ) -> None:
        super().__init__()
        self.idx = rel_index
        self.rel_probs = self._filter_by_idx(rel_probs)
        self.keys = list(self.rel_probs.keys())
    
    def _filter_by_idx(self, rel_probs):
        return {
            k: v for k, v in rel_probs.items()
            if all(r in self.idx for r in k.split("|"))
        }
        
    def __getitem__(self, index):
        key = self.keys[index]
        target_probability = self.rel_probs[key]
        rels = key.split("|")
        rel_ids = [self.idx[r] + 1 for r in rels]
        if len(rel_ids) == 1:
            rel_ids.append(0)
            rels.append("unconditioned")

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
        batch_size: int= 2,
        shuffle: bool = False,
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
            self.idx
        )
        self.val_dataset = RelToBoxDataset(
            self.rel_probs,
            self.idx
        )
        self.num_rels = len(self.idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate
        )
    
    def collate(self, batch):
        rel_ids, rel_labels, target_probability = zip(*[item.values() for item in batch])
        rel_ids = torch.tensor(list(rel_ids), dtype=torch.long)
        target_probability = torch.tensor(list(target_probability))
        return {
            "rel_ids": rel_ids,
            "rel_labels": rel_labels,
            "target_probability": target_probability
        }

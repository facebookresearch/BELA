import json
import logging
import mmap
import random
from unittest.mock import NonCallableMagicMock

import torch
import h5py
from pytorch_lightning import LightningDataModule
from duck.common.utils import list_to_tensor, load_json, load_jsonl

from mblink.utils.utils import EntityCatalogue, order_entities
from mblink.transforms.blink_transform import BlinkTransform

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from tqdm import tqdm

logger = logging.getLogger()


class RelationCatalogue:
    def __init__(self, data_path, idx_path):
        self.data_file = h5py.File(data_path, "r")
        self.data = self.data_file["data"][:]

        logger.info(f"Reading relation catalogue index {idx_path}")
        self.idx = {}
        with open(idx_path, "rt") as fd:
            for idx, line in enumerate(fd):
                rel = line.strip()
                self.idx[rel] = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, relations):
        is_string = isinstance(relations, str)
        if is_string:
            relations = [relations]
        rel_indices = [self.idx[rel] for rel in relations]
        value = self.data[rel_indices].tolist()
        if self.data.dtype == int or self.data.dtype == np.int32:
            value = [v[1 : v[0] + 1] for v in value]
        if is_string:
            return rel_indices[0], value[0]
        return rel_indices, value

    def __contains__(self, relation):
        return relation in self.idx


class EdDuckDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        ent_catalogue: EntityCatalogue,
        rel_catalogue: RelationCatalogue,
        ent_to_rel: Dict[str, List[str]],
        neighbors: Dict[str, List[str]],
        num_neighbors_per_entity=1,
        stop_rels: Optional[Set[str]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.data = EdGenreDataset(path, ent_catalogue)
        self.rel_catalogue = rel_catalogue
        self.ent_to_rel = ent_to_rel
        self.neighbors_dataset = None
        self.neighbors_dataset = EntEmbDuckDataset(
            neighbors,
            ent_catalogue,
            rel_catalogue,
            ent_to_rel,
            stop_rels=stop_rels,
            relation_threshold=0,
            num_neighbors_per_entity=num_neighbors_per_entity
        )
        self.count = 0
        self.batch_size = None
        self.jump_to_batch = None
        if "jump_to_batch" in kwargs:
            self.jump_to_batch = kwargs["jump_to_batch"]
            self.batch_size = kwargs["batch_size"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.jump_to_batch is not None and \
            self.count < self.jump_to_batch * self.batch_size:
                self.count += 1
                return None
        item = self.data[index]
        entity_id = item["entity_id"]
        additional_attributes = self.neighbors_dataset.get_by_entity(entity_id)
        item.update(additional_attributes)
        return item


class EntEmbDuckDataset(torch.utils.data.Dataset):
    def __init__(self,
        duck_neighbors: Dict[str, List[str]],
        ent_catalogue: EntityCatalogue,
        rel_catalogue: RelationCatalogue,
        ent_to_rel: Dict[str, List[str]],
        stop_rels: Optional[Set[str]] = None,
        relation_threshold = 0,
        num_neighbors_per_entity=1
    ):
        super().__init__()
        self.duck_neighbors = duck_neighbors
        self.ent_catalogue = ent_catalogue
        self.rel_catalogue = rel_catalogue
        self.ent_to_rel = ent_to_rel
        if stop_rels is not None:
            self.ent_to_rel = {
                e: list(set(rels) - stop_rels) for e, rels in ent_to_rel.items()
            }
        self.entities = [
            e for e in ent_catalogue.idx
            if len(ent_to_rel[e]) >= relation_threshold
        ]
        self.num_neighbors_per_entity = num_neighbors_per_entity
        self.stop_rels = stop_rels or set()

    def _get_entity_dict(self, entity_id):
        entity_index, entity_tokens = self.ent_catalogue[entity_id]
        rels = self.ent_to_rel[entity_id]
        relation_indices, relation_data = self.rel_catalogue[rels]

        return {
            "entity_id": entity_id,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
            "relation_labels": rels,
            "relation_indexes": relation_indices,
            "relation_data": relation_data
        }

    def get_by_entity(self, entity_id):
        result = self._get_entity_dict(entity_id)
        neighbors = self.duck_neighbors[entity_id][:self.num_neighbors_per_entity]
        neighbors = [self._get_entity_dict(n) for n in neighbors]
        result["neighbors"] = neighbors
        return result

    def __len__(self):
        return len(self.duck_neighbors)
    
    def __get_item__(self, index):
        entity_id = self.entities[index]
        return self.get_by_entity(entity_id)


class EdGenreDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset for entity disambiguation with GENRE open-source format.
    (https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh)
    Each example in this dataset contains one mention.
    """
    def __init__(
        self, path, ent_catalogue
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
            assert len(data["output"]) == 1
            if data["output"][0]["answer"] in self.ent_catalogue:
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

        entity_id = data["output"][0]["answer"]
        entity_index, entity_tokens = self.ent_catalogue[entity_id]

        return {
            "context_left": data["meta"]["left_context"],
            "mention": data["meta"]["mention"],
            "context_right": data["meta"]["right_context"],
            "entity_id": entity_id,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
        }


class DuckTransform(BlinkTransform):
    def __init__(
        self,
        model_path: str = "bert-large-uncased",
        mention_start_token: int = 1,
        mention_end_token: int = 2,
        max_mention_len: int = 128,
        max_entity_len: int = 128,
        max_relation_len: int = 64,
        add_eos_bos: bool = False,
    ):
        super().__init__(
            model_path=model_path,
            mention_start_token=mention_start_token,
            mention_end_token=mention_end_token,
            max_mention_len=max_mention_len,
            max_entity_len=max_entity_len,
            add_eos_bos_to_entity=add_eos_bos
        )
        self.max_relation_len = max_relation_len
        self.add_eos_bos = add_eos_bos

    def _transform_relations(
        self,
        relation_token_ids: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        result = []
        for i, relation in enumerate(relation_token_ids):
            result.append([])
            for token_ids in relation:
                if self.add_eos_bos:
                    token_ids = [self.bos_idx] + token_ids + [self.eos_idx]
                if len(token_ids) > self.max_relation_len:
                    token_ids = token_ids[:self.max_relation_len]
                    token_ids[-1] = self.eos_idx
                result[i].append(token_ids)
        return result
    
    def _transform_neighbors(
        self,
        neighbor_token_ids
    ):
        return [self._transform_entity(neighbors) for neighbors in neighbor_token_ids]
    
    def _transform_neighbor_relations(
        self,
        neighbor_relation_token_ids
    ):
        return [self._transform_relations(rels) for rels in neighbor_relation_token_ids]

    def _list_to_tensor(
        self,
        data,
        pad_value=None
    ):
        if pad_value is None:
            pad_value = self.pad_token_id
        tensor, attention_mask = list_to_tensor(list(data), pad_value=pad_value)
        attention_mask = attention_mask.int()
        return {
            'data': tensor,
            'attention_mask': attention_mask
        }
    
    def _to_tensor(self, token_ids, attention_mask_pad_idx=0):
        return self._list_to_tensor(token_ids, pad_value=None)

    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        batch = dict(batch)
        batch["token_ids"] = batch["entity_token_ids"]
        mention_tensor, entity_tensor = super().forward(batch)
        relation_data = batch["relation_data"]
        relation_pad = 0.0
        if torch.jit.isinstance(relation_data, List[List[List[int]]]):
            relation_data = self._transform_relations(relation_data)
            relation_pad = None
        relations_tensor = self._list_to_tensor(relation_data, pad_value=relation_pad)

        neighbors_tensor = None
        neighbor_token_ids = batch["neighbor_token_ids"]
        if neighbor_token_ids is not None:
            torch.jit.isinstance(neighbor_token_ids, List[List[List[int]]])
            neighbor_token_ids = self._transform_neighbors(neighbor_token_ids)
            neighbors_tensor = self._list_to_tensor(neighbor_token_ids)
        
        neighbor_relation_tensor = None
        neighbor_relation_data = batch["neighbor_relation_data"]
        if neighbor_relation_data is not None:
            relation_pad = 0.0
            if torch.jit.isinstance(neighbor_relation_data, List[List[List[List[int]]]]):
                neighbor_relation_data = self._transform_neighbor_relations(
                    neighbor_relation_data
                )
                relation_pad = None
            neighbor_relation_tensor = self._list_to_tensor(neighbor_relation_data)

        return {
            "mentions": mention_tensor,
            "entities": entity_tensor,
            "relations": relations_tensor,
            "neighbors": neighbors_tensor,
            "neighbor_relations": neighbor_relation_tensor
        }


class EdDuckDataModule(LightningDataModule):
    def __init__(
        self,
        transform: DuckTransform,
        train_path: str,
        val_path: str,
        test_path: str,
        ent_to_rel_path: str,
        ent_catalogue_data_path: str,
        ent_catalogue_idx_path: str,
        rel_catalogue_data_path: str,
        rel_catalogue_idx_path: str,
        neighbors_path: str,
        stop_rels_path: Optional[str] = None,
        pretrained_relations: bool = True,
        batch_size: int = 2,
        num_neighbors_per_entity: int = 1,
        shuffle: bool = True,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_neighbors_per_entity = num_neighbors_per_entity

        self.transform = transform
        self.ent_catalogue = EntityCatalogue(
            ent_catalogue_data_path, ent_catalogue_idx_path
        )
        self.rel_catalogue = RelationCatalogue(
            rel_catalogue_data_path, rel_catalogue_idx_path
        )
        self.pretrained_relations = pretrained_relations
        logger.info(f"Reading mapping from entities to relations: {ent_to_rel_path}")
        self.ent_to_rel = load_json(ent_to_rel_path)

        self.num_workers = 0
        self.duck_neighbors = None
        if neighbors_path is not None:
            logger.info(f"Reading neighbors: {neighbors_path}")
            self.duck_neighbors = load_json(neighbors_path)
        
        self.shuffle = shuffle

        stop_rels = None
        if stop_rels_path is not None:
            stop_rels = set(r["id"] for r in load_jsonl(stop_rels_path))

        self.datasets = {
            "train": EdDuckDataset(
                train_path,
                self.ent_catalogue,
                self.rel_catalogue,
                self.ent_to_rel,
                neighbors=self.duck_neighbors,
                num_neighbors_per_entity=num_neighbors_per_entity,
                stop_rels=stop_rels,
                batch_size=batch_size,
                **kwargs
            ),
            "valid": EdDuckDataset(
                val_path,
                self.ent_catalogue,
                self.rel_catalogue,
                self.ent_to_rel,
                neighbors=self.duck_neighbors,
                num_neighbors_per_entity=num_neighbors_per_entity,
                stop_rels=stop_rels
            ),
            "test": EdDuckDataset(
                test_path,
                self.ent_catalogue,
                self.rel_catalogue,
                self.ent_to_rel,
                neighbors=self.duck_neighbors,
                num_neighbors_per_entity=num_neighbors_per_entity,
                stop_rels=stop_rels
            ),
        }

        self.count = 0
        self.jump_to_batch = None
        if "jump_to_batch" in kwargs:
            self.jump_to_batch = kwargs["jump_to_batch"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_train,
            shuffle=self.shuffle
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
        if self.jump_to_batch is not None and self.count < self.jump_to_batch:
            self.count += 1
            return None
        left_context, mention, right_context, \
            entity_labels, entity_ids, entity_token_ids, \
            relation_labels, relation_ids, relation_data, \
            neighbors = zip(
            *[item.values() for item in batch]
        )
        neighbor_ids = None
        neighbor_relation_data = None
        neighbor_token_ids = None
        neighbor_labels = None
        if is_train and neighbors is not None:
            all_neighbors = [n for ent_neigh in neighbors for n in ent_neigh]
            for ent_neigh in neighbors:
                if len(ent_neigh) < self.num_neighbors_per_entity:
                    difference = self.num_neighbors_per_entity - len(ent_neigh)
                    sample = random.sample(all_neighbors, difference)
                    ent_neigh.extend(sample)
            neighbor_ids = [
                [n["entity_index"] for n in ent_neigh[:self.num_neighbors_per_entity]]
                for ent_neigh in neighbors
            ]
            neighbor_labels = [
                [n["entity_id"] for n in ent_neigh[:self.num_neighbors_per_entity]]
                for ent_neigh in neighbors
            ]
            neighbor_token_ids = [
                [n["entity_tokens"] for n in ent_neigh[:self.num_neighbors_per_entity]]
                for ent_neigh in neighbors
            ]
            neighbor_relation_data = [
                [n["relation_data"] for n in ent_neigh[:self.num_neighbors_per_entity]]
                for ent_neigh in neighbors
            ]
            neighbor_relation_ids = [
                [n["relation_indexes"] for n in ent_neigh[:self.num_neighbors_per_entity]]
                for ent_neigh in neighbors
            ]
            neighbor_relation_labels = [
                [n["relation_labels"] for n in ent_neigh[:self.num_neighbors_per_entity]]
                for ent_neigh in neighbors
            ]

        entity_token_ids, entity_ids, targets = order_entities(
            entity_token_ids,
            entity_ids,
            None,
            None,
            0,
        )
        pad_length = (
            len(batch) - len(entity_token_ids)
        )
        entity_tensor_mask = [1] * len(entity_token_ids) + [0] * pad_length
        entity_token_ids += [
            [self.transform.bos_idx, self.transform.eos_idx]
        ] * pad_length
        entity_ids += [0] * pad_length

        result = self.transform(
            {
                "left_context": left_context,
                "mention": mention,
                "right_context": right_context,
                "entity_token_ids": entity_token_ids,
                "relation_data": relation_data,
                "neighbor_token_ids": neighbor_token_ids,
                "neighbor_relation_data": neighbor_relation_data
            }
        )

        result["entity_labels"] = entity_labels
        result["entity_ids"] = torch.tensor(entity_ids, dtype=torch.long)
        result["relation_labels"] = relation_labels
        result["relation_ids"] = [torch.tensor(rids) for rids in relation_ids]
        result["neighbor_relation_ids"] = [
            [torch.tensor(rids) for rids in neigh_rels]
            for neigh_rels in neighbor_relation_ids
        ]
        result["neighbor_ids"] = torch.tensor(neighbor_ids)
        result["neighbor_labels"] = neighbor_labels
        result["neighbor_relation_labels"] = neighbor_relation_labels
        result["targets"] = torch.tensor(targets, dtype=torch.long)
        result["entity_tensor_mask"] = torch.tensor(entity_tensor_mask, dtype=torch.long)
        return result

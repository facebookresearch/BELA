import json
import logging
import mmap
import random
from unittest.mock import NonCallableMagicMock

import torch
import h5py
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from duck.common.lmdb_wrapper import LmdbImmutableDict
from duck.common.utils import concatenate_dicts_of_lists, flatten_dict_of_lists, list_to_tensor, load_json, load_jsonl, any_none, load_pkl

from mblink.utils.utils import EntityCatalogue
from mblink.transforms.blink_transform import BlinkTransform

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from tqdm import tqdm

logger = logging.getLogger()


class RelationCatalogue:
    def __init__(self, idx_path, data_path=None, return_data=False):
        self.data = None
        if data_path is not None:
            self.data_file = h5py.File(data_path, "r")
            self.data = self.data_file["data"][:]
        logger.info(f"Reading relation catalogue index {idx_path}")
        self.idx = {}
        self.return_data = return_data
        if data_path is None:
            assert not return_data
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
        rel_indices = [self.idx[rel] for rel in relations] + [-1]
        if not self.return_data:
            if is_string:
                rel_indices = rel_indices[0]
            return rel_indices, None
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
        label_to_id: Dict[str, str],
        ent_to_rel: Dict[str, List[str]],
        neighbors: Dict[str, List[str]],
        num_neighbors_per_entity: int = 3,
        stop_rels: Optional[Set[str]] = None,
        entity_priors=None,
        num_priors_per_mention: int = 3,
        relation_threshold: int = 0,
        **kwargs
    ) -> None:
        super().__init__()
        self.data = EdGenreDataset(
            path,
            ent_catalogue,
            label_to_id=label_to_id,
            relation_threshold=relation_threshold,
            ent_to_rel=ent_to_rel,
            stop_rels=stop_rels
        )
        self.rel_catalogue = rel_catalogue
        self.ent_to_rel = ent_to_rel
        self.label_to_id = label_to_id
        self.neighbors_dataset = NeighborsDataset(
            neighbors,
            ent_catalogue,
            rel_catalogue,
            ent_to_rel,
            label_to_id,
            stop_rels=stop_rels,
            num_neighbors_per_entity=num_neighbors_per_entity
        )
        self.priors_dataset = EntityPriorsDataset(
            entity_priors,
            ent_catalogue,
            rel_catalogue,
            ent_to_rel,
            label_to_id=label_to_id,
            stop_rels=stop_rels,
            num_priors_per_mention=num_priors_per_mention
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
        entity_label = item["entity_label"]
        additional_attributes = self.neighbors_dataset.get_by_entity(entity_id)
        item.update(additional_attributes)
        additional_attributes = self.priors_dataset.get_by_mention(item["mention"])
        item.update(additional_attributes)
        prior_indexes = [p["entity_index"] for p in item["priors"][:2]]
        for index in prior_indexes:
            if index not in item["candidate_indexes"]:
                item["candidate_indexes"].append(index)
        item["entity_label"] = entity_label
        return item


class EntityPriorsDataset():
    def __init__(self,
        priors,
        ent_catalogue: EntityCatalogue,
        rel_catalogue: RelationCatalogue,
        ent_to_rel: Dict[str, List[str]],
        label_to_id: Dict[str, str],
        stop_rels: Optional[Set[str]] = None,
        num_priors_per_mention=1
    ):
        super().__init__()
        self.label_to_id = label_to_id
        self.priors = priors
        self.ent_catalogue = ent_catalogue
        self.rel_catalogue = rel_catalogue
        self.ent_to_rel = ent_to_rel
        if stop_rels is not None:
            self.ent_to_rel = {
                e: list(set(rels) - stop_rels) for e, rels in self.ent_to_rel.items()
            }
        self.num_priors_per_mention = num_priors_per_mention
        self.stop_rels = stop_rels or set()
        self.id_to_label = {eid: label for label, eid in label_to_id.items()}

    def _get_entity_dict(self, entity_id, prob):
        entity_label = self.id_to_label[entity_id]
        entity_index, entity_tokens = self.ent_catalogue[entity_id]
        rels = self.ent_to_rel[entity_id]
        relation_indices, relation_data = self.rel_catalogue[rels]

        return {
            "entity_id": entity_id,
            'entity_label': entity_label,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
            "relation_labels": rels,
            "relation_indexes": relation_indices,
            "relation_data": relation_data,
            "probability": prob
        }

    def get_by_mention(self, mention):
        result = {"priors": []}
        if self.priors is not None and mention.lower() in self.priors:
            priors = self.priors[mention.lower()][:self.num_priors_per_mention]
            result["priors"] = [
                self._get_entity_dict(prior[0], prior[1])
                for prior in priors
                if prior[0] in self.ent_catalogue
            ]
        return result

    def __len__(self):
        return len(self.priors)


class NeighborsDataset(torch.utils.data.Dataset):
    def __init__(self,
        duck_neighbors: Dict[str, List[str]],
        ent_catalogue: EntityCatalogue,
        rel_catalogue: RelationCatalogue,
        ent_to_rel: Dict[str, List[str]],
        label_to_id: Dict[str, str],
        stop_rels: Optional[Set[str]] = None,
        relation_threshold = 0,
        num_neighbors_per_entity=1
    ):
        super().__init__()
        self.label_to_id = label_to_id
        self.duck_neighbors = duck_neighbors
        self.ent_catalogue = ent_catalogue
        self.rel_catalogue = rel_catalogue
        self.ent_to_rel = ent_to_rel
        if stop_rels is not None:
            self.ent_to_rel = {
                e: list(set(rels) - stop_rels) for e, rels in self.ent_to_rel.items()
            }
        if relation_threshold > 0:
            self.entities = [
                e for e in ent_catalogue.idx
                if len(self.ent_to_rel[e]) >= relation_threshold
            ]
        else:
            self.entities = [e for e in ent_catalogue.idx]
        self.num_neighbors_per_entity = num_neighbors_per_entity
        self.stop_rels = stop_rels or set()
        self.id_to_label = {eid: label for label, eid in label_to_id.items()}

    def _get_entity_dict(self, entity_id):
        entity_label = self.id_to_label[entity_id]
        entity_index, entity_tokens = self.ent_catalogue[entity_id]
        rels = self.ent_to_rel[entity_id]
        relation_indices, relation_data = self.rel_catalogue[rels]

        return {
            "entity_id": entity_id,
            'entity_label': entity_label,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
            "relation_labels": rels,
            "relation_indexes": relation_indices,
            "relation_data": relation_data
        }

    def get_by_entity(self, entity_id):
        result = self._get_entity_dict(entity_id)
        if self.duck_neighbors is None:
            result["neighbors"] = None
            return result
        neighbors = self.duck_neighbors[entity_id][:self.num_neighbors_per_entity]
        neighbors = [self._get_entity_dict(n) for n in neighbors]
        result["neighbors"] = neighbors
        return result

    def __len__(self):
        return len(self.ent_catalogue)
    
    def __getitem__(self, index):
        entity_id = self.ent_catalogue.entities[index]
        return self.get_by_entity(entity_id)

    def get_slice(self, start, end):
        return [self[i] for i in range(start, end)]


class EdGenreDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset for entity disambiguation with GENRE open-source format.
    (https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh)
    Each example in this dataset contains one mention.
    """
    def __init__(
        self,
        path,
        ent_catalogue,
        label_to_id,
        relation_threshold=0,
        ent_to_rel=None,
        stop_rels=None
    ):
        self.ent_catalogue = ent_catalogue
        self.label_to_id = label_to_id
        self.file = open(path, mode="r")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.offsets = []
        self.count = 0
        stop_rels = stop_rels or set()

        logger.info(f"Build mmap index for {path}")
        line = self.mm.readline()
        offset = 0
        total_lines = 0
        while line:
            data = json.loads(line)
            assert len(data["output"]) == 1
            label = data["output"][0]["answer"]
            if label in self.label_to_id and self.label_to_id[label] in self.ent_catalogue:
                qcode = self.label_to_id[label]
                if ent_to_rel is None or (
                    qcode in ent_to_rel and \
                    len(set(ent_to_rel[qcode]) - stop_rels) >= relation_threshold
                ):
                    self.offsets.append(offset)
                    self.count += 1
            offset = self.mm.tell()
            line = self.mm.readline()
            total_lines += 1
        coverage = self.count / total_lines
        logger.info(f"Coverage of {path}: {coverage:.4f}")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        offset = self.offsets[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        data = json.loads(line)

        entity_label = data["output"][0]["answer"]
        entity_id = self.label_to_id[entity_label]
        entity_index, entity_tokens = self.ent_catalogue[entity_id]

        candidate_indexes = []
        candidate_labels = []
        if "candidates" in data:
            candidate_indexes = [
                self.ent_catalogue[self.label_to_id[c]][0]
                for c in data["candidates"]
                if c in self.label_to_id and self.label_to_id.get(c) in self.ent_catalogue
            ]
            candidate_labels = data["candidates"]

        return {
            "context_left": data["meta"]["left_context"],
            "mention": data["meta"]["mention"],
            "context_right": data["meta"]["right_context"],
            "entity_id": entity_id,
            "entity_label": entity_label,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
            "candidate_indexes": candidate_indexes,
            "candidate_labels": candidate_labels
        }


class DuckTransform(BlinkTransform):
    def __init__(
        self,
        model_path: str = "bert-large-uncased",
        mention_start_token: Optional[int] = None,
        mention_end_token: Optional[int] = None,
        max_mention_len: int = 256,
        max_entity_len: int = 256,
        max_relation_len: int = 64,
        add_eos_bos: bool = False,
        max_num_rels = None
    ):
        sep_token_id = AutoTokenizer.from_pretrained(model_path).sep_token_id
        mention_start_token = mention_start_token or sep_token_id
        mention_end_token = mention_end_token or sep_token_id
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
        self.max_num_rels = max_num_rels

    
    def _transform_relation_token_ids(
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
    
    def _transform_neighbor_relation_token_ids(
        self,
        neighbor_relation_token_ids
    ): 
        return [self._transform_relation_token_ids(rels) for rels in neighbor_relation_token_ids]

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
    
    def _transform_relation_ids(self, relation_ids):
        tensor, attention_mask = list_to_tensor(
            list(relation_ids),
            pad_value=-1,
            dtype=torch.long,
            size=self.max_num_rels
        )
        attention_mask = attention_mask.int()
        return {
            'data': (tensor + 1).long(),
            'attention_mask': attention_mask
        }
    
    def _transform_relations(self, relation_data):
        if any_none(relation_data):
            return None
        relation_pad = 0.0
        if torch.jit.isinstance(relation_data, List[List[List[int]]]):
            relation_data = self._transform_relation_token_ids(relation_data)
            relation_pad = None
        return self._list_to_tensor(relation_data, pad_value=relation_pad)
    
    def _transform_neighbor_relations(self, neighbor_relation_data):
        if any_none(neighbor_relation_data):
            return None
        relation_pad = 0.0
        if torch.jit.isinstance(neighbor_relation_data, List[List[List[List[int]]]]):
            neighbor_relation_data = self._transform_neighbor_relation_token_ids(
                neighbor_relation_data
            )
            relation_pad = None
        return self._list_to_tensor(neighbor_relation_data, pad_value=relation_pad)

    def _to_tensor(self, token_ids, attention_mask_pad_idx=0):
        return self._list_to_tensor(token_ids, pad_value=None)

    
    def transform_ent_data(self, ent_data):
        """
        {
            "entity_id": entity_id,
            'entity_label': entity_label,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
            "relation_labels": rels,
            "relation_indexes": relation_indices,
            "relation_data": relation_data
        }
        """
        entity_ids = [e["entity_index"] for e in ent_data]
        entity_token_ids = [e["entity_tokens"] for e in ent_data]
        relation_ids = [e["relation_indexes"] for e in ent_data]
        entity_labels = [e["entity_label"] for e in ent_data]
        relation_labels = [e["relation_labels"] for e in ent_data]
        relation_data = [e["relation_data"] for e in ent_data]

        relations_tensor = self._transform_relations(relation_data)
        entity_token_ids = self._transform_entity(entity_token_ids)
        entity_tensor = self._to_tensor(
            entity_token_ids
        )
        relation_ids = self._transform_relation_ids(relation_ids)
        entity_ids = torch.tensor(entity_ids, dtype=torch.long)
        return {
            "entities": entity_tensor,
            "relations": relations_tensor,
            "relation_ids": relation_ids,
            "entity_labels": entity_labels,
            "relation_labels": relation_labels
        }

    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        batch = dict(batch)
        batch["token_ids"] = batch["entity_token_ids"]
        mention_tensor, entity_tensor = super().forward(batch)
        relation_data = batch["relation_data"]
        relations_tensor = None
        if relation_data is not None:
            relations_tensor = self._transform_relations(relation_data)
        
        # neighbors_tensor = None
        # neighbor_token_ids = batch["neighbor_token_ids"]
        # if neighbor_token_ids is not None:
        #     torch.jit.isinstance(neighbor_token_ids, List[List[int]])
        #     neighbor_token_ids = self._transform_entity(neighbor_token_ids)
        #     neighbors_tensor = self._list_to_tensor(neighbor_token_ids)
        
        # neighbor_relation_tensor = None
        # neighbor_relation_data = batch["neighbor_relation_data"]
        # if neighbor_relation_data is not None:
        #     neighbor_relation_tensor = self._transform_relations(neighbor_relation_data)

        relation_ids = self._transform_relation_ids(batch["relation_ids"])
        # neighbor_relation_ids = batch["neighbor_relation_ids"]
        # if neighbor_relation_ids is not None:
        #     neighbor_relation_ids = self._transform_relation_ids(neighbor_relation_ids)
        
        candidates = None
        if "candidates" in batch and any(len(cands) > 0 for cands in batch["candidates"]):
            candidates = self._list_to_tensor(batch["candidates"])
        
        return {
            "mentions": mention_tensor,
            "entities": entity_tensor,
            "relations": relations_tensor,
            "relation_ids": relation_ids,
            "candidates": candidates
        }


class EdDuckDataModule(LightningDataModule):
    def __init__(
        self,
        transform: DuckTransform,
        train_path: str,
        val_paths: Dict[str, str],
        test_paths: Dict[str, str],
        ent_to_rel_path: str,
        wikipedia_to_wikidata_path: str,
        ent_catalogue_idx_path: str,
        ent_catalogue_data_path: str,
        rel_catalogue_idx_path: str,
        rel_catalogue_data_path: Optional[str] = None,
        entity_priors_path: Optional[str] = None,
        neighbors_path: Optional[str] = None,
        stop_rels_path: Optional[str] = None,
        pretrained_relations: bool = True,
        batch_size: int = 16,
        num_neighbors_per_entity: int = 1,
        num_priors_per_mention: int = 1,
        max_num_entities_per_batch: int = 32,
        relation_threshold: int = 0,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__()
        self.debug = False or kwargs.get("debug")
        self.batch_size = batch_size
        self.num_neighbors_per_entity = num_neighbors_per_entity

        self.transform = transform
        self.ent_catalogue = EntityCatalogue(
            ent_catalogue_data_path, ent_catalogue_idx_path
        )
        self.rel_catalogue = RelationCatalogue(
            rel_catalogue_idx_path, rel_catalogue_data_path
        )

        self.dataset_cache = {}

        logger.info(f"Reading mapping from Wikipedia titles to Wikidata IDs: {wikipedia_to_wikidata_path}")
        self.label_to_id = load_pkl(wikipedia_to_wikidata_path)
        self.label_to_id = {k: sorted(list(v))[0] for k, v in self.label_to_id.items()}

        self.pretrained_relations = pretrained_relations
        logger.info(f"Reading mapping from entities to relations: {ent_to_rel_path}")
        self.ent_to_rel = load_json(ent_to_rel_path)
        if self.label_to_id is not None:
            self.ent_to_rel = {
                self.label_to_id[e]: rels for e, rels in self.ent_to_rel.items()
                if e in self.label_to_id
            }

        self.num_workers = num_workers if not self.debug else 0
        self.duck_neighbors = None
        if neighbors_path is not None:
            logger.info(f"Reading neighbors: {neighbors_path}")
            self.duck_neighbors = load_json(neighbors_path)
            # self._map_neighbors_to_id()
            
        self.transform.max_num_rels = max(len(rels) for rels in self.ent_to_rel.values())
        self.shuffle = shuffle

        self.stop_rels = None
        if stop_rels_path is not None and not self.debug:
            self.stop_rels = set(r["id"] for r in load_jsonl(stop_rels_path))

        val_paths = list(val_paths.values())
        test_paths = list(test_paths.values())

        # if self.debug:
        #     val_paths = val_paths[:1]
        #     test_paths = test_paths[:1]

        self.entity_priors = {}
        if entity_priors_path is not None:
            self.entity_priors = LmdbImmutableDict(entity_priors_path)
        self.num_priors_per_mention = num_priors_per_mention

        self.train_dataset = self._duck_dataset(train_path, relation_threshold=relation_threshold)
        self.val_datasets = [self._duck_dataset(val_path) for val_path in val_paths]
        self.test_datasets = [self._duck_dataset(test_path) for test_path in test_paths]
        self.count = 0
        self.jump_to_batch = None
        if "jump_to_batch" in kwargs:
            self.jump_to_batch = kwargs["jump_to_batch"]
        
        self.use_all_relations = False
        self.max_num_entities_per_batch = max_num_entities_per_batch
    
    def _map_neighbors_to_id(self):
        if self.label_to_id is None:
            return
        self.duck_neighbors = {
            self.label_to_id[e]: [self.label_to_id[n] for n in neighbors]
            for e, neighbors in self.duck_neighbors.items()
            if e in self.label_to_id
        }
        return self.duck_neighbors
    
    def _duck_dataset(self, path, **kwargs):
        if path in self.dataset_cache:
            return self.dataset_cache[path]
        dataset = EdDuckDataset(
                path,
                self.ent_catalogue,
                self.rel_catalogue,
                self.label_to_id,
                self.ent_to_rel,
                neighbors=self.duck_neighbors,
                num_neighbors_per_entity=self.num_neighbors_per_entity,
                stop_rels=self.stop_rels,
                batch_size=self.batch_size,
                entity_priors=self.entity_priors,
                num_priors_per_mention=self.num_priors_per_mention,
                **kwargs
            )
        self.dataset_cache[path] = dataset
        return dataset
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_train,
            shuffle=self.shuffle
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_eval,
            )
            for val_dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_eval,
            )
            for test_dataset in self.test_datasets
        ]

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
            _, entity_labels, entity_indexes, entity_token_ids, \
            candidate_indexes, candidate_labels, \
            relation_labels, relation_ids, relation_data, \
            neighbors, priors = zip(
            *[item.values() for item in batch]
        )
        neighbors = self._process_entity_lists(
            neighbors, low=1, high=(self.num_neighbors_per_entity + 1)
        )
        priors = self._process_entity_lists(
            priors, low=0, high=(self.num_priors_per_mention)
        )

        additional_entities = concatenate_dicts_of_lists(priors, neighbors)
        additional_entities = flatten_dict_of_lists(additional_entities)
        prior_probabilities = priors["probabilities"]

        targets = None
        entity_token_ids = list(entity_token_ids)
        entity_indexes = list(entity_indexes)
        entity_labels = list(entity_labels)
        relation_ids = list(relation_ids)
        relation_labels = list(relation_labels)
        bsz = len(entity_indexes)

        if is_train:
            entity_token_ids += additional_entities["token_ids"]
            entity_indexes += additional_entities["indexes"]
            entity_labels += additional_entities["labels"]
            relation_ids += additional_entities["relation_ids"]
            relation_labels += additional_entities["relation_labels"]
            
        if relation_data is not None and any(rels is not None for rels in relation_data):
            relation_data = list(relation_data) + additional_entities["relation_data"]
        else:
            relation_data = None
        
        max_length = len(entity_indexes)
        if is_train:
            entity_token_ids, entity_indexes, entity_labels, \
            relation_ids, relation_labels, targets = self._order_entities(
                entity_token_ids,
                entity_indexes,
                entity_labels,
                relation_ids,
                relation_labels
            )
            targets = targets[:bsz]
        
        entity_token_ids = entity_token_ids[:self.max_num_entities_per_batch]
        entity_indexes = entity_indexes[:self.max_num_entities_per_batch]
        entity_labels = entity_labels[:self.max_num_entities_per_batch]
        relation_ids = relation_ids[:self.max_num_entities_per_batch]
        relation_labels = relation_labels[:self.max_num_entities_per_batch]
        if relation_data is not None:
            relation_data = relation_data[:self.max_num_entities_per_batch]
        
        prior_probabilities, prior_prob_mask = self._transform_prior_probabilities(
            prior_probabilities, entity_indexes
        )

        pad_length = max_length - len(entity_token_ids)
        entity_tensor_mask = [1] * len(entity_token_ids) + [0] * pad_length
        entity_token_ids += [
            [self.transform.bos_idx, self.transform.eos_idx]
        ] * pad_length
        entity_indexes += [0] * pad_length
       
        relation_ids, relation_data, ent_rel_mask = self._order_relations(
            relation_ids, relation_data
        )

        result = self.transform(
            {
                "left_context": left_context,
                "mention": mention,
                "right_context": right_context,
                "entity_token_ids": entity_token_ids,
                "relation_ids": relation_ids,
                "relation_data": relation_data,
                "candidates": candidate_indexes
            }
        )
        result["mention_text"] = list(mention)
        result["entity_labels"] = entity_labels
        result["entity_ids"] = torch.tensor(entity_indexes, dtype=torch.long)
        result["relation_labels"] = relation_labels
        result["targets"] = torch.tensor(targets, dtype=torch.long) if targets is not None else None
        result["entity_tensor_mask"] = torch.tensor(entity_tensor_mask, dtype=torch.long).bool()
        result["ent_rel_mask"] = ent_rel_mask
        result["candidate_labels"] = candidate_labels
        result["prior_probabilities"] = prior_probabilities
        result["prior_prob_mask"] = prior_prob_mask
        return result

    def _process_entity_lists(self, entity_lists, low=0, high=1):
        indexes = []
        relation_data = []
        token_ids = []
        labels = []
        relation_labels = []
        relation_ids = []
        probabilities = []

        if any(lst is not None for lst in entity_lists):
            indexes = [
                [e["entity_index"] for e in lst[low:high]]
                for lst in entity_lists
            ]
            labels = [
                [e["entity_label"] for e in lst[low:high]]
                for lst in entity_lists
            ]
            token_ids = [
                [e["entity_tokens"] for e in lst[low:high]]
                for lst in entity_lists
            ]
            relation_data = [
                [e["relation_data"] for e in lst[low:high]]
                for lst in entity_lists
            ]
            relation_ids = [
                [e["relation_indexes"] for e in lst[low:high]]
                for lst in entity_lists
            ]
            relation_labels = [
                [e["relation_labels"] for e in lst[low:high]]
                for lst in entity_lists
            ]
            if any("probability" in e for lst in entity_lists for e in lst):
                probabilities = [
                    {e["entity_index"]: e["probability"] for e in lst[low:high]}
                    for lst in entity_lists
                ]

            if all(rels is None for rels in relation_data):
                relation_data = None
            
            return {
                "indexes": indexes,
                "relation_data": relation_data,
                "token_ids": token_ids,
                "labels": labels,
                "relation_labels": relation_labels,
                "relation_ids": relation_ids,
                "probabilities": probabilities
            }

    def _transform_prior_probabilities(self, prior_probabilities, entity_indexes):
        index_map = {index: i for i, index in enumerate(entity_indexes)}
        result = []
        mask = []
        for probs in prior_probabilities:
            row = torch.zeros(len(index_map))
            mask_row = row.bool()
            for (e, p) in probs.items():
                if e in index_map:
                    row[index_map[e]] = p
                    mask_row[index_map[e]] = True
            result.append(row)
            mask.append(mask_row)
        return torch.stack(result), torch.stack(mask)

    def _order_entities(
        self,
        entity_data,
        entity_indices,
        entity_labels,
        relation_ids,
        relation_labels
    ):
        ent_index_map = {}
        targets = []
        filtered_entity_data = []
        filtered_entity_idxs = []
        filtered_entity_labels = []
        filtered_relation_ids = []
        filtered_relation_labels = []

        tuples = zip(
            entity_data,
            entity_indices,
            entity_labels,
            relation_ids,
            relation_labels
        )

        for tuple in tuples:
            ent_data, ent_idx, ent_label, rel_ids, rel_labels = tuple
            if ent_idx in ent_index_map:
                targets.append(ent_index_map[ent_idx])
            else:
                target = len(ent_index_map)
                targets.append(target)
                ent_index_map[ent_idx] = target
                filtered_entity_data.append(ent_data)
                filtered_entity_idxs.append(ent_idx)
                filtered_entity_labels.append(ent_label)
                filtered_relation_ids.append(rel_ids)
                filtered_relation_labels.append(rel_labels)

        return (
            filtered_entity_data,
            filtered_entity_idxs,
            filtered_entity_labels,
            filtered_relation_ids,
            filtered_relation_labels,
            targets
        )

    def _order_relations(
        self,
        relation_ids,
        relation_data
    ):
        if self.use_all_relations:
            assert relation_data is None
            flat_relation_ids = list(range(len(self.rel_catalogue))) + [-1]
        else:
            flat_relation_ids = [r for rels in relation_ids for r in rels]
            flat_relation_data = []
            if relation_data is not None:
                [r for rels in relation_data for r in rels] if relation_data is not None else []
            
        filtered_relation_data = []
        rel_id_map = {}
        for i, r in enumerate(flat_relation_ids):
            if r not in rel_id_map:
                rel_id_map[r] = len(rel_id_map)
                if relation_data is not None:
                    filtered_relation_data.append(flat_relation_data[i])
        
        filtered_relation_ids = list(rel_id_map.keys())
        if self.use_all_relations:
            assert filtered_relation_ids == flat_relation_ids
        ent_rel_mask = torch.full(
            (len(relation_ids), len(filtered_relation_ids)),
            False
        )
        for i, rels in enumerate(relation_ids):
            indexes = [rel_id_map[r] for r in rels]
            ent_rel_mask[i][indexes] = True

        return filtered_relation_ids, filtered_relation_data, ent_rel_mask

import json
import logging
import mmap

import torch
import h5py
from pytorch_lightning import LightningDataModule
from duck.common.utils import list_to_tensor, load_json

from mblink.utils.utils import EntityCatalogue, order_entities
from mblink.transforms.blink_transform import BlinkTransform

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger()


class RelationCatalogue:
    def __init__(self, tokens_path, idx_path):
        self.data_file = h5py.File(tokens_path, "r")
        self.data = self.data_file["data"]

        logger.info(f"Reading relation catalogue index {idx_path}")
        self.idx = {}
        with open(idx_path, "rt") as fd:
            for idx, line in enumerate(fd):
                rel = line.strip()
                self.idx[rel] = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, relation):
        rel_index = self.idx[relation]
        value = self.data[rel_index].tolist()
        value = value[1 : value[0] + 1]
        return rel_index, value

    def __contains__(self, relation):
        return relation in self.idx


class EdDuckDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        ent_catalogue: EntityCatalogue,
        rel_catalogue: RelationCatalogue,
        ent_to_rel: Dict[str, List[str]],
        neighbors: Optional[Dict[str, List[str]]] = None
    ) -> None:
        super().__init__()
        self.data = EdGenreDataset(path, ent_catalogue)
        self.rel_catalogue = rel_catalogue
        self.ent_to_rel = ent_to_rel
        self.neighbors_dataset = None
        if neighbors is not None:
            self.neighbors_dataset = EntEmbDuckDataset(
                neighbors,
                ent_catalogue,
                rel_catalogue,
                ent_to_rel,
                relation_threshold=0
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        entity_id = item["entity_id"]
        rels = self.ent_to_rel[entity_id]
        idx_tok_pairs = [self.rel_catalogue[r] for r in rels]
        additional_attributes = {
            "relation_indexes": [p[0] for p in idx_tok_pairs],
            "relation_tokens": [p[1] for p in idx_tok_pairs],
            "neighbors": None
        }
        if self.neighbors_dataset is not None:
            additional_attributes["neighbors"] = \
                self.neighbors_dataset.get_by_entity(entity_id)["neighbors"]
        item.update(additional_attributes)
        return item


class EntEmbDuckDataset(torch.utils.data.Dataset):
    def __init__(self,
        duck_neighbors: Dict[str, List[str]],
        ent_catalogue: EntityCatalogue,
        rel_catalogue: RelationCatalogue,
        ent_to_rel: Dict[str, List[str]],
        relation_threshold = 0
    ):
        super().__init__()
        self.duck_neighbors = duck_neighbors
        self.ent_catalogue = ent_catalogue
        self.rel_catalogue = rel_catalogue
        self.entities = [
            e for e in ent_catalogue.idx
            if len(ent_to_rel[e]) >= relation_threshold
        ]
        self.ent_to_rel = ent_to_rel

    def _get_entity_dict(self, entity_id):
        entity_index, entity_tokens = self.ent_catalogue[entity_id]
        rels = self.ent_to_rel[entity_id]
        idx_tok_pairs = [self.rel_catalogue[r] for r in rels]

        return {
            "entity_id": entity_id,
            "entity_index": entity_index,
            "entity_tokens": entity_tokens,
            "relation_indexes": [p[0] for p in idx_tok_pairs],
            "relation_tokens": [p[1] for p in idx_tok_pairs],
        }

    def get_by_entity(self, entity_id):
        result = self._get_entity_dict(entity_id)
        neighbors = self.duck_neighbors[entity_id]
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

    def _to_tensor(
        self,
        token_ids,
        attention_mask_pad_idx=0
    ):
        input_ids = list_to_tensor(token_ids, pad_value=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).int()
        attention_mask[attention_mask == 0] = attention_mask_pad_idx
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        batch = dict(batch)
        batch["token_ids"] = batch["entity_token_ids"]
        mention_tensor, entity_tensor = super().forward(batch)
        relation_token_ids = batch["relation_token_ids"]
        torch.jit.isinstance(relation_token_ids, List[List[List[int]]])

        relation_token_ids = self._transform_relations(relation_token_ids)
        relations_tensor = self._to_tensor(relation_token_ids)
        
        return (
            mention_tensor,
            entity_tensor,
            relations_tensor
        )


class EdDuckDataModule(LightningDataModule):
    """
    Read data from EL datatset and prepare mention/entity pairs tensors
    """
    def __init__(
        self,
        transform: BlinkTransform,
        train_path: str,
        val_path: str,
        test_path: str,
        ent_catalogue_path: str,
        ent_catalogue_idx_path: str,
        rel_catalogue_path: str,
        rel_catalogue_idx_path: str,
        ent_to_rel_path: str,
        neighbors_path: Optional[str] = None,
        batch_size: int = 2,
        negatives: bool = False,
        max_negative_entities_in_batch: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.negatives = negatives
        self.max_negative_entities_in_batch = max_negative_entities_in_batch

        self.transform = transform
        self.ent_catalogue = EntityCatalogue(
            ent_catalogue_path, ent_catalogue_idx_path
        )
        self.rel_catalogue = RelationCatalogue(
            rel_catalogue_path, rel_catalogue_idx_path
        )
        logger.info(f"Reading mapping from entities to relations: {ent_to_rel_path}")
        self.ent_to_rel = load_json(ent_to_rel_path)

        self.num_workers = 0
        self.duck_neighbors = None
        if neighbors_path is not None:
            logger.info(f"Reading neighbors: {neighbors_path}")
            self.duck_neighbors = load_json(neighbors_path)

        self.datasets = {
            "train": EdDuckDataset(
                train_path,
                self.ent_catalogue,
                self.rel_catalogue,
                self.ent_to_rel,
                neighbors=self.duck_neighbors
            ),
            "valid": EdDuckDataset(
                val_path,
                self.ent_catalogue,
                self.rel_catalogue,
                self.ent_to_rel,
                neighbors=self.duck_neighbors
            ),
            "test": EdDuckDataset(
                test_path,
                self.ent_catalogue,
                self.rel_catalogue,
                self.ent_to_rel,
                neighbors=self.duck_neighbors
            ),
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
        left_context, mention, right_context, _, \
            entity_ids, entity_token_ids, \
            relation_ids, relation_token_ids, neighbors = zip(
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

        mention_tensors, entity_tensors, relation_tensors = self.transform(
            {
                "left_context": left_context,
                "mention": mention,
                "right_context": right_context,
                "entity_token_ids": entity_token_ids,
                "relation_token_ids": relation_token_ids
            }
        )

        entity_ids = torch.tensor(entity_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        entity_tensor_mask = torch.tensor(entity_tensor_mask, dtype=torch.long)

        return {
            "mentions": mention_tensors,
            "entities": entity_tensors,
            "relations": relation_tensors,
            "entity_ids": entity_ids,
            "relation_ids": relation_ids,
            "targets": targets,
            "entity_tensor_mask": entity_tensor_mask,
        }

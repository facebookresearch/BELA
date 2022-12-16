import json
import logging

from enum import Enum
from typing import List

import torch
import h5py


logger = logging.getLogger()


class EntityCatalogueType(Enum):
    SIMPLE = "simple"
    MULTI = "multi"


class ElDatasetType(Enum):
    BLINK = "blink"
    MATCHA = "matcha"


class NegativesStrategy(Enum):
    HIGHER = "higher"
    ALL = "all"


def assert_equal_tensor_dict(test_case, result, expected):
    """
    Compare tensors/values in the dict and assert if they are not equal.
    The dict could countain multiple levels of nesting.
    """
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_equal_tensor_dict(test_case, result[key], value)
        else:
            if isinstance(value, torch.Tensor):
                test_case.assertTrue(
                    torch.equal(result[key], value), f"{key} is not equal"
                )
            else:
                test_case.assertEqual(result[key], value, f"{key} is not equal")


def get_seq_lengths(batch: List[List[int]]):
    return [len(example) for example in batch]


class EntityCatalogue:
    def __init__(self, local_path, idx_path):
        self.data_file = h5py.File(local_path, "r")
        self.data = self.data_file["data"]

        logger.info(f"Reading entity catalogue index {idx_path}")
        self.idx = {}
        self.entities = []
        with open(idx_path, "rt") as fd:
            for idx, line in enumerate(fd):
                ent_id = line.strip()
                self.idx[ent_id] = idx
                self.entities.append(ent_id)

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, entity_id):
        ent_index = self.idx[entity_id]
        value = self.data[ent_index].tolist()
        value = value[1 : value[0] + 1]
        return ent_index, value

    def __contains__(self, entity_id):
        return entity_id in self.idx


class MultilangEntityCatalogue:
    """
    Entity catalogue where each entity id has descriptions in different languages
    Index is a json file, where keys are entity ids. The value is dict where key is
    language id and value is triplet (title, count, index). Title is a wikipedia title
    of the entity in that language, count is a number of mentions to the entity in
    that language and index is a pos of entity tokens in tokens array.

    Index example:
    {
        ...
        "Q17": {
            "en": ["Japan", 230, 10],
            "ru": ["Япония", 111, 55]
        }
        ...
    }

    Tokens file is an h5py file, where datasets keys are language ids and stored arrays
    are ent tokens.
    """

    def __init__(self, local_path, idx_path):
        self.data = h5py.File(local_path, "r")

        logger.info(f"Reading entity catalogue index {idx_path}")
        with open(idx_path, "rt") as fd:
            self.idx = json.load(fd)

        # assign unique index number to each entity
        for idx, ent_value in enumerate(self.idx.values()):
            ent_value["idx"] = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, entity_id):
        ent_lang_map = self.idx[entity_id]
        # now choose language with most mentions
        selected_lang = None
        max_count = -1
        for lang, val in ent_lang_map.items():
            if lang == "idx":
                continue
            _, count, _ = val
            if count > max_count:
                max_count = count
                selected_lang = lang
        assert selected_lang is not None

        ent_index = ent_lang_map[selected_lang][2]
        value = self.data[selected_lang][ent_index].tolist()
        value = value[1 : value[0] + 1]
        return ent_lang_map["idx"], value

    def __contains__(self, entity_id):
        return entity_id in self.idx


def order_entities(
    entities_data,
    entity_ids,
    neg_entities_ids=None,
    neg_entities_tokens=None,
    max_negative_entities_in_batch=None,
):
    """
    This function removes duplicated entities in the entities batch and
    constructs the targets.

    In bi-encoder model we train on in-batch random and hard negatives. In this
    case each mention should have one positive entity class in enttiteis batch.
    But it could happen there are two or more mentions in the batch that
    referes to the same entitty (this entity would be in the batch 2 and more
    times). In this case we could predict class correctly and calculate loss.
    To resolve this problem we filter entities and left only one example of
    each in the batch.

    Returns:
        filteres_entities - filtered entities tokens
        filtered_entity_ids - filtered entities_ids
        targets - array, where each i-th element is a position in embedding's
                  matrix of entity embedding of i-th corresponding mention.
    """
    ent_indexes_map = {}
    targets = []
    filteres_entities = []
    filtered_entity_ids = []

    for ent_id, ent_data in zip(entity_ids, entities_data):
        if ent_id in ent_indexes_map:
            targets.append(ent_indexes_map[ent_id])
        else:
            ent_idx = len(ent_indexes_map)
            targets.append(ent_idx)
            ent_indexes_map[ent_id] = ent_idx
            filteres_entities.append(ent_data)
            filtered_entity_ids.append(ent_id)

    # Append `max_negative_entities_in_batch` entities to the end of batch
    neg_entities_ids = neg_entities_ids or []
    neg_entities_tokens = neg_entities_tokens or []

    neg_filteres_entities = []
    neg_filtered_entity_ids = []
    for item_neg_entities_ids, item_neg_entities_tokens in zip(
        neg_entities_ids,
        neg_entities_tokens,
    ):
        for neg_entity_id, neg_entity_tokens in zip(
            item_neg_entities_ids, item_neg_entities_tokens
        ):
            if neg_entity_id not in ent_indexes_map:
                ent_idx = len(ent_indexes_map)
                ent_indexes_map[neg_entity_id] = ent_idx
                neg_filteres_entities.append(neg_entity_tokens)
                neg_filtered_entity_ids.append(neg_entity_id)

    if max_negative_entities_in_batch is not None:
        neg_filteres_entities = neg_filteres_entities[:max_negative_entities_in_batch]
        neg_filtered_entity_ids = neg_filtered_entity_ids[
            :max_negative_entities_in_batch
        ]

    filteres_entities.extend(neg_filteres_entities)
    filtered_entity_ids.extend(neg_filtered_entity_ids)

    return filteres_entities, filtered_entity_ids, targets
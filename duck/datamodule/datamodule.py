import json
import logging
import mmap
from typing import List

import torch
from pytorch_lightning import LightningDataModule

logger = logging.getLogger()

class EDGenreDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset for entity disambiguation with GENRE open-source format.
    (https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh)
    Each example in this dataset contains one mention.
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

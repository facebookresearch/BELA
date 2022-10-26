import json
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
import faiss

from mblink.utils.utils import EntityCatalogue
from duck.datamodule import RelationCatalogue
from duck.common.utils import device, load_json, load_jsonl
import hydra
from omegaconf import OmegaConf, DictConfig
import math


logger = logging.getLogger()


class DuckIndex:
    def __init__(self,
        entities: List[str],
        relations: List[str],
        ent_to_rel: Dict[str, List[str]],
        stop_rels: Optional[Set[str]] = None,
        gpu: bool = False,
        num_shards = 1,
        batch_size: int = 1000,
        hnsw_args: Optional[Dict] = None
    ) -> None:
        self.entities = entities
        self.rel_catalogue = {r: i for i, r in enumerate(relations)}
        self.ent_to_rel = ent_to_rel
        self.gpu = gpu
        self.batch_size = batch_size
        self.stop_rels = stop_rels or set()
        nrels = len(relations)
        self.dim = nrels
        if nrels % 8 != 0:
            self.dim = nrels + 8 - (nrels % 8)
        self.eye = np.eye(self.dim).astype(np.uint8)
        if gpu:
            logger.info("Instantiating flat index on GPU")
            self.indices = []
            self.res = faiss.StandardGpuResources()
            for _ in range(num_shards):
                index = faiss.IndexBinaryFlat(self.dim)
                index = faiss.GpuIndexBinaryFlat(self.res, index)
                self.indices.append(index)
            return
        logger.info("Instantiating HNSW index on CPU")
        self.indices = [self.reinit_hnsw(hnsw_args)]
        for index in self.indices:
            assert index.is_trained

    def reinit_hnsw(self, hnsw_args):
        branching_factor = 32
        if "branching_factor" in hnsw_args and hnsw_args["branching_factor"] is not None:
            branching_factor = hnsw_args["branching_factor"]
        index = faiss.IndexBinaryHNSW(self.dim, branching_factor)
        if "ef_search" in hnsw_args and hnsw_args["ef_search"] is not None:
            index.hnsw.efSearch = hnsw_args["ef_search"]
        if "ef_construction" in hnsw_args and hnsw_args["ef_construction"] is not None:
            index.hnsw.efConstruction = hnsw_args["ef_construction"]
        return index
        
    def _build_repr(
        self, 
        entities: Union[str, List[str]]
    ) -> np.array:
        if isinstance(entities, str):
            entities = [entities]
        one_hots = []
        for e in entities:
            rels = set(self.ent_to_rel[e]) - self.stop_rels
            indexes = [self.rel_catalogue[r] for r in rels]
            one_hot = np.sum(self.eye[indexes], axis=0)
            one_hots.append(one_hot)
        data = np.stack(one_hots)
        return np.packbits(data, axis=-1)

    def build(
        self,
        limit: Optional[int] = None
    ):  
        max_size = len(self.entities)
        if limit is not None and limit < max_size:
            max_size = limit
        num_shards = len(self.indices)
        max_shard_size = int(math.ceil(max_size / num_shards))
        j = 0
        for i in tqdm(range(0, max_size, self.batch_size)):
            batch = self.entities[i : (i + self.batch_size)]
            data = self._build_repr(batch)
            self.indices[j].add(data)
            if self.indices[j].ntotal >= max_shard_size:
                j += 1
    
    def search(
        self,
        query: Union[str, List[str]],
        k: int,
        distinct: bool = True
    ) -> Dict[str, List[str]]:
        if isinstance(query, str):
            query = [query]
        
        result = {}

        for i in tqdm(range(0, len(query), self.batch_size)):
            batch = query[i : (i + self.batch_size)]
            batch_neighbors = self._search_batch(batch, k, distinct=distinct)
            result.update(batch_neighbors)
        return result

    def _search_batch(
        self,
        batch: List[str],
        k: int,
        distinct: bool = True
    ) -> Dict[str, List[str]]:
        distances = []
        ids = []
        batch_repr = self._build_repr(batch)
        offset = 0
        for index in self.indices:
            shard_dist, shard_ids = index.search(batch_repr, k=k + 1)
            shard_ids += offset
            distances.append(shard_dist)
            ids.append(shard_ids)
            offset += index.ntotal

        distances = np.concatenate(distances, axis=-1)
        ids = np.concatenate(ids, axis=-1)
        argsort = np.argsort(distances, axis=-1)

        neighbors = {}
        for i, ent in enumerate(batch):
            neighbors[ent] = []
            sorted_ids = ids[i][argsort[i]]
            sorted_distances = distances[i][argsort[i]]
            neighbor_rels = set()
            for j, neighbor_id in enumerate(sorted_ids):
                neighbor = self.entities[neighbor_id]
                if neighbor == ent:
                    continue
                unseen_neighbor = True
                if distinct:
                    rels = frozenset(self.ent_to_rel[neighbor])
                    unseen_neighbor = rels not in neighbor_rels
                    if unseen_neighbor:
                        neighbor_rels.add(rels)
                    unseen_neighbor = unseen_neighbor and sorted_distances[j] > 0
                if (not distinct) or unseen_neighbor:
                    neighbors[ent].append(neighbor)
                if len(neighbors[ent]) == k:
                    break
        return neighbors
            
    def _dump_shard(
        self,
        shard_index: int, 
        output_path: str
    ):
        index = self.indices[shard_index]
        if self.gpu:
            index = faiss.IndexBinaryFlat(self.dim)
            self.indices[shard_index].copyTo(index)
        faiss.write_index_binary(index, output_path)
    
    def dump(
        self,
        output_path: str
    ):  
        num_shards = len(self.indices)
        if len(self.indices) > 1:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            for i in range(num_shards):
                filename = f"shard{i:02d}.faiss"
                self._dump_shard(i, str(output_path / filename))
            return
        self._dump_shard(0, output_path)

    def _load_shard(
        self, 
        shard_index: int,
        input_path: str
    ):
        index = faiss.read_index_binary(input_path)
        if not self.gpu:
            self.indices[shard_index] = index
            return
        self.indices[shard_index].copyFrom(index)
    
    def load(
        self,
        input_path
    ):
        num_shards = len(self.indices)
        if num_shards == 1:
            self.load_shard(0, input_path)
            return
        input_path = Path(input_path)
        assert input_path.is_dir(), \
            "The index has multiple shard but only one file was given"
        shard_files = list(input_path.glob("shard*.faiss"))
        shard_files = sorted(shard_files)
        assert len(shard_files) == num_shards, \
            f"Expected {num_shards} shards, got {len(shard_files)}"
        for i, file in enumerate(shard_files):
            self._load_shard(i, str(file))

    @staticmethod
    def empty_index(config):
        ent_catalogue = EntityCatalogue(
            config.ent_catalogue_path,
            config.ent_catalogue_idx_path
        )
        rel_catalogue = RelationCatalogue(
            config.rel_catalogue_path,
            config.rel_catalogue_idx_path
        )

        logger.info("Reading mapping from entities to relations")
        ent_to_rel = load_json(config.ent_to_rel_path)
        
        assert list(ent_catalogue.idx.values()) == list(range(len(ent_catalogue)))
        assert list(rel_catalogue.idx.values()) == list(range(len(rel_catalogue)))

        entities = list(ent_catalogue.idx.keys())
        relations = list(rel_catalogue.idx.keys())

        stop_rels = set()
        if config.stop_rels_path is not None:
            stop_rels = load_jsonl(config.stop_rels_path)
            stop_rels = set(r["id"] for r in stop_rels)

        duck_index = DuckIndex(
            entities,
            relations,
            ent_to_rel,
            stop_rels=stop_rels,
            gpu=config.gpu,
            num_shards=config.num_shards,
            batch_size=config.batch_size,
            hnsw_args=OmegaConf.to_container(config.hnsw)
        )
        return duck_index

    @staticmethod
    def build_index(config):
        duck_index = DuckIndex.empty_index(config)
        logger.info("Building index")
        duck_index.build()

        logger.info("Dumping index")
        duck_index.dump(config.index_path)
        return duck_index
    
    @staticmethod
    def build_or_load(config):
        if not Path(config.index_path).exists():
            return DuckIndex.build_index(config)
        duck_index = DuckIndex.empty_index(config)
        logger.info(f"Loading index at {config.index_path}")
        duck_index.load(config.index_path)
        return duck_index


@hydra.main(config_path="../conf/preprocessing", config_name="duck_neighbors")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    DuckIndex.build_index(config)


if __name__ == "__main__":
    main()

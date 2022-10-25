import json
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
import faiss

from mblink.utils.utils import EntityCatalogue
from duck.datamodule import RelationCatalogue
from duck.common.utils import load_json, load_jsonl, most_frequent_relations
import hydra
from omegaconf import OmegaConf, DictConfig


logger = logging.getLogger()


class DuckIndex:
    def __init__(self,
        entities: List[str],
        relations: List[str],
        ent_to_rel: Dict[str, List[str]],
        stop_rels: Optional[Set[str]] = None,
        gpu: bool = False,
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
            self.index = faiss.IndexBinaryFlat(self.dim)
            self.res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexBinaryFlat(self.res, self.index)
            return
        logger.info("Instantiating HNSW index on CPU")
        self.index = self.reinit_hnsw(hnsw_args)
        assert self.index.is_trained

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
        for i in tqdm(range(0, max_size, self.batch_size)):
            batch = self.entities[i : (i + self.batch_size)]
            data = self._build_repr(batch)
            self.index.add(data)
    
    def search(
        self,
        query: Union[str, List[str]],
        k: int
    ) -> Dict[str, List[str]]:
        if isinstance(query, str):
            query = [query]
        
        result = {}

        for i in tqdm(range(0, len(query), self.batch_size)):
            batch = query[i : (i + self.batch_size)]
            batch_repr = self._build_repr(batch)
            _, ids = self.index.search(batch_repr, k=k + 1)
        
            for i, ent in enumerate(batch):
                result[ent] = []
                for ent_id in ids[i]:
                    neighbor = self.entities[ent_id]
                    if neighbor != ent:
                        result[ent].append(neighbor)
                result[ent] = result[ent][:k]
        return result
            
    def dump(
        self,
        output_path: str
    ):
        index = self.index
        if self.gpu:
            index = faiss.IndexBinaryFlat(self.dim)
            self.index.copyTo(index)
        faiss.write_index_binary(index, output_path)

    def load(
        self, 
        input_path
    ):
        index = faiss.read_index_binary(input_path)
        if not self.gpu:
            self.index = index
            return
        self.index.copyFrom(index)
        
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
        duck_index = DuckIndex.empty_index(config)
        if Path(config.index_path).exists():
            logger.info(f"Loading index at {config.index_path}")
            duck_index.load(config.index_path)
            return duck_index
        return duck_index.build_index()


@hydra.main(config_path="../conf/preprocessing", config_name="duck_index")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    DuckIndex.build_index(config)


if __name__ == "__main__":
    main()

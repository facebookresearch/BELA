from tqdm import tqdm
import logging
from duck.preprocessing.duck_index import DuckIndex

import hydra
from omegaconf import OmegaConf, DictConfig
import json

logger = logging.getLogger()


def print_example(duck_index, distinct=True):
    print()
    title = "Neighbors"
    if distinct:
        title = "Distinct Neighbors"
    print("=" * 8 + f" {title} " + "=" * 8)
    example = duck_index.search([
        "Italy",
        "Donald Trump",
        "Cristiano Ronaldo",
        "Justin Bieber",
        "London",
        "Rome",
        "Lion",
        "Jaguar",
        "Jaguar Cars",
        "Maserati",
        "Alan Turing",
        "Gianluigi Buffon",
        "Ada Lovelace",
        "Fields Medal"
    ], k=4, distinct=distinct)
    for ent, neighbors in example.items():
        neighbors_str = ", ".join(neighbors)
        print(ent + ": " + neighbors_str)
    print("=" * 27)
    print()


def build_neighbors(config):
    duck_index = DuckIndex.build_or_load(config)
    print_example(duck_index, distinct=False)
    print_example(duck_index, distinct=True)
    logger.info("Building neighbors")
    entities = duck_index.entities
    neighbors = duck_index.search(entities, k=config.num_neighbors, distinct=config.distinct)
    logger.info("Dumping neighbors")
    with open(config.neighbors_path, "w") as f:
        json.dump(neighbors, f)


@hydra.main(config_path="../conf/preprocessing", config_name="duck_neighbors", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    build_neighbors(config)


if __name__ == "__main__":
    main()

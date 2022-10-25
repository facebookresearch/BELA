from tqdm import tqdm
import logging
from duck.preprocessing.duck_index import DuckIndex

import hydra
from omegaconf import OmegaConf, DictConfig
import json

logger = logging.getLogger()


def print_example(duck_index):
    print()
    print("=" * 8 + " Example " + "=" * 8)
    example = duck_index.search([
        "Italy",
        "Donald Trump",
        "Cristiano Ronaldo",
        "Justin Bieber",
        "London",
        "Lion"
    ], k=4)
    for ent, neighbors in example.items():
        neighbors_str = ", ".join(neighbors)
        print(ent + ": " + neighbors_str)
    print("=" * 25)
    print()


def build_negatives(config):
    duck_index = DuckIndex.build_or_load(config)
    print_example(duck_index)
    logger.info("Building negatives")
    entities = duck_index.entities
    negatives = duck_index.search(entities, k=config.num_negatives)
    logger.info("Dumping negatives")
    with open(config.negatives_path, "w") as f:
        json.dump(negatives, f)


@hydra.main(config_path="../conf/preprocessing", config_name="duck_index")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    build_negatives(config)

if __name__ == "__main__":
    main()

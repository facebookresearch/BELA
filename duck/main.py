import hydra
from omegaconf import OmegaConf, DictConfig
from duck.common.utils import load_json
from duck.datamodule import EdGenreDataset
from duck.datamodule.datamodule import DuckTransform, EdDuckDataModule, EdDuckDataset, RelationCatalogue
from mblink.transforms.blink_transform import BlinkTransform
from mblink.utils.utils import EntityCatalogue

import pickle
from transformers import AutoTokenizer
import logging

logger = logging.getLogger()

@hydra.main(config_path="conf", config_name="duck_conf")
def main(config: DictConfig):
    transform = DuckTransform(
        config.language_model
    )

    data = EdDuckDataModule(
        transform,
        config.data.train_path,
        config.data.val_path,
        config.data.test_path,
        config.data.ent_catalogue_path,
        config.data.ent_catalogue_idx_path,
        config.data.rel_catalogue_path,
        config.data.rel_catalogue_idx_path,
        config.data.ent_to_rel_path,
        config.batch_size
    )

    dataloader = data.train_dataloader()
    for batch in dataloader:
        print("yuhuuuu")
        break


if __name__ == "__main__":
    main()

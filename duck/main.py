import hydra
from omegaconf import OmegaConf, DictConfig
from duck.datamodule.datamodule import DuckTransform, EdDuckDataModule

import logging

logger = logging.getLogger()

@hydra.main(config_path="conf", config_name="duck_conf", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    transform = DuckTransform(
        config.language_model,
        max_mention_len=config.max_mention_len,
        max_entity_len=config.max_entity_len,
        max_relation_len=config.max_relation_len
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
        neighbors_path=config.data.neighbors_path,
        batch_size=config.batch_size
    )

    dataloader = data.train_dataloader()
    for batch in dataloader:
        print("yuhuuuu")
        break


if __name__ == "__main__":
    main()

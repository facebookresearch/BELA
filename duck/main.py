import hydra
from omegaconf import OmegaConf, DictConfig
from duck.datamodule import EDGenreDataset
from mblink.utils.utils import EntityCatalogue


@hydra.main(config_path="conf", config_name="duck_conf")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    ent_catalogue = EntityCatalogue(
        config.data.ent_catalogue_path,
        config.data.ent_catalogue_idx_path
    )
    dataset = EDGenreDataset(config.data.train_path, ent_catalogue)
    print(dataset[0])


if __name__ == "__main__":
    main()

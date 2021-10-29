# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
from bela.conf.config import MainConfig
import os.path
from bela.datamodule.entity_encoder import embed

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
import os
seed_everything(1)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    print(OmegaConf.to_yaml(cfg))

    test_data_name = cfg.datamodule.test_path.split('/')[-1].split(".")[0]
    print(cfg.task)
    
    if not os.path.isdir(cfg.task.save_embeddings_path):
        os.mkdir(cfg.task.save_embeddings_path)
    cfg.task.save_embeddings_path += test_data_name
    cfg.task.save_embeddings_path += '/'
    if not os.path.isdir(cfg.task.save_embeddings_path):
        os.mkdir(cfg.task.save_embeddings_path)

    task = hydra.utils.instantiate(cfg.task, _recursive_=False)


    assert cfg.task.model.model_path == cfg.task.transform.model_path
    transform = hydra.utils.instantiate(cfg.task.transform)
    datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)
    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint_callback)
    trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback])

    ckpt_path = cfg.task.load_from_checkpoint
    trainer.test(
        model=task,
        ckpt_path=ckpt_path,
        verbose=True,
        datamodule=datamodule,
    )

if __name__ == "__main__":
    main()

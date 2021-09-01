# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
from bela.conf.config import MainConfig

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    print(OmegaConf.to_yaml(cfg))
    # cfg.task.datamodule = None
    task = hydra.utils.instantiate(cfg.task, _recursive_=False)

    assert cfg.task.model.model_path == cfg.task.transform.model_path
    transform = hydra.utils.instantiate(cfg.task.transform)
    datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)
    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint_callback)
    trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback])

    if cfg.test_only:
        ckpt_path = cfg.task.pretrained_checkpoint_path
        trainer.test(
            model=task,
            ckpt_path=ckpt_path,
            verbose=True,
            datamodule=datamodule,
        )
    else:
        trainer.fit(task, datamodule=datamodule)
        print(f"*** Best model path is {checkpoint_callback.best_model_path}")
        trainer.test(
            model=None,
            ckpt_path="best",
            verbose=True,
            datamodule=datamodule,
        )


if __name__ == "__main__":
    main()

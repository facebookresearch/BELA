import os
import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning.trainer import Trainer
from duck.common.utils import make_reproducible
from duck.task.duck_entity_disambiguation import Duck
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path


def configure_wandb_logger(config, model):
    if ("debug" in config and config.debug) or "fast_dev_run" in config.trainer:
        return None 
    run_name = config.run_name if "run_name" in config else None
    logger = WandbLogger(
        project="duck",
        name=run_name,
        log_model=True,
        save_dir=config.log_dir,
        config=dict(config)
    )
    wandb.watch(model, log="all", log_freq=1)
    return logger
    

@hydra.main(config_path="conf", config_name="duck_conf", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    make_reproducible(ngpus=config.trainer.devices)
    
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "2"

    if config.get("debug_mode"):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    os.environ["PL_SKIP_CPU_COPY_ON_DDP_TEARDOWN"] = "1"

    assert config.duck.base_model.model_path == config.data.transform.model_path

    Path(config.log_dir).mkdir(exist_ok=True, parents=True)
    Path(config.ckpt_dir).mkdir(exist_ok=True, parents=True)

    task = Duck(config) 
    datamodule = hydra.utils.instantiate(config.data)
    logger = configure_wandb_logger(config, task)
    checkpoint_callback = hydra.utils.instantiate(config.checkpoint_callback)
    checkpoint_callback.CHECKPOINT_NAME_LAST = config.checkpoint_callback.filename + "_last"

    trainer = Trainer(
        **config.trainer,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()

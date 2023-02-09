import os
import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning.trainer import Trainer
from duck.common.utils import seed_prg
from duck.task.duck_entity_disambiguation import Duck
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def configure_wandb_logger(config):
    if not config.get("wandb"):
        return None
    if config.get("debug") or "fast_dev_run" in config.trainer:
        return None 
    run_name = config.run_name if "run_name" in config else None
    wandb_logger = WandbLogger(
        project="duck",
        name=run_name,
        log_model=True,
        save_dir=config.log_dir,
        config=dict(config)
    )
    # wandb.watch(model, log="all", log_freq=100)
    return wandb_logger
    

@hydra.main(config_path="conf", config_name="duck", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    seed_prg(ngpus=config.trainer.devices)

    # torch.set_float32_matmul_precision("high")
    
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "2"

    if config.get("debug"):
        torch.autograd.set_detect_anomaly(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    os.environ["PL_SKIP_CPU_COPY_ON_DDP_TEARDOWN"] = "1"
    os.environ["HYDRA_FULL_ERROR"] = "1"

    Path(config.log_dir).mkdir(exist_ok=True, parents=True)
    Path(config.ckpt_dir).mkdir(exist_ok=True, parents=True)

    datamodule = hydra.utils.instantiate(config.data, debug=config.debug)
    task = hydra.utils.instantiate(config.task, config, data=datamodule)     

    callbacks = []
    if not config.get("debug") and config.get("checkpoint_callback") is not None:
        checkpoint_callback = hydra.utils.instantiate(config.checkpoint_callback)
        checkpoint_callback.CHECKPOINT_NAME_LAST = config.checkpoint_callback.filename + "_last"
        callbacks.append(checkpoint_callback)
    
    if config.get("early_stopping") is not None:
        early_stopping = hydra.utils.instantiate(config.early_stopping)
        callbacks.append(early_stopping)

    logger = configure_wandb_logger(config)

    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=logger
    )
    trainer.fit(task, datamodule=datamodule, ckpt_path=config.get("resume_path"))


if __name__ == "__main__":
    main()

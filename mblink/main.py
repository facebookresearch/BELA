# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import hydra
from mblink.conf.config import MainConfig

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    print(OmegaConf.to_yaml(cfg))

    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "2"

    if cfg.get("debug_mode"):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    os.environ["PL_SKIP_CPU_COPY_ON_DDP_TEARDOWN"] = "1"

    task = hydra.utils.instantiate(cfg.task, _recursive_=False)

    assert cfg.task.model.model_path == cfg.task.transform.model_path
    transform = hydra.utils.instantiate(cfg.task.transform)
    datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)
    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint_callback)
    trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback])

    if cfg.test_only:
        ckpt_path = cfg.task.load_from_checkpoint
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

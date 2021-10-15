# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
from bela.conf.config import MainConfig
import os.path
from bela.datamodule.entity_encoder import embed

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    print(OmegaConf.to_yaml(cfg))
    # cfg.task.datamodule = None

    if cfg.datamodule.novel_entity_idx_path:
        # TODO: externalize into conf folder
        params = {'lower_case': True,
                  'path_to_model': '/data/home/kassner/BELA/data/blink/biencoder_wiki_large.bin',
                  'data_parallel': True,
                  'no_cuda': False,
                  'bert_model': 'bert-large-uncased',
                  'lowercase': True,
                  'out_dim': 1,
                  'pull_from_layer': -1,
                  'add_linear': False,
                  'entity_dict_path': '/data/home/kassner/BELA/data/blink/novel_entities.jsonl',
                  'debug': False,
                  'max_cand_length': 128,
                  'encode_batch_size': 8,
                  'silent': False,
                  }
        cfg.task.novel_entity_embeddings_path = '.'.join(params['entity_dict_path'].split('.')[:-1]) + ".t7"
        if not os.path.isfile(cfg.task.novel_entity_embeddings_path):
            embed(params)
    print(cfg.task)

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

from typing import Any
from tqdm import tqdm
import logging
import torch
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
import json

from duck.task.duck_entity_disambiguation import Duck
import pytorch_lightning as pl
import math
from pytorch_lightning.trainer import Trainer
from einops import rearrange

logger = logging.getLogger()


from typing import Any, Sequence
from tqdm import tqdm
import logging
import torch
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
import json

from duck.task.duck_entity_disambiguation import Duck
import pytorch_lightning as pl
import math
from pytorch_lightning.trainer import Trainer
from einops import rearrange

logger = logging.getLogger()


### SEQUENTIAL
class DuckHardNegativeMiner:
    def __init__(self, config):
        self.config = config
        self.duck = Duck.load_from_checkpoint(config.ckpt_path, strict=False)
        self.duck.eval()
        with open_dict(self.duck.config):
            self.duck.config.debug = config.debug
            self.duck.config.trainer.devices = config.devices
            self.duck.config.num_nodes = config.num_nodes
        self.index = None
        self.device = torch.device("cuda")
        self.duck = self.duck.cuda().to(self.device)
        self.duck.setup_entity_index_sequential()
        self.index = self.duck.ent_index.detach().to(self.device)
        if self.config.get("output_index_path"):
            torch.save(self.index.cpu(), config.output_index_path)
        
    def mine_negatives(self):
        self.duck.eval()
        torch.set_grad_enabled(False)
        bsz = self.config.batch_size
        index_size = self.index.size(0)
        if self.config.debug:
            index_size = 10000
            logger.info(f"Debug mode: using an index size of {index_size}")
        negative_indices = []
        for i in tqdm(range(0, index_size, bsz)):
            batch = self.index[i:i + bsz]
            scores = torch.matmul(
                batch,
                self.duck.ent_index.transpose(0, 1)
            )
            topk = scores.topk(k=self.config.num_negatives + 1, dim=-1).indices
            negative_indices.append(topk)
        negative_indices = torch.cat(negative_indices).cpu().tolist()
        result = {}
        labels = self.duck.data.ent_catalogue.entities
        for i, negatives in enumerate(negative_indices):
            result[labels[i]] = [labels[n] for n in negatives]
        with open(self.config.output_path, "w") as f:
            json.dump(result, f)


@hydra.main(config_path="../conf/preprocessing", config_name="negative_mining", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    miner = DuckHardNegativeMiner(config)
    if not config.only_save_index:
        miner.mine_negatives()


if __name__ == "__main__":
    main()


### DISTRIBUTED
# class DuckHardNegativeMiner(pl.LightningModule):
#     def __init__(self, config):
#         super(DuckHardNegativeMiner, self).__init__()
#         self.config = config
#         self.duck = Duck.load_from_checkpoint(config.ckpt_path)
#         with open_dict(self.duck.config):
#             self.duck.config.debug = config.debug
#             self.duck.config.trainer = config.lightning
#         self.data = None
#         self.world_size = self.config.lightning.devices * self.config.lightning.num_nodes
#         self.automatic_optimization =  False
#         self.linear = torch.nn.Linear(10, 10)

#     def setup(self, stage):
#         # self.duck = self.duck.to(self.device)
#         self.duck = self.duck.to(f"cuda:{self.local_rank}")
#         self.duck.update_entity_index()
#         num_entities = self.duck.ent_index.size(0)
#         local_data_size = math.ceil(num_entities / self.world_size)
#         start = self.global_rank * local_data_size
#         end = min((self.global_rank + 1) * local_data_size, num_entities)
#         self.data = self.duck.ent_index[start:end]
    
#     def train_dataloader(self):
#         # self.prepare_data()
#         return torch.utils.data.DataLoader(
#             self.data,
#             batch_size=self.config.batch_size,
#             shuffle=False
#         )
    
#     def transfer_batch_to_device(
#         self,
#         batch: Any,
#         device: torch.device = None,
#         dataloader_idx: int = 0
#     ) -> Any:
#         device = device or self.device
#         return super().transfer_batch_to_device(batch, device, dataloader_idx)

#     def training_step(self, batch, batch_idx):
#         self.eval()
#         torch.set_grad_enabled(False)
#         scores = torch.matmul(
#             batch,
#             self.duck.ent_index.transpose(0, 1)
#         )
#         topk = scores.topk(k=self.config.num_negatives + 1, dim=-1).indices
#         return {
#             "topk": topk
#         }
    
#     def training_epoch_end(self, outputs):
#         topk = torch.cat([o["topk"] for o in outputs])
#         topk = self.all_gather(topk)
#         if topk.dim() == 3:
#             topk = rearrange(topk, "w e k -> (w e) k")
#         topk = topk.cpu().tolist()
#         labels = self.duck.data.ent_catalogue.entities
#         result = {}
#         for i, label in enumerate(labels):
#             result[label] = [labels[n] for n in topk[i][1:]]
#         with open(self.config.output_path, "w") as f:
#             json.dump(result, f)
        
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.0001)


# @hydra.main(config_path="../conf/preprocessing", config_name="negative_mining", version_base=None)
# def main(config: DictConfig):
#     print(OmegaConf.to_yaml(config))
#     miner = DuckHardNegativeMiner(config)
#     trainer = Trainer(**config.lightning, max_epochs=1)
#     trainer.fit(
#         model=miner
#     )


# if __name__ == "__main__":
#     main()

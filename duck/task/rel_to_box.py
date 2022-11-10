import pytorch_lightning as pl
from duck.box_tensors.intersection import Intersection
from duck.box_tensors.volume import Volume
from duck.common.utils import tiny_value_of_dtype
from duck.modules import BoxEmbedding
import hydra
from torch import nn
import torch
import logging

logger = logging.getLogger()

class BinaryKLDivLoss(nn.Module):
    def __init__(self) -> None:
        super(BinaryKLDivLoss, self).__init__()
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.eps = tiny_value_of_dtype(torch.float)
    
    def forward(self, log_prediction, target):
        prediction = torch.stack([
            log_prediction,
            torch.log(1 - log_prediction.exp() + self.eps)
        ], dim=-1)
        target = torch.stack([target, 1 - target], dim=-1)
        return self.kldiv(prediction, target)


class RelToBox(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super(RelToBox, self).__init__()
        logger.info("Instantiating RelToBox model")
        self.config = config
        self.save_hyperparameters(config)
        self.num_embeddings = kwargs["data"].num_rels + 1
        self.dim = config.rel_to_box.dim
        self.box_embedding = BoxEmbedding(
            self.num_embeddings,
            self.dim,
            universe_idx=0
        )
        self.intersection = Intersection(
            intersection_temperature=config.rel_to_box.boxes.intersection_temperature
        )
        self.volume = Volume(
            intersection_temperature=config.rel_to_box.boxes.intersection_temperature,
            volume_temperature=config.rel_to_box.boxes.volume_temperature
        )
        self.loss = BinaryKLDivLoss()
    
    def forward(self, rel_ids):
        boxes = self.box_embedding(rel_ids)
        box_r0 = boxes[:, 0]
        box_r1 = boxes[:, 1]
        intersection = self.intersection(box_r0, box_r1)
        return self.volume(intersection)

    def training_step(self, batch, batch_idx):
        rel_ids = batch["rel_ids"]
        target_probability = batch["target_probability"]
        log_intersection = self(rel_ids)
        train_loss = self.loss(log_intersection, target_probability)
        if self.logger is not None:
            self.log_dict({
                "train_loss": train_loss
            })
        return train_loss
    
    def configure_optimizers(self):
        return hydra.utils.instantiate(
            self.config.optim, self.parameters(), _recursive_=False
        )
    

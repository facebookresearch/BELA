import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import open_dict
from torch import nn

from duck.box_tensors.intersection import Intersection
from duck.box_tensors.regularization import L2BoxSideRegularizer
from duck.box_tensors.volume import Volume
from duck.common.utils import log1mexp, mean_over_batches, metric_dict_to_string, prefix_suffix_keys, tiny_value_of_dtype
from duck.modules import BoxEmbedding
import torchmetrics
from tqdm import tqdm


logger = logging.getLogger()

class BinaryKLDivLoss(nn.Module):
    def __init__(self) -> None:
        super(BinaryKLDivLoss, self).__init__()
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.eps = tiny_value_of_dtype(torch.float)
    
    def forward(self, log_prediction, target):
        prediction = torch.stack([
            log_prediction,
            log1mexp(log_prediction)
        ], dim=-1)
        target = torch.stack([target, 1.0 - target], dim=-1)
        return self.kldiv(prediction, target)


class RelToBox(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super(RelToBox, self).__init__()
        logger.info("Instantiating RelToBox model")
        self.config = config
        self.num_embeddings = config.get("num_embeddings")
        if self.num_embeddings is None:
            self.num_embeddings = kwargs["data"].num_rels + 1
            with open_dict(self.config):
                self.config.num_embeddings = self.num_embeddings
        self.save_hyperparameters(config)
        self.dim = config.rel_to_box.dim
        self.box_embedding = None
        self.intersection = None
        self.volume = None
        self.kldiv = None
        self.regularizer = None
        self.setup(None)
        
    
    def setup(self, stage: str):
        if stage == "test":
            return
        self.call_configure_sharded_model_hook = False
        self.box_embedding = BoxEmbedding(
            self.num_embeddings,
            self.dim,
            box_parametrizaton=self.config.rel_to_box.boxes.parametrization,
            universe_idx=0
        )
        self.intersection = Intersection(
            intersection_temperature=self.config.rel_to_box.boxes.intersection_temperature
        )
        self.volume = Volume(
            intersection_temperature=self.config.rel_to_box.boxes.intersection_temperature,
            volume_temperature=self.config.rel_to_box.boxes.volume_temperature
        )
        self.kldiv = BinaryKLDivLoss()
        self.regularizer = None
        if "regularization" in self.config.rel_to_box:
            self.regularizer = hydra.utils.instantiate(
                self.config.rel_to_box.regularization
            )

    
    def forward(self, rel_ids):
        return self.box_embedding(rel_ids)

    def log_prob(self, rel_ids):
        boxes = self(rel_ids)
        return self.boxes_logprob(boxes)

    def boxes_logprob(self, boxes):
        box_r0 = boxes[:, 0]
        box_r1 = boxes[:, 1]
        intersection = self.intersection(box_r0, box_r1)
        return self.volume(intersection) - self.volume(box_r1)

    def metrics(
        self,
        log_prediction,
        target,
        prefix=None,
        suffix=None,
        is_train=False,
        **kwargs
    ):
        kldiv = self.kldiv(log_prediction, target)
        regularization = 0.0
        if self.regularizer is not None and "boxes" in kwargs:
            regularization = self.regularizer(kwargs["boxes"])
        loss = kldiv + regularization
        metrics = {
            "kldiv": kldiv,
            "regularization": regularization,
            "loss": loss
        }
        if not is_train:
            pearson = torchmetrics.functional.pearson_corrcoef(log_prediction.exp(), target)
            if pearson.isnan().any().item():
                pearson = torch.tensor(0.0)
            spearman = torchmetrics.functional.spearman_corrcoef(log_prediction.exp(), target)
            metrics.update({
                "pearson": pearson,
                "spearman": spearman
            })
        return prefix_suffix_keys(metrics, prefix=prefix, suffix=suffix)
        
    def training_step(self, batch, batch_idx):
        rel_ids = batch["rel_ids"]
        target_probability = batch["target_probability"]
        boxes = self(rel_ids)
        log_pred = self.boxes_logprob(boxes)
        metrics = self.metrics(
            log_pred, target_probability, suffix="train", is_train=True, boxes=boxes
        )
        self.log_dict(metrics)
        metrics["loss"] = metrics["loss_train"]
        return metrics
    
    def training_epoch_end(self, outputs):
        mean_metrics = mean_over_batches(outputs, prefix="avg")
        self.log_dict(mean_metrics, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        rel_ids = batch["rel_ids"]
        target_probability = batch["target_probability"]
        boxes = self(rel_ids)
        log_pred = self.boxes_logprob(boxes)
        return {
            "log_prediction": log_pred,
            "target": target_probability
        }
    
    def validation_epoch_end(self, outputs):
        log_pred = [output["log_prediction"] for output in outputs]
        target = [output["target"] for output in outputs]
        log_pred = torch.cat(log_pred, dim=0)
        target = torch.cat(target, dim=0)
        metrics = self.metrics(log_pred, target, suffix="val")
        self.log_dict(metrics, sync_dist=True)
        metrics_to_print = ["kldiv_val", "pearson_val", "spearman_val"]
        message = metric_dict_to_string({
            k: v for k, v in metrics.items() if k in metrics_to_print
        })
        tqdm.write("\n" + message + "\n")

    def configure_optimizers(self):
        return hydra.utils.instantiate(
            self.config.optim, self.parameters(), _recursive_=False
        )
    

import math
from typing import Any, List
import pytorch_lightning as pl
import hydra
import logging
import torch
import torch.nn as nn
from torch import Tensor
from duck.box_tensors.box_tensor import BoxTensor
from duck.common.utils import cartesian_to_spherical, expand_box_with_mask, expand_with_mask, list_to_tensor, log1mexp, logexpm1, logsubexp, mean_over_batches, tensor_set_difference, prefix_suffix_keys
from duck.modules.modules import EmbeddingToBox, JointEntRelsEncoder, TransformerSetEncoder
from mblink.task.blink_task import ElBiEncoderTask
from einops import rearrange, repeat
from duck.box_tensors.volume import Volume
from duck.box_tensors.intersection import Intersection
from duck.modules import BoxEmbedding, HFSetToBoxTransformer, SetToBoxTransformer
from duck.task.rel_to_box import RelToBox
import transformers
from abc import ABC, abstractmethod
import faiss
import os
from tqdm import tqdm
import torchmetrics
from torchmetrics.classification import MulticlassF1Score
import wandb
from torchmetrics.retrieval import RetrievalRecall
from torchmetrics import MeanMetric
from pytorch_lightning.strategies import DDPShardedStrategy, DDPStrategy
from omegaconf import open_dict


transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class Q2BDistance(nn.Module):
    def __init__(
        self,
        inside_weight: float = 0.0,
        reduction: str = "none",
        norm: int = 2
    ):
        super(Q2BDistance, self).__init__()
        self.inside_weight = inside_weight
        self.reduction = reduction
        self.norm = norm
    
    def _dist_outside(self, entities, boxes):
        left_delta = boxes.left - entities
        right_delta = entities - boxes.right
        left_delta = torch.max(left_delta, torch.zeros_like(left_delta))
        right_delta = torch.max(right_delta, torch.zeros_like(right_delta))
        return torch.linalg.vector_norm(left_delta + right_delta, ord=self.norm, dim=-1)
    
    def _dist_inside(self, entities, boxes):
        distance =  boxes.center - torch.min(
            boxes.right,
            torch.max(
                boxes.left,
                entities
            )
        )
        return torch.linalg.vector_norm(distance, ord=self.norm, dim=-1)
    
    def _reduce(self, distance):
        if self.reduction == "mean":
            return distance.mean()
        elif self.reduction == "sum":
            return distance.sum()
        elif self.reduction == "none":
            return distance
        raise ValueError(f"Unsupported reduction {self.reduction}")
    
    def forward(self, entities, boxes):
        inside_distance = 0.0
        if self.inside_weight > 0:
            inside_distance = self._dist_inside(entities, boxes)
        outside_distance = self._dist_outside(entities, boxes)
        return outside_distance + self.inside_weight * inside_distance


class BoxEDistance(nn.Module):
    def __init__(
        self,
        reduction: str = "none",
        norm: int = 2
    ):
        super(BoxEDistance, self).__init__()
        self.reduction = reduction
        self.norm = norm

    def forward(self, entity, box):
        width = box.right - box.left
        widthp1 = width + 1
        dist_inside = torch.abs(entity - box.center) / widthp1
        outside_mask = (entity < box.left) | (entity > box.right)
        outside_mask = outside_mask.clone().detach()
        kappa = 0.5 * width * (widthp1 - (1 / widthp1))
        dist_outside = torch.abs(entity - box.center) * widthp1 - kappa
        dist = torch.where(outside_mask, dist_outside, dist_inside)
        return torch.linalg.vector_norm(dist, ord=self.norm, dim=-1).clone()
    

class DuckDistanceRankingLoss(nn.Module):
    def __init__(
        self,
        distance_function=None,
        margin: float = 1.0,
        reduction: str = "mean",
        return_logging_metrics: bool = True,
        inside_weight=0.2,
        norm=1
    ):
        super(DuckDistanceRankingLoss, self).__init__()
        self.distance_function = distance_function or Q2BDistance(
            inside_weight=inside_weight,
            norm=norm
        )
        self.margin = margin
        self.reduction = reduction
        self.return_logging_metrics = return_logging_metrics
    
    def _reduce(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction {self.reduction}")

    def forward(
        self,
        entities: Tensor,
        positive_boxes: BoxTensor,
        negative_boxes: BoxTensor,
        **kwargs
    ):
        negative_boxes = negative_boxes.rearrange("b n d -> n b d")
        if positive_boxes.left.dim() == 3:
            positive_boxes = positive_boxes.rearrange("b n d -> n b d")
        else:
            positive_boxes = positive_boxes.rearrange("b d -> 1 b d")
        
        positive_dist = self.distance_function(entities, positive_boxes)
        negative_dist = self.distance_function(entities, negative_boxes).mean(dim=0)
        
        rel_ids = kwargs.get("rel_ids")
        mask = torch.full_like(positive_dist, True).bool()
        if rel_ids is not None:
            mask = rel_ids["attention_mask"].bool()
            mask = rearrange(mask, "b n -> n b")
        positive_dist[~mask] = 0.0
        positive_dist = positive_dist.sum(dim=0) / mask.sum(dim=0)
        delta = positive_dist - negative_dist + self.margin
        loss = torch.max(delta, torch.zeros_like(delta))
        if not self.return_logging_metrics:
            return loss
        return {
            "loss": self._reduce(loss),
            "positive_distance": wandb.Histogram(positive_dist.detach().cpu().numpy()),
            "negative_distance": wandb.Histogram(negative_dist.detach().cpu().numpy()),
            "positive_distance_mean": positive_dist.mean(),
            "negative_distance_mean": negative_dist.mean()
        }


class DuckBoxMarginLoss(nn.Module):
    def __init__(self,
        margin: float = 0.1,
        reduction: str = "mean"
    ):
        super(DuckBoxMarginLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def _reduce(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction {self.reduction}")

    def forward(
        self,
        entities: Tensor,
        positive_boxes: BoxTensor,
        negative_boxes: BoxTensor,
        **kwargs
    ):
        negative_boxes = negative_boxes.rearrange("b n d -> n b d")
        if positive_boxes.left.dim() == 3:
            positive_boxes = positive_boxes.rearrange("b n d -> n b d")
        else:
            positive_boxes = positive_boxes.rearrange("b d -> 1 b d")

        left_delta_pos = positive_boxes.left - entities + self.margin
        right_delta_pos = entities - positive_boxes.right + self.margin
        
        rel_ids = kwargs.get("rel_ids")
        mask = torch.full_like(left_delta_pos, True).bool()
        if rel_ids is not None:
            mask = rel_ids["attention_mask"].bool()
            mask = rearrange(mask, "b n -> n b")

        left_delta_pos[~mask] = 0.0
        right_delta_pos[~mask] = 0.0

        half_width = (negative_boxes.right - negative_boxes.left) / 2
        delta_neg = half_width - torch.abs(entities - negative_boxes.center) + self.margin
        
        loss_pos = torch.relu(left_delta_pos) + torch.relu(right_delta_pos)
        loss_neg = torch.relu(delta_neg)

        loss = loss_pos.mean(dim=0) + loss_neg.mean(dim=0)

        return {
            "loss": self._reduce(loss)
        }


class DuckNegativeSamplingLoss(nn.Module):
    def __init__(self,
        distance_function=None,
        margin: float = 1.0,
        reduction: str = "mean"
    ):
        super(DuckNegativeSamplingLoss, self).__init__()
        self.distance_function = distance_function
        self.margin = margin
        self.reduction = reduction
        
    def _reduce(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction {self.reduction}")

    def forward(
        self,
        entities: Tensor,
        positive_boxes: BoxTensor,
        negative_boxes: BoxTensor,
        **kwargs
    ):
        negative_boxes = negative_boxes.rearrange("b n d -> n b d")
        if positive_boxes.left.dim() == 3:
            positive_boxes = positive_boxes.rearrange("b n d -> n b d")
        else:
            positive_boxes = positive_boxes.rearrange("b d -> 1 b d")
        
        positive_dist = self.distance_function(entities, positive_boxes)
        negative_dist = self.distance_function(entities, negative_boxes).mean(dim=0)
        
        rel_ids = kwargs.get("rel_ids")
        mask = torch.full_like(positive_dist, True).bool()
        if rel_ids is not None:
            mask = rel_ids["attention_mask"].bool()
            mask = rearrange(mask, "b n -> n b")
        positive_dist[~mask] = 0.0
        positive_dist = positive_dist.sum(dim=0) / mask.sum(dim=0)

        positive_term = torch.nn.functional.logsigmoid(self.margin - positive_dist).mean(dim=0)
        negative_term = torch.nn.functional.logsigmoid(negative_dist - self.margin).mean(dim=0)

        loss = -positive_term - negative_term
        
        return {
            "loss": self._reduce(loss),
            "positive_distance": wandb.Histogram(positive_dist.detach().cpu().numpy()),
            "negative_distance": wandb.Histogram(negative_dist.detach().cpu().numpy()),
            "positive_distance_mean": positive_dist.mean(),
            "negative_distance_mean": negative_dist.mean()
        }

class GumbelBoxProbabilisticMembership(nn.Module):
    def __init__(
        self,
        intersection_temperature: float = 1.0,
        log_scale=True,
        clamp=True,
        dropout: float = 0.9,
        dim=-1
    ):
        super(GumbelBoxProbabilisticMembership, self).__init__()
        self.intersection_temperature = intersection_temperature
        self.log_scale = log_scale
        self.dim = dim
        self.clamp = clamp
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, entity, box):
        logp_gt_left = -torch.exp(
            ((box.left - entity) / self.intersection_temperature).clamp(-100.0, +10.0)
        )

        logp_lt_right = -torch.exp(
            ((entity - box.right) / self.intersection_temperature).clamp(-100.0, +10.0)
        )

        if self.clamp:
            # If the entity is far from the box along any dimension,
            # the value of the probabilistic membership function approaches zero very quickly,
            # so the logarithm becomes -inf. We clamp it to a minimum of -100.0 as in
            # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
            logp_gt_left.clamp_(-100.0, 0.0)
            logp_lt_right.clamp_(-100.0, 0.0)

        lse = torch.logaddexp(logp_gt_left, logp_lt_right)
        result = logexpm1(lse)

        if self.clamp:
            result.clamp_(-100.0, 0.0)

        result = self.dropout(result)
        
        result = result.sum(dim=-1)

        if self.clamp:
            result.clamp_(-100.0, 0.0)
        
        if not self.log_scale:
            result = result.exp()

        return result


class GumbelBoxMembershipNLLLoss(nn.Module):
    def __init__(
        self,
        intersection_temperature: float = 1.0,
        reduction="mean"
    ):
        super(GumbelBoxMembershipNLLLoss, self).__init__()
        self.probabilistic_membership = GumbelBoxProbabilisticMembership(
            intersection_temperature=intersection_temperature,
            log_scale=True,
            dim=-1
        )
        self.nll = nn.NLLLoss(reduction=reduction)
    
    def forward(
        self,
        entities,
        entity_boxes,
        neighbor_boxes,
        **kwargs
    ):
        if entity_boxes.left.dim() == 2:
            entity_boxes = entity_boxes.rearrange("b d -> b 1 d")
        boxes = entity_boxes.cat(neighbor_boxes, dim=1)
        with torch.no_grad():
            target = torch.zeros_like(boxes.left[..., 0]).long()
            target[:, 0:entity_boxes.box_shape[1]] = 1
            target = target.detach()
        entities = repeat(entities, "b d -> b n d", n=boxes.box_shape[1])
        logp = self.probabilistic_membership(entities, boxes)
        logp = rearrange(logp, "b n -> (b n)")
        logp = torch.stack([
            log1mexp(logp),
            logp
        ], dim=-1)
        target = rearrange(target, "b n -> (b n)")
        return self.nll(logp, target)


class AttentionBasedGumbelIntersection(nn.Module):
    def __init__(
        self,
        size=1024,
        attn_heads=8,
        dropout=0.1,
        intersection_temperature=1.0,
        dim=0
    ):
        super(AttentionBasedGumbelIntersection, self).__init__()
        self.gumbel_intersection = Intersection(
            intersection_temperature=intersection_temperature,
            dim=dim
        )
        self.attn = nn.MultiheadAttention(size, attn_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(2 * size, 2 * size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(2 * size, size)
        )
        self.dim = dim
    
    def forward(self, boxes):
        boxes_center = boxes.center
        intersection = self.gumbel_intersection(boxes)
        intersection_center = intersection.center
        centers = torch.cat([repeat(intersection_center, "b d -> r b d", r=boxes_center.size(0)), boxes_center], dim=self.dim)
        center, _ = self.attn(centers, centers, centers)
        center = center.mean(dim=self.dim)
        offset = intersection.right - intersection.left
        box_data = torch.cat([boxes.left, boxes.right], dim=-1)
        box_data = self.ffn(box_data).mean(dim=self.dim)
        offset = offset * torch.sigmoid(box_data)
        left = center - offset / 2
        right = center + offset / 2
        return BoxTensor((left, right))


class Duck(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super(Duck, self).__init__()
        logger.info("Instantiating Duck for entity disambiguation")
        self.config = config
        self.biencoder_task = None
        self.mention_encoder = None
        self.entity_encoder = None
        self.rel_encoder = None
        self.optimizer = None
        self.volume = None
        self.intersection = None
        self.duck_loss = None
        self.duck_point_loss = None
        self.ent_index = None
        self.dim = None
        self.ed_loss = nn.CrossEntropyLoss()
        self.gather_on_ddp = True
        self.data = kwargs.get("data")
        self.no_box_ablation = False
        self.rel_set_transformer = None
        self.joint_ent_rel_transformer = None
        self.ablations = self.config.get("ablations") or {}
        self.relations_as_points = self.ablations.get("relations_as_points") or False
        self.joint_ent_rel_encoding = self.ablations.get("joint_ent_rel_encoding")
        if self.joint_ent_rel_encoding:
            self.relations_as_points = True
        if self.config.get("ablations"):
            self.no_box_ablation = self.ablations.get("blink") \
                or self.relations_as_points \
                or self.joint_ent_rel_encoding or False
        stage = self.config.get("stage") or kwargs.get("stage")
        self.setup(stage)
        with open_dict(self.config):
            self.config.stage = "loading"

        self.datasets = {
            "val": [d.upper() for d in dict(self.config.data.val_paths).keys()],
            "test": [d.upper() for d in dict(self.config.data.test_paths).keys()]
        }
        
        self.regularizer = None
        if self.config.duck.boxes.get("regularization") is not None:
            self.regularizer = hydra.utils.instantiate(
                self.config.duck.boxes.regularization
            )
        self.box_dropout_dist = None
        if self.config.duck.get("box_dropout"):
            p = self.config.duck.box_dropout
            self.box_dropout_dist = torch.distributions.binomial.Binomial(probs=p)

        self.entity_dropout = nn.Dropout(p=self.config.duck.get("dropout") or 0.0)
        self.mention_dropout = nn.Dropout(p=self.config.duck.get("dropout") or 0.0)
        
        self.duck_loss_weight = self.config.duck.get("duck_loss_weight") or 1.0
        
        self.save_hyperparameters(self.config)

    def on_validation_start(self) -> None:
        self.setup_entity_index()

    def on_validation_end(self) -> None:
        logger.info("Freeing index")
        self.free_index()

    def setup_entity_index(self) -> None:
        torch.cuda.empty_cache()
        num_entities = len(self.data.ent_catalogue.entities)
        world_size = self.config.trainer.devices * self.config.trainer.num_nodes
        device_data_size = math.ceil(num_entities / world_size)
        dim = self.entity_encoder.transformer.config.hidden_size
        
        bsz = 512

        local_index = torch.zeros(
            device_data_size,
            dim,
            device=self.device,
            dtype=torch.float16
        ).detach()

        print(f"Local index size: {local_index.size()}")

        start = self.global_rank * device_data_size
        end = min((self.global_rank + 1) * device_data_size, num_entities)
        
        ent_emb_dataset = self.data.train_dataset.ent_emb_dataset
        with torch.no_grad():
            if self.config.get("debug"):
                logger.info("Debug mode: skipping index update")
            else:
                logger.info(f"Updating entity index from entity {start} to {end} on local rank {self.global_rank} ({str(self.device)})")
                for i in tqdm(range(start, end, bsz)):
                    end_index = min(i + bsz, end)
                    ent_batch = ent_emb_dataset.get_slice(i, end_index)
                    ent_batch = self.data.transform.transform_ent_data(ent_batch)
                    ent_batch = self.batch_to_device(ent_batch)
                    entity_repr = self.encode_entity(ent_batch)
                    local_index[i - start:end_index - start] = entity_repr.half().detach()
            logger.info("Gathering local indices")
            torch.cuda.empty_cache()
            indexes = self.all_gather(local_index.detach())
            ent_index_cpu = indexes.cpu()
            if indexes.dim() == 3:
                assert indexes.size(0) == world_size
                ent_index_cpu = rearrange(ent_index_cpu, "w e d -> (w e) d") #.float()
            indexes = None
            torch.cuda.empty_cache()
            self.ent_index = ent_index_cpu.detach().to(self.device)
    
    def setup_entity_index_sequential(self) -> None:
        num_entities = len(self.data.ent_catalogue.entities)
        ent_emb_dataset = self.data.train_dataset.ent_emb_dataset

        if self.config.get("debug"):
            logger.info("Debug mode: instantiating random index")
            self.ent_index = torch.zeros(
                num_entities,
                self.mention_encoder.transformer.config.hidden_size,
                device=self.device
            ).detach()
            return

        bsz = 512

        self.ent_index = torch.zeros(
            num_entities,
            self.mention_encoder.transformer.config.hidden_size,
            device=self.device
        ).detach()

        logger.info(f"Updating entity index")
        with torch.no_grad():
            for i in tqdm(range(0, num_entities, bsz)):
                end_index = min(i + bsz, num_entities)
                ent_batch = ent_emb_dataset.get_slice(i, end_index)
                ent_batch = self.data.transform.transform_ent_data(ent_batch)
                ent_batch = self.batch_to_device(ent_batch)
                entity_repr = self.encode_entity(ent_batch)
                self.ent_index[i:i + bsz] = entity_repr.detach()
    
    def batch_to_device(
        self,
        batch: Any,
        device: torch.device = None,
        dataloader_idx: int = 0
    ) -> Any:
        device = device or self.device
        return self.transfer_batch_to_device(batch, device, dataloader_idx)
    
    def encode_entity(self, ent_batch):
        entities = ent_batch["entities"]
        entity_repr, _ = self.entity_encoder(
            entities["data"],
            attention_mask=entities["attention_mask"],
        )
        
        # entity_boxes = None
        # if not self.no_box_ablation:
        #     rel_ids = ent_batch["relation_ids"]["data"]
        #     entity_boxes = self.relations_to_box(rel_ids)
        #     relation_mask = ent_batch["relation_ids"]["attention_mask"].bool()
        #     centers = entity_boxes.center
        #     centers[relation_mask] = 0.0
        #     center = centers.sum(dim=1) / relation_mask.sum(dim=-1).unsqueeze(-1)
        #     entity_repr = entity_repr + center  
        # elif self.joint_ent_rel_encoding:
            #relations = self.rel_encoder(ent_batch["relation_ids"]["data"])
            # entity_repr = self.joint_ent_rel_transformer(
            #     entity_repr,
            #     relations,
            #     ent_batch["relation_ids"]["attention_mask"]
            # )
        return entity_repr
    
    def free_index(self):
        self.ent_index = None
        torch.cuda.empty_cache()

    def setup(self, stage: str):
        if stage == "test":
            return
        if stage == "loading" or self.data is None:
            with open_dict(self.config):
                self.config.data.val_paths = {}
                self.config.data.test_paths = {}
                self.config.data.train_path = "/fsx/matzeni/data/GENRE/aida-train-kilt.jsonl"
            self.data = hydra.utils.instantiate(self.config.data)
        
        self.call_configure_sharded_model_hook = False

        self.biencoder_task = ElBiEncoderTask(self.config.duck.base_model, self.config.optim)
        self.biencoder_task.setup(stage)
        self.mention_encoder = self.biencoder_task.mention_encoder
        self.entity_encoder = self.biencoder_task.entity_encoder
        self.dim = self.entity_encoder.transformer.config.hidden_size
        self.box_size = self.dim
        if self.config.duck.boxes.get("dimensions") is not None:
            dimensions = self.config.duck.boxes.get("dimensions") 
            if isinstance(dimensions, float):
                self.box_size = int(dimensions * self.dim)
            else:
                self.box_size = dimensions
        self.intersection = Intersection(
            intersection_temperature=self.config.duck.boxes.intersection_temperature,
            dim=0
        )
        
        if self.relations_as_points:
            if self.joint_ent_rel_encoding:
                self.joint_ent_rel_transformer = JointEntRelsEncoder(self.dim)
            else: 
                self.rel_set_transformer = TransformerSetEncoder(self.dim)

        self.setup_rel_encoder()
        self.optimizer = hydra.utils.instantiate(
            self.config.optim,
            [p for p in self.parameters() if p.requires_grad],
            _recursive_=False
        )

        if self.config.duck.duck_loss is not None and not self.no_box_ablation:
            self.duck_loss = hydra.utils.instantiate(
                self.config.duck.duck_loss
            )
        
        if self.relations_as_points:
            self.duck_point_loss = nn.BCEWithLogitsLoss()

        
    def _setup_pretrained_box_embeddings(self):
        logger.info(f"Setting up pretrained box embeddings")
        with torch.no_grad():
            rel2box = RelToBox.load_from_checkpoint(self.config.duck.rel_to_box_model)
            self.rel_encoder = rel2box.box_embedding
            all_boxes = self.rel_encoder.all_boxes()
            data = torch.cat((all_boxes.left, all_boxes.right), -1)
            self.rel_encoder.weight.copy_(data)
            self.rel_encoder.parametrization = "uniform"
            self.rel_encoder.box_constructor = self.rel_encoder._set_box_constructor()

    def _setup_untrained_box_embeddings(self):
        parametrization = self.config.duck.boxes.parametrization
        logger.info(f"Setting up box embeddings with {parametrization} parametrization")
        self.rel_encoder = BoxEmbedding(
            len(self.data.rel_catalogue),
            self.box_size,
            box_parametrizaton=parametrization,
            universe_idx=0
        )
        self.rel_encoder.universe_min = -100.0
        self.rel_encoder.universe_max = -2 * self.rel_encoder.universe_min
        self.rel_encoder.reinit()

    def _setup_rel_description_to_box_encoder(self):
        parametrization = self.config.duck.boxes.parametrization
        logger.info(f"Setting up relation encoder with {parametrization} box parametrization")
        embeddings = self.data.rel_catalogue.data
        embeddings = torch.from_numpy(embeddings).to(self.device)
        embeddings = torch.cat([
            torch.zeros_like(embeddings)[0].unsqueeze(0),
            embeddings
        ])
        self.rel_encoder = EmbeddingToBox(
            embeddings=embeddings,
            box_parametrization=parametrization,
            padding_idx=0,
            output_size=self.box_size
        )

    def _setup_rel_to_point_encoder(self):
        logger.info(f"Setting up relations to point encoder")
        embeddings = self.data.rel_catalogue.data
        embeddings = torch.from_numpy(embeddings).to(self.device)
        embeddings = torch.cat([
            torch.zeros_like(embeddings)[0].unsqueeze(0),
            embeddings
        ])
        self.rel_encoder = nn.Embedding.from_pretrained(embeddings, padding_idx=0)

    def setup_rel_encoder(self):
        if self.relations_as_points:
            self._setup_rel_to_point_encoder()
            return
        if self.config.duck.get("rel_to_box_model") is not None:
            self._setup_pretrained_box_embeddings(self)
            return
        if self.data is not None and self.data.rel_catalogue.data is None:
            self._setup_untrained_box_embeddings()
            return
        self._setup_rel_description_to_box_encoder()

    def sim_score(self, mentions_repr: Tensor, entities_repr: Tensor):
        # eps = 1e-8
        # mentions_repr = mentions_repr / (torch.norm(mentions_repr, p=2, dim=-1).unsqueeze(-1) + eps)
        # entities_repr = entities_repr / (torch.norm(entities_repr, p=2, dim=-1).unsqueeze(-1) + eps)
        scores = torch.matmul(
            mentions_repr.to(entities_repr.dtype),
            torch.transpose(entities_repr, 0, 1)
        )
        return scores
    
    def encode_with_boxes(self, entity, relations, relation_mask):
        entity_boxes = self.relations_to_box(relations)
        # relation_mask = relation_mask.bool()
        # if self.config.duck.boxes.parametrization != "spherical":
        #     centers = entity_boxes.center
        #     centers[relation_mask] = 0.0
        #     center = centers.sum(dim=1) / relation_mask.sum(dim=-1).unsqueeze(-1)
        #     entity = entity + center
        return entity, entity_boxes

    def encode_entity_with_relations(
        self,
        entity,
        relation_ids,
        entity_tensor_mask,
        ent_rel_mask
    ):
        entity_repr, _ = self.entity_encoder(
            entity["data"],
            attention_mask=entity["attention_mask"],
        )

        # entity_repr = entity_repr[entity_tensor_mask]
        
        entity_boxes = None
        if not self.no_box_ablation:
            entity_repr, entity_boxes = self.encode_with_boxes(
                entity_repr,
                relation_ids["data"],
                relation_ids["attention_mask"].bool()
            )

        relation_set_embeddings = None
        if self.relations_as_points:
            relations = self.rel_encoder(relation_ids["data"])
            relations = expand_with_mask(relations, ent_rel_mask)
            if self.joint_ent_rel_encoding:
                entity_repr = self.joint_ent_rel_transformer(
                    entity_repr,
                    relations,
                    ent_rel_mask
                )
            else:
                relation_set_embeddings = self.rel_set_transformer(
                    relations,
                    ent_rel_mask
                )
        return entity_repr, entity_boxes, relation_set_embeddings

    def forward(self, batch):
        mentions = batch["mentions"] 
        entities = batch["entities"]
        relation_ids = batch["relation_ids"]
        neighbors = batch["neighbors"]
        ent_rel_mask = batch["ent_rel_mask"]

        entity_tensor_mask = batch["entity_tensor_mask"].bool()
        # neighbor_relation_ids = batch["neighbor_relation_ids"]

        assert mentions["data"].size(-1) <= self.config.data.transform.max_mention_len
        assert entities["data"].size(-1) <= self.config.data.transform.max_entity_len
        if neighbors is not None:
            assert neighbors["data"].size(-1) <= self.config.data.transform.max_entity_len

        mention_repr, _ = self.mention_encoder(
            mentions["data"], attention_mask=mentions["attention_mask"]
        )
        
        entity_repr, entity_boxes, relation_sets = self.encode_entity_with_relations(
            entities,
            relation_ids,
            entity_tensor_mask,
            ent_rel_mask
        )

        neighbor_repr = None
        neighbor_boxes = None
        neighbor_relation_sets = None
        
        ## Neighbors are now concatenated to entities because they are not used to learn boxes
        # 
        # if neighbors is not None:
        #     neighbor_repr, neighbor_boxes, neighbor_relation_sets = self.encode_entity_with_relations(
        #         neighbors,
        #         neighbor_relation_ids
        #     )
        
        return {
            "mentions": mention_repr,
            "entities": entity_repr,
            "neighbors": neighbor_repr,
            "entity_boxes": entity_boxes,
            "neighbor_boxes": neighbor_boxes,
            "relation_set_embeddings": relation_sets,
            "neighbor_relation_sets": neighbor_relation_sets
        }
    
    def relations_to_box(self, rel_ids):
        # rel_ids = rearrange(batch["relation_ids"]["data"], "b r -> r b")
        return self.rel_encoder(rel_ids.long())
        # return boxes.rearrange("r b d -> b r d")
        # return self.intersection(boxes)

    def append_neighbors_to_entities(self, batch):
        neighbors = batch["neighbors"]
        entities = batch["entities"]
        relation_ids = batch["relation_ids"]
        neighbor_relation_ids = batch["neighbor_relation_ids"]

        entities["data"] = torch.cat(
            [entities["data"], neighbors["data"]]
        )
        entities["attention_mask"] = torch.cat(
            [entities["attention_mask"], neighbors["attention_mask"]]
        )
        relation_ids["data"] = torch.cat(
            [relation_ids["data"], neighbor_relation_ids["data"]]
        )
        relation_ids["attention_mask"] = torch.cat(
            [relation_ids["attention_mask"], neighbor_relation_ids["attention_mask"]]
        )
        mask = batch["entity_tensor_mask"].bool()
        mask = torch.cat(
            [mask, torch.full((neighbors["data"].size(0), ), True, device=self.device)]
        )
        batch["entity_tensor_mask"] = mask
        batch["entity_ids"] = torch.cat([
            batch["entity_ids"],
            batch["neighbor_ids"]
        ])

    def limit_neighbors(self, batch):
        neighbors = batch["neighbors"]
        max_neighbors = self.config.data.get("max_num_neighbors_per_batch")
        neighbor_relation_ids = batch["neighbor_relation_ids"]

        neighbors["data"] = rearrange(
            neighbors["data"], "b n l -> (b n) l"
        )
        neighbors["attention_mask"] = rearrange(
            neighbors["attention_mask"], "b n l -> (b n) l"
        )
        neighbor_relation_ids["data"] = rearrange(
            neighbor_relation_ids["data"], "b n l -> (b n) l"
        )
        neighbor_relation_ids["attention_mask"] = rearrange(
            neighbor_relation_ids["attention_mask"], "b n l -> (b n) l"
        )
        neighbor_ids = rearrange(batch["neighbor_ids"], "b n -> (b n)")
        if max_neighbors is not None:
            neighbors["data"] = neighbors["data"][:max_neighbors]
            neighbors["attention_mask"] = neighbors["attention_mask"][:max_neighbors]
            neighbor_relation_ids["data"] = neighbor_relation_ids["data"][:max_neighbors]
            neighbor_relation_ids["attention_mask"] = neighbor_relation_ids["attention_mask"][:max_neighbors]
            batch["neighbor_ids"] = neighbor_ids[:max_neighbors]

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None  # for debug
            
        if batch["neighbors"] is not None:
            self.limit_neighbors(batch)
            self.append_neighbors_to_entities(batch)
   
        representations = self(batch)
        local_entities = representations["entities"][batch["entity_tensor_mask"].bool()]
        representations, batch = self._gather_representations(representations, batch)
        if self.gather_on_ddp and isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
            assert local_entities.size(0) < representations["entities"].size(0)
            diff = representations["entities"][:local_entities.size(0)] - local_entities
            diff = diff.abs()
            assert diff.max() < 1e-4, f"{diff.max().item()}"

        mentions = representations["mentions"]
        entities = representations["entities"]
        entity_boxes = representations["entity_boxes"]
        target = batch["targets"]
        relation_set_embeddings = representations["relation_set_embeddings"]
        ent_rel_mask = batch["ent_rel_mask"]
        # entity_tensor_mask = batch["entity_tensor_mask"].bool()
        # entities = entities[entity_tensor_mask]
        
        duck_loss = 0.0
        duck_loss_metrics = {}
        regularization = 0.0

        if self.duck_loss is not None:
            rel_ids = {
                "data": expand_with_mask(batch["relation_ids"]["data"], ent_rel_mask),
                "attention_mask": ent_rel_mask
            }
            negative_boxes = self.sample_negative_boxes(rel_ids["data"])
            duck_loss_metrics = self.compute_duck_loss_with_boxes(
                entities, mentions, entity_boxes, negative_boxes, rel_ids
            )
            duck_loss = duck_loss_metrics["loss"]
            regularization = duck_loss_metrics["regularization"]
            duck_loss_metrics = {
                k: v for k, v in duck_loss_metrics.items()
                if k not in ["loss", "regularization"]
            }
            duck_loss_metrics = prefix_suffix_keys(duck_loss_metrics, prefix="train/")
            entities = self.entity_dropout(entities)
            mentions = self.mention_dropout(mentions)
            
        if self.relations_as_points and not self.joint_ent_rel_encoding:
            ent_to_rel_target = torch.eye(entities.size(0)).to(self.device)
            ent_to_rel_scores = torch.matmul(
                relation_set_embeddings,
                entities.transpose(0, 1)
            )
            duck_loss = self.duck_point_loss(ent_to_rel_scores, ent_to_rel_target)
        
        scores = self.sim_score(mentions, entities)
        ed_loss = self.ed_loss(scores, target)
        loss = ed_loss + self.duck_loss_weight * (duck_loss + regularization)
       
        metrics = {
            "train/duck_loss": duck_loss,
            "train/ed_loss": ed_loss,
            "train/loss": loss,
            "train/regularization": regularization
        }
        metrics.update({
            k: v
            for k, v in duck_loss_metrics.items()
            if k != "loss" and isinstance(v, torch.Tensor) 
        })

        if self.logger is not None:
            self.log_dict(metrics)
            if not isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
                wandb.log({
                    k: v
                    for k, v in duck_loss_metrics.items()
                    if k not in metrics
                })
        
        metrics["loss"] = loss

        return metrics

    def compute_duck_loss_with_boxes(self, entities, mentions, entity_boxes, negative_boxes, rel_ids):
        if self.config.duck.get("gaussian_box_regularization") and self.training:
            std = self.config.duck.gaussian_box_regularization
            entities.add_(std * torch.rand_like(entities))

        result = {}
        ent_rel_mask = rel_ids["attention_mask"]
        entity_boxes = expand_box_with_mask(entity_boxes, ent_rel_mask)
        entities_ = entities[..., :self.box_size].clone()
        mentions_ = mentions[..., :self.box_size].clone()
        entity_boxes = entity_boxes[..., :self.box_size].clone()
        negative_boxes = negative_boxes[..., :self.box_size].clone()
        if self.config.duck.boxes.parametrization == "spherical":
            entity_boxes, negative_boxes, entities_ = self.handle_spherical_coord(
                entity_boxes, negative_boxes, entities_, mentions_
            )
        
        entities_ = entities_[:entity_boxes.box_shape[0], :].clone()
        
        dropout_mask = torch.full_like(entities_, False).bool()
        if self.box_dropout_dist is not None and self.training:
            dropout_mask = self.box_dropout_dist.sample(entities_.size()).bool()
        dropout_mask = dropout_mask.to(entities_.device)

        duck_loss = self.duck_loss(
            entities_.masked_fill(dropout_mask, 0.0),
            entity_boxes.masked_fill(dropout_mask.unsqueeze(1), 0.0),
            negative_boxes.masked_fill(dropout_mask.unsqueeze(1), 0.0),
            rel_ids=rel_ids
        )
        regularization = 0.0
        if self.regularizer is not None:
            regularization = self.regularizer(
                entity_boxes[ent_rel_mask].cat(negative_boxes.rearrange("b n d -> (b n) d"))
            )

        box_metrics = self.compute_box_metrics(entities_, entity_boxes, negative_boxes, ent_rel_mask)
        result.update(box_metrics)

        if isinstance(duck_loss, dict):
            result.update(duck_loss)
        else:
            result["loss"] = duck_loss
        
        result["regularization"] = regularization
        return result

    def compute_box_metrics(self, entities, entity_boxes, negative_boxes, ent_rel_mask):
        result = {}
        entities = entities.unsqueeze(1)
        in_pos = (entities > entity_boxes.left) & (entities < entity_boxes.right)
        in_pos = in_pos[ent_rel_mask]
        result["strict_containment_positive_boxes"] = in_pos.all(dim=-1).float().mean()
        result["dim_wise_containment_positive_boxes"] = wandb.Histogram(
            in_pos.float().mean(dim=-1).detach().cpu().numpy()
        )
        in_neg = (entities > negative_boxes.left) & (entities < negative_boxes.right)
        result["strict_containment_negative_boxes"] = in_neg.all(dim=-1).float().mean()
        result["dim_wise_containment_negative_boxes"] = wandb.Histogram(
            in_neg.float().mean(dim=-1).detach().cpu().numpy()
        )
        entity_boxes = entity_boxes[ent_rel_mask]
        result["box_left_distribution"] = wandb.Histogram(entity_boxes.left.detach().cpu().numpy())
        result["box_right_distribution"] = wandb.Histogram(entity_boxes.right.detach().cpu().numpy())
        result["box_left_mean"] = entity_boxes.left.mean()
        result["box_right_mean"] = entity_boxes.right.mean()
        box_size = entity_boxes.right - entity_boxes.left
        result["box_size_distribution"] = wandb.Histogram(box_size.detach().cpu().numpy())
        result["box_size_mean"] = box_size.mean()
        return result

    def handle_spherical_coord(self, entity_boxes, negative_boxes, entities, mentions):
        entities[..., -1] = torch.abs(entities[..., -1].clone())
        mentions[..., -1] = torch.abs(mentions[..., -1].clone())
        _, entities = cartesian_to_spherical(entities)
        # entities_ = entities_[..., :-1]  # drop last coord to keep the range [0, pi]
        entity_boxes = entity_boxes[..., :entities.size(-1)].clone()
        negative_boxes = negative_boxes[..., :entities.size(-1)].clone()
        return entity_boxes, negative_boxes, entities

    def training_epoch_end(self, outputs):
        mean_metrics = mean_over_batches(outputs, suffix="epoch")
        self.log_dict(mean_metrics, sync_dist=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        representations = self(batch)
        mentions = representations["mentions"]
        entities = self.ent_index
        target = batch["entity_ids"]
        scores = self.sim_score(mentions, entities)
        preds = scores.argmax(dim=-1)

        candidate_preds = None
        if batch["candidates"] is not None:
            candidate_indexes = batch["candidates"]["data"].long()
            candidates = self.ent_index[candidate_indexes]
            candidate_scores = torch.bmm(
                mentions.unsqueeze(1).to(candidates.dtype),
                candidates.transpose(1, -1)
            ).squeeze(1)
            candidate_mask = batch["candidates"]["attention_mask"].bool()
            candidate_scores[~candidate_mask] = 0
            candidate_preds = candidate_indexes.gather(
                1, candidate_scores.argmax(dim=-1).unsqueeze(1)
            ).squeeze(dim=1)
            no_candidates_mask = ~(candidate_mask.any(dim=-1))
            candidate_preds[no_candidates_mask] = preds[no_candidates_mask]

        return {
            "preds": preds,
            "candidate_preds": candidate_preds,
            "target": target,
            "topk": scores.topk(100)
        }
    
    def all_gather_flat(self, tensor):
        shape = tensor.size()
        all = self.all_gather(tensor)
        if len(shape) < len(all.size()):
            grouped_dim = all.size(0) * shape[0]
            all = all.view(grouped_dim, *shape[1:])
        return all

    def validation_epoch_end(self, outputs):
        metrics = {}
        
        if len(self.trainer.val_dataloaders) == 1:
            outputs = [outputs]
        datasets = self.datasets["val"]
        assert len(datasets) == len(outputs)
        for i, dataset in enumerate(datasets):
            dataset_outputs = outputs[i]
            has_candidates = any(o["candidate_preds"] is not None for o in dataset_outputs)
            preds = torch.cat([o["preds"] for o in dataset_outputs])
            target = torch.cat([o["target"] for o in dataset_outputs])
            topk = [o["topk"] for o in dataset_outputs]

            preds = self.all_gather_flat(preds)
            target = self.all_gather_flat(target)

            candidate_preds = None
            if has_candidates:
                candidate_preds = torch.cat([o["candidate_preds"] for o in dataset_outputs])
                candidate_preds = self.all_gather_flat(candidate_preds)
            
            micro_f1 = torchmetrics.functional.classification.multiclass_f1_score(
                preds, target, num_classes=len(self.data.ent_catalogue), average='micro'
            )
            if has_candidates:
                micro_f1_candidate_set = torchmetrics.functional.classification.multiclass_f1_score(
                    candidate_preds, target, num_classes=len(self.data.ent_catalogue), average='micro'
                )
                metrics[f"Micro-F1_candidate_set/{dataset}"] = micro_f1_candidate_set
                tqdm.write(f"Micro F1 on {dataset} (with candidate set): \t{micro_f1_candidate_set:.4f}")
            metrics[f"Micro-F1/{dataset}"] = micro_f1
            tqdm.write(f"Micro F1 on {dataset}: \t{micro_f1:.4f}")

            recall_steps = [10, 30, 50, 100]
            for k in recall_steps:
                recall = self.recall_at_k(topk, target, k)
                metrics[f"Recall@{k}/{dataset}"] = recall
        
        metric_names = set(k.split("/")[0] for k in metrics)
        for avg_metric_key in metric_names:
            value = 0
            for dataset_name in self.datasets["test"]:
                value += metrics[f"{avg_metric_key}/{dataset_name}"].item()
            metrics[avg_metric_key + "/Average"] = value / len(self.datasets["test"])

        tqdm.write(f"Average Micro F1: {metrics['Micro-F1/Average']:.4f}")
        
        metrics["avg_micro_f1"] = metrics["Micro-F1/Average"]
        metrics["val_f1"] = metrics.get("Micro-F1/BLINK_DEV") or metrics["avg_micro_f1"]
        self.log_dict(metrics, sync_dist=True)
    
    def recall_at_k(self, top, target, k):
        top_scores = torch.cat([t.values for t in top])[:, :k]
        top_indices = torch.cat([t.indices for t in top])[:, :k]

        top_scores = self.all_gather_flat(top_scores)
        top_indices = self.all_gather_flat(top_indices)
        target = (top_indices.transpose(0, 1) == target).transpose(0, 1)

        index = repeat(
            torch.arange(top_scores.size(0), device=self.device),
            "b -> b k", k=top_scores.size(1)
        )
        return torchmetrics.RetrievalRecall(k=k)(
            top_scores.contiguous().view(-1),
            target.contiguous().view(-1),
            index.contiguous().view(-1),
        )

    def configure_optimizers(self):
        return self.optimizer
    
    def sample_negative_boxes(self, relation_ids):
        with torch.no_grad():
            ids_range = torch.arange(len(self.data.rel_catalogue)).to(self.device) + 1
            negative_ids = []
            for rels in relation_ids:
                negative_pool = tensor_set_difference(ids_range, rels)
                sample_indices = torch.multinomial(torch.ones_like(negative_pool).float(), num_samples=self.config.duck.num_negative_boxes)
                sample = negative_pool[sample_indices]
                negative_ids.append(sample)
            negative_ids = torch.stack(negative_ids).detach()
        return self.rel_encoder(negative_ids)

    # def _extend_with_in_batch_neighbors(
    #     self,
    #     neighbors,
    #     neighbor_boxes
    # ):  
    #     bsz = neighbors.size(0)
    #     num_in_batch_neighbors = self.config.duck.in_batch_neighbors
    #     for _ in range(num_in_batch_neighbors):
    #         num_neighbors = neighbors.size(1)
    #         perm = torch.randperm(bsz * num_neighbors).to(neighbors.device)
    #         in_batch_boxes = neighbor_boxes.rearrange("b n d -> (b n) d")[perm] \
    #             .rearrange("(b n) d -> b n d", b=bsz)[:, :1, :]
    #         neighbor_boxes = neighbor_boxes.cat(in_batch_boxes, dim=1)
    #         in_batch_neighbors = rearrange(neighbors, "b n d -> (b n) d")[perm]
    #         in_batch_neighbors = rearrange(in_batch_neighbors, "(b n) d -> b n d", b=bsz)[:, :1, :]
    #         neighbors = torch.cat([neighbors, in_batch_neighbors], dim=1)
    #     return neighbors, neighbor_boxes
   
    def _gather_representations(self, representations, batch):
        entities = representations["entities"]
        mentions = representations["mentions"]
        target = batch["targets"]
        entity_ids = batch["entity_ids"]
        # rel_ids = batch["relation_ids"]
        entity_tensor_mask = batch["entity_tensor_mask"].bool()
        relation_set_embeddings = representations["relation_set_embeddings"]

        if not self.gather_on_ddp or not isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
            representations["entities"] = entities[entity_tensor_mask]
            if relation_set_embeddings is not None:
                representations["relation_set_embeddings"] = relation_set_embeddings[entity_tensor_mask]
            return representations, batch

        mentions_to_send = mentions.detach()
        entities_to_send = entities.detach()
        # all_boxes = None

        # num_devs = self.config.trainer.devices
        # start_rank = int(math.floor(self.global_rank / num_devs) * num_devs)
        # end_rank = start_rank + num_devs

        # if entity_boxes is not None:
        #     entity_boxes_to_send = entity_boxes.detach()
        #     all_boxes = BoxTensor((
        #         self.all_gather(entity_boxes_to_send.left),
        #         self.all_gather(entity_boxes_to_send.right)
        #     ))

        all_mentions_repr = self.all_gather(mentions_to_send)  # num_workers x bs
        all_entities_repr = self.all_gather(entities_to_send)

        all_relation_set_embeddings = None
        if relation_set_embeddings is not None:
            all_relation_set_embeddings = self.all_gather(relation_set_embeddings.detach())

        all_targets = self.all_gather(target)
        # we are not filtering duplicated entities now
        all_entity_ids = self.all_gather(entity_ids)
        # all_rel_ids = {
        #     "data": self.all_gather(rel_ids["data"].detach()),
        #     "attention_mask": self.all_gather(rel_ids["attention_mask"].detach())
        # }
        all_mask = self.all_gather(entity_tensor_mask)

        # offset = 0
        all_mentions_list = []
        all_entities_list = []
        all_entity_ids_list = []
        all_targets_list = []
        # all_boxes_list = []
        # all_rel_ids_list = []
        all_relation_set_embeddings_list = []

        # Add current device representations first.
        all_mentions_list.append(mentions)
        entities = entities[entity_tensor_mask]
        # entity_boxes = entity_boxes[entity_tensor_mask] if entity_boxes is not None else None
        all_entities_list.append(entities)
        all_entity_ids_list.append(entity_ids[entity_tensor_mask].tolist())
        all_targets_list.append(target)
        # all_boxes_list.append(entity_boxes)
        # all_rel_ids_list.append(rel_ids)
        all_relation_set_embeddings_list.append(relation_set_embeddings)
        # offset += entities_repr.size(0)

        for i in range(all_targets.size(0)):
            if i != self.local_rank:
                all_mentions_list.append(all_mentions_repr[i])
                all_entities_list.append(all_entities_repr[i][all_mask[i]])
                all_entity_ids_list.append(
                    all_entity_ids[i][all_mask[i].bool()].tolist()
                )
                all_targets_list.append(all_targets[i])
                # if all_boxes is not None:
                #     all_boxes_list.append(all_boxes[i][all_mask[i].bool()])
                # else:
                #     all_boxes_list.append(None)
                # all_rel_ids_list.append({
                #     "data": all_rel_ids["data"][i][all_mask[i].bool()],
                #     "attention_mask": all_rel_ids["attention_mask"][i][all_mask[i].bool()]
                # })
                if all_relation_set_embeddings is not None:
                    all_relation_set_embeddings_list.append(
                        all_relation_set_embeddings[i]  # [all_mask[i].bool()]
                    )
                else:
                    all_relation_set_embeddings_list.append(None)

        mentions = torch.cat(all_mentions_list, dim=0)  # total_ctx x dim
        # entities_repr = torch.cat(all_entities_list, dim=0)  # total_query x dim
        # targets = torch.cat(all_targets_list, dim=0)
        entities, target, relation_set_embeddings = self.filter_representations(
            all_entities_list,
            all_entity_ids_list,
            # all_boxes_list,
            # all_rel_ids_list,
            all_targets_list,
            all_relation_set_embeddings_list
        )
        
        representations["mentions"] = mentions
        representations["entities"] = entities
        # representations["entity_boxes"] = boxes
        representations["relation_set_embeddings"] = relation_set_embeddings
        batch["targets"] = target
        # batch["relation_ids"] = rel_ids

        return representations, batch
    
    def filter_representations(
        self,
        all_entities_list,
        all_entity_ids_list,
        # all_boxes_list,
        # all_rel_ids_list,
        all_targets_list,
        all_relation_set_embeddings
    ):
        filtered_entities_repr = []
        filtered_targets = []
        # filtered_boxes = []
        # filtered_rel_ids_data = []
        # filtered_rel_ids_mask = []
        filtered_relation_set_embeddings = []
        ent_indexes_map = {}

        for entities_repr, entity_ids, targets, relation_set_embeddings in zip(
            all_entities_list,
            all_entity_ids_list,
            # all_boxes_list,
            # all_rel_ids_list,
            all_targets_list,
            all_relation_set_embeddings
        ):
            for i, entity_repr in enumerate(entities_repr):
                ent_id = entity_ids[i]
                # box = boxes[i] if boxes is not None else None
                relation_set_emb = relation_set_embeddings[i] if relation_set_embeddings is not None else None
                if ent_id not in ent_indexes_map:
                    ent_idx = len(ent_indexes_map)
                    ent_indexes_map[ent_id] = ent_idx
                    filtered_entities_repr.append(entity_repr)
                    # filtered_boxes.append(box)
                    # filtered_rel_ids_data.append(rel_ids["data"][i])
                    # filtered_rel_ids_mask.append(rel_ids["attention_mask"][i])
                    filtered_relation_set_embeddings.append(relation_set_emb)
            for target in targets.tolist():
                filtered_targets.append(ent_indexes_map[entity_ids[target]])

        filtered_entities_repr = torch.stack(filtered_entities_repr, dim=0)
        # if not self.no_box_ablation:
        #     filtered_boxes = BoxTensor((
        #         torch.stack([box.left for box in filtered_boxes], dim=0),
        #         torch.stack([box.right for box in filtered_boxes], dim=0)
        #     ))
        # else:    
        #     filtered_boxes = None
        
        if self.relations_as_points and not self.joint_ent_rel_encoding:
            filtered_relation_set_embeddings = torch.stack(filtered_relation_set_embeddings, dim=0)
        else:
            filtered_relation_set_embeddings = None

        # filtered_rel_ids = {
        #     "data": torch.stack(filtered_rel_ids_data, dim=0),
        #     "attention_mask": torch.stack(filtered_rel_ids_mask, dim=0)
        # }

        filtered_targets = torch.tensor(
            filtered_targets,
            dtype=torch.long,
            device=filtered_entities_repr.get_device(),
        )

        return filtered_entities_repr, \
            filtered_targets, \
            filtered_relation_set_embeddings

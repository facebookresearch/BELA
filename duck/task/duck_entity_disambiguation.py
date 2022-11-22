from typing import List
import pytorch_lightning as pl
import hydra
import logging
import torch
import torch.nn as nn
from torch import Tensor
from duck.box_tensors.box_tensor import BoxTensor
from duck.common.utils import list_to_tensor, logsubexp, tiny_value_of_dtype
from mblink.task.blink_task import ElBiEncoderTask
from einops import rearrange
from duck.box_tensors.volume import Volume
from duck.box_tensors.intersection import Intersection
from duck.modules import BoxEmbedding, HFSetToBoxTransformer, SetToBoxTransformer
from duck.task.rel_to_box import RelToBox
import transformers
from abc import ABC, abstractmethod


transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class DuckLoss(nn.Module, ABC):
    def __init__(
        self,
        intersection_temperature: float = 1.0,
        volume_temperature: float = 1.0,
        rel_threshold: int = 1
    ):
        super(DuckLoss, self).__init__()
        self.intersection = Intersection(
            intersection_temperature=intersection_temperature
        )
        self.volume = Volume(
            intersection_temperature=intersection_temperature,
            volume_temperature=volume_temperature
        )
        self.eps = tiny_value_of_dtype(torch.float)
        self.rel_threshold = rel_threshold
    
    @abstractmethod
    def target(self, entity_relations, neighbor_relations):
        pass

    def reduce(self, loss):
        return loss.mean()
    
    def mask_loss(self, loss, entity_relations, neighbor_relations):
        mask_neighbor_indices = [
            (i, j)
            for i, rel_sets in enumerate(neighbor_relations)
            for j, t in enumerate(rel_sets) if t.size(0) < self.rel_threshold
        ]
        if len(mask_neighbor_indices) > 0:
            neigh_i, neigh_j = zip(*mask_neighbor_indices)
            loss[list(neigh_i), list(neigh_j)] = 0.
        mask_ent_indices = [i for i, t in enumerate(entity_relations) if t.size(0) < self.rel_threshold]
        loss[mask_ent_indices] = 0.
        return loss


class DuckJaccardLoss(DuckLoss, ABC):
    def __init__(
        self,
        intersection_temperature: float = 1.0,
        volume_temperature: float = 1.0,
        rel_threshold: int = 1
    ):
        super(DuckJaccardLoss, self).__init__(
            intersection_temperature=intersection_temperature,
            volume_temperature=volume_temperature,
            rel_threshold=rel_threshold
        )
        self.clamp_value = 10

    @abstractmethod
    def criterion(self, log_pred_jaccard, target_jaccard):
        pass
    
    def forward(
        self,
        entity_boxes: BoxTensor,
        neighbor_boxes: BoxTensor,
        entity_relations: List[Tensor],
        neighbor_relations: List[List[Tensor]]
    ):
        log_pred_jaccard = self.log_box_jaccard(entity_boxes, neighbor_boxes)
        target_jaccard = self.target(entity_relations, neighbor_relations)
        loss = self.criterion(log_pred_jaccard, target_jaccard)
        loss = self.mask_loss(loss, entity_relations, neighbor_relations)
        return self.reduce(loss)
    
    def log_box_jaccard(self,
        entity_boxes: BoxTensor,
        neighbor_boxes: BoxTensor
    ):
        num_neighbors = neighbor_boxes.box_shape[1]
        entity_boxes = entity_boxes.repeat("b d -> b n d", n=num_neighbors)
        log_intersection = self.volume(self.intersection(entity_boxes, neighbor_boxes))
        log_ent_volume = self.volume(entity_boxes)
        log_neigh_volume = self.volume(neighbor_boxes)
        log_sum = torch.logaddexp(log_ent_volume, log_neigh_volume)
        log_union = logsubexp(log_sum, log_intersection)
        return log_intersection - log_union

    def target(self, entity_relations, neighbor_relations):
        result = []
        for i, ent_rels in enumerate(entity_relations):
            result.append([])
            ent_neighbors = neighbor_relations[i]
            for neigh_rels in ent_neighbors:
                concat = torch.cat([ent_rels, neigh_rels])
                _, counts = torch.unique(concat, return_counts=True)
                intersection = sum(counts > 1)
                union = concat.size(0)
                jaccard_score = torch.tensor(0.)
                if union > 0:
                    jaccard_score = intersection / (union + self.eps)
                result[i].append(jaccard_score)
        return torch.tensor(result, device=result[0][0].device)


class DuckJaccardKLDivLoss(DuckJaccardLoss):
    def __init__(
        self,
        intersection_temperature: float = 1.0,
        volume_temperature: float = 1.0,
        rel_threshold: int = 1
    ):
        super(DuckJaccardKLDivLoss, self).__init__(
            intersection_temperature=intersection_temperature,
            volume_temperature=volume_temperature,
            rel_threshold=rel_threshold
        )
        self.kldiv = nn.KLDivLoss(reduction="none")
    
    def criterion(self, log_pred_jaccard, gt_jaccard):
        return self.kldiv(log_pred_jaccard, gt_jaccard)

    def reduce(self, loss):
        return loss.sum() / loss.size(0)


class DuckJaccardMSELoss(DuckJaccardLoss):
    def __init__(
        self,
        intersection_temperature: float = 1.0,
        volume_temperature: float = 1.0,
        rel_threshold: int = 1,
        log_scale: bool = True
    ):
        super(DuckJaccardMSELoss, self).__init__(
            intersection_temperature=intersection_temperature,
            volume_temperature=volume_temperature,
            rel_threshold=rel_threshold
        )
        self.log_scale = log_scale
    
    def criterion(
        self,
        log_pred_jaccard,
        target_jaccard
    ):
        if not self.log_scale:
            return torch.square(target_jaccard - torch.exp(log_pred_jaccard).clamp(0, 1))
        return torch.square(torch.log(target_jaccard + self.eps) - log_pred_jaccard)


class DuckDoubleMSELoss(DuckLoss):
    def __init__(
        self,
        intersection_temperature: float = 1.0,
        volume_temperature: float = 1.0,
        rel_threshold: int = 1,
        log_scale: bool = True
    ):
        super(DuckDoubleMSELoss, self).__init__(
            intersection_temperature=intersection_temperature,
            volume_temperature=volume_temperature,
            rel_threshold=rel_threshold
        )
        self.log_scale = log_scale

    def forward(
        self,
        entity_boxes: BoxTensor,
        neighbor_boxes: BoxTensor,
        entity_relations: List[Tensor],
        neighbor_relations: List[List[Tensor]]
    ):
        num_neighbors = neighbor_boxes.box_shape[1]
        entity_boxes = entity_boxes.repeat("b d -> b n d", n=num_neighbors)
        log_intersection = self.volume(self.intersection(entity_boxes, neighbor_boxes))
        log_ent_volume = self.volume(entity_boxes)
        log_neigh_volume = self.volume(neighbor_boxes)
        log_ent_neigh_prob = log_intersection - log_ent_volume
        log_neigh_ent_prob = log_intersection - log_neigh_volume
        log_probs = rearrange(
            [log_ent_neigh_prob, log_neigh_ent_prob],
             "probs b n -> b n probs"
        )
        gt_probs = self.target(entity_relations, neighbor_relations)
        if self.log_scale:
            mse = torch.square(torch.log(gt_probs + self.eps) - log_probs)
        else:
            mse = torch.square(gt_probs - torch.exp(log_probs).clamp(0, 1))
        # mse = self.mask_loss(mse, entity_relations, neighbor_relations)
        return mse.mean()

    def target(self, entity_relations, neighbor_relations):
        probs = []
        for i, ent_rels in enumerate(entity_relations):
            probs.append([])
            ent_neighbors = neighbor_relations[i]
            for neigh_rels in ent_neighbors:
                concat = torch.cat([ent_rels, neigh_rels])
                _, counts = torch.unique(concat, return_counts=True)
                intersection = sum(counts > 1)
                ent_neigh_prob = intersection / (neigh_rels.size(0) + self.eps)
                neigh_ent_prob = intersection / (ent_rels.size(0) + self.eps)
                probs[i].append([ent_neigh_prob, neigh_ent_prob])
        return torch.tensor(probs, device=probs[0][0][0].device)


class Duck(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super(Duck, self).__init__()
        logger.info("Instantiating Duck for entity disambiguation")
        self.config = config
        self.save_hyperparameters(config)
        self.biencoder_task = None
        self.mention_encoder = None
        self.entity_encoder = None
        self.relation_box_encoder = None
        self.box_embedding = None
        self.optimizer = None
        self.volume = None
        self.intersection = None
        if config.duck.duck_loss == "jaccard_mse":
            self.duck_loss = DuckJaccardMSELoss(**self.config.duck.boxes)
        elif config.duck.duck_loss == "double":
            self.duck_loss = DuckDoubleMSELoss(**self.config.duck.boxes)
        elif config.duck.duck_loss == "jaccard_kldiv":
            self.duck_loss = DuckJaccardKLDivLoss(**self.config.duck.boxes)
        else:
            raise ValueError(f"Unsupported loss {config.loss}")
        self.disambiguation_loss = None
        self.setup(None)

    def setup(self, stage: str):
        if stage == "test":
            return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False

        self.biencoder_task = ElBiEncoderTask(self.config.duck.base_model, self.config.optim)
        self.biencoder_task.setup(stage)
        self.mention_encoder = self.biencoder_task.mention_encoder
        self.entity_encoder = self.biencoder_task.entity_encoder
        # if not self.config.data.pretrained_relations:
        #     self.relation_box_encoder = HFSetToBoxTransformer(
        #     hydra.utils.instantiate(self.config.duck.base_model), batched=False
        # )
        # else:
        #     self.relation_box_encoder = SetToBoxTransformer(
        #         dim=self.mention_encoder.transformer.config.hidden_size,
        #         hidden_dim=self.mention_encoder.transformer.config.hidden_size,
        #         **self.config.duck.box_encoder
        #     )
        self.box_embedding = self.setup_box_embedding_layer()
        self.optimizer = hydra.utils.instantiate(
            self.config.optim, self.parameters(), _recursive_=False
        )
    
    def setup_box_embedding_layer(self):
        rel2box = RelToBox.load_from_checkpoint(self.config.duck.rel_to_box_model)
        self.box_embedding = rel2box.box_embedding.freeze()

    def sim_score(self, mentions_repr, entities_repr):
        scores = torch.matmul(mentions_repr, torch.transpose(entities_repr, 0, 1))
        return scores
    
    def forward(self, batch):
        mentions = batch["mentions"] 
        entities = batch["entities"]
        relations = batch["relations"]
        neighbors = batch["neighbors"]
        neighbor_relations = batch["neighbor_relations"]

        bsz = mentions["data"].size(0)
        assert bsz == entities["data"].size(0)
        assert bsz == relations["data"].size(0)
        assert bsz == neighbors["data"].size(0)
        assert bsz == neighbor_relations["data"].size(0)
        assert mentions["data"].size(-1) <= self.config.data.transform.max_mention_len
        assert entities["data"].size(-1) <= self.config.data.transform.max_entity_len
        assert neighbors["data"].size(-1) <= self.config.data.transform.max_entity_len
        if self.config.data.pretrained_relations:
            dim = self.mention_encoder.transformer.config.hidden_size
            assert relations["data"].size(-1) == dim
            assert neighbor_relations["data"].size(-1) == dim
        else:
            assert relations["data"].size(-1) <= self.config.data.transform.max_relation_len
            assert neighbor_relations["data"].size(-1) <=  self.config.data.transform.max_relation_len

        num_rels = torch.any(batch["relations"]["attention_mask"].bool(), dim=-1).sum(dim=-1).tolist()
        assert num_rels == [rels.size(0) for rels in batch["relation_ids"]]
        assert num_rels == [len(rels) for rels in batch["relation_labels"]]

        mention_repr, _ = self.mention_encoder(
            mentions["data"], attention_mask=mentions["attention_mask"]
        )
        entity_repr, _ = self.entity_encoder(
            entities["data"],
            attention_mask=entities["attention_mask"],
        )
        bsz = neighbors["data"].size(0)
        neighbor_repr, _ = self.entity_encoder(
            rearrange(neighbors["data"], "b n l -> (b n) l"),
            rearrange(neighbors["attention_mask"], "b n l -> (b n) l")
        )
        neighbor_repr = rearrange(neighbor_repr, "(b n) d -> b n d", b=bsz)
        entity_boxes = self.relation_box_encoder(
            relations["data"],
            attention_mask=relations["attention_mask"]
        )
        neighbor_boxes = self.relation_box_encoder(
            rearrange(neighbor_relations["data"], "b n r l -> (b n) r l"),
            rearrange(neighbor_relations["attention_mask"], "b n r l -> (b n) r l"),
        )
        neighbor_boxes = neighbor_boxes.rearrange("(b n) d -> b n d", b=bsz)

        return {
            "mentions": mention_repr,
            "entities": entity_repr,
            "entity_boxes": entity_boxes,
            "neighbors": neighbor_repr,
            "neighbor_boxes": neighbor_boxes
        }
        
    def training_step(self, batch, batch_idx):
        if batch is None:
            return None  # for debug
            
        representations = self(batch)
        mentions = representations["mentions"]
        entities = representations["entities"]
        entity_boxes = representations["entity_boxes"]
        neighbors = representations["neighbors"]
        neighbor_boxes = representations["neighbor_boxes"]

        entity_relation_ids = batch["relation_ids"]
        neighbor_relation_ids = batch["neighbor_relation_ids"]
        
        mask = batch["entity_tensor_mask"]
        entities = entities[mask.bool()]
        sim_score = self.sim_score(mentions, entities)

        neighbors, neighbor_boxes, neighbor_relation_ids = self._extend_with_in_batch_neighbors(
            neighbors,
            neighbor_boxes,
            neighbor_relation_ids
        )

        loss = self.duck_loss(
            entity_boxes,
            neighbor_boxes,
            entity_relation_ids,
            neighbor_relation_ids
        )
        if self.logger is not None:
            self.log_dict({
                "duck_loss": loss
            })
        return loss

    def configure_optimizers(self):
        return self.optimizer
    
    def _extend_with_in_batch_neighbors(
        self,
        neighbors,
        neighbor_boxes,
        neighbor_relation_ids,
    ):
        
        bsz = neighbors.size(0)
        num_in_batch_neighbors = self.config.duck.in_batch_neighbors
        for _ in range(num_in_batch_neighbors):
            num_neighbors = neighbors.size(1)
            perm = torch.randperm(bsz * num_neighbors).to(neighbors.device)
            in_batch_boxes = neighbor_boxes.rearrange("b n d -> (b n) d")[perm] \
                .rearrange("(b n) d -> b n d", b=bsz)[:, :1, :]
            neighbor_boxes = neighbor_boxes.cat(in_batch_boxes, dim=1)
            in_batch_neighbors = rearrange(neighbors, "b n d -> (b n) d")[perm]
            in_batch_neighbors = rearrange(in_batch_neighbors, "(b n) d -> b n d", b=bsz)[:, :1, :]
            neighbors = torch.cat([neighbors, in_batch_neighbors], dim=1)
            neighbor_rel_tensor, mask = list_to_tensor(neighbor_relation_ids, pad_value=-1)
            neighbor_rel_tensor = rearrange(neighbor_rel_tensor, "b n r -> (b n) r")[perm]
            neighbor_rel_tensor = rearrange(neighbor_rel_tensor, "(b n) r -> b n r", b=bsz)
            mask = mask.bool()
            mask = rearrange(mask, "b n r -> (b n) r")[perm]
            mask = rearrange(mask, "(b n) r -> b n r", b=bsz)
            for i, neigh_rels in enumerate(neighbor_relation_ids):
                neigh_rels.append(
                    neighbor_rel_tensor[i, 0, :][mask[i, 0]]
                )
        return neighbors, neighbor_boxes, neighbor_relation_ids

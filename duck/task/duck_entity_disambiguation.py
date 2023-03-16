import math
from typing import Any
import pytorch_lightning as pl
import hydra
import logging
import torch
import torch.nn as nn
from torch import Tensor
from duck.common.utils import cartesian_to_spherical, expand_box_with_mask, expand_with_mask, prefix_suffix_keys
from duck.modules.modules import EmbeddingToBox, JointEntRelsEncoder, TransformerSetEncoder
from mblink.task.blink_task import ElBiEncoderTask
from einops import rearrange, repeat
from duck.box_tensors.intersection import Intersection
from duck.modules import BoxEmbedding
from duck.task.rel_to_box import RelToBox
import transformers
from tqdm import tqdm
import torchmetrics
import wandb
from pytorch_lightning.strategies import DDPShardedStrategy, DDPStrategy
from omegaconf import open_dict
# Needed for loading checkpoints before code refactoring
from duck.task.duck_loss import DuckNegativeSamplingLoss, DuckNegativeSamplingLossWithTemperature, BoxEDistance


transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class Duck(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super(Duck, self).__init__()
        logger.info("Instantiating Duck for entity disambiguation")
        self.config = config
        self.biencoder_task = None
        self.mention_encoder = None
        self.entity_encoder = None
        self.rel_encoder = None
        self.volume = None
        self.intersection = None
        self.duck_loss = None
        self.duck_point_loss = None
        self.ent_index = None
        self.dim = transformers.AutoConfig.from_pretrained(self.config.language_model).hidden_size
        self.ed_loss = nn.CrossEntropyLoss()
        self.prior_loss = nn.NLLLoss()
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
        
        self.prior_ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(p=self.config.duck.get("dropout") or 0.0),
            nn.ReLU(),
            nn.Linear(self.dim, 2),
            nn.LogSoftmax(dim=-1)
        )
        # self.prior_projection = nn.Linear(2, 1)
        stage = self.config.get("stage") or kwargs.get("stage")
        self.setup(stage)
        with open_dict(self.config):
            self.config.stage = "loading"
        
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
        
        neighbors_dataset = self.data.train_dataset.neighbors_dataset
        with torch.no_grad():
            if self.config.get("debug"):
                logger.info("Debug mode: skipping index update")
            else:
                logger.info(f"Updating entity index from entity {start} to {end} on local rank {self.global_rank} ({str(self.device)})")
                for i in tqdm(range(start, end, bsz)):
                    end_index = min(i + bsz, end)
                    ent_batch = neighbors_dataset.get_slice(i, end_index)
                    ent_batch = self.data.transform.transform_ent_data(ent_batch)
                    ent_batch = self.batch_to_device(ent_batch)
                    entity_repr = self.encode_entity(ent_batch)
                    local_index[i - start:end_index - start] = entity_repr.half().detach()
            logger.info("Gathering local indices")
            torch.cuda.empty_cache()
            indices = self.all_gather(local_index.detach())
            ent_index_cpu = indices.cpu()
            if indices.dim() == 3:
                assert indices.size(0) == world_size
                ent_index_cpu = rearrange(ent_index_cpu, "w e d -> (w e) d") #.float()
            indices = None
            torch.cuda.empty_cache()
            self.ent_index = ent_index_cpu.detach().to(self.device)
    
    def setup_entity_index_sequential(self, spherical_coord=False) -> None:
        num_entities = len(self.data.ent_catalogue.entities)
        neighbors_dataset = self.data.train_dataset.neighbors_dataset

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
                ent_batch = neighbors_dataset.get_slice(i, end_index)
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

        if self.config.duck.duck_loss is not None and not self.no_box_ablation:
            self.duck_loss = hydra.utils.instantiate(
                self.config.duck.duck_loss
            )
        
        if self.relations_as_points:
            self.duck_point_loss = nn.BCEWithLogitsLoss()

        if self.config.get("ckpt_path") is not None:
            with open(self.config.ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location=torch.device("cpu"))
            self.load_state_dict(checkpoint["state_dict"], strict=False)

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
            output_size=self.box_size,
            freeze=True
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

    def ed_score(self, mentions: Tensor, entities: Tensor):
        if entities.dim() == 2:
            return torch.matmul(
                mentions.to(entities.dtype),
                entities.transpose(0, 1)
            )
        return torch.bmm(
            mentions.unsqueeze(1).to(entities.dtype),
            entities.transpose(1, -1)
        ).squeeze(1)

    def log_combined_score(self, mentions, ed_scores, prior_probabilities):
        no_prior_mask = (prior_probabilities == 0.0).all(dim=-1)
        prior_probabilities[no_prior_mask] = 1.0
        prior_probabilities = prior_probabilities / prior_probabilities.sum(dim=-1).unsqueeze(-1)
        logweights = self.prior_ffn(mentions)
        return torch.logaddexp(
            logweights[:, :1] + ed_scores.log_softmax(dim=-1),
            logweights[:, 1:] + prior_probabilities.log()
        )

    def encode_with_boxes(self, entity, relations, relation_mask):
        entity_boxes = self.relations_to_box(relations)
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

    def encode_mention(self, mention):
        return self.mention_encoder(
            mention["data"], attention_mask=mention["attention_mask"]
        )[0]
    
    def forward(self, batch):
        mentions = batch["mentions"] 
        entities = batch["entities"]
        relation_ids = batch["relation_ids"]
        ent_rel_mask = batch["ent_rel_mask"]

        entity_tensor_mask = batch["entity_tensor_mask"].bool()

        assert mentions["data"].size(-1) <= self.config.data.transform.max_mention_len
        assert entities["data"].size(-1) <= self.config.data.transform.max_entity_len

        mention_repr = self.encode_mention(mentions)
        
        entity_repr, entity_boxes, relation_sets = self.encode_entity_with_relations(
            entities,
            relation_ids,
            entity_tensor_mask,
            ent_rel_mask
        )
        
        return {
            "mentions": mention_repr,
            "entities": entity_repr,
            "entity_boxes": entity_boxes,
            "relation_set_embeddings": relation_sets
        }
    
    def relations_to_box(self, rel_ids):
        # rel_ids = rearrange(batch["relation_ids"]["data"], "b r -> r b")
        return self.rel_encoder(rel_ids.long())
        # return boxes.rearrange("r b d -> b r d")
        # return self.intersection(boxes)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None  # for debug
   
        representations = self(batch)
        bsz = representations["mentions"].size(0)
        representations, batch = self._gather_representations(representations, batch)

        mentions = representations["mentions"]
        entities = representations["entities"]
        entity_boxes = representations["entity_boxes"]
        target = batch["targets"]
        relation_set_embeddings = representations["relation_set_embeddings"]
        ent_rel_mask = batch["ent_rel_mask"]
        
        duck_loss_entity = 0.0
        duck_loss_mention = 0.0
        duck_metrics = {}
        regularization = 0.0

        if self.duck_loss is not None and not self.config.duck.entity_priors:
            rel_ids = batch["relation_ids"]["data"]
            duck_metrics = self.compute_box_metrics(entity_boxes)
            duck_metrics = prefix_suffix_keys(duck_metrics, "Boxes/")

            rel_ids = expand_with_mask(rel_ids, ent_rel_mask)

            duck_entity_metrics, negative_boxes = self.compute_duck_loss_with_boxes(
                entities, mentions, entity_boxes, rel_ids, ent_rel_mask
            )
            duck_loss_entity = duck_entity_metrics["loss"]
            regularization = duck_entity_metrics["regularization"]
            duck_entity_metrics = {
                k: v for k, v in duck_entity_metrics.items()
                if k not in ["loss", "regularization"]
            }
            duck_entity_metrics = prefix_suffix_keys(duck_entity_metrics, prefix="Entity/")
            duck_metrics.update(duck_entity_metrics)

            if self.config.duck.mention_in_box:
                mention_rel_mask = ent_rel_mask[target[:bsz].clone()]
                negative_boxes = negative_boxes[target[:bsz].clone()]
                duck_mention_metrics, _ = self.compute_duck_loss_with_boxes(
                    mentions, entities, entity_boxes,
                    rel_ids[target[:bsz].clone()], mention_rel_mask.clone(),
                    negative_boxes=negative_boxes
                )
                duck_loss_mention = duck_mention_metrics["loss"]
                duck_mention_metrics = {
                    k: v for k, v in duck_mention_metrics.items()
                    if k not in ["loss", "regularization"]
                }
                duck_mention_metrics = prefix_suffix_keys(duck_mention_metrics, prefix="Mention/")
                duck_metrics.update(duck_mention_metrics)
                regularization = 2 * regularization

            entities = self.entity_dropout(entities)
            mentions = self.mention_dropout(mentions)
            
        if self.relations_as_points and not self.joint_ent_rel_encoding:
            ent_to_rel_target = torch.eye(entities.size(0)).to(self.device)
            ent_to_rel_scores = torch.matmul(
                relation_set_embeddings,
                entities.transpose(0, 1)
            )
            duck_loss_entity = self.duck_point_loss(ent_to_rel_scores, ent_to_rel_target)
        
        ed_score = self.ed_score(mentions, entities)
        if self.config.duck.entity_priors:
            prior_probabilities = batch["prior_probabilities"]
            log_combined_score = self.log_combined_score(mentions, ed_score, prior_probabilities)
            ed_loss = self.prior_loss(log_combined_score, target)
            loss = ed_loss
        else:
            ed_loss = self.ed_loss(ed_score, target)
            loss = ed_loss + self.duck_loss_weight * (duck_loss_entity + duck_loss_mention + regularization)

        metrics = {
            "train/duck_loss_entity": duck_loss_entity,
            "train/duck_loss_mention": duck_loss_mention,
            "train/ed_loss": ed_loss,
            "train/loss": loss,
            "train/regularization": regularization
        }
        metrics.update({
            k: v
            for k, v in duck_metrics.items()
            if k != "loss" and isinstance(v, torch.Tensor) 
        })

        if self.logger is not None:
            self.log_dict(metrics)
            if not isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
                wandb.log({
                    k: v
                    for k, v in duck_metrics.items()
                    if k not in metrics
                })

        return loss

    def compute_duck_loss_with_boxes(
            self,
            entities,
            mentions,
            entity_boxes,
            rel_ids,
            mask,
            negative_boxes=None
        ):
        if self.config.duck.get("gaussian_box_regularization") and self.training:
            std = self.config.duck.gaussian_box_regularization
            entities.add_(std * torch.randn_like(entities))

        result = {}
        entity_boxes = expand_box_with_mask(entity_boxes, mask)
        entities_ = entities[..., :self.box_size].clone()
        mentions_ = mentions[..., :self.box_size].clone()
        entity_boxes = entity_boxes[..., :self.box_size].clone()
        if self.config.duck.boxes.parametrization == "spherical":
            entity_boxes, entities_ = self.handle_spherical_coord(
                entity_boxes, entities_, mentions_
            )
        
        entities_ = entities_[:entity_boxes.box_shape[0], :].clone()

        if negative_boxes is None:
            if self.config.duck.negative_box_sampling == "uniform":
                negative_boxes = self.uniform_negative_box_sampling(rel_ids)
            else:
                negative_boxes = self.self_adversarial_negative_box_sampling(entities_, rel_ids)
            negative_boxes = negative_boxes[..., :self.box_size].clone()
            negative_boxes = negative_boxes[..., :entities_.size(-1)].clone()
        
        dropout_mask = torch.full_like(entities_, False).bool()
        if self.box_dropout_dist is not None and self.training:
            dropout_mask = self.box_dropout_dist.sample(entities_.size()).bool()
        dropout_mask = dropout_mask.to(entities_.device)

        duck_loss = self.duck_loss(
            entities_.masked_fill(dropout_mask, 0.0),
            entity_boxes.masked_fill(dropout_mask.unsqueeze(1), 0.0),
            negative_boxes.masked_fill(dropout_mask.unsqueeze(1), 0.0),
            mask=mask
        )
        regularization = 0.0
        if self.regularizer is not None:
            regularization = self.regularizer(
                entity_boxes[mask].cat(negative_boxes.rearrange("b n d -> (b n) d"))
            )

        box_metrics = self.compute_box_containment_metrics(entities_, entity_boxes, rel_ids, mask)
        result.update(box_metrics)
        f1_metrics = self.compute_box_f1(entities_, rel_ids)
        result.update(f1_metrics)

        if isinstance(duck_loss, dict):
            result.update(duck_loss)
        else:
            result["loss"] = duck_loss
        
        result["regularization"] = regularization
        return result, negative_boxes

    def compute_box_containment_metrics(self, entities, entity_boxes, relation_ids, ent_rel_mask):
        result = {}
        entities = entities.unsqueeze(1).half()
        entity_boxes = entity_boxes.half()
        negative_boxes = self.uniform_negative_box_sampling(relation_ids, num_samples=128)[..., :entities.size(-1)].half()
        in_pos = (entities > entity_boxes.left) & (entities < entity_boxes.right)
        in_pos = in_pos[ent_rel_mask]
        result["containment_positive_boxes"] = in_pos.all(dim=-1).float().mean()
        result["dim_wise_containment_positive_boxes"] = wandb.Histogram(
            in_pos.float().mean(dim=-1).detach().cpu().numpy()
        )
        in_neg = (entities > negative_boxes.left) & (entities < negative_boxes.right)
        result["containment_negative_boxes"] = in_neg.all(dim=-1).float().mean()
        result["dim_wise_containment_negative_boxes"] = wandb.Histogram(
            in_neg.float().mean(dim=-1).detach().cpu().numpy()
        )
        return result
    
    def compute_box_f1(self, entities, relation_ids):
        result = {}
        entities = entities.unsqueeze(1).half()
        ids_range = torch.arange(0, len(self.data.rel_catalogue) + 1, device=self.device)
        all_boxes = self.rel_encoder(ids_range).half()
        all_boxes = all_boxes[:, :entities.size(-1)]
        predicted_mask = ((all_boxes.left < entities) & (entities < all_boxes.right)).all(dim=-1)
        gold_mask = torch.full_like(predicted_mask, False).scatter_(1, relation_ids, True)
        # dist = hydra.utils.instantiate(self.config.duck.duck_loss.distance_function)
        # distances = dist(entities, all_boxes)
        # sort_perm = torch.argsort(distances, dim=-1)
        # predicted_mask = predicted_mask.gather(-1, sort_perm)
        # gold_mask = gold_mask.gather(-1, sort_perm)
        matches = (predicted_mask & gold_mask)
        precision = torch.nan_to_num((matches.sum(dim=-1) / predicted_mask.sum(dim=-1)))
        recall =  matches.sum(dim=-1) / gold_mask.sum(dim=-1)
        f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
        result["box_precision"] = precision.mean()
        result["box_recall"] = recall.mean()
        result["box_f1"] = f1.mean()
        # result["box_p@10"] = torch.nan_to_num(
        #     (matches[:, :10].sum(dim=-1) / predicted_mask[:, :10].sum(dim=-1))
        # ).mean()
        return result

    def compute_box_metrics(self, entity_boxes):
        result = {}
        result["box_left_distribution"] = wandb.Histogram(entity_boxes.left.detach().cpu().numpy())
        result["box_right_distribution"] = wandb.Histogram(entity_boxes.right.detach().cpu().numpy())
        result["box_left_mean"] = entity_boxes.left.mean()
        result["box_right_mean"] = entity_boxes.right.mean()
        box_size = entity_boxes.right - entity_boxes.left
        result["box_size_distribution"] = wandb.Histogram(box_size.detach().cpu().numpy())
        result["box_size_mean"] = box_size.mean()
        return result

    def handle_spherical_coord(self, entity_boxes, entities, mentions):
        # abs on last coord to keep range [0, pi]
        entities[..., -1] = torch.abs(entities[..., -1].clone()) 
        mentions[..., -1] = torch.abs(mentions[..., -1].clone())
        _, entities = cartesian_to_spherical(entities)
        # entities_ = entities_[..., :-1]  # drop last coord to keep the range [0, pi]
        entity_boxes = entity_boxes[..., :entities.size(-1)].clone()
        return entity_boxes, entities

    def score(self, mentions, entities, prior_probabilities=None):
        ed_score = self.ed_score(mentions, entities).float()
        if prior_probabilities is None:
            return ed_score.softmax(dim=-1)
        return self.log_combined_score(mentions, ed_score, prior_probabilities).float().exp()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mentions = self.encode_mention(batch["mentions"])
        entities = self.ent_index
        target = batch["entity_ids"]
        prior_probabilities = None
        if self.config.duck.entity_priors:
            prior_indices = batch["prior_indices"]
            prior_probabilities = torch.zeros((mentions.size(0), entities.size(0)), device=mentions.device)
            prior_probabilities = prior_probabilities.scatter_(
                1, prior_indices.long(), batch["prior_probabilities"]
            )
        scores = self.score(mentions, entities, prior_probabilities=prior_probabilities)
        preds = scores.argmax(dim=-1)

        candidate_preds = preds
        candidate_preds_at_30 = preds
        candidate_target = target
        if batch["candidates"] is not None:
            candidate_indices = batch["candidates"]["data"].long()
            candidates = self.ent_index[candidate_indices]
            if self.config.duck.entity_priors:
                prior_probabilities = prior_probabilities.gather(1, candidate_indices)
            candidate_scores = self.score(mentions, candidates, prior_probabilities=prior_probabilities)
            candidate_mask = batch["candidates"]["attention_mask"].bool()
            candidate_scores[~candidate_mask] = 0
            candidate_preds = candidate_indices.gather(
                1, candidate_scores.argmax(dim=-1).unsqueeze(1)
            ).squeeze(dim=1)
            candidate_preds_at_30 = candidate_indices.gather(
                1, candidate_scores[:, :30].argmax(dim=-1).unsqueeze(1)
            ).squeeze(dim=1)
            # candidate_mask = candidate_mask.any(dim=-1)
            # candidate_preds = candidate_preds[candidate_mask]
            # candidate_preds_at_30 = candidate_preds_at_30[candidate_mask]
            # candidate_target = candidate_target[candidate_mask]
        
        return {
            "preds": preds,
            "candidate_preds": candidate_preds,
            "target": target,
            "topk": scores.topk(100),
            "candidate_preds_at_30": candidate_preds_at_30,
            "candidate_target": candidate_target
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
            preds = torch.cat([o["preds"] for o in dataset_outputs])
            target = torch.cat([o["target"] for o in dataset_outputs])
            candidate_target = torch.cat([o["candidate_target"] for o in dataset_outputs])
            topk = [o["topk"] for o in dataset_outputs]
            candidate_preds = torch.cat([o["candidate_preds"] for o in dataset_outputs])
            candidate_preds_at_30 = torch.cat([o["candidate_preds_at_30"] for o in dataset_outputs])

            preds = self.all_gather_flat(preds)
            target = self.all_gather_flat(target)
            candidate_preds = self.all_gather_flat(candidate_preds)
            candidate_preds_at_30 = self.all_gather_flat(candidate_preds_at_30)
            candidate_target = self.all_gather_flat(candidate_target)
            
            micro_f1 = torchmetrics.functional.classification.multiclass_f1_score(
                preds, target, num_classes=len(self.data.ent_catalogue), average='micro'
            ).item()
            micro_f1_candidate_set = torchmetrics.functional.classification.multiclass_f1_score(
                candidate_preds, candidate_target, num_classes=len(self.data.ent_catalogue), average='micro'
            ).item()
            micro_f1_candidate_set_at_30 = torchmetrics.functional.classification.multiclass_f1_score(
                candidate_preds_at_30, candidate_target, num_classes=len(self.data.ent_catalogue), average='micro'
            ).item()


            metrics[f"Micro-F1/{dataset}"] = micro_f1
            tqdm.write(f"Micro F1 on {dataset}: \t{micro_f1:.4f}")
            metrics[f"Micro-F1_candidate_set/{dataset}"] = micro_f1_candidate_set
            tqdm.write(f"Micro F1 on {dataset} (with candidate set): \t{micro_f1_candidate_set:.4f}")
            metrics[f"Micro-F1_candidate_set_at_30/{dataset}"] = micro_f1_candidate_set_at_30
            tqdm.write(f"Micro F1 on {dataset} (with candidate set at 30): \t{micro_f1_candidate_set_at_30:.4f}")

            recall_steps = [10, 30, 50, 100]
            for k in recall_steps:
                recall = self.recall_at_k(topk, target, k)
                metrics[f"Recall@{k}/{dataset}"] = recall
            
        metric_names = list(dict.fromkeys([k.split("/")[0] for k in metrics]))
        for avg_metric_key in metric_names:
            value = 0
            for dataset_name in self.datasets["test"]:
                value += metrics[f"{avg_metric_key}/{dataset_name}"]
            metrics[avg_metric_key + "/Average"] = value / len(self.datasets["test"])

        tqdm.write(f"Average Micro F1: {metrics['Micro-F1/Average']:.4f}")
        
        metrics["avg_micro_f1"] = metrics["Micro-F1_candidate_set/Average"]
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
        ).item()

    def configure_optimizers(self):
        trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        if self.config.duck.entity_priors:
            for n, p in self.named_parameters():
                if "prior" not in n:
                    p.requires_grad = False
            trainable_parameters = [p for n, p in self.named_parameters() if "prior" in n and p.requires_grad]
        optimizer = hydra.utils.instantiate(
            self.config.optim,
            trainable_parameters,
            _recursive_=False
        )
        if self.config.get("lr_scheduler") is None:
            return optimizer
        if self.config.lr_scheduler.get("_target_") is not None:
            scheduler = hydra.utils.instantiate(
                self.config.lr_scheduler, optimizer=optimizer
            )
        else:
            num_training_steps = self.config.trainer.max_epochs * len(self.data.train_dataloader())
            accumulate_grad_batches = self.config.trainer.get("accumulate_grad_batches")
            if accumulate_grad_batches is not None:
                num_training_steps /= accumulate_grad_batches
            num_warmup_steps = self.config.lr_scheduler.num_warmup_steps
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def self_adversarial_negative_box_sampling(self, entities, relation_ids, sampling_temperature=0.5):
        sampling_temperature = self.config.duck.duck_loss.sampling_temperature or sampling_temperature
        with torch.no_grad():
            ids_range = torch.arange(0, len(self.data.rel_catalogue) + 1, device=self.device)
            all_boxes = self.rel_encoder(ids_range).half()[..., :entities.size(-1)]
            dist = hydra.utils.instantiate(self.config.duck.duck_loss.distance_function)
            distances = dist(entities.unsqueeze(1).half(), all_boxes)
            distances.scatter_(1, relation_ids, float("inf"))
            if self.config.duck.negative_box_sampling == "topk":
                topk = distances.topk(self.config.duck.num_negative_boxes, dim=-1, largest=False)
                negative_ids = topk.indices
            else:
                distribution = torch.nn.functional.softmax(-sampling_temperature * distances, dim=-1)
                negative_ids = torch.multinomial(distribution, num_samples=self.config.duck.num_negative_boxes) 
            if self.config.debug:
                for i, row in enumerate(negative_ids):
                    set_pos = set(rid.item() for rid in relation_ids[i]) 
                    set_neg = set(rid.item() for rid in row)
                    assert len(set_pos & set_neg) == 0
        return self.rel_encoder(negative_ids.detach())

    def uniform_negative_box_sampling(self, relation_ids, num_samples=None):
        num_samples = num_samples or self.config.duck.num_negative_boxes
        with torch.no_grad():
            ids_range = torch.arange(0, len(self.data.rel_catalogue) + 1, device=self.device)
            weights = torch.ones((relation_ids.size(0), ids_range.size(0)), device=self.device)
            weights.scatter_(1, relation_ids, float("-inf"))
            distribution = torch.nn.functional.softmax(weights, dim=-1)
            negative_ids = torch.multinomial(distribution, num_samples=num_samples).detach()
            if self.config.debug:
                for i, row in enumerate(negative_ids):
                    set_pos = set(rid.item() for rid in relation_ids[i]) 
                    set_neg = set(rid.item() for rid in row)
                    assert len(set_pos & set_neg) == 0
        return self.rel_encoder(negative_ids)
   
    def _gather_representations(self, representations, batch):
        entities = representations["entities"]
        mentions = representations["mentions"]
        target = batch["targets"]
        entity_ids = batch["entity_ids"]
        # rel_ids = batch["relation_ids"]
        entity_tensor_mask = batch["entity_tensor_mask"].bool()
        relation_set_embeddings = representations["relation_set_embeddings"]
        prior_probabilities = batch["prior_probabilities"]

        if not self.gather_on_ddp or not isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
            representations["entities"] = entities[entity_tensor_mask]
            if relation_set_embeddings is not None:
                representations["relation_set_embeddings"] = relation_set_embeddings[entity_tensor_mask]
            batch["prior_probabilities"] = prior_probabilities[:, entity_tensor_mask]
            return representations, batch

        mentions_to_send = mentions.detach()
        entities_to_send = entities.detach()
    
        all_mentions_repr = self.all_gather(mentions_to_send)  # num_workers x bs
        all_entities_repr = self.all_gather(entities_to_send)

        all_relation_set_embeddings = None
        if relation_set_embeddings is not None:
            all_relation_set_embeddings = self.all_gather(relation_set_embeddings.detach())

        all_targets = self.all_gather(target)
        all_entity_ids = self.all_gather(entity_ids)
        all_mask = self.all_gather(entity_tensor_mask)
        all_prior_probabilities = self.all_gather(prior_probabilities.detach())

        # offset = 0
        all_mentions_list = []
        all_entities_list = []
        all_entity_ids_list = []
        all_targets_list = []
        # all_boxes_list = []
        # all_rel_ids_list = []
        all_relation_set_embeddings_list = []
        all_prior_probabilities_list = []

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
        all_prior_probabilities_list.append(prior_probabilities[:, entity_tensor_mask])
        # offset += entities_repr.size(0)

        for i in range(all_targets.size(0)):
            if i != self.local_rank:
                all_mentions_list.append(all_mentions_repr[i])
                all_entities_list.append(all_entities_repr[i][all_mask[i]])
                all_entity_ids_list.append(
                    all_entity_ids[i][all_mask[i].bool()].tolist()
                )
                all_targets_list.append(all_targets[i])
                all_prior_probabilities_list.append(all_prior_probabilities[i][:, all_mask[i]])
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
        entities, target, relation_set_embeddings, prior_probabilities = self.filter_representations(
            all_entities_list,
            all_entity_ids_list,
            # all_boxes_list,
            # all_rel_ids_list,
            all_targets_list,
            all_relation_set_embeddings_list,
            all_prior_probabilities_list
        )
        
        representations["mentions"] = mentions
        representations["entities"] = entities
        # representations["entity_boxes"] = boxes
        representations["relation_set_embeddings"] = relation_set_embeddings
        batch["targets"] = target
        batch["prior_probabilities"] = prior_probabilities
        # batch["relation_ids"] = rel_ids

        return representations, batch
    
    def filter_representations(
        self,
        all_entities_list,
        all_entity_ids_list,
        all_targets_list,
        all_relation_set_embeddings,
        all_prior_probabilities_list
    ):
        filtered_entities_repr = []
        filtered_targets = []
        # filtered_boxes = []
        # filtered_rel_ids_data = []
        # filtered_rel_ids_mask = []
        filtered_relation_set_embeddings = []
        ent_indices_map = {}

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
                if ent_id not in ent_indices_map:
                    ent_idx = len(ent_indices_map)
                    ent_indices_map[ent_id] = ent_idx
                    filtered_entities_repr.append(entity_repr)
                    # filtered_boxes.append(box)
                    # filtered_rel_ids_data.append(rel_ids["data"][i])
                    # filtered_rel_ids_mask.append(rel_ids["attention_mask"][i])
                    filtered_relation_set_embeddings.append(relation_set_emb)
            for target in targets.tolist():
                filtered_targets.append(ent_indices_map[entity_ids[target]])

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
        
        with torch.no_grad():
            filtered_prior_probabilities = torch.zeros((
                sum(p.size(0) for p in all_prior_probabilities_list),
                filtered_entities_repr.size(0)
            ), device=filtered_entities_repr.get_device())

            low = 0
            for prior_probabilities in all_prior_probabilities_list:
                high = low + prior_probabilities.size(0)
                filtered_prior_probabilities[low:high, :prior_probabilities.size(1)] = prior_probabilities
                low += prior_probabilities.size(0)

        return filtered_entities_repr, \
            filtered_targets, \
            filtered_relation_set_embeddings, \
            filtered_prior_probabilities.clone().detach()

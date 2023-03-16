from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from duck.box_tensors.box_tensor import BoxTensor
from duck.common.utils import log1mexp, logexpm1
from einops import rearrange, repeat
from duck.box_tensors.intersection import Intersection
import wandb


class DuckNegativeSamplingLoss(nn.Module):
    def __init__(self,
        distance_function=None,
        margin_pos: float = 1.0,
        margin_neg: float = 1.0,
        reduction: str = "mean",
        **kwargs
    ):
        super(DuckNegativeSamplingLoss, self).__init__()
        self.distance_function = distance_function
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
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
        
        mask = kwargs.get("mask")
        if mask is None:
            mask = torch.full_like(positive_dist, True).bool()
        else:
            mask = rearrange(mask, "b n -> n b").bool()

        positive_dist[~mask] = 0.0
        positive_dist = positive_dist.sum(dim=0) / mask.sum(dim=0)

        positive_term = torch.nn.functional.logsigmoid(self.margin_pos - positive_dist).mean(dim=0)
        negative_term = torch.nn.functional.logsigmoid(negative_dist - self.margin_neg).mean(dim=0)

        loss = -positive_term - negative_term
        
        return {
            "loss": self._reduce(loss),
            "positive_distance": wandb.Histogram(positive_dist.detach().cpu().numpy()),
            "negative_distance": wandb.Histogram(negative_dist.detach().cpu().numpy()),
            "positive_distance_mean": positive_dist.mean(),
            "negative_distance_mean": negative_dist.mean()
        }


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
        
        mask = kwargs.get("mask")
        if mask is None:
            mask = torch.full_like(positive_dist, True).bool()
        else:
            mask = rearrange(mask, "b n -> n b").bool()
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
        
        mask = kwargs.get("mask")
        if mask is None:
            mask = torch.full_like(left_delta_pos, True).bool()
        else:
            mask = rearrange(mask, "b n -> n b").bool()

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


class DuckNCELoss(nn.Module):
    def __init__(self,
        distance_function=None,
        margin_pos: float = 1.0,
        margin_neg: float = 1.0,
        reduction: str = "mean",
        intersect_positive_boxes: bool = False,
        **kwargs
    ):
        super(DuckNCELoss, self).__init__()
        self.distance_function = distance_function
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.intersect_positive_boxes = intersect_positive_boxes
        self.intersection = Intersection(intersection_temperature=0.0001)
        
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
        negative_dist = self.distance_function(entities, negative_boxes)

        distances = torch.cat([
            positive_dist - self.margin_pos,
            negative_dist - self.margin_neg
        ], dim=0)
        # margin_dist = self.margin - distances
        logits = torch.stack([distances, -distances])
        logits = rearrange(logits, "c n b -> b c n")
        target = torch.zeros_like(logits[:, 0, :]).long()
        target[:, :positive_dist.size(0)] = 1
        loss = self.criterion(logits, target.long())
        
        mask = kwargs.get("mask")
        if mask is None:
            mask = torch.full_like(positive_dist, True).bool().t()
        mask = torch.cat([
            mask, torch.ones_like(negative_dist.t()).bool()
        ], dim=1)
        
        loss.masked_fill_(~mask, 0.0)

        return {
            "loss": self._reduce(loss),
            "positive_distance": wandb.Histogram(positive_dist.detach().cpu().numpy()),
            "negative_distance": wandb.Histogram(negative_dist.detach().cpu().numpy()),
            "positive_distance_mean": positive_dist.mean(),
            "negative_distance_mean": negative_dist.mean()
        }


class DuckNegativeSamplingLossWithTemperature(nn.Module):
    def __init__(self,
        distance_function=None,
        margin_pos: float = 1.0,
        margin_neg: float = 1.0,
        sampling_temperature: Optional[float] = None,
        reduction: str = "mean",
        intersect_positive_boxes: bool = False
    ):
        super(DuckNegativeSamplingLossWithTemperature, self).__init__()
        self.distance_function = distance_function
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.reduction = reduction
        self.intersect_positive_boxes = intersect_positive_boxes
        self.intersection = Intersection(intersection_temperature=0.0001)
        self.sampling_temperature = sampling_temperature
        
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
        
        mask = kwargs.get("mask")
        if mask is None:
            mask = torch.full_like(positive_dist, True).bool()
        else:
            mask = rearrange(mask, "b n -> n b").bool()

        if self.intersect_positive_boxes:
            positive_boxes.left[~mask] = float("-inf")
            positive_boxes.right[~mask] = float("inf")
            positive_boxes = self.intersection(positive_boxes).rearrange("b d -> 1 b d")

        positive_dist = self.distance_function(entities, positive_boxes)
        negative_dist = self.distance_function(entities, negative_boxes)
        
        positive_term = torch.nn.functional.logsigmoid(self.margin_pos - positive_dist)
        negative_term = torch.nn.functional.logsigmoid(negative_dist - self.margin_neg)

        if not self.intersect_positive_boxes:
            positive_term[~mask] = 0.0
            positive_dist[~mask] = 0.0

        positive_term = positive_term.sum(dim=0) / mask.sum(dim=0)
        if self.sampling_temperature is not None:
            weights = torch.nn.functional.softmax(-self.sampling_temperature * negative_dist, dim=0)
            negative_term = (negative_term * weights).sum(dim=0)
        else:
            negative_term = negative_term.mean(dim=0)

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


class GaussianBoxProbabilisticMembership(nn.Module):
    def __init__(
            self,
            sigma=0.1,
            log_scale=True,
            dim=-1
    ):
        super(GaussianBoxProbabilisticMembership, self).__init__()
        self.dim = dim
        self.log_scale = log_scale
        self.sigma = sigma
        self.gamma = 1.702

    def forward(self, entity, box):
        temperature = self.gamma / self.sigma
        eps = 1e-7
        result = torch.sigmoid(temperature * (entity - box.left)) - torch.sigmoid(temperature * (entity - box.right))
        result = result + result * (1 - torch.sigmoid(temperature * (box.center - box.left)) + torch.sigmoid(temperature * (box.center - box.right)))
        if self.log_scale:
            return result.clamp_min(eps).log()
        return result


class GaussianBoxMembershipNLLLoss(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,
        reduction="mean",
        **kwargs
    ):
        super(GaussianBoxMembershipNLLLoss, self).__init__()
        self.probabilistic_membership = GaussianBoxProbabilisticMembership(
            sigma=sigma,
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
            target = target.detach().clone()
        entities = repeat(entities, "b d -> b n d", n=boxes.box_shape[1])
        logp = self.probabilistic_membership(entities, boxes).mean(dim=-1)
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
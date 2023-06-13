# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional, List


@dataclass
class Entity:
    entity_id: str  # E.g. "Q3312129"
    offset: int
    length: int
    text: str
    entity_type: Optional[str] = None  # E.g. wiki
    md_score: Optional[float] = None
    el_score: Optional[float] = None

    @property
    def mention(self):
        return self.text[self.offset : self.offset + self.length]

    @property
    def extended_mention(self):
        """Mentin in surrounding context (10 chars), with the mention in brackets"""
        left_context = self.text[max(0, self.offset - 10) : self.offset]
        right_context = self.text[self.offset + self.length : self.offset + self.length + 10]
        # Add ... if the context is truncated
        if self.offset - 10 > 0:
            left_context = "..." + left_context
        if self.offset + self.length + 10 < len(self.text):
            right_context = right_context + "..."
        return f"{left_context}[{self.mention}]{right_context}"

    def __repr__(self):
        str_repr = f'Entity<mention="{self.extended_mention}", entity_id={self.entity_id}'
        if self.md_score is not None and self.el_score is not None:
            str_repr += f", md_score={self.md_score:.2f}, el_score={self.el_score:.2f}"
        str_repr += ">"
        return str_repr

    def __eq__(self, other):
        return self.offset == other.offset and self.length == other.length and self.entity_id == other.entity_id


class Sample:
    text: str
    sample_id: Optional[str] = None
    ground_truth_entities: Optional[List[Entity]] = None
    predicted_entities: Optional[List[Entity]] = None

    def __init__(self, text, sample_id=None, ground_truth_entities=None, predicted_entities=None):
        self.text = text
        self.sample_id = sample_id
        self.ground_truth_entities = ground_truth_entities
        self.predicted_entities = predicted_entities
        if self.ground_truth_entities is not None and self.predicted_entities is not None:
            self.compute_scores() 

    def compute_scores(self):
        self.true_positives = [
            predicted_entity
            for predicted_entity in self.predicted_entities
            if predicted_entity in self.ground_truth_entities
        ]
        self.false_positives = [
            predicted_entity
            for predicted_entity in self.predicted_entities
            if predicted_entity not in self.ground_truth_entities
        ]
        self.false_negatives = [
            ground_truth_entity
            for ground_truth_entity in self.ground_truth_entities
            if ground_truth_entity not in self.predicted_entities
        ]
        # Bag of entities
        self.ground_truth_entity_ids = set(
            [ground_truth_entity.entity_id for ground_truth_entity in self.ground_truth_entities]
        )
        self.predicted_entity_ids = set(
            [predicted_entity.entity_id for predicted_entity in self.predicted_entities]
        )
        self.true_positives_boe = [
            predicted_entity_id
            for predicted_entity_id in self.predicted_entity_ids
            if predicted_entity_id in self.ground_truth_entity_ids
        ]
        self.false_positives_boe = [
            predicted_entity_id
            for predicted_entity_id in self.predicted_entity_ids
            if predicted_entity_id not in self.ground_truth_entity_ids
        ]
        self.false_negatives_boe = [
            ground_truth_entity_id
            for ground_truth_entity_id in self.ground_truth_entity_ids
            if ground_truth_entity_id not in self.predicted_entity_ids
        ]

    def __repr__(self):
        repr_str = f'Sample(text="{self.text[:100]}..."'
        if self.ground_truth_entities is not None:
            repr_str += f", ground_truth_entities={self.ground_truth_entities[:3]}..."
        if self.predicted_entities is not None:
            repr_str += f", predicted_entities={self.predicted_entities[:3]}..."
        repr_str += ")"
        return repr_str

    def print(self, max_display_length=1000):
        print(f"{self.text[:max_display_length]=}")
        if self.ground_truth_entities is not None:
            print("***************** Ground truth entities *****************")
            print(f"{len(self.ground_truth_entities)=}")
            for ground_truth_entity in self.ground_truth_entities:
                if ground_truth_entity.offset + ground_truth_entity.length > max_display_length:
                    continue
                print(ground_truth_entity)
        if self.predicted_entities is not None:
            print("***************** Predicted entities *****************")
            print(f"{len(self.predicted_entities)=}")
            for predicted_entity in self.predicted_entities:
                if predicted_entity.offset + predicted_entity.length > max_display_length:
                    continue
                print(predicted_entity)


def convert_jsonl_data_to_samples(jsonl_data: List[Dict[str, Any]]) -> List[Sample]:
    """Converts the jsonl data to a list of samples."""
    samples = []
    for example in jsonl_data:
        ground_truth_entities = [
            Entity(entity_id=entity_id, offset=offset, length=length, text=example["original_text"])
            for _, _, entity_id, _, offset, length in example["gt_entities"]
        ]
        sample = Sample(text=example["original_text"], ground_truth_entities=ground_truth_entities)
        samples.append(sample)
    return samples


def convert_predictions_to_entities(example_predictions: Dict[str, List], text) -> List[Entity]:
    """Converts the predictions of a single example to a list of entities."""
    predicted_entities = [
        Entity(entity_id=entity_id, offset=offset, length=length, md_score=md_score, el_score=el_score, text=text)
        for offset, length, entity_id, md_score, el_score in zip(
            example_predictions["offsets"],
            example_predictions["lengths"],
            example_predictions["entities"],
            example_predictions["md_scores"],
            example_predictions["el_scores"],
        )
    ]
    return predicted_entities


def convert_jsonl_data_and_predictions_to_samples(
    jsonl_data: List[Dict[str, Any]], predictions: Dict[str, List], md_threshold, el_threshold
) -> List[Sample]:
    samples = convert_jsonl_data_to_samples(jsonl_data)
    for sample, example_predictions in zip(samples, predictions):
        predicted_entities = convert_predictions_to_entities(example_predictions, sample.text)
        predicted_entities = [
            entity for entity in predicted_entities if entity.el_score > el_threshold and entity.md_score > md_threshold
        ]
        sample.predicted_entities = predicted_entities
        sample.compute_scores()
        samples.append(sample)
    return samples

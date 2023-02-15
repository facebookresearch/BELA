from dataclasses import dataclass
from pathlib import Path
import yaml
from hydra.experimental import compose, initialize_config_module
import hydra
import torch
from tqdm import tqdm
import json
import faiss
import logging

from typing import Optional, Union, List, Dict, Any, Tuple

from bela.transforms.spm_transform import convert_sp_to_char_offsets


logger = logging.getLogger(__name__)


def load_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    all_data = []
    with open(path, 'rt') as fd:
        for line in tqdm(fd):
            data = json.loads(line)
            all_data.append(data)
    return all_data


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

    def __repr__(self):
        str_repr = f"Entity<mention=\"{self.mention}\", entity_id={self.entity_id}"
        if self.md_score is not None and self.el_score is not None:
            str_repr += f", md_score={self.md_score:.2f}, el_score={self.el_score:.2f}"
        str_repr += ">"
        return str_repr

    def __eq__(self, other):
        return self.offset == other.offset and self.length == other.length and self.entity_id == other.entity_id


class Sample:
    text: str
    ground_truth_entities: List[Entity]
    predicted_entities: Optional[List[Entity]] = None

    def __init__(self, text, ground_truth_entities, predicted_entities=None):
        self.text = text
        self.ground_truth_entities = ground_truth_entities
        self.predicted_entities = predicted_entities
        if self.predicted_entities is not None:
            # Compute scores
            self.true_positives = [predicted_entity for predicted_entity in self.predicted_entities if predicted_entity in self.ground_truth_entities]
            self.false_positives = [predicted_entity for predicted_entity in self.predicted_entities if predicted_entity not in self.ground_truth_entities]
            self.false_negatives = [ground_truth_entity for ground_truth_entity in self.ground_truth_entities if ground_truth_entity not in self.predicted_entities]
            # Bag of entities
            self.ground_truth_entity_ids = set([ground_truth_entity.entity_id for ground_truth_entity in self.ground_truth_entities])
            self.predicted_entity_ids = set([predicted_entity.entity_id for predicted_entity in self.predicted_entities])
            self.true_positives_boe = [predicted_entity_id for predicted_entity_id in self.predicted_entity_ids if predicted_entity_id in self.ground_truth_entity_ids]
            self.false_positives_boe = [predicted_entity_id for predicted_entity_id in self.predicted_entity_ids if predicted_entity_id not in self.ground_truth_entity_ids]
            self.false_negatives_boe = [ground_truth_entity_id for ground_truth_entity_id in self.ground_truth_entity_ids if ground_truth_entity_id not in self.predicted_entity_ids]


    def __repr__(self):
        repr_str = f"Sample(text=\"{self.text[:100]}...\", ground_truth_entities={self.ground_truth_entities[:3]}..."
        if self.predicted_entities is not None:
            repr_str += f", predicted_entities={self.predicted_entities[:3]}..."
        repr_str += ")"
        return repr_str


    def print(self, max_display_length=1000):
        print(f"{self.text[:max_display_length]=}")
        print("***************** Ground truth entities *****************")
        print(f"{len(self.ground_truth_entities)=}")
        for ground_truth_entity in self.ground_truth_entities:
            if ground_truth_entity.offset + ground_truth_entity.length > max_display_length:
                continue
            print(ground_truth_entity)
        if self.predicted_entities is None:
            print("***************** Predicted entities *****************")
            print(f"{len(self.predicted_entities)=}")
            for predicted_entity in self.predicted_entities:
                if predicted_entity.offset + predicted_entity.length > max_display_length:
                    continue
                print(predicted_entity)


class ModelEval:
    def __init__(self, checkpoint_path, config_name="joint_el_mel"):
        self.device = torch.device("cuda:0")
        
        logger.info("Create task")
        with initialize_config_module("bela/conf"):
            cfg = compose(config_name=config_name)
            cfg.task.load_from_checkpoint = checkpoint_path  # Overwrite checkpoint path in config
        self.checkpoint_path = checkpoint_path
            
        self.transform = hydra.utils.instantiate(cfg.task.transform)
        # TODO: The datamodule instanciation takes 90s due to the memory map in joint_el_datamodule.py: prun output: 91.197   91.197 joint_el_datamodule.py:166(__init__)
        datamodule = hydra.utils.instantiate(cfg.datamodule, transform=self.transform)
        self.task = hydra.utils.instantiate(cfg.task, datamodule=datamodule, _recursive_=False)
        
        self.task.setup("train")
        self.task = self.task.eval()
        self.task = self.task.to(self.device)
        self.embeddings = self.task.embeddings
        self.faiss_index = self.task.faiss_index
        
        # logger.info("Create GPU index")
        # self.create_gpu_index()
        
        logger.info("Create ent index")
        self.ent_idx = []
        for ent in datamodule.ent_catalogue.idx:
            self.ent_idx.append(ent)
        
        logger.info("Load checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.task.load_state_dict(checkpoint["state_dict"])
        
    def create_gpu_index(self, gpu_id=0):
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = gpu_id
        flat_config.useFloat16 = True

        res = faiss.StandardGpuResources()

        self.faiss_index = faiss.GpuIndexFlatIP(res, embeddings.shape[1], flat_config)
        self.faiss_index.add(self.embeddings)
        
    def lookup(
        self,
        query: torch.Tensor,
    ):
        scores, indices = self.faiss_index.search(query, k=1)

        return scores.squeeze(-1).to(self.device), indices.squeeze(-1).to(self.device)
    
    def process_batch(self, texts): 
        batch: Dict[str, Any] = {"texts": texts}
        model_inputs = self.transform(batch)

        token_ids = model_inputs["input_ids"].to(self.device)
        text_pad_mask = model_inputs["attention_mask"].to(self.device)
        tokens_mapping = model_inputs["tokens_mapping"].to(self.device)
        sp_tokens_boundaries = model_inputs["sp_tokens_boundaries"].tolist()

        with torch.no_grad():
            _, last_layer = self.task.encoder(token_ids)
            text_encodings = last_layer
            text_encodings = self.task.project_encoder_op(text_encodings)

            mention_logits, mention_bounds = self.task.mention_encoder(
                text_encodings, text_pad_mask, tokens_mapping
            )

            (
                chosen_mention_logits,
                chosen_mention_bounds,
                chosen_mention_mask,
                mention_pos_mask,
            ) = self.task.mention_encoder.prune_ctxt_mentions(
                mention_logits,
                mention_bounds,
                num_cand_mentions=50,
                threshold=self.task.md_threshold,
            )

            mention_offsets = chosen_mention_bounds[:, :, 0]
            mention_lengths = (
                chosen_mention_bounds[:, :, 1] - chosen_mention_bounds[:, :, 0] + 1
            )
            mention_lengths[mention_offsets == 0] = 0

            mentions_repr = self.task.span_encoder(
                text_encodings, mention_offsets, mention_lengths
            )

            # flat mentions and entities indices (mentions_num x embedding_dim)
            flat_mentions_repr = mentions_repr[mention_lengths != 0]
            mentions_scores = torch.sigmoid(chosen_mention_logits)

            # retrieve candidates top-1 ids and scores
            cand_scores, cand_indices = self.lookup(
                flat_mentions_repr.detach()
            )

            entities_repr = self.embeddings[cand_indices].to(self.device)

            chosen_mention_limits: List[int] = (
                chosen_mention_mask.int().sum(-1).detach().cpu().tolist()
            )
            flat_mentions_scores = mentions_scores[mention_lengths != 0].unsqueeze(-1)
            cand_scores = cand_scores.unsqueeze(-1)

            el_scores = torch.sigmoid(
                self.task.el_encoder(
                    flat_mentions_repr,
                    entities_repr,
                    flat_mentions_scores,
                    cand_scores,
                )
            ).squeeze(1)

        predictions = []
        cand_idx = 0

        # mention_offsets include cls_token, but we don't use it here when converting to char offsets.
        mention_offsets =  (mention_offsets - 1).clamp(0)
        # TODO: it is not clear why we need to subtract 1 for the mention lengths.
        mention_lengths = (mention_lengths - 1).clamp(0)
        for text, offsets, lengths, md_scores in zip(
            texts, mention_offsets, mention_lengths, mentions_scores
        ):
            char_offsets = []
            char_lengths = []
            ex_entities = []
            ex_md_scores = []
            ex_el_scores = []
            for offset, length, md_score in zip(offsets, lengths, md_scores):
                if length != 0:
                    if md_score >= self.task.md_threshold:
                        # Convert to char offsets
                        sp_offset = offset.detach().cpu().item()
                        sp_length = length.detach().cpu().item()
                        char_offset, char_length = convert_sp_to_char_offsets(text, sp_offset, sp_length, self.transform.processor)
                        char_offsets.append(char_offset)
                        char_lengths.append(char_length)
                        ex_entities.append(self.ent_idx[cand_indices[cand_idx].detach().cpu().item()])
                        ex_md_scores.append(md_score.item())       
                        ex_el_scores.append(el_scores[cand_idx].item())     
                    cand_idx += 1


            # Debug
            #sample_token_ids = token_ids[example_idx]
            #mention_token_ids = sample_token_ids[ex_sp_offsets[example_idx] : ex_sp_offsets[example_idx] + ex_sp_lengths[example_idx]]
            #decoded_mention_tokens = [self.transform.processor.decode([token_id - 1]) for token_id in mention_token_ids.tolist()]
            #mention = text[char_offsets[example_idx] : char_offsets[example_idx] + char_lengths[example_idx]]
            #print(f"{sample_token_ids=}")
            #print(f"{list(zip(mention_token_ids.tolist(), decoded_mention_tokens))=}")
            #print(f"{mention=}")

            predictions.append(
                {
                    "offsets": char_offsets,
                    "lengths": char_lengths,
                    "entities": ex_entities,
                    "md_scores": ex_md_scores,
                    "el_scores": ex_el_scores,
                }
            )

        return predictions
    
    def process_disambiguation_batch(self, texts, mention_offsets, mention_lengths, entities):
        batch: Dict[str, Any] = {
            "texts": texts,
            "mention_offsets": mention_offsets,
            "mention_lengths": mention_lengths,
            "entities": entities,
        }
        model_inputs = self.transform(batch)

        token_ids = model_inputs["input_ids"].to(self.device)
        mention_offsets = model_inputs["mention_offsets"]
        mention_lengths = model_inputs["mention_lengths"]
        tokens_mapping = model_inputs["tokens_mapping"].to(self.device)
        sp_tokens_boundaries = model_inputs["sp_tokens_boundaries"].tolist()

        with torch.no_grad():
            _, last_layer = self.task.encoder(token_ids)
            text_encodings = last_layer
            text_encodings = self.task.project_encoder_op(text_encodings)

            mentions_repr = self.task.span_encoder(
                text_encodings, mention_offsets, mention_lengths
            )

            flat_mentions_repr = mentions_repr[mention_lengths != 0]
            # retrieve candidates top-1 ids and scores
            cand_scores, cand_indices = self.lookup(
                flat_mentions_repr.detach()
            )
            predictions = []
            cand_idx = 0
            example_idx = 0
            for offsets, lengths in zip(
                mention_offsets, mention_lengths,
            ):
                ex_sp_offsets = []
                ex_sp_lengths = []
                ex_entities = []
                ex_dis_scores = []
                for offset, length in zip(offsets, lengths):
                    if length != 0:
                        ex_sp_offsets.append(offset.detach().cpu().item())
                        ex_sp_lengths.append(length.detach().cpu().item())
                        ex_entities.append(self.ent_idx[cand_indices[cand_idx].detach().cpu().item()])
                        ex_dis_scores.append(cand_scores[cand_idx].detach().cpu().item())           
                        cand_idx += 1

                char_offsets, char_lengths = convert_sp_to_char_offsets(
                    texts[example_idx],
                    ex_sp_offsets,
                    ex_sp_lengths,
                    sp_tokens_boundaries[example_idx],
                )

                predictions.append({
                    "offsets": char_offsets,
                    "lengths": char_lengths,
                    "entities": ex_entities,
                    "scores": ex_dis_scores
                })
                example_idx+= 1

        return predictions
    
    def get_predictions(self, test_data, batch_size=256):
        all_predictions = []
        for batch_start in tqdm(range(0,len(test_data),batch_size)):
            batch = test_data[batch_start:batch_start+batch_size]
            texts = [example['original_text'] for example in batch]
            predictions = self.process_batch(texts)
            all_predictions.extend(predictions)
        return all_predictions
    
    def get_disambiguation_predictions(self, test_data, batch_size=256):
        all_predictions = []
        for batch_start in tqdm(range(0,len(test_data),batch_size)):
            batch = test_data[batch_start:batch_start+batch_size]
            texts = [example['original_text'] for example in batch]
            mention_offsets = [[offset for _,_,_,_,offset,_ in example['gt_entities']] for example in batch]
            mention_lengths = [[length for _,_,_,_,_,length in example['gt_entities']] for example in batch]
            entities = [[0 for _,_,_,_,_,_ in example['gt_entities']] for example in batch]

            predictions = self.process_disambiguation_batch(texts, mention_offsets, mention_lengths, entities)
            all_predictions.extend(predictions)
        return all_predictions

    @staticmethod
    def convert_data_and_predictions_to_samples(data, predictions, md_threshold, el_threshold) -> List[Sample]:
        samples = []
        for example, example_predictions in zip(data, predictions):

            ground_truth_entities = [
                Entity(entity_id=entity_id, offset=offset, length=length, text=example['original_text'])
                for _, _, entity_id, _, offset, length in example['gt_entities']
            ]
            predicted_entities = [
                Entity(entity_id=entity_id, offset=offset, length=length, md_score=md_score, el_score=el_score, text=example['original_text'])
                for offset, length, entity_id, md_score, el_score in zip(
                    example_predictions['offsets'],
                    example_predictions['lengths'],
                    example_predictions['entities'],
                    example_predictions['md_scores'],
                    example_predictions['el_scores'],
                )
            ]
            predicted_entities = [entity for entity in predicted_entities if entity.el_score > el_threshold and entity.md_score > md_threshold]
            sample = Sample(text=example['original_text'], ground_truth_entities=ground_truth_entities, predicted_entities=predicted_entities)
            samples.append(sample)
        return samples

    @staticmethod
    def compute_scores(data, predictions, md_threshold=0.2, el_threshold=0.05):
        tp, fp, support = 0, 0, 0
        tp_boe, fp_boe, support_boe = 0, 0, 0

        predictions_per_example = []
        samples = ModelEval.convert_data_and_predictions_to_samples(data, predictions, md_threshold, el_threshold)
        for sample in samples:
            predictions_per_example.append((len(sample.ground_truth_entities), len(sample.predicted_entities)))
            support += len(sample.ground_truth_entities)
            tp += len(sample.true_positives)
            fp += len(sample.false_positives)
            support_boe += len(sample.ground_truth_entity_ids)
            tp_boe += len(sample.true_positives_boe)
            fp_boe += len(sample.false_positives_boe)

        def safe_division(a, b):
            if b == 0:
                return 0
            else:
                return a / b

        def compute_f1_p_r(tp, fp, fn):
            precision = safe_division(tp, (tp + fp))
            recall = safe_division(tp, (tp + fn))
            f1 = safe_division(2 * tp, (2 * tp + fp + fn))
            return f1, precision, recall

        fn = support - tp
        fn_boe = support_boe - tp_boe
        return compute_f1_p_r(tp, fp, fn), compute_f1_p_r(tp_boe, fp_boe, fn_boe)
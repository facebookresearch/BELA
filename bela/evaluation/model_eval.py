import yaml
from hydra.experimental import compose, initialize_config_module
import hydra
import torch
from tqdm import tqdm
import json
import faiss
import logging

from typing import Optional, List, Dict, Any, Tuple


logger = logging.getLogger(__name__)


def load_file(path):
    all_data = []
    with open(path, 'rt') as fd:
        for line in tqdm(fd):
            data = json.loads(line)
            all_data.append(data)
    return all_data


def convert_sp_to_char_offsets(
    text: str,
    sp_offsets: List[int],
    sp_lengths: List[int],
    sp_tokens_boundaries: List[List[int]],
) -> Tuple[List[int], List[int]]:
    """
    Function convert sentecepiece offsets and lengths to character level
    offsets and lengths for a given `text`.
    """
    char_offsets: List[int] = []
    char_lengths: List[int] = []
    text_utf8_chars: List[str] = [char for char in text]

    for sp_offset, sp_length in zip(sp_offsets, sp_lengths):
        # sp_offsets include cls_token, while boundaries doesn't
        if sp_offset == 0:
            continue

        sp_offset = sp_offset - 1
        char_offset = sp_tokens_boundaries[sp_offset][0]
        char_end = sp_tokens_boundaries[sp_offset + sp_length - 1][1]

        # sp token boundaries include whitespaces, so remove them
        while text_utf8_chars[char_offset].isspace():
            char_offset += 1
            assert char_offset < len(text_utf8_chars)

        char_offsets.append(char_offset)
        char_lengths.append(char_end - char_offset)

    return char_offsets, char_lengths
    

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
        example_idx = 0
        for offsets, lengths, md_scores in zip(
            mention_offsets, mention_lengths, mentions_scores
        ):
            ex_sp_offsets = []
            ex_sp_lengths = []
            ex_entities = []
            ex_md_scores = []
            ex_el_scores = []
            for offset, length, md_score in zip(offsets, lengths, md_scores):
                if length != 0:
                    if md_score >= self.task.md_threshold:
                        ex_sp_offsets.append(offset.detach().cpu().item())
                        ex_sp_lengths.append(length.detach().cpu().item())
                        ex_entities.append(self.ent_idx[cand_indices[cand_idx].detach().cpu().item()])
                        ex_md_scores.append(md_score.item())       
                        ex_el_scores.append(el_scores[cand_idx].item())     
                    cand_idx += 1

            char_offsets, char_lengths = convert_sp_to_char_offsets(
                texts[example_idx],
                ex_sp_offsets,
                ex_sp_lengths,
                sp_tokens_boundaries[example_idx],
            )

            predictions.append(
                {
                    "offsets": char_offsets,
                    "lengths": char_lengths,
                    "entities": ex_entities,
                    "md_scores": ex_md_scores,
                    "el_scores": ex_el_scores,
                }
            )
            example_idx += 1

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
    def compute_scores(data, predictions, md_threshold=0.2, el_threshold=0.05):
        tp, fp, support = 0, 0, 0
        tp_boe, fp_boe, support_boe = 0, 0, 0

        predictions_per_example = []
        for example, example_predictions in zip(data, predictions):

            example_targets = {
                (offset,length):ent_id
                for _,_,ent_id,_,offset,length in example['gt_entities']
            }

            example_predictions = {
                (offset, length):ent_id
                for offset, length, ent_id, md_score, el_score in zip(
                    example_predictions['offsets'],
                    example_predictions['lengths'],
                    example_predictions['entities'],
                    example_predictions['md_scores'],
                    example_predictions['el_scores'],
                )
                if (el_score > el_threshold and md_score > md_threshold) 
            }

            predictions_per_example.append((len(example_targets), len(example_predictions)))

            for pos, ent in example_targets.items():
                support += 1
                if pos in example_predictions and example_predictions[pos] == ent:
                    tp += 1
            for pos, ent in example_predictions.items():
                if pos not in example_targets or example_targets[pos] != ent:
                    fp += 1

            example_targets_set = set(example_targets.values())
            example_predictions_set = set(example_predictions.values())

            for ent in example_targets_set:
                support_boe += 1
                if ent in example_predictions_set:
                    tp_boe += 1
            for ent in example_predictions_set:
                if ent not in example_targets_set:
                    fp_boe += 1

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
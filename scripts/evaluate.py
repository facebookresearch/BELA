import warnings
warnings.filterwarnings('ignore')

import yaml
from hydra.experimental import compose, initialize_config_module
import hydra
import torch
from tqdm import tqdm
import json
import faiss
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from bela.transforms.spm_transform import SPMTransform
from bela.evaluation.model_eval import ModelEval, load_file
from bela.utils.prediction_utils import get_predictions_using_windows, get_sp_transform


def evaluate_model_e2e(checkpoint_path, datasets, md_threshold = 0.2, el_threshold = 0.4, embeddings_path=None, ent_catalogue_idx_path=None):
    print(f"Loading model from checkpoint {checkpoint_path}")
    model_eval = ModelEval(
        checkpoint_path=checkpoint_path,
        config_name="joint_el_mel_new",
        embeddings_path=embeddings_path,
        ent_catalogue_idx_path=ent_catalogue_idx_path
    )

    model_eval.task.md_threshold = md_threshold
    model_eval.task.el_threshold = el_threshold

    for test_data_path in datasets:
        print(f"Processing {test_data_path}")
        test_data = load_file(test_data_path)
        
        for sample in test_data:
            if "data_example_id" in sample:
                sample["document_id"] = sample["data_example_id"]
        predictions = get_predictions_using_windows(model_eval, test_data, window_length=254, window_overlap=10, do_merge_predictions=True)
        (f1, precision, recall), (f1_boe, precision_boe, recall_boe) = ModelEval.compute_scores(test_data, predictions)
        
        print(f"F1 = {f1:.4f}, precision = {precision:.4f}, recall = {recall:.4f}")


def convert_examples_for_disambiguation(test_data, transform, skip_unknown_ent_ids=False, ent_idx=None):
    old_max_seq_len = transform.max_seq_len
    transform.max_seq_len = 10000
    
    new_examples = []
    max_mention_token_pos_in_text = 192
    skipped_ent_ids = 0

    for example in tqdm(test_data):
        text = example['original_text']
        outputs = transform(dict(texts=[text]))
        sp_token_boundaries = outputs['sp_tokens_boundaries'][0]

        for _ , _ , ent_id, _ , offset, length in example['gt_entities']:
            if skip_unknown_ent_ids and ent_idx is not None and ent_id not in ent_idx:
                skipped_ent_ids += 1
                continue
                
            token_pos = 0
            while token_pos < len(sp_token_boundaries) and offset >= sp_token_boundaries[token_pos][1]:
                token_pos += 1

            new_text = text
            new_offset = offset
            if token_pos > max_mention_token_pos_in_text:
                shift = sp_token_boundaries[token_pos-max_mention_token_pos_in_text][0].item()
                new_text = new_text[shift:]
                new_offset = new_offset - shift

            assert text[offset:offset+length] == new_text[new_offset:new_offset+length]

            new_example = {
                'original_text': new_text,
                'gt_entities': [[0,0,ent_id,'wiki',new_offset,length]],
            }
            new_examples.append(new_example)
    
    transform.max_seq_len = old_max_seq_len
    return new_examples, skipped_ent_ids


def metrics_disambiguation(test_data, predictions):
    support = 0
    correct = 0

    for example_idx, (example, prediction) in tqdm(enumerate(zip(test_data, predictions))):
        if len(prediction['entities']) == 0:
            continue
        target = example['gt_entities'][0][2]
        prediction = prediction['entities'][0]
        correct += (target == prediction)
        support += 1

    accuracy = correct/support

    return accuracy, support


def evaluate_model_dis(checkpoint_path, datasets, embeddings_path=None, ent_catalogue_idx_path=None):
    print(f"Loading model from checkpoint {checkpoint_path}")
    model_eval = ModelEval(
        checkpoint_path=checkpoint_path,
        config_name="joint_el_mel_new",
        embeddings_path=embeddings_path,
        ent_catalogue_idx_path=ent_catalogue_idx_path
    )

    for test_data_path in datasets:
        print(f"Processing {test_data_path}")
        test_data = load_file(test_data_path)
        
        test_data_for_disambgiation, skipped = convert_examples_for_disambiguation(test_data, model_eval.transform)
        predictions = model_eval.get_disambiguation_predictions(test_data_for_disambgiation)
        accuracy, support = metrics_disambiguation(test_data_for_disambgiation, predictions)
        print(f"Accuracty {accuracy}, support {support}, skipped {skipped}")


embeddings_path = "./models/embeddings.pt"
ent_catalogue_idx_path = "./models/index.txt"


print("End-to-end EL performance on Mewsli-9’-test (under-labeled for end-to-end linking)")
checkpoint_path = "./models/model_mewsli.ckpt"
datasets = [
    './data/mewsli-9-splitted/ar.jsonl_test',
    './data/mewsli-9-splitted/de.jsonl_test',
    './data/mewsli-9-splitted/en.jsonl_test',
    './data/mewsli-9-splitted/es.jsonl_test',
    './data/mewsli-9-splitted/fa.jsonl_test',
    './data/mewsli-9-splitted/ja.jsonl_test',
    './data/mewsli-9-splitted/sr.jsonl_test',
    './data/mewsli-9-splitted/ta.jsonl_test',
    './data/mewsli-9-splitted/tr.jsonl_test',
]
evaluate_model_e2e(
    checkpoint_path=checkpoint_path,
    datasets=datasets,
    embeddings_path=embeddings_path,
    ent_catalogue_idx_path=ent_catalogue_idx_path,
)


print("End-to-end EL performance on Mewsli-9’-test (labeled for end-to-end linking)")
checkpoint_path = "./models/model_e2e.ckpt"
datasets = [
    './data/mewsli-9-labelled/ar_labelled.jsonl',
    './data/mewsli-9-labelled/de_labelled.jsonl',
    './data/mewsli-9-labelled/en_labelled.jsonl',
    './data/mewsli-9-labelled/es_labelled.jsonl',
    './data/mewsli-9-labelled/fa_labelled.jsonl',
    './data/mewsli-9-labelled/ja_labelled.jsonl',
    './data/mewsli-9-labelled/sr_labelled.jsonl',
    './data/mewsli-9-labelled/ta_labelled.jsonl',
    './data/mewsli-9-labelled/tr_labelled.jsonl',
]
evaluate_model_e2e(
    checkpoint_path=checkpoint_path,
    datasets=datasets,
    embeddings_path=embeddings_path,
    ent_catalogue_idx_path=ent_catalogue_idx_path,
)


print("End-to-end results on AIDA")
checkpoint_path = "./models/model_aida.ckpt"
datasets = [
    './data/aida/aida_testb.jsonl_wikidata',
]
evaluate_model_e2e(
    checkpoint_path=checkpoint_path,
    datasets=datasets,
    embeddings_path=embeddings_path,
    ent_catalogue_idx_path=ent_catalogue_idx_path,
)


print("ED accuracy on Mewsli-9")
checkpoint_path = "./models/model_wiki.ckpt"
datasets = [
    './data/mewsli-9/ar.jsonl',
    './data/mewsli-9/de.jsonl',
    './data/mewsli-9/en.jsonl',
    './data/mewsli-9/es.jsonl',
    './data/mewsli-9/fa.jsonl',
    './data/mewsli-9/ja.jsonl',
    './data/mewsli-9/sr.jsonl',
    './data/mewsli-9/ta.jsonl',
    './data/mewsli-9/tr.jsonl',
]
evaluate_model_dis(
    checkpoint_path=checkpoint_path,
    datasets=datasets,
    embeddings_path=embeddings_path,
    ent_catalogue_idx_path=ent_catalogue_idx_path,
)
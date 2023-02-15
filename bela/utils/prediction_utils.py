from collections import defaultdict
from functools import lru_cache

from bela.evaluation.model_eval import ModelEval
from bela.transforms.spm_transform import SPMTransform


@lru_cache
def get_sp_transform():
    return SPMTransform(max_seq_len=100000)


def get_windows(text):
    sp_transform = get_sp_transform()
    tokens = sp_transform([text])[0]
    tokens = tokens[1:-1]
    window_length = 254
    windows = []
    for window_start in range(0, len(tokens), window_length // 2):
        start_pos = tokens[window_start][1]
        if window_start + window_length >= len(tokens):
            end_pos = tokens[-1][2]
        else:
            end_pos = tokens[window_start + window_length][2]
        windows.append((start_pos, end_pos))
    return windows


def convert_predictions_to_dict(example_predictions):
    if len(example_predictions) > 0:
        offsets, lengths, entities, md_scores, el_scores = zip(*example_predictions)
    else:
        offsets, lengths, entities, md_scores, el_scores = [], [], [], [], []
    return {
        "offsets": offsets,
        "lengths": lengths,
        "entities": entities,
        "md_scores": md_scores,
        "el_scores": el_scores,
    }


def group_predictions_by_example(all_predictions, extended_examples):
    grouped_predictions = defaultdict(list)
    for prediction, extended_example in zip(all_predictions, extended_examples):
        window_start = extended_example["window_start"]
        prediction = dict(prediction)
        prediction["offsets"] = [
            offset + window_start for offset in prediction["offsets"]
        ]
        grouped_predictions[extended_example["document_id"]].append((prediction))

    predictions = {}
    for document_id, example_prediction_list in grouped_predictions.items():
        example_predictions = []
        for prediction in example_prediction_list:
            for offset, length, ent, md_score, el_score in zip(
                prediction["offsets"],
                prediction["lengths"],
                prediction["entities"],
                prediction["md_scores"],
                prediction["el_scores"],
            ):
                example_predictions.append((offset, length, ent, md_score, el_score))
                example_predictions = sorted(example_predictions)
        predictions[document_id] = example_predictions

    return predictions


def merge_predictions(example_predictions):
    filtered_example_predictions = []

    current_end = None
    current_offset = None
    current_length = None
    current_ent_id = None
    current_md_score = None
    current_el_score = None

    for offset, length, ent_id, md_score, el_score in example_predictions:
        if current_end is None:
            current_end = offset + length
            current_offset = offset
            current_length = length
            current_ent_id = ent_id
            current_md_score = md_score
            current_el_score = el_score
            continue

        if offset < current_end:
            # intersection of two predictions
            if md_score > current_md_score:
                current_ent_id = ent_id
                current_offset = offset
                current_length = length
                current_md_score = md_score
                current_el_score = el_score
        else:
            filtered_example_predictions.append(
                (
                    current_offset,
                    current_length,
                    current_ent_id,
                    current_md_score,
                    current_el_score,
                )
            )
            current_ent_id = ent_id
            current_offset = offset
            current_length = length
            current_md_score = md_score
            current_el_score = el_score

        current_end = offset + length

    if current_offset is not None:
        filtered_example_predictions.append(
            (
                current_offset,
                current_length,
                current_ent_id,
                current_md_score,
                current_el_score,
            )
        )

    return filtered_example_predictions


def correct_mention_offsets(example_predictions, text):
    # TODO: Won't that break mentions that end with a separator? Double check in evaluations datasets
    TEXT_SEPARATORS = [" ", ".", ",", "!", "?", "-", "\n"]
    corrected_example_predictions = []
    for offset, length, ent_id, md_score, el_score in example_predictions:
        while (
            offset != 0
            and offset < len(text)
            and (
                # Break when the previous character is a separator and the current one is not
                text[offset - 1] not in TEXT_SEPARATORS
                or text[offset] in TEXT_SEPARATORS
            )
        ):
            offset += 1
            length -= 1
        while (
            offset + length < len(text)
            # Break as soon as the last character is a separator
            and text[offset + length] not in TEXT_SEPARATORS
        ):
            length += 1
        corrected_example_predictions.append(
            (offset, length, ent_id, md_score, el_score)
        )
    return corrected_example_predictions


def get_predictions_using_windows(model_eval: ModelEval, test_data, batch_size=1024):
    extended_examples = []

    for example in test_data:
        text = example["original_text"]
        windows = get_windows(text)
        for idx, (start_pos, end_pos) in enumerate(windows):
            new_text = text[start_pos:end_pos]
            extended_examples.append(
                {
                    "document_id": example["document_id"],
                    "original_text": new_text,
                    "gt_entities": example["gt_entities"],
                    "window_idx": idx,
                    "window_start": start_pos,
                    "window_end": end_pos,
                }
            )

    all_predictions = model_eval.get_predictions(
        extended_examples, batch_size=batch_size
    )
    predictions_dict = group_predictions_by_example(all_predictions, extended_examples)

    predictions = []
    for example in test_data:
        document_id = example["document_id"]
        text = example["original_text"]
        example_predictions = predictions_dict[document_id]
        example_predictions = merge_predictions(example_predictions)
        example_predictions = correct_mention_offsets(example_predictions, text)
        example_predictions = convert_predictions_to_dict(example_predictions)
        predictions.append(example_predictions)

    return predictions

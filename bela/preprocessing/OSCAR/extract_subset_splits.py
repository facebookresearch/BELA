import argparse
import gzip
import json
import glob
from newsplease import NewsPlease
import tqdm
import os
from transformers import AutoTokenizer

from bela.utils.utils_preprocess import split_paragraph_max_seq_length


def filter_data(base_path, name):
    with open(base_path + "/subset/" + name + ".json") as f:
        idx_dict = json.load(f)

    # collect text
    num_articles = 0
    for idx in tqdm.tqdm(idx_dict):
        file_path = base_path + "/splits/en_part_" + str(idx) + ".txt.gz"
        j = 0
        idx_dict[idx][j]['text'] = ''
        current_offset = idx_dict[idx][j]['offset']
        current_nb_sentences = idx_dict[idx][j]['nb_sentences']
        if os.path.exists(file_path):
            with gzip.open(file_path, 'rb') as f:
                for i, line in enumerate(f):
                    if i in range(current_offset, current_offset + current_nb_sentences):
                        text = line.decode('UTF-8').strip()
                        if 'image caption' not in text:
                            if idx_dict[idx][j]['text']=='':
                                text = text.strip()
                            idx_dict[idx][j]['text'] += text
                            idx_dict[idx][j]['text'] += "."
                    if i == current_offset + current_nb_sentences:
                        if j >= len(idx_dict[idx]) - 1:
                            break
                        else:
                            j += 1
                            num_articles += 1
                            idx_dict[idx][j]['text'] = ''
                            current_offset = idx_dict[idx][j]['offset']
                            current_nb_sentences = idx_dict[idx][j]['nb_sentences']

    with open(base_path + "/subset/" + name + "_filled.json", "w") as f:
        json.dump(idx_dict, f)

    return idx_dict


def prep4labeling(data_dict, base_path, name):
    f_out = open(base_path + "/subset/" + name + '_4labeling.jsonl', 'w')
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    for idx in data_dict:
        for article in data_dict[idx]:
            if 'text' in article and len(article['text'])!=0:
                split_paragraph_max_seq_length(article['text'], f_out, tokenizer, idx)
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
    )
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.base_path + "/subset/" + args.name + "_filled.json"):
        idx_dict = filter_data(args.base_path, args.name)
    else:
        with open(args.base_path + "/subset/" + args.name + "_filled.json") as f:
            idx_dict = json.load(f)
    prep4labeling(idx_dict, args.base_path, args.name)
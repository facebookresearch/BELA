import argparse
import gzip
import json
import glob
import tqdm
import logging
import os
from transformers import AutoTokenizer
import concurrent.futures

from bela.utils.utils_preprocess import split_paragraph_max_seq_length

logger = logging.getLogger(__name__)


def filter_data(idx_dict, idx, base_path):
    # collect text
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
                        idx_dict[idx][j]['text'] += text
                        idx_dict[idx][j]['text'] += " "
                if i == current_offset + current_nb_sentences:
                    if j >= len(idx_dict[idx]) - 1:
                        break
                    else:
                        j += 1
                        idx_dict[idx][j]['text'] = ''
                        current_offset = idx_dict[idx][j]['offset']
                        current_nb_sentences = idx_dict[idx][j]['nb_sentences']

                
def prep4labeling(data_dict, base_path, name):
    f_out = open(base_path + "/processed/" + name + '_4labeling.jsonl', 'w')
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    for idx in data_dict:
        num_article = 0
        for article in data_dict[idx]:
            if 'text' in article and len(article['text'])!=0:
                identifier = str(idx) + "_" + str(num_article)
                split_paragraph_max_seq_length(article, f_out, tokenizer, identifier)
                num_article += 1
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="Name of subset",
    )
    parser.add_argument(
        "--base_path",
        type=str,
    )
    args, _ = parser.parse_known_args()
    dict_path = args.base_path + "/processed/" + args.name
    if not os.path.exists(dict_path + "_filled.json"):
        logger.info("Collect data from OSCAR dumps")
        with open(dict_path + ".json") as f:
            idx_dict = json.load(f)
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            {executor.submit(filter_data, idx_dict, idx, args.base_path): idx for idx in idx_dict}
        with open(dict_path + "_filled.json", "w") as f:
            json.dump(idx_dict, f)
        logger.info(dict_path + "_filled.json")

    else:
        with open(dict_path + "_filled.json") as f:
            idx_dict = json.load(f)

    prep4labeling(idx_dict, args.base_path, args.name)

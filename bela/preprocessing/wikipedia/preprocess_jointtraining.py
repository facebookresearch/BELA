import argparse
import nltk
import tqdm
from nltk.tokenize import word_tokenize
import pickle
import json
import random
import os
import glob

nltk.download('punkt')
random.seed(10)

def write_out(data, f_out):
    paragraph = data['text']
    diff = 0
    for entity in data['entities']:
        entity['offset'] += diff

        start = int(entity['offset'])
        end = int(entity['offset']) + int(entity['length'])
        if end < len(paragraph):
            if paragraph[end] == "-":
                paragraph = paragraph[:end] + " - " + paragraph[end + 1:]
                diff += 2
        if start!=0:
            try:
                if paragraph[start - 1] == "-":
                    paragraph = paragraph[:start - 1] + " - " + paragraph[start:]
                    diff += 2
                    entity['offset'] += 2
            except:
                pass

    paragraph_tokenized = []
    gt_entities = []
    pre = 0
    for entity in data['entities']:
        start = int(entity['offset'])
        end = int(entity['offset']) + int(entity['length'])
        pre_paragraph = word_tokenize(paragraph[pre:start])
        entity_tokenized = word_tokenize(paragraph[start:end])
        pre = end
        paragraph_tokenized.extend(pre_paragraph)
        gt_entities.append([len(paragraph_tokenized), len(entity_tokenized), entity['entity_id'], "OSCAR"])
        paragraph_tokenized.extend(entity_tokenized)

    post_paragraph = word_tokenize(paragraph[pre:])
    paragraph_tokenized.extend(post_paragraph)
    template = {
        "data_example_id": data["id"],
        "text": paragraph_tokenized,
        "gt_entities": gt_entities}
    f_out.write(json.dumps(template))
    f_out.write("\n")


def process_wiki_based_data(base_path):
    with open(base_path + "t1_newsplit/processed/joint_all.jsonl", 'w') as f_out:
        dataset_splits = glob.glob(base_path + "t1_newsplit/processed/*")
        for dataset_split in dataset_splits:
            with open(dataset_split) as f:
                for line in f:
                    line = json.loads(line)
                    line["id"] = "wiki" + "_" + line["id"]
                    write_out(line, f_out)


def split_data_t1(base_path, num_jointrain=20000, num_jointval=5000):
    with open(base_path + "t1_newsplit/processed/joint_all.jsonl") as f, \
            open(base_path + 't1_newsplit/jointtrain.jsonl', 'w') as f_jointtrain, \
            open(base_path + 't1_newsplit/jointdev.jsonl', 'w') as f_jointvalid, \
            open(base_path + 't1_newsplit/test.jsonl', 'w') as f_test:
        num_instances = sum(1 for _ in f)
        f.seek(0)

        num_test = num_instances-num_jointrain-num_jointval

        p_jointtrain = num_jointrain / (num_instances -(num_test/2))

        p_jointval = 1 - (num_jointval / (num_instances -(num_test/2)))

        n_jointrain = 0
        n_joinval = 0
        ids_train = []
        ids_dev = []
        for line in f:
            line_ = json.loads(line)
            r = random.random()
            if n_jointrain < num_jointrain and r < p_jointtrain:
                f_jointtrain.write(line)
                ids_train.append(line_["data_example_id"])
                n_jointrain += 1
            elif n_joinval < num_jointrain and r > p_jointval:
                f_jointvalid.write(line)
                ids_dev.append(line_["data_example_id"])
                n_joinval += 1
            else:
                f_test.write(line)
    return ids_train, ids_dev


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
    )
    parser.add_argument(
        "--base_path",
        type=str,
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
    )
    parser.add_argument(
        "--base_wikipedia",
        type=str,
    )
    parser.add_argument(
        "--time_split",
        type=str,
    )

    args, _ = parser.parse_known_args()
    base_dataset = "_".join(args.datasets.split(','))
    if not os.path.exists(args.base_path + 't1/' + base_dataset + "_matcha_jointtrain.jsonl"):
        print("Preprocess t1")
        ids_train, ids_dev = split_data_t1(args.base_path, args.datasets, args.time_split)
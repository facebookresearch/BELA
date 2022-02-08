import argparse
import copy
import json
import logging
import nltk
from nltk.tokenize import word_tokenize
import os
import pickle
import random
import tqdm

from bela.utils.utils_preprocess import sort_time

nltk.download('punkt')
random.seed(10)

logger = logging.getLogger()

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
        "text_raw": data["text_raw"],
        "gt_entities": gt_entities,
        "time_stamp": data["time_stamp"],
        "entities_raw": data["entities_raw"]}
    f_out.write(json.dumps(template))
    f_out.write("\n")


def process_OSCAR_based_data(base_path, base_datasets):

    f_out = open(base_path + "_".join(base_datasets.split(',')) + ".jsonl", "w")
    for base_dataset in base_datasets.split(','):
        with open(base_path + base_dataset + "_4labeling.jsonl_processed", "rb") as f:
            with open(base_path + base_dataset + "_4labeling.jsonl", "rb") as g:
                for line, line_raw in zip(f, g):
                    line_raw = json.loads(line_raw)
                    line = json.loads(line)
                    line["id"] = base_dataset + "_" + line["id"]
                    line["text_raw"] = line_raw["text"]
                    line["entities_raw"] = copy.deepcopy(line["entities"])
                    write_out(line, f_out)
    f_out.close()


def split_data_t2(base_path, base_datasets, time_split, ids_t1_train_dev, num_train=20000, num_val=5000):

    year_ref, month_ref = time_split.split("_")
    month_ref = int(month_ref)
    year_ref = int(year_ref)

    '''with open(base_wikidata + "en_title2wikidataID.pkl", "rb") as f:
        title2wikidataID = pickle.load(f)

    novel_entities = {}
    with open(base_wikipedia + "enwiki-20210701-post-kilt.kilt_format.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            if line["wikipedia_title"] in title2wikidataID:
                novel_entities[line["wikipedia_title"]] = 0'''

    base_dataset = "_".join(base_datasets.split(','))
    with open(base_path + base_dataset + ".jsonl") as f, \
            open(base_path + 't2/' + base_dataset + "_train.jsonl", 'w') as f_train, \
            open(base_path + 't2/' + base_dataset + "_dev.jsonl", 'w') as f_valid, \
            open(base_path + 't2/' + base_dataset + "_test.jsonl", 'w') as f_test:
        
        idcs_t2 = []
        for line in f:
            line = json.loads(line)
            year, month = line['time_stamp'].split("_")
            year = int(year)
            month = int(month)
            if year>year_ref or (year==year_ref and month>=month_ref):
                idcs_t2.append(line["data_example_id"])
        f.seek(0)

        idcs_train = idcs_t2 + ids_t1_train_dev
        random.shuffle(idcs_train)

        start = 0
        idcs_train = idcs_train[start:num_train]
        start+=num_train
        idcs_dev = idcs_train[start:start+num_val]
        start+=num_val

        data_test = {}
        for line in f:
            line_ = json.loads(line)
            idx = line_["data_example_id"]
            if idx in idcs_train:
                f_train.write(line)
            elif idx in idcs_dev:
                f_valid.write(os.linesep)
            elif idx not in ids_t1_train_dev:
                time = line_['time_stamp']
                if time in data_test:
                    data_test[time].append(line)
                else:
                    data_test[time] = [line]

        # writeout sorted by time
        for time in sorted(data_test):
            for line in data_test[time]:
                f_test.write(line)
        
def split_data_t1(base_path, base_datasets, time_split, num_train=20000, num_val=5000):

    year_ref, month_ref = time_split.split("_")
    month_ref = int(month_ref)
    year_ref = int(year_ref)

    base_dataset = "_".join(base_datasets.split(','))
    with open(base_path + base_dataset + ".jsonl") as f, \
            open(base_path + 't1/' + base_dataset + "_train.jsonl", 'w') as f_train, \
            open(base_path + 't1/' + base_dataset + "_dev.jsonl", 'w') as f_valid, \
            open(base_path + 't1/' + base_dataset + "_test.jsonl", 'w') as f_test:
        idcs_t1 = []
        for line in f:
            line = json.loads(line)
            year, month = line['time_stamp'].split("_")
            year = int(year)
            month = int(month)
            if year<year_ref or (year==year_ref and month<month_ref):
                idcs_t1.append(line["data_example_id"])

        f.seek(0)

        random.shuffle(idcs_t1)

        start = 0
        idcs_train = idcs_t1[start:num_train]
        start+=num_train
        idcs_dev = idcs_t1[start:start+num_val]
        start+=num_val
        idcs_test = idcs_t1[start:]

        logger.info("Number of training samples %d", len(idcs_train))
        logger.info("Number of dev samples %d", len(idcs_dev))
        logger.info("Number of test samples %d", len(idcs_test))

        data_test = {}
        for line in f:
            line_ = json.loads(line)
            idx = line_["data_example_id"]
            if idx in idcs_train:
                f_train.write(line)
            elif idx in idcs_dev:
                f_valid.write(line)
            elif idx in idcs_test:
                time = line_['time_stamp']
                if time in data_test:
                    data_test[time].append(line)
                else:
                    data_test[time] = [line]
        # writeout sorted by time
        for time in sorted(data_test):
            for line in data_test[time]:
                f_test.write(line)
                

    return idcs_train, idcs_dev

def collect_ids(base_path, base_datasets):
    ids_train, ids_dev = [], []
    base_dataset = "_".join(base_datasets.split(','))
    with open(base_path + 't1/' + base_dataset + "_train.jsonl") as f:
        for line in f:
            line = json.loads(line)
            ids_train.append(line['data_example_id'])

    with open(base_path + 't1/' + base_dataset + "_dev.jsonl") as f:
        for line in f:
            line = json.loads(line)
            ids_dev.append(line['data_example_id'])
    
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
        "--time_split",
        type=str,
    )

    args, _ = parser.parse_known_args()
    base_dataset = "_".join(args.datasets.split(','))

    dataset_path = args.base_path + base_dataset + ".jsonl"
    if not os.path.exists(dataset_path):
        logger.info('Collect subsets')
        process_OSCAR_based_data(args.base_path, args.datasets)

    if not os.path.exists(args.base_path + 't1/' + base_dataset + "_train.jsonl"):
        logger.info("Preprocess t1")
        if not os.path.isdir(args.base_path + 't1/'):
            os.mkdir(args.base_path + 't1/')
        ids_train, ids_dev = split_data_t1(args.base_path, args.datasets, args.time_split)
    else:
        logger.info("Collect ids t1")   
        ids_train, ids_dev = collect_ids(args.base_path, args.datasets)
    
    if not os.path.exists(args.base_path + 't2/' + base_dataset + "_train.jsonl"): 
        logger.info("Preprocess t2")   
        if not os.path.isdir(args.base_path + 't2/'):
            os.mkdir(args.base_path + 't2/')
        idcs_train_dev = ids_train + ids_dev
        split_data_t2(args.base_path, args.datasets, args.time_split, idcs_train_dev)
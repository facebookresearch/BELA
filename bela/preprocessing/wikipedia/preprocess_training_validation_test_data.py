import argparse
import nltk
import tqdm
from nltk.tokenize import word_tokenize
import pickle
import json
import random

nltk.download('punkt')


def write_out(entities, paragraph, data_example_id, f_out):
    paragraph = paragraph.strip()
    diff = 0
    for entity in entities:
        entity['start'] += diff
        entity['end'] += diff

        start = int(entity['start'])
        end = int(entity['end'])
        if end < len(paragraph):
            if paragraph[end] == "-":
                paragraph = paragraph[:end] + " - " + paragraph[end + 1:]
                diff += 2
        if start!=0:
            try:
                if paragraph[start - 1] == "-":
                    paragraph = paragraph[:start - 1] + " - " + paragraph[start:]
                    diff += 2
                    entity['start'] += diff
                    entity['end'] += diff
            except:
                print(start, end, diff, entities, entity)
                print("error", paragraph)

    paragraph_tokenized = []
    gt_entities = []
    pre = 0
    for entity in entities:
        start = int(entity['start'])
        end = int(entity['end'])
        pre_paragraph = word_tokenize(paragraph[pre:start])
        entity_tokenized = word_tokenize(paragraph[start:end])
        pre = end
        paragraph_tokenized.extend(pre_paragraph)
        gt_entities.append([len(paragraph_tokenized), len(entity_tokenized), entity['text'], "wiki"])
        paragraph_tokenized.extend(entity_tokenized)

    post_paragraph = word_tokenize(paragraph[pre:])
    paragraph_tokenized.extend(post_paragraph)

    template = {
        "data_example_id": data_example_id,
        "text": paragraph_tokenized,
        "gt_entities": gt_entities}
    f_out.write(json.dumps(template))
    f_out.write("\n")


def process_wiki_based_data(base_dataset, lang):

    with open(base_dataset + "/" + lang + "/" + lang + "wiki0.pkl", "rb") as f:
        data = pickle.load(f)
    f_out = open(base_dataset + "/" + lang + "_matcha_.jsonl", "w")
    data_example_id = 0
    for d in tqdm.tqdm(data):
        if len(data[d]['anchors']) > 0:
            paragraph_id = data[d]['anchors'][0]['paragraph_id']
            entities = []
            for anchor in data[d]['anchors']:
                if anchor['wikidata_src'] == 'wikipedia':
                    paragraph_id_current = anchor['paragraph_id']
                    if paragraph_id_current == paragraph_id:
                        entities.append(anchor)
                    else:
                        if paragraph_id > 1 and len(entities) > 0:
                            if len(data[d]['paragraphs'][paragraph_id])>0:
                                write_out(entities, data[d]['paragraphs'][paragraph_id], data_example_id, f_out)
                                data_example_id += 1
                            else:
                                print("toot short", data[d]['paragraphs'][paragraph_id])
                        paragraph_id = anchor['paragraph_id']
                        entities = [anchor]
    f_out.close()


def split_data(base_dataset, lang, num_train=17000000, num_val=5000):

    with open(base_dataset + "/" + lang + "_matcha.jsonl") as f, \
            open(base_dataset + "/" + lang + "_matcha_train.jsonl", 'w') as f_train, \
            open(base_dataset + "/" + lang + "_matcha_dev.jsonl", 'w') as f_valid, \
            open(base_dataset + "/" + lang + "_matcha_test.jsonl", 'w') as f_test:
        num_instances = sum(1 for _ in f)
        f.seek(0)
        percentage = num_train/num_instances
        percentage_val = num_val/num_instances

        i = 0
        j = 0
        for line in f:
            r = random.random()
            if i < num_train and r < percentage:
                f_train.write(line)
                i += 1
            if j < num_val and r < percentage_val:
                f_valid.write(line)
                j += 1
            else:
                f_test.write(line)

def process_oscar_based_data():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--base_dataset",
        type=str,
    )
    parser.add_argument(
        "--lang",
        type=str,
    )

    args, _ = parser.parse_known_args()
    if args.data_type == "wiki":
        process_wiki_based_data(args.base_dataset, args.lang)

    split_data(args.base_dataset, args.lang)

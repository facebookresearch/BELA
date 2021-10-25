import argparse
import nltk
import tqdm
from nltk.tokenize import word_tokenize
import pickle
import json
import random

nltk.download('punkt')

def write_out_for_labeling(paragraph, data_example_id, f_out):
    paragraph = paragraph.strip()
    output = {'text': paragraph, 'id': data_example_id}
    f_out.write(json.dumps(output))
    f_out.write("\n")

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
                    entity['start'] += 2
                    entity['end'] += 2
            except:
                pass

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
    p = paragraph_tokenized
    template = {
        "data_example_id": data_example_id,
        "text": paragraph_tokenized,
        "gt_entities": gt_entities}
    f_out.write(json.dumps(template))
    f_out.write("\n")


def process_wiki_based_data(base_dataset, lang):

    with open(base_dataset + "/" + lang + "/" + lang + "wiki0.pkl", "rb") as f:
        data = pickle.load(f)
    f_out = open(base_dataset + "/" + lang + "_matcha.jsonl", "w")
    f_out_l = open(base_dataset + "/" + lang + "_internal.jsonl", "w")
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
                            keep = True
                            paragraph = data[d]['paragraphs'][paragraph_id]
                            if len(data[d]['paragraphs'][paragraph_id])>10:
                                for ent in entities:
                                    start_id = ent['start']
                                    end_id = ent['end']
                                    if start_id >=len(paragraph)-1 or end_id > len(paragraph)-1 or start_id==end_id:
                                        keep = False
                                if keep:
                                    write_out_for_labeling(data[d]['paragraphs'][paragraph_id], data_example_id, f_out_l)
                                    # write_out(entities, data[d]['paragraphs'][paragraph_id], data_example_id, f_out)
                                    data_example_id += 1
                        paragraph_id = anchor['paragraph_id']
                        entities = [anchor]
    f_out.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--training_type",
        type=str,
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
    #if args.data_type == "wiki":
    process_wiki_based_data(args.base_dataset, args.lang)
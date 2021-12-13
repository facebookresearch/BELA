import argparse
import json
import matplotlib.pyplot as plt
import pickle
import tqdm


def stats(base_dataset, base_wikidata, base_wikipedia):

    with open(base_wikidata + "en_title2wikidataID.pkl", "rb") as f:
        title2wikidataID = pickle.load(f)

    novel_entities = {}
    known_entities = {}
    out_of_wikidata = 0
    with open(base_wikipedia + "enwiki-20210701-post-kilt.kilt_format.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            if line["wikipedia_title"] in title2wikidataID:
                novel_entities[line["wikipedia_title"]] = 0
            else:
                out_of_wikidata += 1
    unknown_ents = set()
    f_out = open('/fsx/kassner/OSCAR/subset/cnn_bbc_novel_blink_format.jsonl', 'w')
    f_out_ = open('/fsx/kassner/OSCAR/subset/cnn_bbc_known_blink_format.jsonl', 'w')
    with open('/fsx/kassner/OSCAR/subset/cnn_4labeling.jsonl_processed') as f:
        for line in f:
            line = json.loads(line)
            output = {"context_left": '', "mention": '', "context_right": '',"mention": '', "query_id": "", "label_id": ""}
            line['id'] = "cnn_" + line['id']
            output["query_id"] = line['id']
            for ent in line["entities"]:
                if ent["entity_id"] in novel_entities:
                    unknown_ents.add(ent["entity_id"])
                    novel_entities[ent["entity_id"]] += 1
                    ent['novel'] = True
                    mention = line['text'][ent['offset']:ent['offset']+ent['length']]
                    context_left = line['text'][:ent['offset']]
                    context_right = line['text'][ent['offset']+ent['length']:]
                    output["context_left"] = context_left
                    output["mention"] = mention
                    output["context_right"] = context_right
                    output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                    f_out.write(json.dumps(output))
                    f_out.write('\n')
                else:
                    try:
                        mention = line['text'][ent['offset']:ent['offset']+ent['length']]
                        context_left = line['text'][:ent['offset']]
                        context_right = line['text'][ent['offset']+ent['length']:]
                        output["context_left"] = context_left
                        output["mention"] = mention
                        output["context_right"] = context_right
                        output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                        f_out_.write(json.dumps(output))
                        f_out_.write('\n')
                    except:
                        pass


    with open('/fsx/kassner/OSCAR/subset/bbc_4labeling.jsonl_processed') as f:
        for line in f:
            line = json.loads(line)
            output = {"context_left": '', "mention": '', "context_right": '',"mention": '', "query_id": "", "label_id": ""}
            line['id'] = "bbc_" + line['id']
            output["query_id"] = line['id']
            for ent in line["entities"]:
                if ent["entity_id"] in novel_entities:
                    unknown_ents.add(ent["entity_id"])
                    novel_entities[ent["entity_id"]] += 1
                    ent['novel'] = True
                    mention = line['text'][ent['offset']:ent['offset']+ent['length']]
                    context_left = line['text'][:ent['offset']]
                    context_right = line['text'][ent['offset']+ent['length']:]
                    output["context_left"] = context_left
                    output["mention"] = mention
                    output["context_right"] = context_right
                    output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                    f_out.write(json.dumps(output))
                    f_out.write('\n')
                else:
                    try:
                        mention = line['text'][ent['offset']:ent['offset']+ent['length']]
                        context_left = line['text'][:ent['offset']]
                        context_right = line['text'][ent['offset']+ent['length']:]
                        output["context_left"] = context_left
                        output["mention"] = mention
                        output["context_right"] = context_right
                        output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                        f_out_.write(json.dumps(output))
                        f_out_.write('\n')
                    except:
                        pass

    f_out.close()
    f_out_.close()
    print(len(unknown_ents))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dataset",
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
    args, _ = parser.parse_known_args()

    stats(args.base_dataset, args.base_wikidata, args.base_wikipedia)

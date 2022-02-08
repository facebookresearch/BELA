import argparse
import json
import matplotlib.pyplot as plt
import pickle
import random
import tqdm


'''def convert2blink_old(title2wikidataID, f_out, dataset_path):
    with open(dataset_path) as f:
        for line in f:
            line = json.loads(line)
            output = {"context_left": '', "mention": '', "context_right": '',"mention": '', "query_id": "", "label_id": ""}
            line['id'] = subset + "_" + line['id']
            output["query_id"] = line['id']
            for ent in line["entities"]:
                mention = line['text'][ent['offset']:ent['offset']+ent['length']]
                context_left = line['text'][:ent['offset']]
                context_right = line['text'][ent['offset'] + ent['length']:]
                output["context_left"] = context_left
                output["mention"] = mention
                output["context_right"] = context_right
                if ent['entity_id'] in title2wikidataID:
                    output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                    f_out.write(json.dumps(output))
                    f_out.write('\n')'''

def convert2blink(title2wikidataID, f_out, dataset_path):
    with open(dataset_path) as f:
        print(dataset_path)
        for line in f:
            line = json.loads(line)
            for j, ent in enumerate(line["entities_raw"]):
                output = {}
                line['id'] = line['data_example_id']
                output["query_id"] = line['data_example_id'] + "_" + str(j)
                mention = line['text_raw'][ent['offset']:ent['offset']+ent['length']]
                context_left = line['text_raw'][:ent['offset']]
                context_right = line['text_raw'][ent['offset'] + ent['length']:]
                output["context_left"] = context_left
                output["mention"] = mention
                output["context_right"] = context_right

                #if ent['entity_id'] in title2wikidataID:
                p = random.random()
                if p<=0.2:
                    output["label_id"] = ent['entity_id']
                    f_out.write(json.dumps(output))
                    f_out.write('\n')

'''def convert2arboEL(title2wikidataID, f_out, dataset_path):
    with open(dataset_path) as f:
        for line in f:
            line = json.loads(line)
            output = {}
            line['context_doc_id'] = line['data_example_id']
            for ent in line["entities_raw"]:
                mention = line['text_raw'][ent['offset']:ent['offset']+ent['length']]
                context_left = line['text_raw'][:ent['offset']]
                context_right = line['text_raw'][ent['offset'] + ent['length']:]
                output["context_left"] = context_left
                output["mention"] = mention
                output["context_right"] = context_right

                output["mention_id"] = ent['entity_id']
                output["type"] = 
                output["label"] = 
                output["label_title"] = ent['entity_id']
                if ent['entity_id'] in title2wikidataID:
                    p = random.random()
                    if p<=0.2:
                        output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                        f_out.write(json.dumps(output))
                        f_out.write('\n')'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_wikidata",
        type=str,
    )
    parser.add_argument(
        "--base_path",
        type=str,
    )
    parser.add_argument(
        "--datasets",
        type=str,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    args, _ = parser.parse_known_args()


    with open(args.base_wikidata + "en_title2wikidataID.pkl", "rb") as f:
        title2wikidataID = pickle.load(f)

    if args.dataset_path is not None:
        datasets = [args.base_path + args.dataset_path]
        output_name = args.dataset_path.split(".")[0].split("_")[-1]
        output_path = args.output_path + output_name  + ".jsonl"
    else:
        datasets = args.datasets.split(',')
        output_path = args.output_path + "_".join(datasets) + ".jsonl"
        for i, subset in enumerate(datasets):
            datasets[i] = args.base_path + subset + '_4labeling.jsonl_processed'

    with open(output_path, "w") as f_out:
        for subset in datasets:
            convert2blink(title2wikidataID, f_out, subset)
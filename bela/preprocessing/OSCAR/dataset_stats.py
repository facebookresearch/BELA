import argparse
import json
import matplotlib.pyplot as plt
import pickle
import tqdm


def plot_histogram(count_dict, title, output_path):
    plt.xlabel('Frequency')
    plt.ylabel('Counts')
    plt.title(title)

    plt.hist(list(count_dict.values()), max(list(count_dict.values())))
    plt.savefig(output_path + title + ".png")


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
    #f_out = ('/fsx/kassner/OSCAR/subset/cnn_bbc_novel.jsonl', 'w')
    with open('/fsx/kassner/OSCAR/subset/cnn_4labeling.jsonl_processed') as f:
        for line in f:
            line = json.loads(line)
            line['id'] = "cnn_" + line['id']
            for ent in line["entities"]:
                if ent["entity_id"] in novel_entities:
                    novel_entities[ent["entity_id"]] += 1
                    ent['novel'] = True
                else:
                    if ent["entity_id"] in known_entities:
                        known_entities[ent["entity_id"]] += 1
                    else:
                        known_entities[ent["entity_id"]] = 1
                    ent['novel'] = False
            #f_out.write(json.dumps(line))
            #f_out.write('\n')


    with open('/fsx/kassner/OSCAR/subset/bbc_4labeling.jsonl_processed') as f:
        for line in f:
            line = json.loads(line)
            line['id'] = "bbc_" + line['id']
            for ent in line["entities"]:
                if ent["entity_id"] in novel_entities:
                    novel_entities[ent["entity_id"]] += 1
                    ent['novel'] = True
                else:
                    if ent["entity_id"] in known_entities:
                        known_entities[ent["entity_id"]] += 1
                    else:
                        known_entities[ent["entity_id"]] = 1
                    ent['novel'] = False
            #f_out.write(json.dumps(line))
            #f_out.write('\n')
    
    #f_out.close()

    plot_histogram(novel_entities, "novel entities", base_dataset)
    plot_histogram(known_entities, "known entities", base_dataset)


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

import argparse
import json
import matplotlib.pyplot as plt
import pickle
import tqdm


def plot_histogram(count_dict, title, output_path):
    plt.xlabel('Frequency')
    plt.ylabel('Counts')
    plt.title(title)

    plt.hist(list(count_dict.values()), 1)
    plt.savefig(output_path + title + ".png")


def stats(base_dataset, lang):
    with open("/fsx/kassner/wikidata/en_title2wikidataID.pkl", "rb") as f:
        title2wikidataID = pickle.load(f)

    novel_entities = {}
    known_entities = {}
    with open("/data/wikipedia/enwiki-20210701-post-kilt.kilt_format.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            novel_entities[title2wikidataID[line["wikipedia_title"]]] = 0

    with open(base_dataset + "/" + lang + "/" + lang + "wiki0.pkl", "rb") as f:
        data = pickle.load(f)

    for d in tqdm.tqdm(data):
        for anchor in data[d]['anchors']:
            novel = False
            for wiki_id in anchor['wikidata_ids']:

                if wiki_id in novel_entities:
                    novel = True
                    break
            if novel:
                novel_entities[wiki_id] += 1
            else:
                if wiki_id in known_entities:
                    known_entities[wiki_id] += 1
                else:
                    known_entities[wiki_id] = 1

    plot_histogram(novel_entities, "novel entities", base_dataset)
    plot_histogram(known_entities, "known entities", base_dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dataset",
        type=str,
    )
    parser.add_argument(
        "--lang",
        type=str,
    )
    args, _ = parser.parse_known_args()

    stats(args.base_dataset, args.lang)

import argparse
import json
import pickle


def collect_novel_entities(base_blink, base_wikidata, base_wikipedia):
    with open(base_wikidata + "en_title2wikidataID.pkl", "rb") as f:
        title2wikidataID = pickle.load(f)

    with open(base_blink + "novel_entities.jsonl", 'w') as f_out:
        with open(base_wikipedia + "enwiki-20210701-post-kilt.kilt_format.jsonl", "r") as f:
            for line in f:
                line = json.loads(line)
                if line["wikipedia_title"] in title2wikidataID:
                    output = {}
                    output['title'] = line["wikipedia_title"]
                    output['text'] = '.'.join(line["text"].split('.')[0:10])
                    output['entity'] = line["wikipedia_title"]
                    output['idx'] = line["wikipedia_id"]
                    f_out.write(json.dump(output))
                    f_out.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_blink",
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
    collect_novel_entities(args.base_blink, args.base_wikidata, args.base_wikipedia)

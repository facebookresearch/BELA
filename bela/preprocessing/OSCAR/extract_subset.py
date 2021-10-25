import argparse
import gzip
import json
import glob
from newsplease import NewsPlease
import tqdm
import os


def filter_data(base_path, url, name):
    filenames = glob.glob(base_path + "/metadata/en_meta_part_*_.jsonl")
    idx_dict = {}

    for filename in tqdm.tqdm(filenames):
        idx = filename.split("/")[-1].split("_")[-2]
        idx_dict[idx] = []
        with open(filename, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')
                line = json.loads(line)
                if url in line['uri']:
                    try:
                        article = NewsPlease.from_url(line['uri'])
                        current_year = article.date_publish.year
                        current_month = article.date_publish.month
                        line["timestamp"] = str(current_year) + '_' + str(current_month)
                        idx_dict[idx].append(line)
                        """if line["timestamp"]:
                            if current_year > year:
                                idx_dict[idx].append(line)
                            elif current_year >= year and current_month >= month:
                                idx_dict[idx].append(line)
                            else:
                                idx_dict[idx].append(line)"""
                    except:
                        pass

    with open(base_path + "/subset/" + name + ".json", "w") as f:
        json.dump(idx_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
    )
    args, _ = parser.parse_known_args()
    filter_data(args.base_path, args.url, args.name)
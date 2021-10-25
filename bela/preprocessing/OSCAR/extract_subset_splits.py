import argparse
import gzip
import json
import glob
from newsplease import NewsPlease
import tqdm
import os


def filter_data(base_path, name):
    with open(base_path + "/subset/" + name + ".json") as f:
        idx_dict = json.load(f)

    # collect text
    num_articles = 0
    for idx in tqdm.tqdm(idx_dict):
        file_path = base_path + "/splits/en_part_" + str(idx) + ".txt.gz"
        j = 0
        idx_dict[idx][j]['text'] = ''
        current_offset = idx_dict[idx]["old"][j]['offset']
        current_nb_sentences = idx_dict[idx]["old"][j]['nb_sentences']
        if os.path.exists(file_path):
            with gzip.open(file_path, 'rb') as f:
                for i, line in enumerate(f):
                    if i in range(current_offset, current_offset + current_nb_sentences):
                        idx_dict[idx][j]['text'] += " "
                        text = line.decode('UTF-8').strip()
                        if 'image caption' not in text:
                            idx_dict[idx][j]['text'] += text
                    if i == current_offset + current_nb_sentences:
                        if j >= len(idx_dict[idx]) - 1:
                            break
                        else:
                            j += 1
                            num_articles += 1
                            idx_dict[idx][j]['text'] = ''
                            current_offset = idx_dict[idx][j]['offset']
                            current_nb_sentences = idx_dict[idx][j]['nb_sentences']
            print(num_articles)
    with open(base_path + "/subset/" + name + "_filled.json", "w") as f:
        json.dump(idx_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    filter_data(args.base_path, args.name)
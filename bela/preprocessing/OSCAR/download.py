import argparse
from datasets import load_dataset
import json
from newsplease import NewsPlease
import os
import tqdm

URLS = {"bbc": "/www.bbc.com/",
        "cnn": "https://www.cnn.com/",
        "dw": "https://www.dw.com/en/",
        "reuters": "https://www.reuters.com/article/",
        "ngeo": "https://www.nationalgeographic.com/",
        "guardian": "https://www.theguardian.com/"}

def downnload_dataset(cache_dir):
    os.mkdir(cache_dir+ 'processed/')
    dataset = load_dataset("oscar-corpus/OSCAR-2109", "deduplicated_en", use_auth_token=True, cache_dir=cache_dir)

def filter(cache_dir, name):
    dataset = load_dataset("oscar-corpus/OSCAR-2109", "deduplicated_en", use_auth_token=True, cache_dir=cache_dir, streaming=True, split='train')
    url = URLS[name]
    with open(cache_dir + "processed/" + name + ".jsonl", "w") as f:
        for d in tqdm.tqdm(dataset):
            d_uri = d['meta']['headers']['warc-target-uri']
            if url in d['meta']['headers']['warc-target-uri']:
                d_id = d["id"]
                if 'content-type' in d['meta']:
                    if 'text/plain'==d['meta']['content-type']:
                        try:
                            article = NewsPlease.from_url(d_uri)
                            current_year = article.date_publish.year
                            current_month = article.date_publish.month
                            time_stamp = str(current_year) + '_' + str(current_month)
                            output = {}
                            output["id"] = d_id
                            output["time_stamp"] = time_stamp
                            output["text"] = d["text"]
                            f.write(json.dump(output))
                            f.write("\n")
                        except:
                            pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Folder to save ",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Folder to save ",
    )

    args, _ = parser.parse_known_args()
    if not os.path.exists(args.cache_dir):
        downnload_dataset(args.cache_dir)
    if not os.path.exists(args.cache_dir + "processed/" + args.name + ".jsonl"):
        filter(args.cache_dir, args.name)
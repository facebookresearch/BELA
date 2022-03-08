import argparse
from datasets import load_dataset
import json
from newsplease import NewsPlease
import os
import tqdm

URLS = {"bbc": "https://www.bbc.com/",
        "cnn": "https://www.cnn.com/",
        "dw": "https://www.dw.com/en/",
        "reuters": "https://www.reuters.com/article/",
        "ngeo": "https://www.nationalgeographic.com/",
        "guardian": "https://www.theguardian.com/",
        "ap": "https://apnews.com/article/", 
        "scientist": "https://www.the-scientist.com/",
        "scientificamerican": "https://www.scientificamerican.com/"}

def downnload_dataset(cache_dir):
    os.mkdir(cache_dir+ 'processed/')
    dataset = load_dataset("oscar-corpus/OSCAR-2109", "deduplicated_en", use_auth_token=True, cache_dir=cache_dir)

def filter(cache_dir, shard_idx, num_shards=1000, total=498789052):
    print(shard_idx)
    # 498789052
    #dataset = load_dataset("oscar-corpus/OSCAR-2109", "deduplicated_en", use_auth_token=True, cache_dir=cache_dir, streaming=True, split='train')
    dataset = load_dataset("oscar-corpus/OSCAR-2109", "deduplicated_en", use_auth_token=True, cache_dir=cache_dir, streaming=True, split='train')
    #dataset_shard = dataset.shard(num_shards=num_shards, index=shard_idx)
    start_idx = int(total/num_shards)*shard_idx
    end_idx = int(total/num_shards)*(shard_idx+1)
    with open(cache_dir + "processed/news_" + str(shard_idx) + ".jsonl", "a") as f:
        for i, d in enumerate(tqdm.tqdm(dataset)):
            if i<start_idx:
                continue
            if i>=end_idx:
                break
            d_uri = d['meta']['headers']['warc-target-uri']
            process = False
            for url_key in URLS:
                url = URLS[url_key]
                if url in d['meta']['headers']['warc-target-uri']:
                    process =True
                    name = url_key
            if process:
                d_id = d["id"]
                if 'content-type' in d['meta']['headers']:
                    if 'text/plain'==d['meta']['headers']['content-type']:
                        try:
                            article = NewsPlease.from_url(d_uri)
                            current_year = article.date_publish.year
                            current_month = article.date_publish.month
                            time_stamp = str(current_year) + '_' + str(current_month)
                            output = {}
                            output["id"] = d_id
                            output["time_stamp"] = time_stamp
                            output["text"] = d["text"]
                            output["data_source_name"] = name
                            output["url"] = d['meta']['headers']['warc-target-uri']
                            f.write(json.dumps(output))
                            f.write("\n")
                        except:
                            pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Folder to save ",
        default="../OSCAR2109/"
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
    )

    args, _ = parser.parse_known_args()
    if not os.path.exists(args.cache_dir):
        downnload_dataset(args.cache_dir)
    if not os.path.exists(args.cache_dir + "processed/news_" + str(args.shard_idx) + ".jsonl"):
        filter(args.cache_dir, args.shard_idx)
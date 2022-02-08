# Preprocessing OSCAR data

URLS
bbc_news https://www.bbc.com/news/
bbc_sport https://www.bbc.com/sports/
bbc_future https://www.bbc.com/future/
bbc_travel https://www.bbc.com/travel/
bbc_culture https://www.bbc.com/culture
cnn https://www.cnn.com/
dw https://www.dw.com/en/
reuters https://www.reuters.com/article/

1. Download OSCAR data from:
Please login with your huggingface credentials as OSCAR 21.09 is behind a [request access feature](https://huggingface.co/docs/transformers/model_sharing#preparation) on HuggingFace side:
```huggingface-cli login```

Tho download the script and set --cache_dir to the location where the data should be stored. At this point, the full dataset is XX large and filtered to XX in later processing steps:
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/download.py --cache_dir data/```

2. Process OSCAR data
   1. Filter OSCAR
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/extract_subset.py --url /www.bbc.com/news/ --name bbc_news --base_path /fsx/kassner/OSCAR```
   2. Process for Labeling
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/process4labeling.py --name bbc_sport --base_path /fsx/kassner/OSCAR```
   3. Train, dev, test split
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/split_dataset.py --datasets cnn,dw,reuters,bbc_news,bbc_sport,bbc_future,bbc_travel,bbc_culture --base_path /fsx/kassner/OSCAR/processed/ --time_split 2019_8```

Temporary:
```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/convert_blink_format.py --base_wikidata /fsx/kassner/wikidata/ --base_path /fsx/kassner/OSCAR/processed/ --datasets cnn,dw,reuters,bbc_news,bbc_sport,bbc_future,bbc_travel,bbc_culture --output_path /fsx/kassner/data_BLINK/```
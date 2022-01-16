# Preprocessing OSCAR data

URLS
bbc_news https://www.bbc.com/news/
bbc_sports https://www.bbc.com/sports/
bbc_future https://www.bbc.com/future/
bbc_travel https://www.bbc.com/travel/
cnn https://www.cnn.com/
dw https://www.dw.com/en/
reuters https://www.reuters.com/article/

1. Download OSCAR data
2. Process OSCAR data
   1. Filter OSCAR
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/extract_subset.py --url /www.bbc.com/news/ --name bbc_news --base_path /fsx/kassner/OSCAR```
   2. Process for Labeling
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/process4labeling.py --name bbc --base_path /fsx/kassner/OSCAR```
   3. Train, dev, test split
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/train_dev_test_split.py --datasets cnn,dw,reuters,bbc_news,bbc_sports,bbc_future,bbc_travel --base_path /fsx/kassner/OSCAR/processed/ --base_wikidata /fsx/kassner/wikidata/ --base_wikipedia /fsx/kassner/wikipedia/t2/ --time_split 2019_9```

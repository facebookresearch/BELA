# Preprocessing OSCAR dta
used urls:
/www.bbc.com/news/
https://www.cnn.com/


1. Download OSCAR data
2. Process OSCAR data
   1. Filter OSCAR
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/extract_subset.py --url /www.bbc.com/news/ --name bbc --base_path /fsx/kassner/OSCAR```
   2. Prep for labeling
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/process4labeling.py --name bbc --base_path /fsx/kassner/OSCAR```
   3. Train, dev, test split
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/train_dev_test_split.py --datasets cnn,bbc --base_path /fsx/kassner/OSCAR/subset/ --base_wikidata /fsx/kassner/wikidata/ --base_wikipedia /fsx/kassner/wikipedia/t2/ --time_split 2019_9```

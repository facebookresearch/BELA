# Preprocessing OSCAR dta

1. Download OSCAR data
2. Process OSCAR data
   1. Filter OSCAR
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/extract_subset.py --url /www.bbc.com/news/ --name bbc --base_path /fsx/kassner/OSCAR```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/extract_subset.py --url https://www.cnn.com/ --name cnn --base_path /fsx/kassner/OSCAR```
   2.
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/extract_subset_splits.py --name bbc --base_path /fsx/kassner/OSCAR```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/OSCAR/extract_subset_splits.py --name cnn --base_path /fsx/kassner/OSCAR```
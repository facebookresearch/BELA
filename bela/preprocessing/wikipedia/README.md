# Preprocessing Wikipedia and Wikinews data

1. Download data and models

```bash download.sh```

2. Process wikipedia data
   1. WikiExtractor with timestamps
   
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/WikiExtractor_timestamp.py data/wikinews/enwikinews-20210901-pages-articles-multistream.xml -o data/wikinews/en -l```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/WikiExtractor_timestamp.py data/wikipedia/enwiki-pages-articles.xml.bz2 -o data/wikipedia/t1/en -l```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/WikiExtractor_timestamp.py  data/wikipedia/enwiki-20220301-pages-articles-multistream.xml.bz2 -o data/wikipedia/t2/en -l```
   
   2. Compress wikidata, generate useful dictionary's, e.g., title -> ID, ID -> title or alias tables:
   
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_wikidata.py --base_wikidata data/wikidata/ compress```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_wikidata.py --base_wikidata data/wikidata/ dicts```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_wikidata.py --base_wikidata data/wikidata/ redirects```
   
   3. Process wikinews

   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_extract.py  --lang en --base_wikipedia data/wikinews/```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikinews/ --base_wikidata /fsx/kassner/wikidata/ prepare```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikinews/ --base_wikidata /fsx/kassner/wikidata/ solve```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikinews/ --base_wikidata /fsx/kassner/wikidata/ fill```
   
   4. Process wikipedia t1
   
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_extract.py  --lang en --base_wikipedia data/wikipedia/t1/```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/t1/ --base_wikidata /fsx/kassner/wikidata/ prepare```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/t1/ --base_wikidata /fsx/kassner/wikidata/ solve```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/t1/ --base_wikidata /fsx/kassner/wikidata/ fill```
   
   5. Process wikipedia t2
   
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_extract.py  --lang en --base_wikipedia data/wikipedia/t2/```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/t2/ --base_wikidata /fsx/kassner/wikidata/ prepare```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/t2/ --base_wikidata /fsx/kassner/wikidata/ solve```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/t2/ --base_wikidata /fsx/kassner/wikidata/ fill```
   
   6. Prepare pretraining data t1
   
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/split_data.py```
   
   7. Prepare pretraining data t2
   
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/split_data.py --t2 True```
   
   7. Prepare novel entity descriptions
   
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/entities_t2.py```


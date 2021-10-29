# Preprocessing Wikipedia data
# TO DO: write shell scripts for these processes
1. Download wikipedia data
2. Process wikipedia data
   1. WikiExtractor with timestamps
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/WikiExtractor_timestamp.py data/wikinews/enwikinews-20210901-pages-articles-multistream.xml -o data/wikinews/en -l```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/WikiExtractor_timestamp.py /fsx/kassner/wikipedia/enwiki-pages-articles.xml.bz2 -o /fsx/kassner/wikipedia/en -l```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/WikiExtractor_timestamp.py  data/wikipedia/enwiki-20210920-pages-articles-multistream.xml -o data/wikipedia/en -l```
   2. Compress wikidata, generate useful dictionary's, e.g., title -> ID, ID -> title or alias tables:
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_wikidata.py --base_wikidata /fsx/kassner/wikidata/ compress```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_wikidata.py --base_wikidata /fsx/kassner/wikidata/ dicts```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_wikidata.py --base_wikidata /fsx/kassner/wikidata/ redirects```
   3. Process wikinews
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_extract.py  --lang en --base_wikipedia data/wikinews/```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikinews/ --base_wikidata /fsx/kassner/wikidata/ prepare```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikinews/ --base_wikidata /fsx/kassner/wikidata/ solve```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikinews/ --base_wikidata /fsx/kassner/wikidata/ fill```
   4. Process wikipedia t1
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_extract.py  --lang en --base_wikipedia /fsx/kassner/wikipedia/```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia /fsx/kassner/wikipedia/ --base_wikidata /fsx/kassner/wikidata/ prepare```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia /fsx/kassner/wikipedia/ --base_wikidata /fsx/kassner/wikidata/ solve```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia /fsx/kassner/wikipedia/ --base_wikidata /fsx/kassner/wikidata/ fill```
   5. Process wikipedia t2
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_extract.py  --lang en --base_wikipedia data/wikipedia/```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/ --base_wikidata /fsx/kassner/wikidata/ prepare```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/ --base_wikidata /fsx/kassner/wikidata/ solve```
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_anchors.py --lang en --base_wikipedia data/wikipedia/ --base_wikidata /fsx/kassner/wikidata/ fill```
   6. Prepare pretraining data
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_training_validation_test_data.py --lang en --base_dataset /fsx/kassner/wikipedia/ --data_type wiki --training_type pretraining```
   7. Prepare novel entity descriptions
   ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/extract_descriptions_novel_entities.py  --base_wikipedia data/wikipedia/ --base_wikidata /fsx/kassner/wikidata/ --base_blink data/blink/```

3. Download BLINK data 
   1. FAISS index
   2. BLINK model


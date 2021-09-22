# Preprocessing steps

1. Download wikipedia data:
   1. Wikidata
   2. Wikinews
      1. Process wikipedia data
         1. WikiExtractor with timestamps:
         ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/WikiExtractor_timestamp.py data/wikipedia/wikinews/enwikinews-20210901-pages-articles-multistream.xml -o en -l```
         2. Compress wikidata, generate useful dictionary's, e.g., title -> ID, ID -> title or alias tables:
         ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_wikidata.py --base_wikidata data/wikipedia compress```
         3. Contruct dictonray from wikinews
         ```PYTHONPATH=.:$PYTHONPATH python bela/preprocessing/wikipedia/preprocess_extract.py  --lang en --base_wikipedia bela/preprocessing/wikipedia/wikinews/```


2. Download BLINK data 
   1. FAISS index
   2. BLINK model
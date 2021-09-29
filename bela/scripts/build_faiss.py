import torch
import faiss

emb_file = '/data/home/kassner/BELA/data/blink/all_entities_large.t7'
ent_embeddings = torch.load(emb_file)

d = ent_embeddings.shape[1] # 768
buffer_size = 50000
store_n = 128
ef_search = 256
ef_construction = 200

index = faiss.IndexHNSWFlat(d, store_n, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efSearch = ef_search
index.hnsw.efConstruction = ef_construction

index.train(ent_embeddings.numpy())
index.add(ent_embeddings.numpy())

faiss.write_index(index, '/data/home/kassner/BELA/data/blink/index_large.faiss')

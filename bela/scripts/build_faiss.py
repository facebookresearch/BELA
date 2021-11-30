import argparse
import torch
import faiss
import numpy as np
import random

import logging
logger = logging.getLogger(__name__)

def build_index(base_path, embedding_file1, output_name, selected_subset="select", embedding_file2=None, filter_fraction=None):
    emb_file = base_path + embedding_file1 + '.t7'
    ent_embeddings = torch.load(emb_file)
    if filter_fraction is not None:
        keep_idcs = random.sample(range(0, len(ent_embeddings)), int(len(ent_embeddings)/filter_fraction))
        keep_idcs = torch.tensor(keep_idcs)
        ent_embeddings = torch.index_select(ent_embeddings, 0, keep_idcs)
        torch.save(keep_idcs, base_path + "filter_idcs_" + str(filter_fraction) + ".t7")
    '''if selected_subset is not None:
        keep_idcs = []
        subset = set()
        with open("/data/home/kassner/BELA/data/blink/entities_10_2018.txt") as f:
            for ent in f:
                subset.add(ent)
        with open("/data/home/kassner/BELA/data/blink/en_bert_ent_idx.txt") as f:
            for i, ent in enumerate(f):
                if ent in subset:
                    keep_idcs.append(i)
        ent_embeddings = torch.index_select(ent_embeddings, 0, keep_idcs)
        logger.info("Number of entities", len(ent_embeddings))'''


    if embedding_file2 is not None:
        emb_file2 = base_path + embedding_file2 + '.t7'
        ent_embeddings2 = torch.load(emb_file2)
        ent_embeddings = torch.cat((ent_embeddings, ent_embeddings2))

    logger.info("loaded embeddings")

    d = ent_embeddings.shape[1]
    buffer_size = 500000
    store_n = 128
    ef_search = 512
    ef_construction = 200

    index = faiss.IndexHNSWFlat(d, store_n, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.hnsw.efConstruction = ef_construction

    index.train(ent_embeddings.numpy())
    logger.info("index trained")

    num_indexed = 0
    n = len(ent_embeddings)
    for i in range(0, n, buffer_size):
        vectors = ent_embeddings[i: i + buffer_size]
        index.add(np.array(vectors))
        num_indexed += buffer_size
        logger.info("data indexed %d", num_indexed)

    faiss.write_index(index, base_path + output_name + '.faiss')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/data/home/kassner/BELA/data/blink/")
    parser.add_argument('--embedding_file1', type=str, default="all_entities_large")
    parser.add_argument('--output_name', type=str, default="faiss_index_t1")
    parser.add_argument('--embedding_file2', type=str, default=None)
    parser.add_argument('--filter_fraction', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    build_index(args.base_path, args.embedding_file1, args.output_name, args.embedding_file2, args.filter_fraction)

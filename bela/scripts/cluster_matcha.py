"""
Greedy nearest neighbor clustering for dense embeddings.
Adapted from https://github.com/rloganiv/streaming-cdc
"""
import argparse
import csv
import logging
import pickle
import numpy as np
import torch
import json
import glob
from bela.datamodule.joint_el_datamodule import EntityCatalogue
from bela.utils.cluster import Grinch
from sklearn import metrics

logger = logging.getLogger(__name__)


class LinkingStrategy:
    def __init__(self, n, limit, threshold, device):
        self._i = 0
        self._n = n
        self._limit = limit
        self._threshold = threshold
        self._device = device

    def __call__(self, row):
        raise NotImplementedError


class Backwards(LinkingStrategy):
    def __init__(self, n, limit, threshold, device):
        super().__init__(n, limit, threshold, device)

    def __call__(self, row):
        row = row.clone().detach()
        if self._limit is not None:
            start = max(0, self._i - self._limit)
        else:
            start = 0
        mask = torch.zeros_like(row, dtype=torch.bool)
        mask[start:self._i+1] = True
        row[~mask] = -1e32
        self._i += 1
        return row > self._threshold


class Diversity(LinkingStrategy):
    def __init__(self, n, limit, threshold, device):
        super().__init__(n, limit, threshold, device)
        self._mask = torch.zeros(n, dtype=torch.bool, device=device)

    def __call__(self, row):
        row = row.clone().detach()
        self._mask[self._i] = True
        row[~self._mask] = -1e32
        # If limit is reached. remove most similar entry to current.
        if self._mask.sum() == self._limit:
            removal_index = torch.argmax(row[:self._i])
            self._mask[removal_index] = False
        self._i += 1
        return row > self._threshold


class Cache(LinkingStrategy):
    def __init__(self, n, limit, threshold, device):
        super().__init__(n, limit, threshold, device)
        self._mask = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_seen = torch.zeros(n, dtype=torch.int64, device=device)
    
    def __call__(self, row):
        row = row.clone().detach()
        self._mask[self._i] = True
        row[~self._mask] = -1e32
        out = row > self._threshold
        self._last_seen[out] = self._i
        if self._mask.sum() == self._limit:
            removal_index = torch.argmin(self._last_seen[:self._i])
            self._mask[removal_index] = False
            self._last_seen[removal_index] = 1e13
        self._i += 1
        return out


class DiversityCache(LinkingStrategy):
    def __init__(self, n, limit, threshold, device):
        super().__init__(n, limit, threshold, device)
        self._mask = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_seen = torch.zeros(n, dtype=torch.int64, device=device)

    def __call__(self, row):
        row = row.clone().detach()
        self._mask[self._i] = True
        row[~self._mask] = -1e32
        out = row > self._threshold
        self._last_seen[out] = self._i
        if self._mask.sum() == self._limit:
            if out[:self._i].any():
                removal_index = torch.argmax(row[:self._i])
            else:
                removal_index = torch.argmin(self._last_seen[:self._i])
            self._last_seen[removal_index] = 1e13
            self._mask[removal_index] = False
        self._i += 1
        return out


LINKING_STRATEGIES = {
    'backwards': Backwards,
    'diversity': Diversity,
    'cache': Cache,
    'diversity-cache': DiversityCache,
}


def score(embeddings):
    logger.info('Scoring')
    with torch.no_grad():
        scores = torch.mm(embeddings, embeddings.transpose(0, 1))
        return scores
    
def find_threshold_grinch(grinch, target, max_iters=100):
    logger.info(f'Finding threshold. Target # of clusts: {target}.')
    bounds = [0.0, 1.0]
    n_clusters = -1
    epsilon = grinch.points.shape[0] / 1000.0
    logger.info(f'Epsilon: {epsilon}')
    i = 0
    while abs(n_clusters - target) > epsilon and i < max_iters:
        i += 1
        threshold = (bounds[0] + bounds[1]) / 2
        clusters = grinch.flat_clustering(threshold)
        n_clusters = len(np.unique(clusters))
        logger.info(f'Threshold: {threshold}, # of clusts: {n_clusters}')
        if n_clusters < target:
            bounds[0] = threshold
        else:
            bounds[1] = threshold
    return clusters

def find_threshold(scores, linking_strategy, target, entity_ids, max_iters=100):
    logger.info(f'Finding threshold. Target # of clusts: {target}.')
    bounds = [0.0, 1.0]
    n_clusters = -1
    epsilon = scores.shape[0] / 1000.0
    logger.info(f'Epsilon: {epsilon}')
    i = 0
    while abs(n_clusters - target) > epsilon:
        threshold = (bounds[0] + bounds[1]) / 2
        linking_strategy._threshold = threshold
        clusters = cluster(scores, linking_strategy)
        n_clusters = len(np.unique(clusters))
        '''labels_true = []
        labels_pred = []
        f1_best = 0
        for ent, cluster in zip(entity_ids, clusters):
            labels_true.append(ent)
            labels_pred.append(cluster)
        h = metrics.homogeneity_score(labels_true, labels_pred)
        c  = metrics.completeness_score(labels_true, labels_pred)
        f1 = metrics.f1_score(labels_true, labels_pred, average='macro')
        '''

        logger.info(f'Threshold: {threshold}, # of clusts: {n_clusters}')
        if n_clusters < target:
            bounds[0] = threshold
        else:
            bounds[1] = threshold
    return clusters


def cluster(scores, linking_strategy):

    # Back fill adjacency.
    adjacency_matrix = torch.zeros_like(scores, dtype=torch.bool)
    n = scores.shape[0]
    for i, row in enumerate(scores):
        with torch.no_grad():
            adjacency_matrix[i] = linking_strategy(row)

    # Transpose adjacency to propagate cluster ids forward.
    clusters = torch.arange(n, device=scores.device)
    for i, row in enumerate(adjacency_matrix.transpose(0, 1)):
        clusters[row] = clusters[i].clone()

    return clusters.cpu().numpy()

def load_embeddings(embeddings_path_list, filter_type, idcs_filter, max_mentions=None):
    embedding_idx = 0
    logger.info('Loading embeddings')
    entity_vocab = set()
    entity_ids = []
    embeddings = []
    num_mentions = 0
    for embedding_path in sorted(embeddings_path_list):
        embeddings_buffer = torch.load(embedding_path, map_location='cpu')
        for embedding_batch in embeddings_buffer:
            for embedding in embedding_batch:
                entity, embedding = embedding[0], embedding[1:]
                embedding_idx +=1
                entity = int(float(entity))
                if filter_type=="entities":
                    if entity in idcs_filter:
                        continue
                if filter_type=="idcs":
                    if embedding_idx not in idcs_filter:
                        continue
                num_mentions +=1
                embedding = [float(x) for x in embedding]
                embeddings.append(embedding)
                entity_vocab.add(entity)
                entity_ids.append(entity)
                if max_mentions is not None:
                    if num_mentions>=max_mentions:
                        return embeddings, entity_vocab, entity_ids
    return embeddings, entity_vocab, entity_ids

def append_embeddings(entity_vocab, entity_ids, embeddings, embeddings_path_list, filter_type, idcs_filter, max_mentions=None):
    embedding_idx = 0
    logger.info('Adding embeddings')
    num_mentions = len(embeddings)
    for embedding_path in sorted(embeddings_path_list):
        embeddings_buffer = torch.load(embedding_path, map_location='cpu')
        for embedding_batch in embeddings_buffer:
            for embedding in embedding_batch:
                entity, embedding = embedding[0], embedding[1:]
                embedding_idx +=1
                entity = int(float(entity))
                if filter_type=="entities":
                    if entity in idcs_filter:
                        continue
                if filter_type=="idcs":
                    if embedding_idx not in idcs_filter:
                        continue
                num_mentions +=1
                embedding = [float(x) for x in embedding]
                embeddings.append(embedding)
                entity_vocab.add(entity)
                entity_ids.append(entity)
                if max_mentions is not None:
                    if num_mentions>=max_mentions:
                        return embeddings, entity_vocab, entity_ids
    return embeddings, entity_vocab, entity_ids

def select_filter_idcs(filter_type, dataset_path, ent_catalogue_idx_path, \
                        timesplit=None, year_ref=2019, month_ref=9):
    idcs_filter = set()
    if filter_type=="idcs" and timesplit is not None: 
        with open(dataset_path + ".jsonl") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                year, month = line["time_stamp"].split("_")
                year, month = int(year), int(month)
                if timesplit=="t2": 
                    if year>year_ref or (year==year_ref and month>month_ref):
                        for _ in range(len(line["gt_entities"])):
                            idcs_filter.add(i)
                elif timesplit=="t1":
                    if year<year_ref or (year==year_ref and month<=month_ref):
                        idcs_filter.add(i)
    if filter_type=="entities":
        '''with open(wikidata_base_path + "en_title2wikidataID.pkl", "rb") as f:
            title2wikidataID = pickle.load(f)
        with open(wikipedia_base_path + "t2/enwiki-20210701-post-kilt.kilt_format.jsonl", "r") as f:
            for line in f:
                line = json.loads(line)
                if line["wikipedia_title"] in title2wikidataID:
                    idcs_filter.add(line["wikipedia_title"])'''
        ent_catalogue = EntityCatalogue(ent_catalogue_idx_path, None, reverse=True)
        idcs_filter = ent_catalogue.idx_referse.keys()
    return idcs_filter

def main(args):
    if args.type=="novel":
        filter_type="entities"
    if args.type=="t1" or args.type=="t2":
        filter_type="idcs"
    logging.info("Filter type: %s", filter_type)
    idcs_filter = select_filter_idcs(filter_type, args.dataset_path, args.ent_catalogue_idx_path, timesplit=args.type)

    input_path = args.input + '*.t7'
    embeddings_path_list = glob.glob(input_path)
    embeddings, entity_vocab, entity_ids = load_embeddings(embeddings_path_list, filter_type, idcs_filter, args.max_mentions)
    if len(embeddings)<=args.max_mentions:
        idcs_filter = select_filter_idcs("idcs", args.dataset_path, args.ent_catalogue_idx_path, timesplit="t2")
        embeddings, entity_vocab, entity_ids = append_embeddings(entity_vocab, entity_ids, embeddings, embeddings_path_list, filter_type, idcs_filter, args.max_mentions)
    logging.info("Number of mentions: %s", len(embeddings))
    if args.cluster_type=="greedy":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        if not args.dot_prod:
            embeddings /= torch.norm(embeddings, dim=-1, keepdim=True)
        entity_ids = torch.tensor(entity_ids, dtype=torch.int64, device=device)


        linking_strategy = LINKING_STRATEGIES[args.strategy](
            n=entity_ids.size(0),
            limit=args.limit,
            threshold=args.threshold,
            device=device,
        )

        scores = score(embeddings)
        if args.threshold is not None:
            clusters = cluster(scores, linking_strategy)
        else:
            target = len(entity_vocab)
            clusters = find_threshold(scores, linking_strategy, target, entity_ids)

        clusters = clusters.tolist()
    
    elif args.cluster_type=="grinch":
        embeddings = np.array(embeddings, dtype=np.float32)

        grinch = Grinch(points=embeddings, active_leaf_limit=args.limit, pruning_strategy=args.strategy)
        grinch.build_dendrogram()

        if args.threshold is not None:
            clusters = grinch.flat_clustering(args.threshold)
        else:
            target = len(entity_vocab)
            clusters = find_threshold_grinch(grinch, target)

    output_name = args.input
    output_name = args.input.split('/')[-3].split('.')[0]
    with open(args.output + output_name + "_" + args.cluster_type + "_" + args.type + ".txt", 'w') as g:
        for t, p in zip(entity_ids, clusters):
            g.write('%i, %i\n' % (t, p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cluster_type', type=str, default="greedy")
    parser.add_argument('--dataset_path', type=str, default='/fsx/kassner/OSCAR/subset/cnn_bbc_matcha')
    parser.add_argument('--novel_entity_idx_path', type=str, default='/data/home/kassner/BELA/data/blink/novel_entities_filtered.jsonl')
    parser.add_argument('--ent_catalogue_idx_path', type=str, default='/data/home/kassner/BELA/data/blink/en_bert_ent_idx.txt')
    parser.add_argument('--wikidata_base_path', type=str, default='/fsx/kassner/wikidata/')
    parser.add_argument('--wikipedia_base_path', type=str, default='/fsx/kassner/wikipedia/')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--strategy', type=str, default='backwards',
                        choices=list(LINKING_STRATEGIES.keys()))
    parser.add_argument('--type', type=str, default='all')
    parser.add_argument('-d', '--dot_prod', action='store_true')
    parser.add_argument('--max_mentions', type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
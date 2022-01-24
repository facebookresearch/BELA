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
        logger.info(f'Threshold: {threshold}, # of clusts: {n_clusters}')
        if n_clusters < target:
            bounds[0] = threshold
        else:
            bounds[1] = threshold
    return clusters

'''def find_threshold(scores, linking_strategy, target, entity_ids, max_iters=100):
    logger.info(f'Finding threshold. Target # of clusts: {target}.')
    bounds = [0.0, 1.0]
    n_clusters = -1
    attempts = 0
    max_attempts = 20
    h_best = 0
    c_best = 0
    best_hc = 0
    logger.info(f'Attempts: {max_attempts}')
    while attempts<max_attempts:
        threshold = (bounds[0] + bounds[1]) / 2
        linking_strategy._threshold = threshold
        clusters = cluster(scores, linking_strategy)
        n_clusters = len(np.unique(clusters))
        labels_true = entity_ids
        labels_pred = clusters
        h = metrics.homogeneity_score(labels_true, labels_pred)
        c  = metrics.completeness_score(labels_true, labels_pred)
        logger.info(f'Threshold: {threshold}, # of clusts: {n_clusters}')
        print(c, h, c_best, h_best)
        #if h>h_best and c>=c_best*0.95 or c>c_best and h>=h_best*0.95:
        if (h+c)>best_hc:    
            bounds[0] = threshold
            h_best = h
            c_best=c
            best_hc = h+c

        else:
            bounds[1] = threshold
        attempts+=1

    return clusters'''

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


def load_embeddings(embeddings_path_list, loaded_idcs, idcs_keep=None, idcs_filter=None, entities_keep=None, entities_filter=None, max_mentions=None):
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
                
                # filter on the basis of idcs
                if idcs_keep is not None:
                    if embedding_idx not in idcs_keep:
                        continue
                if idcs_filter is not None:
                    if embedding_idx in idcs_filter:
                        continue

                # filter on the basis of entities
                if entities_keep is not None:
                    if entity not in entities_keep:
                        continue
                if entities_filter is not None:
                    if entity in entities_filter:
                        continue

                num_mentions +=1
                embedding = [float(x) for x in embedding]
                embeddings.append(embedding)
                entity_vocab.add(entity)
                entity_ids.append(entity)
                loaded_idcs.append(embedding_idx)
                if max_mentions is not None:
                    if num_mentions>=max_mentions:
                        return embeddings, entity_vocab, entity_ids
    return embeddings, entity_vocab, entity_ids, loaded_idcs

def append_embeddings(entity_vocab, entity_ids, embeddings, embeddings_path_list, loaded_idcs, idcs_keep=None, idcs_filter=None, entities_keep=None, entities_filter=None, max_mentions=None):
    embedding_idx = 0
    num_mentions = len(embeddings)
    num_entites_pre = len(entity_vocab)
    logger.info('Number of entities %s', num_entites_pre)
    logger.info('Adding embeddings')
    for embedding_path in sorted(embeddings_path_list):
        embeddings_buffer = torch.load(embedding_path, map_location='cpu')
        for embedding_batch in embeddings_buffer:
            for embedding in embedding_batch:
                entity, embedding = embedding[0], embedding[1:]
                embedding_idx +=1
                entity = int(float(entity))
                if len(entity_vocab)>=2*num_entites_pre:
                    if entity not in entity_vocab:
                        continue
                # filter on the basis of idcs
                if idcs_keep is not None:
                    if embedding_idx not in idcs_keep:
                        continue
                if idcs_filter is not None:
                    if embedding_idx in idcs_filter:
                        continue

                # filter on the basis of entities
                if entities_keep is not None:
                    if entity not in entities_keep:
                        continue
                if entities_filter is not None:
                    if entity in entities_filter:
                        continue

                num_mentions +=1
                embedding = [float(x) for x in embedding]
                embeddings.append(embedding)
                entity_vocab.add(entity)
                entity_ids.append(entity)
                loaded_idcs.append(embedding_idx)
                if max_mentions is not None:
                    if num_mentions>=max_mentions:
                        return embeddings, entity_vocab, entity_ids
    
    return embeddings, entity_vocab, entity_ids, loaded_idcs

def load_entity_embeddings(entity_vocab, embeddings):
    logging.info('Loading entity embeddings %d', len(embeddings))
    entity_embeddings = torch.load("data/blink/all_entities_large.t7")
    #entity_embeddings2 = torch.load("data/blink/novel_entities_filtered.t7")
    #entity_embeddings = torch.cat((entity_embeddings, entity_embeddings2),0)
    for i, emb in enumerate(entity_embeddings):
        if i in entity_vocab:
            embeddings.append(emb)
    return embeddings


def select_idcs_keep(dataset_path, timesplit=None, year_ref=2019, month_ref=9):
    idcs_keep = set()
    i = 0
    with open(dataset_path + ".jsonl") as f:
        for line in f:
            line = json.loads(line)
            year, month = line["time_stamp"].split("_")
            year, month = int(year), int(month)
            for _ in range(len(line["gt_entities"])):
                i += 1
                if timesplit=="t2": 
                    if year>year_ref or (year==year_ref and month>month_ref):
                        idcs_keep.add(i)
                elif timesplit=="t1":
                    if year<year_ref or (year==year_ref and month<=month_ref):
                        idcs_keep.add(i)
    return idcs_keep

def select_entities(ent_catalogue_idx_path):
    ent_catalogue = EntityCatalogue(ent_catalogue_idx_path, None, reverse=True)
    entities_selected = ent_catalogue.idx_referse.keys()

    entities_selected = set(entities_selected)
    return entities_selected

def main(args):


    output_name = args.input
    output_name = args.input.split('/')[-3].split('.')[0]
    with_entities = ""
    if args.with_entities:
        with_entities = "_with_entities"
    output_name = args.output + output_name + "_" + args.cluster_type + "_" + str(args.type_time) + "_" + str(args.type_ent) + "_" + str(args.threshold) + "_" + str(args.max_mentions) + with_entities
    
    idcs_filter = None
    idcs_keep = None
    entities_keep = None
    entities_filter = None
    loaded_idcs = []

    # collect idcs to filter/keep
    if args.type_time:
        idcs_keep = select_idcs_keep(args.dataset_path, args.type_time)

    # collect entities to filter/keep
    if args.type_ent is not None:
        entities_selected = select_entities(args.ent_catalogue_idx_path)
        if args.type_ent=="known":
            entities_keep = entities_selected
        elif args.type_ent=="unknown":
            entities_filter = entities_selected

    input_path = args.input + '*.t7'
    embeddings_path_list = glob.glob(input_path)
    #embeddings, entity_vocab, entity_ids = load_embeddings(embeddings_path_list, filter_type, idcs_filter, args.max_mentions)
    embeddings, entity_vocab, entity_ids, loaded_idcs =  load_embeddings(embeddings_path_list, loaded_idcs, idcs_keep=idcs_keep, idcs_filter=idcs_filter, entities_keep=entities_keep, entities_filter=entities_filter, max_mentions=args.max_mentions)

    logging.info('Number of mentions %d', len(embeddings))
    logging.info("Number of entities: %s", len(entity_vocab))
    if args.max_mentions is not None:
        if len(embeddings)<=args.max_mentions:
            idcs_keep = select_idcs_keep(args.dataset_path, "t2")
            embeddings, entity_vocab, entity_ids, loaded_idcs = append_embeddings(entity_vocab, entity_ids, embeddings, embeddings_path_list, loaded_idcs, idcs_keep=idcs_keep, idcs_filter=None, entities_keep=entities_filter, entities_filter=None, max_mentions=args.max_mentions)
    
    logging.info("Number of mentions: %s", len(embeddings))
    logging.info("Number of entities: %s", len(entity_vocab))
    if args.with_entites:
        embeddings = load_entity_embeddings(entity_vocab, embeddings)
        logging.info("Number of mentions: %s", len(embeddings))

    torch.save(embeddings, output_name + ".t7")
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
        if not args.dot_prod:
            embeddings /= torch.norm(embeddings, dim=-1, keepdim=True)

        embeddings = np.array(embeddings, dtype=np.float32)

        grinch = Grinch(points=embeddings, active_leaf_limit=args.limit, pruning_strategy=args.strategy)
        grinch.build_dendrogram()

        if args.threshold is not None:
            clusters = grinch.flat_clustering(args.threshold)
        else:
            target = len(entity_vocab)
            clusters = find_threshold_grinch(grinch, target)

    logger.info("Save clusters to %s", output_name)
    with open(output_name + ".txt", 'w') as g:
        for t, p, i in zip(entity_ids, clusters, loaded_idcs):
            g.write('%i, %i, %i\n' % (t, p, i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/checkpoints/kassner/hydra_outputs/main/2022-01-17-142422/0/")
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cluster_type', type=str, default="greedy")
    parser.add_argument('--dataset_path', type=str, default='/fsx/kassner/OSCAR/processed/cnn_bbc_news')
    parser.add_argument('--novel_entity_idx_path', type=str, default='/data/home/kassner/BELA/data/blink/novel_entities_filtered.jsonl')
    parser.add_argument('--ent_catalogue_idx_path', type=str, default='/data/home/kassner/BELA/data/blink/en_bert_ent_idx.txt')
    #parser.add_argument('--wikidata_base_path', type=str, default='/fsx/kassner/wikidata/')
    #parser.add_argument('--wikipedia_base_path', type=str, default='/fsx/kassner/wikipedia/')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--strategy', type=str, default='backwards')
    parser.add_argument('--type_time', type=str)
    parser.add_argument('--type_ent', type=str)
    parser.add_argument('-d', '--dot_prod', action='store_true')
    parser.add_argument('-e', '--with_entities', action='store_true')
    parser.add_argument('--max_mentions', type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
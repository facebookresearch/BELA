"""
Greedy nearest neighbor clustering for dense embeddings.
"""
import argparse
import csv
import logging

import numpy as np
import torch
import json

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
        return torch.mm(embeddings, embeddings.transpose(0, 1))
    

def find_threshold(scores, linking_strategy, target, max_iters=100):
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


def main(args):
    logger.info('Loading embeddings')
    entity_vocab = {}
    entity_ids = []
    embeddings = []
    with open(args.input, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            entity, *embedding = line
            entity = int(entity.split("tensor(")[1][:-1])
            embedding = [float(x) for x in embedding]
            embeddings.append(embedding)
            if entity not in entity_vocab:
                entity_vocab[entity] = len(entity_vocab)
            entity_id = entity_vocab[entity]
            entity_ids.append(entity_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        clusters = find_threshold(scores, linking_strategy, target)

    clusters = clusters.tolist()
    output_name = args.input
    output_name = args.input.split('/')[-1].split('.')[0]
    with open(args.output + output_name + "_greedy.txt", 'w') as g:
        for t, p in zip(entity_ids, clusters):
            g.write('%i, %i\n' % (t, p))

    with open(args.output + output_name + "_vocab.json", 'w') as fp:
        json.dump(entity_vocab, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--strategy', type=str, default='backwards',
                        choices=list(LINKING_STRATEGIES.keys()))
    parser.add_argument('-d', '--dot_prod', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
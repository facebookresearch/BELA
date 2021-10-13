"""
Cluster embeddings using GRINCH.
"""
import argparse
import csv
import logging

import numpy as np

from bela.utils.cluster import Grinch

logger = logging.getLogger(__name__)


def find_threshold(grinch, target, max_iters=100):
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


def main(args):
    # Load embeddings
    logger.info('Loading embeddings')
    entity_vocab = {}
    entity_ids = []
    embeddings = []
    with open(args.input, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            uid, entity, *embedding = line
            embedding = [float(x) for x in embedding]
            embeddings.append(embedding)
            if entity not in entity_vocab:
                entity_vocab[entity] = len(entity_vocab)
            entity_id = entity_vocab[entity]
            entity_ids.append(entity_id)
    embeddings = np.array(embeddings, dtype=np.float32)

    grinch = Grinch(points=embeddings, active_leaf_limit=args.limit, pruning_strategy=args.strategy)
    grinch.build_dendrogram()

    if args.threshold is not None:
        clusters = grinch.flat_clustering(args.threshold)
    else:
        target = len(entity_vocab)
        clusters = find_threshold(grinch, target)

    with open(args.output, 'w') as g:
        for t, p in zip(entity_ids, clusters):
            g.write('%i, %i\n' % (t, p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--strategy', type=str, default='similarity')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
import argparse
import collections
import json
import statistics

import scipy.sparse as sparse
from scipy.optimize import linear_sum_assignment


def _get_allowed_mids(train, test, choice):
    seen_ids = set()
    with open(train, 'r') as f:
        for line in f:
            data = json.loads(line)
            seen_ids.add(data['entity_id'])
    allowed_mids = set()
    with open(test, 'r') as f:
        for mid, line in enumerate(f):
            data = json.loads(line)
            seen = data['entity_id'] in seen_ids
            if seen and choice == 'seen':
                allowed_mids.add(mid)
            elif not seen and choice == 'unseen':
                allowed_mids.add(mid)
    return allowed_mids


def _create_muc_lookup(clusters):
    out = {}
    for cluster_id, cluster in clusters.items():
        for element in list(cluster):
            out[element] = cluster_id
    return out


def _create_cluster_lookup(clusters):
    out = {}
    for cluster in clusters.values():
        for element in list(cluster):
            out[element] = cluster
    return out


def muc(
    true_clusters,
    pred_clusters,
):
    true_muc_lookup = _create_muc_lookup(true_clusters)
    pred_muc_lookup = _create_muc_lookup(pred_clusters)

    precision_numerator = 0
    precision_denominator = 0
    for pred_cluster in pred_clusters.values():
        size = len(pred_cluster)
        partitions = len(set(true_muc_lookup[i] for i in pred_cluster))
        precision_numerator += size - partitions
        precision_denominator += size - 1
    muc_precision = precision_numerator / (precision_denominator + 1e-13)
    print(f'MUC Precision: {muc_precision}')

    recall_numerator = 0
    recall_denominator = 0
    for true_cluster in true_clusters.values():
        size = len(true_cluster)
        partitions = len(set(pred_muc_lookup[i] for i in true_cluster))
        recall_numerator += size - partitions
        recall_denominator += size - 1
    muc_recall = recall_numerator / (recall_denominator + 1e-13)
    print(f'MUC Recall: {muc_recall}')

    muc_f1 = 2 * muc_precision * muc_recall / (muc_precision + muc_recall + 1e-13)
    print(f'MUC F1: {muc_f1}')

    return muc_precision, muc_recall, muc_f1


def b3(
    true_clusters,
    pred_clusters,
    total,
):
    true_lookup = _create_cluster_lookup(true_clusters)
    pred_lookup = _create_cluster_lookup(pred_clusters)

    # Now do B-Cubed!
    b3_precision = 0
    b3_recall = 0
    for i in true_lookup.keys():
        numerator = len(true_lookup[i] & pred_lookup[i])
        b3_precision += numerator / len(pred_lookup[i])
        b3_recall += numerator / len(true_lookup[i])
    b3_precision /= total
    b3_recall /= total
    b3_f1 = 2 * b3_precision * b3_recall / (b3_precision + b3_recall)
    print(f'B3 Precision: {b3_precision}')
    print(f'B3 Recall: {b3_recall}')
    print(f'B3 F1: {b3_f1}')
    return b3_precision, b3_recall, b3_f1


def sparse_from_set(clusters, total):
    row_ind = []
    col_ind = []
    data = []
    for i, cluster in enumerate(clusters.values()):
        for j in cluster:
            row_ind.append(i)
            col_ind.append(j)
            data.append(1)
    M = len(clusters)
    N = total
    return sparse.csr_matrix((data, (row_ind, col_ind)), (M,N))
        

def phi_4(k, r):
    """
    k : (keys, ents)
    r : (responses, ents)
    """
    intersections = k * r.transpose() # (keys, responses)
    k_counts = k.sum(axis=-1).reshape(-1, 1)
    r_counts = r.sum(axis=-1).reshape(1, -1)
    score = 2 * intersections / (k_counts + r_counts)
    return score


def ceaf_e(
    true_clusters,
    pred_clusters,
    total,
):
    # Now do CEAF!
    k = sparse_from_set(true_clusters, total)
    r = sparse_from_set(pred_clusters, total)
    scores = phi_4(k, r)
    row_opt, col_opt = linear_sum_assignment(scores, maximize=True)
    numerator = scores[row_opt, col_opt].sum()
    ceaf_precision = numerator / len(true_clusters)
    ceaf_recall = numerator / len(pred_clusters)
    ceaf_f1 = 2 * ceaf_precision * ceaf_recall / (ceaf_precision + ceaf_recall)
    print(f'CEAF-e Precision: {ceaf_precision}')
    print(f'CEAF-e Recall: {ceaf_recall}')
    print(f'CEAF-e F1: {ceaf_f1}')
    return ceaf_precision, ceaf_recall, ceaf_f1


def error_analysis(
    true_clusters,
    pred_clusters,
):
    """
    Count the number of divided vs conflated entities.
    """
    # MUC lookups let us map mentions to their cluster ids.
    true_lookup = _create_muc_lookup(true_clusters)
    pred_lookup = _create_muc_lookup(pred_clusters)

    # Map each element in the predicted clusters to its true id to get the
    # number of conflated entities.
    num_conflated = 0
    for cluster in pred_clusters.values():
        num_conflated += len(set(true_lookup[e] for e in cluster)) - 1

    # Map each element in the true clusters to 
    num_divided = 0
    for cluster in true_clusters.values():
        num_divided += len(set(pred_lookup[e] for e in cluster)) - 1

    return num_conflated, num_divided


def main(args):
    # Indexed by cluster id
    true_clusters = collections.defaultdict(set)
    pred_clusters = collections.defaultdict(set)

    if args.train and args.test:
        allowed_mids = _get_allowed_mids(
            args.train,
            args.test,
            args.choice
        )
    else:
        allowed_mids = None

    total = 0
    with open(args.input, 'r') as f:
        for mid, line in enumerate(f):
            if allowed_mids is not None:
                if mid not in allowed_mids:
                    continue
            t, p = [x.strip() for x in line.split(',')]
            true_clusters[t].add(total)
            pred_clusters[p].add(total)
            total += 1
    median_size = statistics.median(len(x) for x in true_clusters)
    print(f'True clusters: {len(true_clusters)}')
    print(f'Median size: {median_size}')
    print(f'Pred clusters: {len(pred_clusters)}')

    muc_precision, muc_recall, muc_f1 = muc(true_clusters, pred_clusters)
    b3_precision, b3_recall, b3_f1 = b3(true_clusters, pred_clusters, total)
    ceaf_precision, ceaf_recall, ceaf_f1 = ceaf_e(true_clusters, pred_clusters, total)

    line = '\t'.join('%0.3f' % x for x in [
        muc_precision,
        muc_recall,
        muc_f1,
        b3_precision,
        b3_recall,
        b3_f1,
        ceaf_precision,
        ceaf_recall,
        ceaf_f1,
        len(pred_clusters),
        statistics.mean((muc_f1, b3_f1, ceaf_f1)),
        # statistics.median(len(x) for x in pred_clusters),
    ])
    print(f'{line}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('-c', '--choice', type=str, default=None,
                        choices=['seen', 'unseen'])
    args = parser.parse_args()

    main(args)
# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from tempfile import NamedTemporaryFile
from argparse import ArgumentParser, RawTextHelpFormatter

from torch import Tensor
import torch.nn.functional as F

from crow import run_feature_processing_pipeline, apply_crow_aggregation, apply_ucrow_aggregation, normalize

weighting_schemes = ['crow', 'ucrow', 'identity']

distance_schemes = ['euclidean', 'cosine']


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Calculate the distance between query set features and gallery set features.

    Args:
        x1 (torch.tensor): query set features.
        x2 (torch.tensor): gallery set features.

    Returns:
        dis (torch.tensor): the cosine distance between query set features and gallery set features.
    """
    assert len(x1.shape) == len(x2.shape) == 2 and x1.shape[1] == x2.shape[1]
    similarity_matrix = F.cosine_similarity(x1.unsqueeze(1),
                                            x2.unsqueeze(0), dim=2)

    return 1 - similarity_matrix


def get_nn(x, data, dis='euclidean', k=None):
    """
    Find the k top indices and distances of index data vectors from query vector x.

    :param ndarray x:
        the query vector
    :param ndarray data:
        the index vectors
    :param int k:
        optional k to truncate return

    :returns ndarray idx:
        the indices of index vectors in ascending order of distance
    :returns ndarray dists:
        the squared distances
    """
    if k is None:
        k = len(data)

    if dis == 'euclidean':
        dists = ((x - data) ** 2).sum(axis=1)
    else:
        dists = cosine_distance(torch.from_numpy(x), torch.from_numpy(data)).numpy()[0]
    idxs = np.argsort(dists)
    dists = dists[idxs]

    return idxs[:k], dists[:k]


def simple_query_expansion(Q, data, inds, top_k=10):
    """
    Get the top-k closest vectors, average and re-query

    :param ndarray Q:
        query vector
    :param ndarray data:
        index data vectors
    :param ndarray inds:
        the indices of index vectors in ascending order of distance
    :param int top_k:
        the number of closest vectors to consider

    :returns ndarray idx:
        the indices of index vectors in ascending order of distance
    :returns ndarray dists:
        the squared distances
    """
    Q += data[inds[:top_k], :].sum(axis=0)
    return normalize(Q)


def load_features(feature_dir, verbose=True):
    """
    Iterate and load numpy pickle files in the provided directory along with the filename.

    :param feature_dir: directory to iterate or list of directories
    :type feature_dir: str or list
    :param bool verbose:
        optional flag to disabled progress printout

    :yields ndarray X:
        the ndarray from the pickle file
    :yields str name:
        the name of the file without file extension
    """
    if type(feature_dir) == str:
        feature_dir = [feature_dir]

    for directory in feature_dir:
        for i, f in enumerate(os.listdir(directory)):
            name = os.path.splitext(f)[0]

            # Print progress
            if verbose and not i % 100:
                sys.stdout.write('\rProcessing file %i' % i)
                sys.stdout.flush()

            X = np.load(os.path.join(directory, f))

            yield X, name

    sys.stdout.write('\n')
    sys.stdout.flush()


def load_and_aggregate_features(feature_dir, agg_fn=None):
    """
    Given a directory of features as numpy pickles, load them, map them
    through the provided aggregation function, and return a list of
    the features and a list of the corresponding file names without the
    file extension.

    :param feature_dir: directory to iterate or list of directories
    :type feature_dir: str or list
    :param callable agg_fn:
        map function for raw features

    :returns list features:
        the list of loaded features
    :returns list names:
        corresponding file names without extension
    """
    print('Loading features %s ...' % str(feature_dir))
    features = []
    names = []
    for X, name in load_features(feature_dir):
        names.append(name)
        if agg_fn is not None:
            X = agg_fn(X)
        features.append(X.reshape(-1))

    return features, names


def get_ap(inds, dists, query_name, index_names, groundtruth_dir, ranked_dir=None):
    """
    Given a query, index data, and path to groundtruth data, perform the query,
    and evaluate average precision for the results by calling to the compute_ap
    script. Optionally save ranked results in a file.

    :param ndarray inds:
        the indices of index vectors in ascending order of distance
    :param ndarray dists:
        the squared distances
    :param str query_name:
        the name of the query
    :param list index_names:
        the name of index items
    :param str groundtruth_dir:
        directory of groundtruth files
    :param str ranked_dir:
        optional path to a directory to save ranked list for query

    :returns float:
        the average precision for this query
    """

    if ranked_dir is not None:
        # Create dir for ranked results if needed
        if not os.path.exists(ranked_dir):
            os.makedirs(ranked_dir)
        rank_file = os.path.join(ranked_dir, '%s.txt' % query_name)
        f = open(rank_file, 'w')
    else:
        f = NamedTemporaryFile(delete=False)
        rank_file = f.name
        f = open(rank_file, 'w')
    # print('rank_file:\n', rank_file)

    f.writelines([index_names[i] + '\n' for i in inds])
    f.close()

    groundtruth_prefix = os.path.join(groundtruth_dir, query_name)
    cmd = './compute_ap %s %s' % (groundtruth_prefix, rank_file)
    ap = os.popen(cmd).read()

    # Delete temp file
    if ranked_dir is None:
        os.remove(rank_file)

    return float(ap.strip())


def fit_whitening(whiten_features, dw, d, agg_fn=None):
    """
    Calculate whitening parameters

    :param str whiten_features: 
        directory of features to fit whitening
    :param callable agg_fn: 
        aggregation function
    :param int d: 
        final feature dimension

    :returns dict params:
        a dict of transformation parameters
    """
    # Load features for fitting whitening
    data, _ = load_and_aggregate_features(whiten_features, agg_fn=agg_fn)

    # Whiten, and reduce dim of features
    # Whitening is trained on the same images that we query against here for expediency
    print('Fitting PCA/whitening wth d=%d on %s ...' % (d, whiten_features))
    _, whiten_params = run_feature_processing_pipeline(data, dw=dw, d=d)

    return whiten_params


def run_eval(queries_dir, groundtruth_dir, index_features,
             dw, d, whiten_params,
             out_dir, dis='euclidean', agg_fn=None, qe_fn=None):
    """
    Run full evaluation pipeline on specified data.

    :param str queries_dir: directory of query features
    :param str groundtruth_dir: directory of groundtruth info
    :param index_features: directory or list of directories of index features
    :type index_features: str or list
    :param str whiten_features: directory of features to fit whitening
    :param str out_dir: directory to save query results
    :param callable agg_fn: aggregation function
    :param callable qe_fn: query expansion function
    """
    data, image_names = load_and_aggregate_features(index_features, agg_fn=agg_fn)
    # print(np.array(data).shape)
    data, _ = run_feature_processing_pipeline(np.vstack(data), dw=dw, d=d, params=whiten_params)
    # print(data.shape)

    # Iterate queries, process them, rank results, and evaluate mAP
    aps = []
    for Q, query_name in tqdm(load_features(queries_dir)):
        # print('raw feature:\n', Q.shape)
        if agg_fn is not None:
            Q = agg_fn(Q)
        # print('aggregated feature:\n', Q.shape)
        if len(Q.shape) == 1 or len(Q.shape) != 2:
            Q = Q.reshape(1, -1)

        # Normalize and PCA to final feature
        Q, _ = run_feature_processing_pipeline(Q, dw=dw, d=d, params=whiten_params)
        # print('preprocessed feature:\n', Q.shape)

        inds, dists = get_nn(Q, data, dis=dis)

        # perform query_expansion
        if qe_fn is not None:
            Q = qe_fn(Q, data, inds)
            inds, dists = get_nn(Q, data, dis=dis)

        ap = get_ap(inds, dists, query_name, image_names, groundtruth_dir, out_dir)
        aps.append(ap)

    return np.array(aps).mean()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--queries', dest='queries', type=str, default='oxford/layer4_queries/',
                        help='directory containing image files')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='oxford/groundtruth/',
                        help='directory containing groundtruth files')
    parser.add_argument('--index_features', dest='index_features', type=str, default='oxford/layer4',
                        help='directory containing raw features to index')

    parser.add_argument('--wt', dest='weighting', type=str, default='crow',
                        choices=weighting_schemes,
                        help='weighting to apply for feature aggregation')

    parser.add_argument('--dw', dest='dw', type=int, default=0,
                        help='the way to reduce dimension.\n'
                             'dw = 0: Do not perform dimension reduction.\n'
                             'dw = 1: Perform L2-Norm only.\n'
                             'dw = 2: Perform L2-Norm -> PCA -> L2-Norm.\n'
                             'dw = 3: Perform L2-Norm -> PCA (Whitening) -> L2-Norm.\n'
                             'Default: 0 ()')
    parser.add_argument('--d', dest='d', type=int, default=128, help='dimension of final feature')
    parser.add_argument('--whiten_features', dest='whiten_features', type=str, default='paris/layer4',
                        help='directory containing raw features to fit whitening')

    parser.add_argument('--dis', dest='dis', type=str, default='euclidean',
                        choices=distance_schemes,
                        help='Distance function. Default: Euclidean distance')

    parser.add_argument('--qe', dest='qe', type=int, default=0,
                        help='perform query expansion with this many top results')
    parser.add_argument('--out', dest='out', type=str, default=None, help='optional path to save ranked output')
    args = parser.parse_args()

    # Select which aggregation function to apply
    if args.weighting == 'crow':
        agg_fn = apply_crow_aggregation
    elif args.weighting == 'ucrow':
        agg_fn = apply_ucrow_aggregation
    else:
        agg_fn = None

    if args.qe > 0:
        qe_fn = partial(simple_query_expansion, top_k=args.qe)
    else:
        qe_fn = None

    # compute whitening params
    if args.dw >= 2:
        whitening_params = fit_whitening(args.whiten_features, args.dw, args.d, agg_fn=agg_fn)
    else:
        whitening_params = None

    # compute aggregated features and run the evaluation
    mAP = run_eval(args.queries, args.groundtruth, args.index_features,
                   args.dw, args.d, whitening_params, args.out, dis=args.dis, agg_fn=agg_fn, qe_fn=qe_fn)
    print('mAP: %f' % mAP)

    exit(0)

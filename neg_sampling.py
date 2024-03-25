from typing import Any

import numpy as np
from tqdm import trange

from pairing import SzudzikPair

neg_sampler_factories = {
    "random_edge_sampler": lambda src, dst: random_edge_sampler(src, dst, src, dst),
}


def random_edge_sampler(src, dst, src_hist, dst_hist):
    """
    Random edges can be any edge between any two nodes in src_hist and dst_hist
    """
    possible_edge_keys = np.vectorize(SzudzikPair.encode)(src_hist, dst_hist)
    observed_edge_keys = SzudzikPair.encode(src, dst)
    mask = ~np.in1d(possible_edge_keys, observed_edge_keys)
    possible_edge_keys = possible_edge_keys[mask]

    neg_edge_keys = np.random.choice(
        possible_edge_keys,
        size=(src.shape[0],),
    )
    neg_src, neg_dst = SzudzikPair.decode(neg_edge_keys)
    return neg_src, neg_dst


def random_dst_neg_sampler(src, dst, n_nodes):
    neg_dst = np.random.randint(0, n_nodes, size=src.shape[0])

    return src, neg_dst


def historical_dst_neg_sampler(src, dst, dst_hist, n_trials=20):
    dst_hist = np.unique(list(dst_hist))
    mask = np.zeros_like(src, dtype=bool)
    neg_dst = np.empty_like(src)
    if len(dst_hist) == 0:
        print("Warning: no dst nodes found in the dataset")
        raise ValueError

    for _ in range(n_trials):
        neg_dst[~mask] = np.random.choice(
            dst_hist,
            size=(~mask).sum(),
        )
        mask = neg_dst != dst
        mask = mask & (neg_dst != src)
        if mask.all():
            break
    return src, neg_dst


def random_neg_sampler(src, dst, n_nodes, max_trials=10):
    """
    Sample negative edges by sampling random source and destination nodes
    """
    mask = np.zeros_like(src, dtype=bool)
    pos_code = SzudzikPair.encode(src, dst)
    sample_s = np.empty_like(src)
    sample_src = np.empty_like(src)
    sample_dst = np.empty_like(dst)
    max_key = SzudzikPair.encode(n_nodes, n_nodes)
    for _ in range(max_trials):
        sample_s[~mask] = np.random.randint(
            0,
            max_key,
            size=((~mask).sum()),
        )
        sample_src, sample_dst = SzudzikPair.decode(sample_s)
        mask = sample_src != sample_dst
        mask = mask & (sample_src != src)
        mask = mask & (sample_dst != dst)
        if mask.all():
            break

    return sample_src, sample_dst


def edge_neg_sampler(src, dst, src_pool, dst_pool, n_trials=20):
    """
    Sample Negative edge from a pool of edges
    Make sure that the negative samples are not equal to the positive samples
    """
    pool_keys = np.unique(SzudzikPair.encode(src_pool, dst_pool))
    mask = np.zeros_like(src, dtype=bool)
    neg_keys = np.empty_like(src)
    for k in range(n_trials):
        size = (~mask).sum()
        neg_keys[~mask] = np.random.choice(pool_keys, size)
        neg_src, neg_dst = SzudzikPair.decode(neg_keys)
        # make sure that the negative samples are not equal to the positive samples
        mask = ~((neg_src == src) & (neg_dst == dst))
        if mask.all():
            break

    return neg_src, neg_dst


def dst_neg_sampler(src, dst, dst_pool, n_trials=20):
    """
    Sample the destination node from a pool of destination nodes
    Make sure that the negative samples are not equal to the positive samples
    """
    pool_keys = np.unique(dst_pool)
    mask = np.zeros_like(src, dtype=bool)
    neg_dst = np.empty_like(src)
    for k in range(n_trials):
        size = (~mask).sum()
        neg_dst[~mask] = np.random.choice(pool_keys, size=size)
        mask = neg_dst != dst
        if mask.all():
            break
    return src, neg_dst

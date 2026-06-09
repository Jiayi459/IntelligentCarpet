"""seed_tactile_stats.py — one-time seeder for the shared tactile stats cache.

Writes `train/com/output/tactile_stats.json` containing the scalar tactile
mean/std that beta / gamma / epsilon all use to standardize the raw tactile
window inputs.

The numbers reproduce phase2_tactile's RNG-replayed sample bit-for-bit:
    np.random.seed(SEED)
    sample_centers = train_centers[np.random.permutation(n_train_tot)[:1000]]
    sample_tactile = np.concatenate([tactile_all[t-99:t+1] for t in sample_centers])
    mean = sample_tactile.mean(); std = sample_tactile.std()

phase2_tactile / phase2_gamma already do this internally on every run (it costs
~10 s); the cache here exists so phase2_epsilon can READ the same numbers
without having to re-run the sampling block, AND so future scripts share
the same standardization without re-deriving it.

Idempotent: re-running produces an identical JSON.

Run:
    python train/com/seed_tactile_stats.py
    python train/com/seed_tactile_stats.py --force        # overwrite existing
"""

import os
import sys
import json
import pickle
import re
import argparse

import numpy as np


_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
sys.path.insert(0, _TRAIN)

_OUT       = os.path.join(_HERE, 'output')
_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')
_STATS     = os.path.join(_OUT, 'tactile_stats.json')


HISTORY = 100
HORIZON = 10
SEED    = 42
N_SAMPLE_WINDOWS = 1000


def main():
    parser = argparse.ArgumentParser(description='Seed the shared tactile mean/std cache.')
    parser.add_argument('--force', action='store_true',
                        help='overwrite tactile_stats.json if it already exists')
    args = parser.parse_args()

    if os.path.exists(_STATS) and not args.force:
        with open(_STATS) as f:
            existing = json.load(f)
        print(f'tactile_stats.json already exists:')
        print(f'  tactile_mean = {existing["tactile_mean"]:.6f}')
        print(f'  tactile_std  = {existing["tactile_std"]:.6f}')
        print(f'use --force to recompute and overwrite.')
        return

    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY} -- build it first with phase2_tactile.py')

    print(f'loading tactile cache: {_CACHE_NPY}')
    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    T = tactile_all.shape[0]
    assert tactile_all.shape == (T, 96, 96), f'unexpected shape {tactile_all.shape}'

    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)
    com_gt = com_results['com_gt']
    assert len(com_gt) == T

    with open(os.path.join(_TRAIN, 'singlePerson_test', 'log.p'), 'rb') as f:
        log = pickle.load(f)
    with open(os.path.join(_TRAIN, 'singlePerson_test', 'fileNames.p'), 'rb') as f:
        file_names = pickle.load(f)

    n_sessions = len(log) - 1
    _SUBJECT_RE = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
    _ = [_SUBJECT_RE.match(n).group(3) for n in file_names]  # validate file names parse

    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))

    # Rebuild centers + train mask, same as phase2_tactile / phase2_gamma
    centers, splits = [], []
    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        valid_t = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t))
        for i, t in enumerate(valid_t):
            centers.append(t)
            splits.append('train' if i < n_train_s else 'test')
    centers = np.asarray(centers)
    splits  = np.asarray(splits)
    assert len(centers) == 17218, f'expected 17218 samples, got {len(centers)}'
    train_centers = centers[splits == 'train']
    n_train_tot   = len(train_centers)
    print(f'samples: total={len(centers)}, train={n_train_tot}')

    # The deterministic RNG-replayed sample (same as phase2_tactile.main())
    np.random.seed(SEED)
    sample_centers = train_centers[np.random.permutation(n_train_tot)[:N_SAMPLE_WINDOWS]]
    sample_tactile = np.concatenate(
        [tactile_all[t - HISTORY + 1 : t + 1] for t in sample_centers], axis=0
    )
    tactile_mean = float(sample_tactile.mean())
    tactile_std  = float(sample_tactile.std())
    print(f'tactile_mean = {tactile_mean:.6f}')
    print(f'tactile_std  = {tactile_std:.6f}')

    payload = {
        'tactile_mean':       tactile_mean,
        'tactile_std':        tactile_std,
        'n_sample_windows':   N_SAMPLE_WINDOWS,
        'history':            HISTORY,
        'seed':               SEED,
        'source':             'phase2_tactile RNG-replayed sample',
        'n_total_samples':    int(len(centers)),
        'n_train_samples':    int(n_train_tot),
    }
    with open(_STATS, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'wrote {_STATS}')


if __name__ == '__main__':
    main()

"""eval_phase2_tactile.py — produce metrics.json + plots from a tactile checkpoint.

Use case: when train_phase2_tactile.py was interrupted before its eval phase ran
(e.g. qrsh wall-time hit), the best-val checkpoint at
{output_dir}/tactile_model_best.pt is saved but metrics.json + the plots never
got written. This script does just that eval + output phase from the checkpoint.

Important: reproduces train_phase2_tactile.py's standardization stats EXACTLY
by replaying the same RNG-driven 1000-window sample. As long as the cache and
SEED are unchanged, tactile_mean and tactile_std match the training-time values
bit-for-bit, so the predictions match what the training script would have
produced if it had reached the eval phase.

Usage:
    python train/com/eval_phase2_tactile.py \
        --output-dir train/com/output/phase2_tactile_200ep

Optional:
    --checkpoint  filename inside output-dir (default tactile_model_best.pt)
    --tactile-cache  path to tactile_all.npy (default train/com/output/tactile_all.npy)
"""

import os
import sys
import json
import pickle
import re
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Make the sibling training module importable (safe now thanks to __main__ guards)
_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _TRAIN)

from train_phase2_tactile import (
    TactileForecaster,
    HISTORY,
    HORIZON,
    SEED,
    BATCH,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a tactile checkpoint, write metrics.json + plots.')
    parser.add_argument('--output-dir', required=True,
                        help='where the checkpoint lives and where outputs will be saved')
    parser.add_argument('--checkpoint', default='tactile_model_best.pt',
                        help='checkpoint filename inside output-dir')
    parser.add_argument('--tactile-cache', default=None,
                        help='path to tactile_all.npy (default: train/com/output/tactile_all.npy)')
    args = parser.parse_args()

    output_dir = args.output_dir
    ckpt_path  = os.path.join(output_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f'no checkpoint at {ckpt_path}')

    _OUT       = os.path.join(_HERE, 'output')
    _CACHE_NPY = args.tactile_cache or os.path.join(_OUT, 'tactile_all.npy')
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY}')

    # Same seed + RNG order as training, so all derived stats match exactly.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    print(f'checkpoint: {ckpt_path}')
    print(f'tactile cache: {_CACHE_NPY}')

    # Tactile (mmap so we don't load 1.2 GB into RAM)
    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    T = tactile_all.shape[0]
    print(f'tactile_all: shape={tactile_all.shape}')

    # CoM + session metadata
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
    subjects_per_sess = [_SUBJECT_RE.match(n).group(3) for n in file_names]

    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))

    # Sample indices + per-session 70/30 split (identical to training)
    centers, sources = [], {'subject': [], 'session': [], 'split': []}
    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        valid_t = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t))
        for i, t in enumerate(valid_t):
            centers.append(t)
            sources['subject'].append(subjects_per_sess[s])
            sources['session'].append(s)
            sources['split'].append('train' if i < n_train_s else 'test')
    centers = np.asarray(centers)
    meta = {k: np.asarray(v) for k, v in sources.items()}
    train_mask = meta['split'] == 'train'
    test_mask  = meta['split'] == 'test'
    assert len(centers) == 17218, f'expected 17218 samples, got {len(centers)}'
    print(f'samples: train={train_mask.sum()}, test={test_mask.sum()}')

    # CoM targets
    ref_all      = com_gt[centers]
    Y_abs_all    = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in centers], axis=0)
    Y_delta_all  = Y_abs_all - ref_all[:, None, :]

    ref_test       = ref_all[test_mask]
    Y_abs_test     = Y_abs_all[test_mask]
    Y_delta_test   = Y_delta_all[test_mask]
    Y_delta_train  = Y_delta_all[train_mask]

    pred_persistence = np.broadcast_to(ref_test[:, None, :],
                                       (len(ref_test), HORIZON, 3)).copy()

    # Tactile mean/std — replay training's RNG-driven 1000-window sample
    sample_centers = centers[train_mask][np.random.permutation(train_mask.sum())[:1000]]
    sample_tactile = np.concatenate([tactile_all[t - HISTORY + 1 : t + 1] for t in sample_centers], axis=0)
    tactile_mean = float(sample_tactile.mean())
    tactile_std  = float(sample_tactile.std())
    print(f'tactile mean={tactile_mean:.4f}  std={tactile_std:.4f}')

    mean_Y = Y_delta_train.reshape(-1, 3).mean(axis=0)
    std_Y  = Y_delta_train.reshape(-1, 3).std(axis=0)
    std_Y  = np.where(std_Y < 1e-6, 1.0, std_Y)
    print(f'Y_delta mean={mean_Y}  std={std_Y}')

    class TactileDataset(Dataset):
        def __init__(self, centers, Y_delta):
            self.centers = centers
            self.Y_delta = Y_delta

        def __len__(self):
            return len(self.centers)

        def __getitem__(self, idx):
            t = int(self.centers[idx])
            window = tactile_all[t - HISTORY + 1 : t + 1]
            window = (window - tactile_mean) / tactile_std
            target = (self.Y_delta[idx] - mean_Y) / std_Y
            return (torch.from_numpy(window.astype(np.float32)),
                    torch.from_numpy(target.astype(np.float32)))

    test_ds     = TactileDataset(centers[test_mask], Y_delta_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # Load model
    model = TactileForecaster().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    # Inference
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    pred_delta_norm = np.concatenate(preds, axis=0)
    pred_delta      = pred_delta_norm * std_Y + mean_Y
    pred_tactile    = pred_delta + ref_test[:, None, :]

    # Also persist tactile_model.pt (same weights as best, so downstream tools
    # that look for either filename work).
    torch.save(model.state_dict(), os.path.join(output_dir, 'tactile_model.pt'))

    # Metrics (same schema as train_phase2_tactile)
    def euc(pred, gt):
        return np.linalg.norm(pred - gt, axis=2)

    methods = {'persistence': pred_persistence, 'phase2_tactile': pred_tactile}
    e_persist      = euc(pred_persistence, Y_abs_test)
    median_persist = float(np.median(e_persist))

    results = {}
    for name, pred in methods.items():
        e_3d = euc(pred, Y_abs_test)
        e_ax = np.abs(pred - Y_abs_test)
        results[name] = {
            'overall_median_3d_mm':       float(np.median(e_3d)),
            'overall_mean_3d_mm':         float(np.mean(e_3d)),
            'overall_p95_3d_mm':          float(np.percentile(e_3d, 95)),
            'per_horizon_median_3d_mm':   [float(np.median(e_3d[:, h]))         for h in range(HORIZON)],
            'per_horizon_p95_3d_mm':      [float(np.percentile(e_3d[:, h], 95)) for h in range(HORIZON)],
            'per_axis_median_mm':         {ax: float(np.median(e_ax[:, :, i])) for i, ax in enumerate('xyz')},
            'per_axis_per_horizon_median_mm': {
                ax: [float(np.median(e_ax[:, h, i])) for h in range(HORIZON)]
                for i, ax in enumerate('xyz')
            },
            'skill_score_vs_persistence': float(np.median(e_3d) / median_persist),
        }

    test_subj = meta['subject'][test_mask]
    e_tact = euc(pred_tactile, Y_abs_test)
    per_subject = {}
    for subj in sorted(np.unique(test_subj)):
        m = test_subj == subj
        per_subject[subj] = {
            'n':      int(m.sum()),
            'median': float(np.median(e_tact[m])),
            'p95':    float(np.percentile(e_tact[m], 95)),
        }
    results['phase2_tactile']['per_subject_median_mm'] = per_subject

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Console summary
    bar = '=' * 78
    print(f'\n{bar}\nPHASE 2 TACTILE-ONLY EVAL (from {args.checkpoint})  '
          f'(n_test = {Y_abs_test.shape[0]}, horizon = {HORIZON} frames = 1.0 s)\n{bar}')

    ordered = ['persistence', 'phase2_tactile']
    print(f'{"method":<22} {"median 3D":>12} {"mean 3D":>10} {"p95 3D":>10} {"skill":>8}')
    for name in ordered:
        r = results[name]
        print(f'  {name:<20}  {r["overall_median_3d_mm"]:>10.1f}   '
              f'{r["overall_mean_3d_mm"]:>8.1f}   '
              f'{r["overall_p95_3d_mm"]:>8.1f}   '
              f'{r["skill_score_vs_persistence"]:>6.3f}')

    print('\nPer-horizon median 3D error (mm):')
    print(f'{"horizon (frame)":<22}' + ''.join(f'{h + 1:>7d}' for h in range(HORIZON)))
    for name in ordered:
        vals = results[name]['per_horizon_median_3d_mm']
        print(f'  {name:<20}' + ''.join(f'{v:>7.1f}' for v in vals))

    print('\nPer-axis median error (mm, averaged across horizons):')
    for name in ordered:
        by_ax = results[name]['per_axis_median_mm']
        print(f'  {name:<22}  x={by_ax["x"]:>5.1f}   y={by_ax["y"]:>5.1f}   z={by_ax["z"]:>5.1f}')

    print('\nPhase 2 tactile per-subject median 3D error (mm):')
    for subj, st in sorted(per_subject.items(), key=lambda kv: kv[1]['median']):
        print(f'  {subj:<14}  n={st["n"]:>5d}   median={st["median"]:>6.1f}   p95={st["p95"]:>6.1f}')

    # Plots
    fig, ax = plt.subplots(figsize=(8, 5))
    hs = np.arange(1, HORIZON + 1) / 10.0
    for name in ordered:
        ax.plot(hs, results[name]['per_horizon_median_3d_mm'], marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median 3D Euclidean error (mm)',
           title=f'Phase 2 tactile-only (from {args.checkpoint}) — error vs horizon')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_horizon.png'), dpi=100)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ordered:
        vals = results[name]['per_axis_per_horizon_median_mm']['z']
        ax.plot(hs, vals, marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median |z| error (mm)',
           title=f'Phase 2 tactile-only (from {args.checkpoint}) — z-axis error vs horizon')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_horizon_z.png'), dpi=100)
    plt.close()

    print(f'\nSaved metrics + plots to {output_dir}')
    print('(no training_curve.png — eval-only run.)')


if __name__ == '__main__':
    main()

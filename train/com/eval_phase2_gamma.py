"""eval_phase2_gamma.py — produce metrics.json + plots from a gamma checkpoint.

Use case: when train_phase2_gamma.py was interrupted before its eval phase ran
(e.g. qrsh wall-time hit, SSH disconnect), the best-val checkpoint at
{output_dir}/gamma_model_best.pt is saved but metrics.json + plots never got
written. This script does just the eval + output phase from the saved checkpoint.

Standardization stats are loaded from {output_dir}/stats.npz (which the training
script writes before training begins), so the standardization is byte-identical
to the training-time values regardless of when the job was interrupted.

Usage:
    python train/com/eval_phase2_gamma.py \
        --output-dir train/com/output/phase2_gamma

Optional:
    --checkpoint   filename inside output-dir (default gamma_model_best.pt)
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


# Make sibling training module importable (safe thanks to __main__ guards)
_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _TRAIN)

from train_phase2_gamma import (
    GammaForecaster,
    HISTORY,
    HORIZON,
    SEED,
    BATCH,
    MOVING_FRAC,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a gamma checkpoint; write metrics + plots.')
    parser.add_argument('--output-dir', required=True,
                        help='where the checkpoint + stats live and where outputs will be saved')
    parser.add_argument('--checkpoint', default='gamma_model_best.pt',
                        help='checkpoint filename inside output-dir')
    parser.add_argument('--tactile-cache', default=None,
                        help='path to tactile_all.npy (default: train/com/output/tactile_all.npy)')
    args = parser.parse_args()

    output_dir = args.output_dir
    ckpt_path  = os.path.join(output_dir, args.checkpoint)
    stats_path = os.path.join(output_dir, 'stats.npz')
    if not os.path.exists(ckpt_path):
        raise SystemExit(f'no checkpoint at {ckpt_path}')
    if not os.path.exists(stats_path):
        raise SystemExit(f'no stats.npz at {stats_path} -- did training start? rerun train_phase2_gamma.py for at least 1 epoch.')

    _OUT       = os.path.join(_HERE, 'output')
    _CACHE_NPY = args.tactile_cache or os.path.join(_OUT, 'tactile_all.npy')
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY}')

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device     : {device}')
    print(f'checkpoint : {ckpt_path}')
    print(f'stats      : {stats_path}')
    print(f'cache      : {_CACHE_NPY}')

    # ---- Load standardization stats (matched training-time values exactly) ----
    stats = np.load(stats_path)
    tactile_mean = float(stats['tactile_mean'])
    tactile_std  = float(stats['tactile_std'])
    com_mean     = stats['com_mean']
    com_std      = stats['com_std']
    mean_Y       = stats['mean_Y']
    std_Y        = stats['std_Y']
    print(f'tactile mean={tactile_mean:.4f}  std={tactile_std:.4f}')
    print(f'CoM history mean={com_mean}  std={com_std}')
    print(f'Y_delta mean={mean_Y}  std={std_Y}')

    # ---- Load tactile cache + CoM data ----
    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    T = tactile_all.shape[0]

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

    # ---- Rebuild sample indices (same as training) ----
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
    test_mask = meta['split'] == 'test'
    test_centers = centers[test_mask]
    test_subj    = meta['subject'][test_mask]
    n_test = len(test_centers)
    assert len(centers) == 17218
    print(f'samples: total={len(centers)}, test={n_test}')

    ref_test   = com_gt[test_centers]
    Y_abs_test = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in test_centers], axis=0)
    com_hist_test = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in test_centers], axis=0)
    com_hist_test_norm = (com_hist_test - com_mean) / com_std

    pred_persistence = np.broadcast_to(ref_test[:, None, :], (n_test, HORIZON, 3)).copy()

    # ---- Load model ----
    model = GammaForecaster().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'gamma model: {n_params} params')

    # ---- Inference (batched over test samples) ----
    print('running inference on test set...')
    preds = np.zeros((n_test, HORIZON, 3), dtype=np.float64)
    with torch.no_grad():
        for i0 in range(0, n_test, BATCH):
            i1 = min(i0 + BATCH, n_test)
            batch_centers = test_centers[i0:i1]
            windows = np.stack(
                [(tactile_all[t - HISTORY + 1 : t + 1] - tactile_mean) / tactile_std
                 for t in batch_centers],
                axis=0
            ).astype(np.float32)
            tact  = torch.from_numpy(windows).to(device)
            com_h = torch.from_numpy(com_hist_test_norm[i0:i1].astype(np.float32)).to(device)
            y = model(tact, com_h).cpu().numpy()
            delta = y * std_Y + mean_Y
            preds[i0:i1] = delta + ref_test[i0:i1][:, None, :]
    pred_gamma = preds

    # Also save tactile_model.pt-style end-of-training name pointing at the same weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'gamma_model.pt'))

    # ---- Motion subset partition ----
    future_xy   = Y_abs_test[:, :, :2]
    step_diffs  = np.diff(future_xy, axis=1)
    step_speeds = np.linalg.norm(step_diffs, axis=2)
    v_future    = step_speeds.max(axis=1)
    threshold   = float(np.quantile(v_future, 1.0 - MOVING_FRAC))
    moving_mask = v_future > threshold
    static_mask = ~moving_mask
    subsets = {'full': np.ones(n_test, dtype=bool), 'static': static_mask, 'moving': moving_mask}
    print(f'subset partition (threshold {threshold:.2f} mm/frame): '
          f'static={static_mask.sum()}, moving={moving_mask.sum()}')

    # ---- Metrics ----
    def euc(pred, gt):
        return np.linalg.norm(pred - gt, axis=2)

    methods = {'persistence': pred_persistence, 'phase2_gamma': pred_gamma}
    persist_e = euc(pred_persistence, Y_abs_test)
    persist_med_per_sub = {sub: float(np.median(persist_e[mask])) for sub, mask in subsets.items()}

    results = {}
    for name, pred in methods.items():
        e_3d = euc(pred, Y_abs_test)
        e_ax = np.abs(pred - Y_abs_test)
        rec = {}
        for sub, mask in subsets.items():
            rec[sub] = {
                'n':                  int(mask.sum()),
                'median_3d_mm':       float(np.median(e_3d[mask])),
                'mean_3d_mm':         float(np.mean(e_3d[mask])),
                'p95_3d_mm':          float(np.percentile(e_3d[mask], 95)),
                'per_horizon_median': [float(np.median(e_3d[mask, h])) for h in range(HORIZON)],
                'per_axis_median':    {ax: float(np.median(e_ax[mask, :, i])) for i, ax in enumerate('xyz')},
                'skill_vs_persistence': float(np.median(e_3d[mask]) / persist_med_per_sub[sub])
                                        if persist_med_per_sub[sub] > 1e-6 else float('nan'),
            }
        results[name] = rec

    out = {
        'n_test':                  n_test,
        'threshold_mm_per_frame':  threshold,
        'n_static':                int(static_mask.sum()),
        'n_moving':                int(moving_mask.sum()),
        'recovered_from':          args.checkpoint,
        'methods':                 results,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # ---- Console summary ----
    bar = '=' * 78
    print(f'\n{bar}\nGAMMA FUSION EVAL (from {args.checkpoint})  (n_test = {n_test})\n{bar}')
    print(f'subsets: full ({n_test}), static ({static_mask.sum()}), moving ({moving_mask.sum()})')

    print(f'\n{"method":<22} {"FULL median":>12} {"STATIC median":>14} {"MOVING median":>14} {"skill@MOVING":>14}')
    for name in ['persistence', 'phase2_gamma']:
        rec = results[name]
        print(f'  {name:<20}  '
              f'{rec["full"]["median_3d_mm"]:>10.1f}    '
              f'{rec["static"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["skill_vs_persistence"]:>12.3f}')

    print(f'\nMOVING-subset per-axis medians (mm):')
    for name in ['persistence', 'phase2_gamma']:
        a = results[name]['moving']['per_axis_median']
        print(f'  {name:<22}  x={a["x"]:>6.1f}  y={a["y"]:>6.1f}  z={a["z"]:>6.1f}')

    # ---- Plots ----
    hs = np.arange(1, HORIZON + 1) / 10.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ['persistence', 'phase2_gamma']:
        ax.plot(hs, results[name]['moving']['per_horizon_median'], marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median 3D Euclidean error (mm), MOVING',
           title=f'Gamma (recovered from {args.checkpoint}) — moving subset')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_horizon_moving.png'), dpi=100)
    plt.close()

    # ---- Decision hint ----
    hms_path = os.path.join(_OUT, 'high_motion', 'metrics.json')
    p1_moving = v2_moving = None
    if os.path.exists(hms_path):
        with open(hms_path) as f:
            hms = json.load(f)
        if 'phase1_gru_com' in hms['methods']:
            p1_moving = hms['methods']['phase1_gru_com']['moving']['median_3d_mm']
        if 'phase2_v2_gru_kp' in hms['methods']:
            v2_moving = hms['methods']['phase2_v2_gru_kp']['moving']['median_3d_mm']

    g_moving = results['phase2_gamma']['moving']['median_3d_mm']
    print(f'\n{bar}\nDECISION HINT (MOVING subset, 1-s horizon)\n{bar}')
    print(f'gamma (recovered):  {g_moving:.1f} mm')
    if p1_moving is not None:
        print(f'Phase 1:            {p1_moving:.1f} mm  (CoM history only)')
    if v2_moving is not None:
        print(f'v2:                 {v2_moving:.1f} mm  (camera 21-keypoint history)')
    if p1_moving is not None and v2_moving is not None:
        if g_moving < v2_moving:
            print('-> gamma BEATS v2 -- tactile adds info beyond camera-pose history. STRONG positive.')
        elif g_moving < p1_moving:
            print('-> gamma BEATS Phase 1 -- tactile adds info on top of CoM history. POSITIVE (but less than camera-pose).')
        else:
            print('-> gamma does NOT beat Phase 1 -- tactile contributes no info beyond CoM history. NEGATIVE.')
    print(f'\nReminder: recovered from partial training. Re-run for full epochs if needed.')


if __name__ == '__main__':
    main()

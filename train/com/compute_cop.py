"""compute_cop.py — Center of Pressure (CoP) from the tactile cache, + persistence baselines.

Motivation:
    The literature converges on CoP as the natural target for tactile/pressure
    sensors (it IS the centroid of what the sensor measures, no camera-OpenPose
    chain). This script (a) computes per-frame 2-D CoP from the existing
    tactile_all.npy cache and (b) measures the persistence-on-CoP forecasting
    floor at horizons 0.1-1.0 s, for direct comparison with the existing
    persistence-on-CoM floor (54.1 mm median 3D at 1 s).

    The headline decision after this script runs:
        - if persistence-on-CoP >> persistence-on-CoM (e.g. 100+ mm vs 54 mm),
          CoP is the better forecasting target -- pivot;
        - if comparable, stay on CoM and proceed with gamma fusion.

How CoP is computed:
    For each (96, 96) tactile frame p:
        total = sum(p)
        if total > eps:
            cop_col_grid = sum(p * col_indices) / total     # in [0, 95]
            cop_row_grid = sum(p * row_indices) / total     # in [0, 95]
        else:
            cop = (nan, nan)                                # no contact
    Mapped to mm using the same carpet extent as compute_com.py:
        cop_x_mm = -100 + cop_col_grid * (1900 / 95)
        cop_y_mm = -100 + cop_row_grid * (1900 / 95)
    Column => x (carpet length), row => y (carpet width). This is consistent
    with the OpenCV/image-array convention and with how the keypoints (which
    use the same downsampled 20x20 grid) are laid out.

    `total_pressure` is also saved per frame -- proxy for vertical GRF if you
    want a "z-like" channel later.

Window / split logic:
    Identical to phase1/phase2 (HISTORY=100, HORIZON=10, per-session 70/30
    chronological split, same gt_outliers mask) so the resulting test-set
    persistence numbers are directly comparable to the existing
    phase1/phase2/phase2_v2 reports.

    Additional filter applied here: skip any window whose [t, t+HORIZON] range
    contains a frame with NaN CoP (no contact). Frames with no detectable
    contact are physically incompatible with persistence-on-CoP (there is no
    "current CoP" to project forward), so we drop those samples cleanly.

Outputs (under train/com/output/):
    cop_results.p             dict with 'cop_mm' (T, 2), 'total_pressure' (T,),
                              'no_contact' (T,) bool, plus the persistence numbers
                              and the metadata fields used (SEED, HISTORY, HORIZON).
    cop_persistence.png       median 3D error vs horizon: CoP vs CoM-xy vs CoM-3D
    cop_persistence_per_axis.png   per-axis (x, y) breakdown for both CoP and CoM

Run (any cwd):
    python train/com/compute_cop.py            # full sanity + persistence
    python train/com/compute_cop.py --no-plot  # skip the matplotlib plots
"""

import os
import sys
import json
import pickle
import re
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_OUT   = os.path.join(_HERE, 'output')


# ---------------------------------------------------------------------------
# Coordinate convention (must match compute_com.py exactly)
# ---------------------------------------------------------------------------

_GRID_N        = 96                       # tactile grid is 96 x 96 sensors
_CARPET_MIN_MM = -100.0
_CARPET_MAX_MM = 1800.0
_GRID_STEP_MM  = (_CARPET_MAX_MM - _CARPET_MIN_MM) / (_GRID_N - 1)   # 1900 / 95 = 20.0 mm

HISTORY = 100
HORIZON = 10
SEED    = 42


# ---------------------------------------------------------------------------
# CoP from the full tactile stack
# ---------------------------------------------------------------------------

def compute_cop_stack(tactile_all):
    """Vectorized CoP over every frame.

    tactile_all : (T, 96, 96) non-negative pressure.

    Returns:
        cop_mm        : (T, 2) float64 -- (cop_x_mm, cop_y_mm) per frame, NaN where no contact.
        total_press   : (T,)  float64 -- sum of pressure across the grid per frame.
        no_contact    : (T,)  bool    -- True where total_press < eps.
    """
    T = tactile_all.shape[0]
    # Convert from mmap (read-only) to float64 only when needed for sums; sums work fine on mmap.
    totals    = tactile_all.sum(axis=(1, 2)).astype(np.float64)            # (T,)
    row_sums  = tactile_all.sum(axis=2).astype(np.float64)                 # (T, 96)  marginal over cols
    col_sums  = tactile_all.sum(axis=1).astype(np.float64)                 # (T, 96)  marginal over rows
    indices   = np.arange(_GRID_N, dtype=np.float64)                       # 0..95

    no_contact = totals < 1e-6

    # Centroid in grid coordinates (0..95)
    safe_totals = np.where(no_contact, 1.0, totals)                        # avoid /0; we'll NaN later
    cop_row_grid = (row_sums * indices[None, :]).sum(axis=1) / safe_totals
    cop_col_grid = (col_sums * indices[None, :]).sum(axis=1) / safe_totals

    # Map to mm; column index => x (carpet length), row index => y (carpet width)
    cop_x_mm = _CARPET_MIN_MM + cop_col_grid * _GRID_STEP_MM
    cop_y_mm = _CARPET_MIN_MM + cop_row_grid * _GRID_STEP_MM

    cop_x_mm[no_contact] = np.nan
    cop_y_mm[no_contact] = np.nan

    return np.stack([cop_x_mm, cop_y_mm], axis=1), totals, no_contact


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute CoP from tactile cache and persistence baselines.')
    parser.add_argument('--tactile-cache', default=os.path.join(_OUT, 'tactile_all.npy'),
                        help='path to tactile_all.npy')
    parser.add_argument('--no-plot', action='store_true', help='skip matplotlib plots')
    args = parser.parse_args()

    if not os.path.exists(args.tactile_cache):
        raise SystemExit(f'no tactile cache at {args.tactile_cache} -- build it with '
                         f'python train/com/train_phase2_tactile.py (which calls extract_tactile_cache)')

    # ---- Load tactile cache and compute CoP ----
    print(f'loading tactile cache: {args.tactile_cache}')
    tactile_all = np.load(args.tactile_cache, mmap_mode='r')
    T = tactile_all.shape[0]
    assert tactile_all.shape == (T, 96, 96), f'unexpected tactile shape {tactile_all.shape}'
    print(f'  tactile_all: (T={T}, 96, 96)  dtype={tactile_all.dtype}')

    print('computing CoP for every frame...')
    cop_mm, total_press, no_contact = compute_cop_stack(tactile_all)
    print(f'  CoP_x mm: min={np.nanmin(cop_mm[:, 0]):.1f}  max={np.nanmax(cop_mm[:, 0]):.1f}  '
          f'mean={np.nanmean(cop_mm[:, 0]):.1f}')
    print(f'  CoP_y mm: min={np.nanmin(cop_mm[:, 1]):.1f}  max={np.nanmax(cop_mm[:, 1]):.1f}  '
          f'mean={np.nanmean(cop_mm[:, 1]):.1f}')
    print(f'  total_pressure: min={total_press.min():.2g}  max={total_press.max():.2g}  '
          f'mean={total_press.mean():.2g}')
    print(f'  frames with no contact: {no_contact.sum()} / {T} '
          f'({100 * no_contact.sum() / T:.3f}%)')

    # Frame-to-frame jump for CoP -- analog of the CoM trajectory-smoothness check
    valid_pair = ~no_contact[:-1] & ~no_contact[1:]
    cop_diff   = cop_mm[1:] - cop_mm[:-1]
    cop_jump   = np.linalg.norm(cop_diff[valid_pair], axis=1)
    print(f'  CoP frame-to-frame jump (valid pairs only):'
          f'  median={np.median(cop_jump):.1f} mm  mean={cop_jump.mean():.1f}  '
          f'p95={np.percentile(cop_jump, 95):.1f}  max={cop_jump.max():.1f}')

    # ---- Load CoM + session metadata for the same-split persistence comparison ----
    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)
    com_gt = com_results['com_gt']                                 # (T, 3) mm
    assert len(com_gt) == T

    with open(os.path.join(_TRAIN, 'singlePerson_test', 'log.p'), 'rb') as f:
        log = pickle.load(f)
    with open(os.path.join(_TRAIN, 'singlePerson_test', 'fileNames.p'), 'rb') as f:
        file_names = pickle.load(f)

    n_sessions = len(log) - 1
    _SUBJECT_RE = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
    subjects_per_sess = [_SUBJECT_RE.match(n).group(3) for n in file_names]

    # Same gt_outliers definition as phase1/phase2 (CoM-domain)
    _in_carpet  = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))

    # Build sample indices (matches phase1/phase2/phase2_v2 windowing exactly).
    # Extra filter: skip windows whose [t, t+HORIZON] contains a no_contact frame
    # (cannot compute persistence-on-CoP for those).
    centers_all   = []
    sources       = {'subject': [], 'session': [], 'split': []}
    n_dropped_for_no_contact = 0
    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        valid_t_phase_window = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t_phase_window))
        for i, t in enumerate(valid_t_phase_window):
            # also need no_contact == False over [t, t+HORIZON] inclusive
            if no_contact[t : t + HORIZON + 1].any():
                n_dropped_for_no_contact += 1
                continue
            centers_all.append(t)
            sources['subject'].append(subjects_per_sess[s])
            sources['session'].append(s)
            sources['split'].append('train' if i < n_train_s else 'test')
    centers = np.asarray(centers_all)
    meta = {k: np.asarray(v) for k, v in sources.items()}
    test_mask = meta['split'] == 'test'
    n_test = test_mask.sum()
    print(f'\nsamples: total={len(centers)}, train={(~test_mask).sum()}, test={n_test} '
          f'(dropped {n_dropped_for_no_contact} for no-contact within window)')

    test_centers = centers[test_mask]                                       # (n_test,)
    test_subj    = meta['subject'][test_mask]

    # ---- Persistence at horizons 1..HORIZON (frames) ----
    # For CoP (2-D): err = || cop(t+h) - cop(t) ||_2
    # For CoM 2-D (xy): same but on com_gt[:, :2]
    # For CoM 3-D: || com_gt(t+h) - com_gt(t) ||_2          (matches the existing project metric)
    horizons = np.arange(1, HORIZON + 1)                                    # 1..10

    cop_now    = cop_mm[test_centers]                                       # (n_test, 2)
    com_xy_now = com_gt[test_centers, :2]                                   # (n_test, 2)
    com_3d_now = com_gt[test_centers]                                       # (n_test, 3)

    cop_future    = np.stack([cop_mm[test_centers + h]        for h in horizons], axis=1)  # (n_test, H, 2)
    com_xy_future = np.stack([com_gt[test_centers + h, :2]    for h in horizons], axis=1)  # (n_test, H, 2)
    com_3d_future = np.stack([com_gt[test_centers + h, :]     for h in horizons], axis=1)  # (n_test, H, 3)

    # Persistence prediction = repeat current value across horizons
    err_cop    = np.linalg.norm(cop_future    - cop_now[:, None, :],    axis=2)            # (n_test, H)
    err_com_xy = np.linalg.norm(com_xy_future - com_xy_now[:, None, :], axis=2)
    err_com_3d = np.linalg.norm(com_3d_future - com_3d_now[:, None, :], axis=2)

    # Per-axis breakdown
    ax_err_cop    = np.abs(cop_future    - cop_now[:, None, :])             # (n_test, H, 2)
    ax_err_com_xy = np.abs(com_xy_future - com_xy_now[:, None, :])          # (n_test, H, 2)
    ax_err_com_3d = np.abs(com_3d_future - com_3d_now[:, None, :])          # (n_test, H, 3)

    def percentiles(arr_2d):
        """arr_2d: (n_test, H) -> dict of per-horizon median/mean/p95 and overall."""
        return {
            'per_horizon_median': [float(np.median(arr_2d[:, h])) for h in range(arr_2d.shape[1])],
            'per_horizon_mean':   [float(np.mean(arr_2d[:, h]))   for h in range(arr_2d.shape[1])],
            'per_horizon_p95':    [float(np.percentile(arr_2d[:, h], 95)) for h in range(arr_2d.shape[1])],
            'overall_median':     float(np.median(arr_2d)),
            'overall_mean':       float(np.mean(arr_2d)),
            'overall_p95':        float(np.percentile(arr_2d, 95)),
        }

    stats = {
        'persistence_cop_2d':       percentiles(err_cop),
        'persistence_com_xy_2d':    percentiles(err_com_xy),
        'persistence_com_3d':       percentiles(err_com_3d),
        'per_axis': {
            'cop_x':    [float(np.median(ax_err_cop[:, h, 0])) for h in range(HORIZON)],
            'cop_y':    [float(np.median(ax_err_cop[:, h, 1])) for h in range(HORIZON)],
            'com_x':    [float(np.median(ax_err_com_xy[:, h, 0])) for h in range(HORIZON)],
            'com_y':    [float(np.median(ax_err_com_xy[:, h, 1])) for h in range(HORIZON)],
            'com_z':    [float(np.median(ax_err_com_3d[:, h, 2])) for h in range(HORIZON)],
        },
    }

    # ---- Console summary ----
    bar = '=' * 78
    print(f'\n{bar}\nPERSISTENCE BASELINE: CoP vs CoM  (n_test = {n_test}, horizons 0.1..1.0 s)\n{bar}')

    print(f'\n{"":<6}  {"CoP 2D":>10}  {"CoM xy":>10}  {"CoM 3D":>10}     (median 3D Euclid mm, predict no change)')
    for i, h in enumerate(horizons):
        sec = h / 10.0
        print(f'  h={sec:.1f}s  '
              f'{stats["persistence_cop_2d"]["per_horizon_median"][i]:>10.1f}  '
              f'{stats["persistence_com_xy_2d"]["per_horizon_median"][i]:>10.1f}  '
              f'{stats["persistence_com_3d"]["per_horizon_median"][i]:>10.1f}')
    print(f'  {"overall":>6}  '
          f'{stats["persistence_cop_2d"]["overall_median"]:>10.1f}  '
          f'{stats["persistence_com_xy_2d"]["overall_median"]:>10.1f}  '
          f'{stats["persistence_com_3d"]["overall_median"]:>10.1f}     (median across all (sample, horizon) pairs)')

    print(f'\n{"":<6}  {"CoP 2D":>10}  {"CoM xy":>10}  {"CoM 3D":>10}     (mean 3D mm)')
    for i, h in enumerate(horizons):
        sec = h / 10.0
        print(f'  h={sec:.1f}s  '
              f'{stats["persistence_cop_2d"]["per_horizon_mean"][i]:>10.1f}  '
              f'{stats["persistence_com_xy_2d"]["per_horizon_mean"][i]:>10.1f}  '
              f'{stats["persistence_com_3d"]["per_horizon_mean"][i]:>10.1f}')

    print(f'\n{"":<6}  {"CoP 2D":>10}  {"CoM xy":>10}  {"CoM 3D":>10}     (p95 3D mm)')
    for i, h in enumerate(horizons):
        sec = h / 10.0
        print(f'  h={sec:.1f}s  '
              f'{stats["persistence_cop_2d"]["per_horizon_p95"][i]:>10.1f}  '
              f'{stats["persistence_com_xy_2d"]["per_horizon_p95"][i]:>10.1f}  '
              f'{stats["persistence_com_3d"]["per_horizon_p95"][i]:>10.1f}')

    print(f'\nPer-axis median error vs horizon (mm):')
    print(f'  axis   ' + '  '.join(f'h={h/10:.1f}s' for h in horizons))
    for axis_name, vals in stats['per_axis'].items():
        print(f'  {axis_name:<6} ' + '  '.join(f'{v:>6.1f}' for v in vals))

    # ---- Save results ----
    out_pickle = {
        'cop_mm':            cop_mm,               # (T, 2)
        'total_pressure':    total_press,          # (T,)
        'no_contact':        no_contact,           # (T,) bool
        'persistence_stats': stats,
        'n_test':            int(n_test),
        'horizons_frames':   horizons.tolist(),
        'HISTORY':           HISTORY,
        'HORIZON':           HORIZON,
        'SEED':              SEED,
    }
    out_path = os.path.join(_OUT, 'cop_results.p')
    with open(out_path, 'wb') as f:
        pickle.dump(out_pickle, f)
    print(f'\nSaved CoP + persistence stats to {out_path}')

    # ---- Plots ----
    if not args.no_plot:
        hs_sec = horizons / 10.0

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hs_sec, stats['persistence_cop_2d']['per_horizon_median'],
                marker='o', label='CoP (2-D, tactile-derived)')
        ax.plot(hs_sec, stats['persistence_com_xy_2d']['per_horizon_median'],
                marker='s', label='CoM xy (2-D, camera-derived)')
        ax.plot(hs_sec, stats['persistence_com_3d']['per_horizon_median'],
                marker='^', label='CoM 3-D (camera-derived)')
        ax.set(xlabel='forecast horizon (seconds)',
               ylabel='median Euclidean error (mm)',
               title='Persistence baseline: CoP vs CoM at 1-s horizon\n(higher = more room for a learned model to add value)')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT, 'cop_persistence.png'), dpi=100)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        for axis_name, vals in stats['per_axis'].items():
            ax.plot(hs_sec, vals, marker='o', label=axis_name)
        ax.set(xlabel='forecast horizon (seconds)',
               ylabel='median |error| (mm)',
               title='Persistence baseline: per-axis error vs horizon')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT, 'cop_persistence_per_axis.png'), dpi=100)
        plt.close()

        print(f'Plots saved to {_OUT}/cop_persistence{{,_per_axis}}.png')

    # ---- Decision hint ----
    print(f'\n{bar}\nDECISION HINT\n{bar}')
    cop_med = stats['persistence_cop_2d']['per_horizon_median'][-1]
    com_med = stats['persistence_com_3d']['per_horizon_median'][-1]
    com_xy_med = stats['persistence_com_xy_2d']['per_horizon_median'][-1]
    ratio = cop_med / com_xy_med if com_xy_med > 1e-6 else float('inf')
    print(f'persistence median at 1-s horizon: CoP-2D = {cop_med:.1f} mm,  CoM-xy-2D = {com_xy_med:.1f} mm,  CoM-3D = {com_med:.1f} mm')
    print(f'CoP-2D / CoM-xy-2D ratio = {ratio:.2f}')
    if ratio > 2.0:
        print('-> CoP has substantially more room than CoM at the persistence baseline.')
        print('   Pivot is worth pursuing: write train_phase2_cop.py and a tactile->CoP forecaster.')
    elif ratio < 0.7:
        print('-> CoP is harder to forecast (lower bar) than expected. Likely too slow on this dataset.')
        print('   Stay on CoM; proceed with gamma fusion.')
    else:
        print('-> CoP and CoM-xy persistence are similar. No headroom advantage to pivoting.')
        print('   Stay on CoM; proceed with gamma fusion.')


if __name__ == '__main__':
    main()

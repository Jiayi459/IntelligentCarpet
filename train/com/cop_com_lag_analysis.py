"""cop_com_lag_analysis.py — diagnostic #1: is tactile a *leading indicator* of CoM?

Scientific question:
    Does the center-of-pressure (CoP, computed directly from tactile) lead or
    follow center-of-mass (CoM, derived from camera keypoints)?

    The forecasting hypothesis the whole IntelligentCarpet line of work depends
    on is "small pressure shifts precede CoM displacement." If true, CoP leads
    CoM by some positive lag k > 0, and forecasting CoM from tactile is
    biomechanically meaningful. If CoP and CoM are synchronous (lag = 0), the
    forecasting framing is fundamentally wrong on this dataset / sampling rate
    -- tactile is a *measurement* of current weight distribution, not a
    predictor of future motion. (If CoP lags CoM, even worse: the camera-CoM
    pipeline is leading the tactile sensor, suggesting our "ground truth" has
    its own latency.)

What this script computes:
    For each session, compute Pearson correlation of *velocity* (first
    difference) between CoP and CoM at lags L ∈ [−10, +10] frames (−1.0 s to
    +1.0 s @ 10 fps). Sign convention:
        corr_L = Pearson( d_cop[t] , d_com[t + L] )
    so positive L means "if CoP at time t correlates with CoM L frames LATER,
    then CoP is a leading indicator." Peak lag at L > 0 supports the
    forecasting hypothesis; peak at L = 0 kills it.

    Two regimes are reported:
        (a) all valid frames (excludes NaN-CoP frames)
        (b) motion-only subset: frames where peak |d_com_xy| > MOTION_THRESHOLD

    Aggregation: per-session correlations, then mean ± std across sessions.

Outputs (under train/com/output/cop_com_lag/):
    lag_correlation.png            corr vs lag, per axis, both regimes
    metrics.json                   numerical values + peak lags + interpretation hint

Run (any cwd, needs both com_results.p and cop_results.p):
    python train/com/cop_com_lag_analysis.py
    python train/com/cop_com_lag_analysis.py --motion-threshold 10.0   # alt threshold
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


_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_OUT   = os.path.join(_HERE, 'output')
_LAGDIR = os.path.join(_OUT, 'cop_com_lag')


LAGS = list(range(-10, 11))                  # −1.0 s to +1.0 s in steps of 0.1 s
MIN_PAIRS_PER_LAG = 50                       # minimum valid pairs to trust a per-session correlation


def crosscorr_at_lag(x, y, lag):
    """Return (Pearson r, n) for corr(x[t], y[t+lag]).

    NaN-skipping; returns (nan, 0) if too few valid pairs or zero variance.
    """
    N = len(x)
    if lag == 0:
        xi, yi = x, y
    elif lag > 0:
        # need t + lag < N -> t < N - lag
        xi = x[:N - lag]
        yi = y[lag:]
    else:  # lag < 0
        # need t + lag >= 0 -> t >= -lag
        xi = x[-lag:]
        yi = y[:N + lag]
    mask = ~(np.isnan(xi) | np.isnan(yi))
    n = int(mask.sum())
    if n < MIN_PAIRS_PER_LAG:
        return np.nan, n
    xi, yi = xi[mask], yi[mask]
    if xi.std() < 1e-9 or yi.std() < 1e-9:
        return np.nan, n
    return float(np.corrcoef(xi, yi)[0, 1]), n


def main():
    parser = argparse.ArgumentParser(description='CoP-vs-CoM lag analysis (diagnostic #1).')
    parser.add_argument('--motion-threshold', type=float, default=5.0,
                        help='velocity threshold (mm/frame) defining the motion-only subset. '
                             'A frame counts as "motion" if max(|d_com_x|, |d_com_y|) > threshold.')
    args = parser.parse_args()

    os.makedirs(_LAGDIR, exist_ok=True)

    # ---- Load CoM ----
    com_pkl = os.path.join(_OUT, 'com_results.p')
    if not os.path.exists(com_pkl):
        raise SystemExit(f'no com_results.p at {com_pkl} -- run compute_com.py first.')
    with open(com_pkl, 'rb') as f:
        com_results = pickle.load(f)
    com_gt = com_results['com_gt']                                   # (T, 3) mm
    T = len(com_gt)
    print(f'com_gt: shape={com_gt.shape}')

    # ---- Load CoP ----
    cop_pkl = os.path.join(_OUT, 'cop_results.p')
    if not os.path.exists(cop_pkl):
        raise SystemExit(f'no cop_results.p at {cop_pkl} -- run compute_cop.py first.')
    with open(cop_pkl, 'rb') as f:
        cop_results = pickle.load(f)
    cop_mm = cop_results['cop_mm']                                   # (T, 2) mm, NaN at no-contact
    no_contact = cop_results.get('no_contact', np.isnan(cop_mm).any(axis=1))
    assert cop_mm.shape == (T, 2), f'shape mismatch: cop {cop_mm.shape}, T={T}'
    print(f'cop_mm: shape={cop_mm.shape}, no_contact frames = {int(no_contact.sum())}/{T}')

    # ---- Session boundaries ----
    log_pkl = os.path.join(_TRAIN, 'singlePerson_test', 'log.p')
    with open(log_pkl, 'rb') as f:
        log = pickle.load(f)
    n_sessions = len(log) - 1
    print(f'n_sessions = {n_sessions}')

    # ---- Per-session cross-correlation ----
    # For each session, compute correlations at each lag for each axis (x, y),
    # over both regimes (all frames / motion only). Store as ragged arrays so
    # we can aggregate per lag at the end.
    per_session = {
        regime: {ax: {L: [] for L in LAGS} for ax in 'xy'}
        for regime in ('all', 'motion')
    }
    per_session_n = {
        regime: {ax: {L: [] for L in LAGS} for ax in 'xy'}
        for regime in ('all', 'motion')
    }

    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        com_s = com_gt[a:b]                                           # (Ts, 3)
        cop_s = cop_mm[a:b]                                           # (Ts, 2)
        if (b - a) < 50:
            continue

        # Velocities (first difference in mm/frame)
        d_com = np.diff(com_s, axis=0)                                # (Ts-1, 3)
        d_cop = np.diff(cop_s, axis=0)                                # (Ts-1, 2)

        # Motion mask on the velocity series (length Ts-1)
        motion_mask = np.maximum(np.abs(d_com[:, 0]), np.abs(d_com[:, 1])) > args.motion_threshold

        for L in LAGS:
            for ax_i, ax_name in enumerate('xy'):
                # All frames
                r, n = crosscorr_at_lag(d_cop[:, ax_i], d_com[:, ax_i], L)
                if not np.isnan(r):
                    per_session['all'][ax_name][L].append(r)
                    per_session_n['all'][ax_name][L].append(n)

                # Motion only (apply mask before slicing — replace non-motion with NaN)
                cop_masked = d_cop[:, ax_i].copy().astype(np.float64)
                com_masked = d_com[:, ax_i].copy().astype(np.float64)
                cop_masked[~motion_mask] = np.nan
                com_masked[~motion_mask] = np.nan
                r_m, n_m = crosscorr_at_lag(cop_masked, com_masked, L)
                if not np.isnan(r_m):
                    per_session['motion'][ax_name][L].append(r_m)
                    per_session_n['motion'][ax_name][L].append(n_m)

    # ---- Aggregate (mean ± std across sessions) ----
    def aggregate(regime):
        agg = {}
        for ax in 'xy':
            vals = [per_session[regime][ax][L] for L in LAGS]
            mean = [float(np.mean(v)) if len(v) > 0 else float('nan') for v in vals]
            std  = [float(np.std(v))  if len(v) > 1 else 0.0           for v in vals]
            n_sess = [len(v) for v in vals]
            n_pairs_total = [int(sum(per_session_n[regime][ax][L])) for L in LAGS]
            # Peak lag = argmax(mean) -- but only over lags where we have data
            valid = [i for i, m in enumerate(mean) if not np.isnan(m)]
            if valid:
                peak_idx = max(valid, key=lambda i: mean[i])
                peak_lag = LAGS[peak_idx]
                peak_r   = mean[peak_idx]
            else:
                peak_lag, peak_r = None, float('nan')
            agg[ax] = {
                'corr_mean':      mean,
                'corr_std':       std,
                'n_sessions':     n_sess,
                'n_pairs_total':  n_pairs_total,
                'peak_lag':       peak_lag,
                'peak_corr':      peak_r,
            }
        return agg

    agg_all = aggregate('all')
    agg_motion = aggregate('motion')

    # ---- Interpretation hint ----
    def interpret(agg, regime):
        lines = []
        for ax in 'xy':
            pl = agg[ax]['peak_lag']
            pr = agg[ax]['peak_corr']
            if pl is None:
                lines.append(f'{regime}/{ax}: no valid data')
                continue
            if pl > 0:
                tag = f'CoP LEADS CoM by {pl} frame{"s" if pl > 1 else ""} ({pl * 0.1:.1f} s)'
            elif pl < 0:
                tag = f'CoM LEADS CoP by {-pl} frame{"s" if -pl > 1 else ""} ({-pl * 0.1:.1f} s)'
            else:
                tag = 'CoP and CoM are SYNCHRONOUS (lag 0)'
            lines.append(f'{regime}/{ax}: peak corr = {pr:.3f} at lag {pl:+d} -> {tag}')
        return lines

    summary_lines = []
    summary_lines.append('=' * 78)
    summary_lines.append('CoP-vs-CoM LAG ANALYSIS — diagnostic #1 result')
    summary_lines.append('=' * 78)
    summary_lines.append(f'motion threshold for motion-only regime: > {args.motion_threshold:.1f} mm/frame')
    summary_lines.append('')
    summary_lines.append('-- ALL FRAMES --')
    summary_lines.extend(interpret(agg_all, 'all'))
    summary_lines.append('')
    summary_lines.append('-- MOTION ONLY --')
    summary_lines.extend(interpret(agg_motion, 'motion'))
    summary_lines.append('')
    summary_lines.append('Reading:')
    summary_lines.append('  positive peak lag -> CoP precedes CoM -> tactile IS a leading indicator')
    summary_lines.append('  zero peak lag     -> synchronous       -> tactile is contemporaneous (forecasting hypothesis weak)')
    summary_lines.append('  negative peak lag -> CoM precedes CoP  -> camera pipeline leads tactile (bias)')
    summary_lines.append('=' * 78)
    for line in summary_lines:
        print(line)

    # ---- Save metrics ----
    out = {
        'lags':                    LAGS,
        'fps':                     10,
        'motion_threshold_mm_per_frame': args.motion_threshold,
        'n_sessions_total':        n_sessions,
        'no_contact_frames':       int(no_contact.sum()),
        'no_contact_fraction':     float(no_contact.sum() / T),
        'all_frames':              agg_all,
        'motion_only':             agg_motion,
        'summary':                 summary_lines,
    }
    with open(os.path.join(_LAGDIR, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved {os.path.join(_LAGDIR, "metrics.json")}')

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, regime, agg in [(0, 'all frames', agg_all), (1, 'motion only', agg_motion)]:
        for row, ax_name in enumerate('xy'):
            ax = axes[row, col]
            mean = np.asarray(agg[ax_name]['corr_mean'])
            std  = np.asarray(agg[ax_name]['corr_std'])
            ax.errorbar(LAGS, mean, yerr=std, marker='o', capsize=3,
                        color=('tab:blue' if col == 0 else 'tab:red'))
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            pl = agg[ax_name]['peak_lag']
            if pl is not None:
                ax.axvline(pl, color='black', linestyle='--', alpha=0.6,
                            label=f'peak lag = {pl:+d}')
                ax.legend(fontsize='small')
            ax.set_title(f'{regime} / {ax_name}-axis  (peak corr = {agg[ax_name]["peak_corr"]:.3f})',
                         fontsize=10)
            if row == 1:
                ax.set_xlabel('lag L (frames)  -- corr(d_cop[t], d_com[t+L])')
            if col == 0:
                ax.set_ylabel('Pearson r')
            ax.grid(alpha=0.3)
    fig.suptitle(f'CoP-CoM velocity cross-correlation  (10 fps; positive lag = CoP leads CoM)',
                 fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(_LAGDIR, 'lag_correlation.png')
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'saved {out_png}')


if __name__ == '__main__':
    main()

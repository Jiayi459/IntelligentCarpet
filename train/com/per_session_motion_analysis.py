"""per_session_motion_analysis.py — characterize motion content of the dataset.

Scientific question (settles the repeatability vs sampling-rate debate):
    Is motion structurally absent across most sessions (-> sampling rate is
    the primary bottleneck) OR is motion present but heterogeneous across
    sessions (-> repeatability is the primary bottleneck) OR both?

What this script computes per session:
    - session length (frames)
    - velocity percentiles (50, 75, 90, 95, 99) on |d_CoM_xy|
    - motion-frame fraction at multiple thresholds (5, 10, 20 mm/frame)
    - number of detected transitions: contiguous runs of motion (>= 3 frames
      at threshold 10 mm/frame), preceded by >= 10 static frames
    - subject identity, train/test classification

Aggregations:
    - per-subject summary (across that subject's sessions)
    - dataset-wide histograms
    - cross-session variability of motion content

Outputs (under train/com/output/per_session_motion/):
    per_session_motion.csv          one row per session
    motion_summary.json             aggregates + interpretation hint
    motion_fraction_distribution.png   histogram of motion fraction across sessions
    transitions_per_session.png        bar chart per session (ordered, colored by subject)
    velocity_box_per_subject.png       box plot of session velocities, per subject

Run (no GPU needed; uses only com_results.p):
    python train/com/per_session_motion_analysis.py
"""

import os
import sys
import json
import csv
import pickle
import re
import argparse
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_OUT   = os.path.join(_HERE, 'output')
_MOTION = os.path.join(_OUT, 'per_session_motion')

HISTORY = 100
HORIZON = 10
SEED    = 42

MOTION_THRESHOLDS_MM_PER_FRAME = [5.0, 10.0, 20.0, 40.0]
# Transition definition: contiguous run of >= MIN_RUN frames at velocity
# > TRANSITION_THRESH, preceded by >= MIN_STATIC frames of static.
TRANSITION_THRESH = 10.0
MIN_RUN           = 3
MIN_STATIC        = 10


def count_transitions(velocities, thresh=TRANSITION_THRESH, min_run=MIN_RUN, min_static=MIN_STATIC):
    """Return the number of "transition" events in a 1-D velocity series.

    A transition starts when velocity rises above `thresh` for at least
    `min_run` consecutive frames, AND the preceding `min_static` frames were
    all below `thresh` (so we don't count continuing motion as new transitions).
    """
    if len(velocities) < min_static + min_run:
        return 0
    is_motion = velocities > thresh
    count = 0
    i = min_static
    while i <= len(velocities) - min_run:
        # check the candidate run
        if is_motion[i:i + min_run].all():
            # check the preceding static window
            if not is_motion[i - min_static:i].any():
                count += 1
                # skip past this run; consume motion until it ends
                j = i + min_run
                while j < len(velocities) and is_motion[j]:
                    j += 1
                i = j + 1
                continue
        i += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='Per-session motion-content analysis.')
    parser.add_argument('--output-dir', type=str, default=_MOTION)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load CoM + session metadata ----
    com_pkl = os.path.join(_OUT, 'com_results.p')
    if not os.path.exists(com_pkl):
        raise SystemExit(f'no com_results.p at {com_pkl}')
    with open(com_pkl, 'rb') as f:
        com_results = pickle.load(f)
    com_gt = com_results['com_gt']                                    # (T, 3) mm
    T = len(com_gt)

    log_pkl = os.path.join(_TRAIN, 'singlePerson_test', 'log.p')
    fn_pkl  = os.path.join(_TRAIN, 'singlePerson_test', 'fileNames.p')
    with open(log_pkl, 'rb') as f:
        log = pickle.load(f)
    with open(fn_pkl, 'rb') as f:
        file_names = pickle.load(f)
    n_sessions = len(log) - 1
    _SUBJECT_RE = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
    subjects_per_sess = [_SUBJECT_RE.match(n).group(3) for n in file_names]
    rounds_per_sess   = [_SUBJECT_RE.match(n).group(4) for n in file_names]
    dates_per_sess    = [_SUBJECT_RE.match(n).group(2) for n in file_names]

    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))

    # ---- Per-session analysis ----
    rows = []                                                           # for CSV
    per_subject_velocities = defaultdict(list)                          # subject -> list of d_com_xy magnitudes
    motion_fractions_at_thresh = {t: [] for t in MOTION_THRESHOLDS_MM_PER_FRAME}
    transitions_per_session = []

    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        com_sess = com_gt[a:b]                                          # (Ts, 3)
        if len(com_sess) < 11:
            continue
        out_mask = gt_outliers[a:b]                                     # (Ts,) bool
        # Compute velocities; treat outlier frames as zero motion (don't count)
        d_xy = np.diff(com_sess[:, :2], axis=0)                         # (Ts-1, 2)
        v_xy = np.linalg.norm(d_xy, axis=1)                             # (Ts-1,) mm/frame
        # Mask out velocities that touch an outlier frame
        out_either = out_mask[:-1] | out_mask[1:]
        v_xy[out_either] = 0.0

        # Velocity percentiles (only on non-outlier-touching frames for honesty)
        v_clean = v_xy[~out_either]
        pct = {}
        if len(v_clean) >= 10:
            for p in (50, 75, 90, 95, 99):
                pct[p] = float(np.percentile(v_clean, p))
        else:
            for p in (50, 75, 90, 95, 99):
                pct[p] = float('nan')

        # Motion fraction at each threshold (over non-outlier frames)
        mf = {}
        for thr in MOTION_THRESHOLDS_MM_PER_FRAME:
            if len(v_clean) > 0:
                mf[thr] = float((v_clean > thr).sum() / len(v_clean))
            else:
                mf[thr] = float('nan')
            motion_fractions_at_thresh[thr].append(mf[thr] if not np.isnan(mf[thr]) else 0.0)

        # Transition count at the TRANSITION_THRESH definition
        n_trans = count_transitions(v_xy)

        subj = subjects_per_sess[s]
        rd   = rounds_per_sess[s]
        dt   = dates_per_sess[s]

        # Train/test classification on the FORECASTING samples that fall in this session
        # (matches the same 70/30 chronological per-session split as forecasting scripts).
        # Compute the sample-center range that falls in this session:
        # valid centers: range(a + HISTORY - 1, b - HORIZON), then 70/30 split.
        valid_t = [t for t in range(a + HISTORY - 1, b - HORIZON)
                   if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()]
        n_valid = len(valid_t)
        n_train = int(0.7 * n_valid)
        n_test  = n_valid - n_train

        per_subject_velocities[subj].extend(v_clean.tolist())
        transitions_per_session.append(n_trans)

        rows.append({
            'session':            s,
            'subject':            subj,
            'date':               dt,
            'round':              rd,
            'frames':             int(b - a),
            'frames_outlier':     int(out_mask.sum()),
            'velocity_p50':       pct[50],
            'velocity_p75':       pct[75],
            'velocity_p90':       pct[90],
            'velocity_p95':       pct[95],
            'velocity_p99':       pct[99],
            'motion_frac_thr5':   mf[5.0],
            'motion_frac_thr10':  mf[10.0],
            'motion_frac_thr20':  mf[20.0],
            'motion_frac_thr40':  mf[40.0],
            'n_transitions':      n_trans,
            'n_forecast_samples_train': n_train,
            'n_forecast_samples_test':  n_test,
        })

    # ---- Save CSV ----
    csv_path = os.path.join(args.output_dir, 'per_session_motion.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {csv_path}  ({len(rows)} sessions)')

    # ---- Aggregate stats ----
    n_subjects = len(per_subject_velocities)
    summary = {
        'n_sessions':              len(rows),
        'n_subjects':              n_subjects,
        'motion_thresholds_used':  MOTION_THRESHOLDS_MM_PER_FRAME,
        'transition_def':          {
            'velocity_threshold': TRANSITION_THRESH,
            'min_run_frames':     MIN_RUN,
            'min_static_frames':  MIN_STATIC,
        },
        'aggregate': {
            'transitions_per_session_mean':   float(np.mean(transitions_per_session)),
            'transitions_per_session_median': float(np.median(transitions_per_session)),
            'transitions_per_session_max':    int(np.max(transitions_per_session)),
            'sessions_with_zero_transitions': int(sum(1 for n in transitions_per_session if n == 0)),
            'total_transitions':              int(sum(transitions_per_session)),
        },
        'motion_fraction_dataset_wide': {
            f'thr{int(thr)}': {
                'mean':   float(np.mean(motion_fractions_at_thresh[thr])),
                'median': float(np.median(motion_fractions_at_thresh[thr])),
                'p95':    float(np.percentile(motion_fractions_at_thresh[thr], 95)),
                'sessions_below_1pct': int(sum(1 for f in motion_fractions_at_thresh[thr] if f < 0.01)),
            }
            for thr in MOTION_THRESHOLDS_MM_PER_FRAME
        },
        'per_subject': {},
    }

    for subj, v_list in per_subject_velocities.items():
        if len(v_list) == 0:
            continue
        va = np.asarray(v_list)
        # Count sessions for this subject
        n_sess_subj = sum(1 for r in rows if r['subject'] == subj)
        n_trans_subj = sum(r['n_transitions'] for r in rows if r['subject'] == subj)
        summary['per_subject'][subj] = {
            'n_sessions':         n_sess_subj,
            'total_frames':       int(len(va)),
            'velocity_p50':       float(np.percentile(va, 50)),
            'velocity_p95':       float(np.percentile(va, 95)),
            'velocity_p99':       float(np.percentile(va, 99)),
            'motion_frac_thr10':  float((va > 10.0).sum() / len(va)),
            'n_transitions':      int(n_trans_subj),
        }

    # ---- Interpretation hint ----
    # Bottleneck verdict heuristics:
    #   - if MEDIAN motion frac at thr=10 is < 0.05 across sessions -> motion absent (sampling rate / target framing wins)
    #   - if motion frac varies enormously across sessions (e.g. p95/median > 5x), repeatability matters
    median_mf10 = summary['motion_fraction_dataset_wide']['thr10']['median']
    p95_mf10    = summary['motion_fraction_dataset_wide']['thr10']['p95']
    sessions_dead = summary['motion_fraction_dataset_wide']['thr10']['sessions_below_1pct']
    n_sess = summary['n_sessions']
    verdict_lines = []
    verdict_lines.append('=' * 78)
    verdict_lines.append('MOTION-CONTENT VERDICT')
    verdict_lines.append('=' * 78)
    verdict_lines.append(f'sessions analyzed: {n_sess}; subjects: {n_subjects}')
    verdict_lines.append(f'motion fraction at >10 mm/frame: median={median_mf10:.3f}  p95={p95_mf10:.3f}')
    verdict_lines.append(f'sessions with <1% motion frames: {sessions_dead}/{n_sess}')
    verdict_lines.append(f'transitions detected: total={summary["aggregate"]["total_transitions"]}, '
                          f'sessions with zero transitions: {summary["aggregate"]["sessions_with_zero_transitions"]}/{n_sess}')
    verdict_lines.append('')
    if median_mf10 < 0.05 and sessions_dead > n_sess * 0.3:
        verdict_lines.append('-> Motion is structurally ABSENT across most of the dataset.')
        verdict_lines.append('   Sampling-rate / target-framing is the dominant bottleneck.')
    elif p95_mf10 > median_mf10 * 3 and median_mf10 > 0.01:
        verdict_lines.append('-> Motion is HIGHLY HETEROGENEOUS across sessions.')
        verdict_lines.append('   Some sessions have rich motion; others have none.')
        verdict_lines.append('   Pattern repeatability across sessions is a real concern.')
        verdict_lines.append('   Path B (event detection on motion-rich sessions only) is viable.')
    else:
        verdict_lines.append('-> Motion content is moderate and roughly homogeneous.')
        verdict_lines.append('   Neither extreme verdict; both factors plausibly contribute.')
    verdict_lines.append('=' * 78)
    summary['verdict_lines'] = verdict_lines
    for line in verdict_lines:
        print(line)

    # ---- Save JSON ----
    json_path = os.path.join(args.output_dir, 'motion_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {json_path}')

    # ---- Plots ----
    # 1. Histogram of motion fraction at thr=10 across sessions
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, thr in zip(axes.flat, MOTION_THRESHOLDS_MM_PER_FRAME):
        fracs = np.asarray(motion_fractions_at_thresh[thr])
        ax.hist(fracs, bins=30, color='tab:blue', alpha=0.85, edgecolor='black', linewidth=0.3)
        ax.axvline(np.median(fracs), color='tab:red', linestyle='--',
                    label=f'median={np.median(fracs):.3f}')
        ax.set(xlabel=f'fraction of frames with |d_CoM_xy| > {int(thr)} mm/frame',
               ylabel='# sessions',
               title=f'motion @ > {int(thr)} mm/frame')
        ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    fig.suptitle(f'Distribution of motion content across {n_sess} sessions', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'motion_fraction_distribution.png'), dpi=100)
    plt.close()

    # 2. Transitions per session, bar chart ordered by count
    sorted_idx = np.argsort(transitions_per_session)[::-1]
    sorted_counts = [transitions_per_session[i] for i in sorted_idx]
    sorted_subjects = [rows[i]['subject'] for i in sorted_idx]
    subject_colors = {subj: plt.cm.tab20(i % 20) for i, subj in enumerate(sorted(set(sorted_subjects)))}
    colors = [subject_colors[s] for s in sorted_subjects]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(sorted_counts)), sorted_counts, color=colors)
    ax.set(xlabel='session (ordered by transition count)', ylabel='# transitions',
           title=f'Transitions per session (def: >={MIN_RUN}-frame motion run preceded by '
                 f'>={MIN_STATIC} static frames, vel threshold {TRANSITION_THRESH} mm/frame)')
    # subject legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=col, label=subj) for subj, col in subject_colors.items()]
    ax.legend(handles=legend_patches, fontsize='small', ncol=2, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'transitions_per_session.png'), dpi=100)
    plt.close()

    # 3. Velocity box plot per subject
    subjects_sorted = sorted(per_subject_velocities.keys())
    data = [np.asarray(per_subject_velocities[s]) for s in subjects_sorted]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    bp = ax.boxplot(data, labels=subjects_sorted, showfliers=False, whis=(5, 95))
    ax.set(xlabel='subject', ylabel='|d_CoM_xy| (mm/frame)',
           title='Velocity distribution per subject (whiskers = 5-95 pct, fliers hidden)')
    ax.axhline(TRANSITION_THRESH, color='tab:red', linestyle='--', alpha=0.6,
                label=f'transition threshold = {TRANSITION_THRESH} mm/frame')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'velocity_box_per_subject.png'), dpi=100)
    plt.close()

    print(f'\nplots saved to {args.output_dir}')


if __name__ == '__main__':
    main()

"""path_b_yiyueluo_multiseed.py — robust evaluation of Path B on YiyueLuo 104.

Single-seed AUROC 0.752 on the prior run is suggestive but noisy (only ~4 test
events spread across 1,468 samples). This script tightens the headline by:

    1. Running the classifier with N seeds (default 5: 42-46). Encoder hidden
       states are computed ONCE (frozen encoder is deterministic), then a fresh
       Linear(128, 1) head is initialized per seed with a fresh val split.
    2. Reporting per-seed AUROC mean +/- std AND combining scores across seeds
       for an "ensemble" PR / ROC curve and optimal-F1 threshold.
    3. Computing AUPRC -- the right rare-event metric (penalizes false
       positives in a way AUROC doesn't).
    4. Producing PR-curve, ROC-curve, and per-seed AUROC bar-chart plots.

Outputs (under train/com/output/path_b_yiyueluo_multiseed/):
    metrics.json
    auroc_per_seed.png      bar chart of per-seed test AUROCs
    pr_curve.png            ensemble + per-seed PR curves
    roc_curve.png           ensemble + per-seed ROC curves
    score_distribution.png  test-score histogram, separated by true label

Run (needs tactile cache + epsilon checkpoint on CRC):
    python train/com/path_b_yiyueluo_multiseed.py
    python train/com/path_b_yiyueluo_multiseed.py --seeds 42 43 44 45 46 47 48 --epochs 400
"""

import os
import sys
import json
import pickle
import re
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_OUT   = os.path.join(_HERE, 'output')
_PATHB_MS = os.path.join(_OUT, 'path_b_yiyueluo_multiseed')

_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')
_STATS     = os.path.join(_OUT, 'tactile_stats.json')

sys.path.insert(0, os.path.join(_TRAIN, 'tactile_direct'))


HISTORY = 100
HORIZON = 10
TARGET_SESSION_INDEX = 104

TRANSITION_THRESH = 10.0
MIN_RUN           = 3
MIN_STATIC        = 10

LR             = 1e-3
DEFAULT_EPOCHS = 400          # higher than path_b's 200 -- val was still decreasing
BATCH          = 32
VAL_FRAC       = 0.10


def detect_transition_starts(velocities, thresh=TRANSITION_THRESH,
                              min_run=MIN_RUN, min_static=MIN_STATIC):
    if len(velocities) < min_static + min_run:
        return []
    is_motion = velocities > thresh
    starts = []
    i = min_static
    while i <= len(velocities) - min_run:
        if is_motion[i:i + min_run].all():
            if not is_motion[i - min_static:i].any():
                starts.append(i)
                j = i + min_run
                while j < len(velocities) and is_motion[j]:
                    j += 1
                i = j + 1
                continue
        i += 1
    return starts


def auroc(y, scores):
    order = np.argsort(scores)
    y_sorted = y[order]
    n_pos = int(y_sorted.sum()); n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = np.arange(1, len(y_sorted) + 1)
    sum_pos_ranks = float(ranks[y_sorted == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def precision_recall_curve(y, scores):
    """Return (precisions, recalls, thresholds) sorted by descending threshold."""
    order = np.argsort(-scores)                                         # high to low
    y_sorted = y[order]
    scores_sorted = scores[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos = int(y_sorted.sum())
    precisions = tp / np.maximum(tp + fp, 1e-9)
    recalls    = tp / max(n_pos, 1)
    return precisions, recalls, scores_sorted


def auprc(precisions, recalls):
    """Average precision = sum_i (R_i - R_{i-1}) * P_i  (sklearn convention).

    Inputs come from precision_recall_curve sorted by descending score, so
    recalls is already monotonically non-decreasing.
    """
    r_prev = np.concatenate([[0.0], recalls[:-1]])
    return float(((recalls - r_prev) * precisions).sum())


def roc_curve(y, scores):
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos = int(y_sorted.sum()); n_neg = len(y_sorted) - n_pos
    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    return fpr, tpr


def main():
    parser = argparse.ArgumentParser(description='Path B multi-seed + threshold tune on YiyueLuo 104.')
    parser.add_argument('--epsilon-checkpoint',
                        default=os.path.join(_OUT, 'phase2_epsilon', 'dynamics_model.pt'),
                        help='path to epsilon dynamics_model.pt')
    parser.add_argument('--output-dir', type=str, default=_PATHB_MS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device     : {device}')
    print(f'output_dir : {args.output_dir}')
    print(f'seeds      : {args.seeds}')
    print(f'epochs     : {args.epochs}')

    # ---- Inputs ----
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY}')
    if not os.path.exists(_STATS):
        raise SystemExit(f'no tactile stats at {_STATS}')
    if not os.path.exists(args.epsilon_checkpoint):
        raise SystemExit(f'no epsilon checkpoint at {args.epsilon_checkpoint}')

    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    with open(_STATS) as f:
        s = json.load(f)
    tactile_mean = float(s['tactile_mean']); tactile_std = float(s['tactile_std'])

    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)
    com_gt = com_results['com_gt']

    with open(os.path.join(_TRAIN, 'singlePerson_test', 'log.p'), 'rb') as f:
        log = pickle.load(f)

    a, b = log[TARGET_SESSION_INDEX], log[TARGET_SESSION_INDEX + 1]
    print(f'session {TARGET_SESSION_INDEX}: global frames [{a}, {b}); {b - a} frames')

    # ---- Detect transitions ----
    com_sess = com_gt[a:b]
    v_xy = np.linalg.norm(np.diff(com_sess[:, :2], axis=0), axis=1)
    starts_local = detect_transition_starts(v_xy)
    transition_frames_global = [int(a + k + 1) for k in starts_local]
    print(f'  transitions: {len(transition_frames_global)} at {transition_frames_global}')

    # ---- Sample centers + labels ----
    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))
    valid_centers = [t for t in range(a + HISTORY - 1, b - HORIZON)
                     if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()]
    valid_centers = np.asarray(valid_centers)
    n_train = int(0.7 * len(valid_centers))
    train_centers = valid_centers[:n_train]
    test_centers  = valid_centers[n_train:]

    event_set = set(transition_frames_global)
    def label(t):
        return int(any((t + k) in event_set for k in range(1, HORIZON + 1)))
    y_train = np.asarray([label(int(t)) for t in train_centers], dtype=np.float32)
    y_test  = np.asarray([label(int(t)) for t in test_centers],  dtype=np.float32)
    print(f'  samples: train={len(train_centers)} (pos={int(y_train.sum())}), '
          f'test={len(test_centers)} (pos={int(y_test.sum())})')

    # ---- Load encoder, encode hidden states ONCE ----
    from model_epsilon import DynamicsModel
    dyn = DynamicsModel().to(device)
    ck = torch.load(args.epsilon_checkpoint, map_location=device, weights_only=False)
    dyn.load_state_dict(ck['dynamics'])
    for p in dyn.parameters():
        p.requires_grad_(False)
    dyn.eval()

    def encode_centers(centers):
        H = np.zeros((len(centers), 128), dtype=np.float32)
        with torch.no_grad():
            for i0 in range(0, len(centers), BATCH):
                i1 = min(i0 + BATCH, len(centers))
                w = np.stack([
                    (np.asarray(tactile_all[t - HISTORY + 1 : t + 1]) - tactile_mean) / tactile_std
                    for t in centers[i0:i1]
                ], axis=0).astype(np.float32)
                x = torch.from_numpy(w).to(device)
                H[i0:i1] = dyn.encode_history(x).cpu().numpy()
        return H

    print('encoding hidden states (one-time; frozen)...')
    t0 = time.time()
    H_train_full = encode_centers(train_centers)
    H_test       = encode_centers(test_centers)
    print(f'  encoded {len(H_train_full)} train + {len(H_test)} test in {time.time() - t0:.1f}s')

    H_test_t = torch.from_numpy(H_test).to(device)

    # ---- Multi-seed training ----
    per_seed_results = []
    test_scores_per_seed = []                                            # (n_seeds, n_test)
    for seed in args.seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        perm = np.random.permutation(len(H_train_full))
        n_val = max(1, int(len(H_train_full) * VAL_FRAC))
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        H_tr = H_train_full[tr_idx]; y_tr = y_train[tr_idx]
        H_val = H_train_full[val_idx]; y_val = y_train[val_idx]

        head = nn.Linear(128, 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=LR)
        pos_weight = max(1.0, (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        H_tr_t  = torch.from_numpy(H_tr).to(device)
        y_tr_t  = torch.from_numpy(y_tr).to(device)
        H_val_t = torch.from_numpy(H_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)

        best_val = float('inf'); best_state = None
        for epoch in range(args.epochs):
            head.train()
            perm_ep = torch.randperm(len(H_tr_t), device=device)
            for i0 in range(0, len(H_tr_t), BATCH):
                idx = perm_ep[i0:i0 + BATCH]
                logit = head(H_tr_t[idx]).squeeze(-1)
                loss = criterion(logit, y_tr_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                v_loss = float(criterion(head(H_val_t).squeeze(-1), y_val_t).item())
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)

        head.eval()
        with torch.no_grad():
            logits = head(H_test_t).squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        au = auroc(y_test, probs)
        precisions, recalls, _ = precision_recall_curve(y_test, probs)
        ap = auprc(precisions, recalls)
        per_seed_results.append({
            'seed':       int(seed),
            'auroc':      float(au) if not np.isnan(au) else None,
            'auprc':      float(ap),
            'best_val':   best_val,
            'pos_weight': float(pos_weight),
        })
        test_scores_per_seed.append(probs)
        print(f'  seed {seed}: AUROC={au:.4f}  AUPRC={ap:.4f}  best_val={best_val:.4f}')

    # ---- Aggregate ----
    aurocs = np.asarray([r['auroc'] for r in per_seed_results if r['auroc'] is not None])
    auprcs = np.asarray([r['auprc'] for r in per_seed_results])
    test_scores_per_seed = np.stack(test_scores_per_seed, axis=0)        # (n_seeds, n_test)
    ensemble_scores = test_scores_per_seed.mean(axis=0)                  # (n_test,)

    au_ensemble = auroc(y_test, ensemble_scores)
    p_ens, r_ens, thr_ens = precision_recall_curve(y_test, ensemble_scores)
    ap_ensemble = auprc(p_ens, r_ens)
    # F1 at each threshold
    f1 = 2 * p_ens * r_ens / np.maximum(p_ens + r_ens, 1e-9)
    best_f1_idx = int(np.argmax(f1))
    best_f1    = float(f1[best_f1_idx])
    best_thr   = float(thr_ens[best_f1_idx])
    best_p     = float(p_ens[best_f1_idx])
    best_r     = float(r_ens[best_f1_idx])

    # Baseline F1: predict 1 for everything (recall=1, precision = pos_rate)
    pos_rate = float(y_test.mean())
    f1_baseline_all_one = 2 * pos_rate * 1.0 / (pos_rate + 1.0)

    bar = '=' * 78
    print(f'\n{bar}\nPATH B MULTI-SEED RESULTS — YiyueLuo session {TARGET_SESSION_INDEX}\n{bar}')
    print(f'  seeds                : {args.seeds}')
    print(f'  per-seed AUROC       : mean = {aurocs.mean():.3f}  std = {aurocs.std():.3f}')
    print(f'                          values = {[round(x, 3) for x in aurocs.tolist()]}')
    print(f'  per-seed AUPRC       : mean = {auprcs.mean():.3f}  std = {auprcs.std():.3f}')
    print(f'                          values = {[round(x, 3) for x in auprcs.tolist()]}')
    print(f'  ensemble AUROC       : {au_ensemble:.3f}')
    print(f'  ensemble AUPRC       : {ap_ensemble:.3f}   (random baseline = {pos_rate:.3f})')
    print(f'  ensemble best F1     : {best_f1:.3f} at threshold {best_thr:.3f}')
    print(f'    -> precision        : {best_p:.3f}')
    print(f'    -> recall           : {best_r:.3f}')
    print(f'  always-predict-1 F1  : {f1_baseline_all_one:.3f}  (sanity baseline)')
    print()
    if aurocs.mean() - 1.96 * aurocs.std() / np.sqrt(len(aurocs)) > 0.65:
        print('  Reading: AUROC is robustly above 0.65 across seeds (95% CI of mean).')
        print('           Path D direction (dataset switch) is well-supported.')
    elif aurocs.mean() > 0.60:
        print('  Reading: mean AUROC ~0.6+, but variance leaves the result uncertain.')
        print('           More seeds or longer training might help; dataset switch still defensible.')
    else:
        print('  Reading: mean AUROC near 0.5; the single-seed 0.752 was a fluke or due to a single')
        print('           lucky head initialization. Reconsider the architecture or the data.')

    # ---- Save metrics ----
    out = {
        'session_index':           TARGET_SESSION_INDEX,
        'n_transitions':           len(transition_frames_global),
        'transition_frames_global': transition_frames_global,
        'n_train_samples':         int(len(train_centers)),
        'n_test_samples':          int(len(test_centers)),
        'n_pos_train':             int(y_train.sum()),
        'n_pos_test':              int(y_test.sum()),
        'seeds':                   args.seeds,
        'epochs':                  args.epochs,
        'per_seed':                per_seed_results,
        'aggregate': {
            'auroc_mean':          float(aurocs.mean()),
            'auroc_std':           float(aurocs.std()),
            'auroc_values':        aurocs.tolist(),
            'auprc_mean':          float(auprcs.mean()),
            'auprc_std':           float(auprcs.std()),
            'auprc_values':        auprcs.tolist(),
            'ensemble_auroc':      float(au_ensemble),
            'ensemble_auprc':      float(ap_ensemble),
            'baseline_auprc_random': pos_rate,
            'best_f1':             best_f1,
            'best_f1_threshold':   best_thr,
            'best_f1_precision':   best_p,
            'best_f1_recall':      best_r,
            'always_one_f1':       float(f1_baseline_all_one),
        },
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved metrics.json')

    # ---- Plots ----
    # 1. AUROC per seed bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seeds_x = [r['seed'] for r in per_seed_results]
    aurocs_x = [r['auroc'] for r in per_seed_results]
    auprcs_x = [r['auprc'] for r in per_seed_results]
    x = np.arange(len(seeds_x))
    ax.bar(x - 0.2, aurocs_x, width=0.4, label='AUROC', color='tab:blue')
    ax.bar(x + 0.2, auprcs_x, width=0.4, label='AUPRC', color='tab:orange')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='AUROC random=0.5')
    ax.axhline(pos_rate, color='tab:orange', linestyle=':', alpha=0.5,
                label=f'AUPRC random={pos_rate:.3f}')
    ax.set_xticks(x); ax.set_xticklabels(seeds_x)
    ax.set(xlabel='seed', ylabel='score',
           title=f'Per-seed AUROC and AUPRC (mean AUROC = {aurocs.mean():.3f} +/- {aurocs.std():.3f})')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'auroc_per_seed.png'), dpi=100)
    plt.close()

    # 2. PR curve (ensemble + per-seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, (seed, probs) in enumerate(zip(seeds_x, test_scores_per_seed)):
        p_k, r_k, _ = precision_recall_curve(y_test, probs)
        ax.plot(r_k, p_k, alpha=0.3, color='gray', label=f'seed {seed}' if k == 0 else None)
    ax.plot(r_ens, p_ens, color='tab:blue', linewidth=2.0,
            label=f'ensemble (AUPRC = {ap_ensemble:.3f})')
    ax.axhline(pos_rate, color='tab:red', linestyle=':',
                label=f'random baseline = {pos_rate:.3f}')
    ax.plot(best_r, best_p, 'ko', markersize=8, label=f'best F1 = {best_f1:.3f}')
    ax.set(xlabel='recall', ylabel='precision', title='Precision-Recall curve',
           xlim=(0, 1.02), ylim=(0, max(0.5, p_ens.max() * 1.1)))
    ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pr_curve.png'), dpi=100)
    plt.close()

    # 3. ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, probs in enumerate(test_scores_per_seed):
        fpr, tpr = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, alpha=0.3, color='gray')
    fpr_ens, tpr_ens = roc_curve(y_test, ensemble_scores)
    ax.plot(fpr_ens, tpr_ens, color='tab:blue', linewidth=2.0,
            label=f'ensemble AUROC = {au_ensemble:.3f}')
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='random')
    ax.set(xlabel='false positive rate', ylabel='true positive rate',
           title='ROC curve')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'), dpi=100)
    plt.close()

    # 4. Score distribution histogram (ensemble), separated by true label
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 40)
    ax.hist(ensemble_scores[y_test == 0], bins=bins, alpha=0.6,
             color='tab:gray', label=f'y=0 (n={int((y_test==0).sum())})', density=True)
    ax.hist(ensemble_scores[y_test == 1], bins=bins, alpha=0.7,
             color='tab:red', label=f'y=1 (n={int((y_test==1).sum())})', density=True)
    ax.axvline(best_thr, color='black', linestyle='--',
                label=f'best-F1 threshold = {best_thr:.2f}')
    ax.set(xlabel='ensemble predicted probability',
           ylabel='density', title='Test-set score distribution by true label')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'score_distribution.png'), dpi=100)
    plt.close()
    print(f'saved plots to {args.output_dir}')


if __name__ == '__main__':
    main()

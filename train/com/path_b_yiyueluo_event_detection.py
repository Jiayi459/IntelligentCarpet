"""path_b_yiyueluo_event_detection.py — proof-of-concept event detection on the
single most event-rich session (YiyueLuo, session 104).

Scientific question:
    Given that we've established the IntelligentCarpet dataset lacks repeated
    discrete events across sessions, can our epsilon pipeline detect events when
    they DO exist? Session 104 has 11 transitions over 5000 frames -- the
    high-water mark. If a linear probe on epsilon's MAE-pretrained encoder + GRU
    can predict "transition in next 1 s" on this session, the architecture
    is sound and the prior failures were data-limited. If not, the
    architecture itself has problems.

What this script does:
    1. Identifies the transition frames in YiyueLuo session 104 using the
       same definition as per_session_motion_analysis.py (vel > 10 mm/frame
       for >=3 consecutive frames preceded by >=10 static frames).
    2. Builds binary classification targets: y(t) = 1 iff a transition
       starts in [t+1, t+HORIZON] (the next 1 s).
    3. Runs epsilon's encoder + GRU (frozen, MAE-pretrained from Stage 1+2) over
       every sample's tactile history -> 128-d hidden state.
    4. Trains a single Linear(128, 1) classifier head with BCE loss, using
       pos_weight to handle the extreme class imbalance.
    5. Evaluates on the test split (last 30 % of valid centers in this
       session): AUROC, accuracy, precision, recall, and a per-sample
       event-score timeline plot.

Outputs (under train/com/output/path_b_yiyueluo/):
    metrics.json
    event_score_timeline.png    model's per-sample event score over time,
                                with true event frames marked
    training_curve.png          train + val BCE loss per epoch

Run (needs tactile cache + epsilon checkpoint on CRC):
    python train/com/path_b_yiyueluo_event_detection.py
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
_PATHB = os.path.join(_OUT, 'path_b_yiyueluo')

_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')
_STATS     = os.path.join(_OUT, 'tactile_stats.json')

# epsilon model classes
sys.path.insert(0, os.path.join(_TRAIN, 'tactile_direct'))


# ---------------------------------------------------------------------------
# Constants — match the project's other forecasting scripts
# ---------------------------------------------------------------------------

HISTORY = 100
HORIZON = 10
SEED    = 42

# Target session: YiyueLuo, 5000 frames, 11 transitions (per per_session_motion.csv row 105)
TARGET_SESSION_INDEX = 104

# Transition definition (must match per_session_motion_analysis.py)
TRANSITION_THRESH = 10.0
MIN_RUN           = 3
MIN_STATIC        = 10

# Training
LR             = 1e-3
EPOCHS         = 200
BATCH          = 32
VAL_FRAC       = 0.10                       # within-train val for best-val checkpointing


def detect_transition_starts(velocities, thresh=TRANSITION_THRESH,
                              min_run=MIN_RUN, min_static=MIN_STATIC):
    """Like count_transitions in per_session_motion_analysis but returns the
    starting frame indices (relative to the input velocity series) of each
    detected transition.
    """
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


def main():
    parser = argparse.ArgumentParser(description='Path B (restricted): event detection on YiyueLuo session 104.')
    parser.add_argument('--epsilon-checkpoint',
                        default=os.path.join(_OUT, 'phase2_epsilon', 'dynamics_model.pt'),
                        help='path to epsilon dynamics_model.pt')
    parser.add_argument('--output-dir',   type=str, default=_PATHB)
    parser.add_argument('--epochs',       type=int, default=EPOCHS)
    parser.add_argument('--finetune',     action='store_true',
                        help='also finetune encoder + GRU (default: linear-probe only with frozen backbone)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device     : {device}')
    print(f'output_dir : {args.output_dir}')
    print(f'session    : {TARGET_SESSION_INDEX}')

    # ---- Inputs ----
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY}')
    if not os.path.exists(_STATS):
        raise SystemExit(f'no tactile stats at {_STATS} -- run seed_tactile_stats.py')
    if not os.path.exists(args.epsilon_checkpoint):
        raise SystemExit(f'no epsilon checkpoint at {args.epsilon_checkpoint}')

    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    T = tactile_all.shape[0]
    print(f'tactile_all: shape={tactile_all.shape}')

    with open(_STATS) as f:
        s = json.load(f)
    tactile_mean = float(s['tactile_mean']); tactile_std = float(s['tactile_std'])
    print(f'tactile_mean={tactile_mean:.4f}  tactile_std={tactile_std:.4f}')

    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)
    com_gt = com_results['com_gt']

    with open(os.path.join(_TRAIN, 'singlePerson_test', 'log.p'), 'rb') as f:
        log = pickle.load(f)
    with open(os.path.join(_TRAIN, 'singlePerson_test', 'fileNames.p'), 'rb') as f:
        file_names = pickle.load(f)
    n_sessions = len(log) - 1
    print(f'n_sessions = {n_sessions}; targeting session {TARGET_SESSION_INDEX}')

    if TARGET_SESSION_INDEX >= n_sessions:
        raise SystemExit(f'session {TARGET_SESSION_INDEX} out of range (max {n_sessions - 1})')

    _SUBJECT_RE = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
    subj = _SUBJECT_RE.match(file_names[TARGET_SESSION_INDEX]).group(3)
    rd   = _SUBJECT_RE.match(file_names[TARGET_SESSION_INDEX]).group(4)
    dt   = _SUBJECT_RE.match(file_names[TARGET_SESSION_INDEX]).group(2)
    a, b = log[TARGET_SESSION_INDEX], log[TARGET_SESSION_INDEX + 1]
    print(f'  subject={subj}, date={dt}, round={rd}')
    print(f'  global frame range: [{a}, {b})  ({b - a} frames)')

    # ---- Detect transitions in this session ----
    com_sess = com_gt[a:b]                                              # (Ts, 3)
    d_xy = np.diff(com_sess[:, :2], axis=0)
    v_xy = np.linalg.norm(d_xy, axis=1)                                  # (Ts-1,) mm/frame
    transition_starts_local = detect_transition_starts(v_xy)             # indices into v_xy
    # The velocity at index k corresponds to motion between frame k and k+1; the
    # "transition" starts at frame k+1 in the session (when motion first crosses threshold).
    transition_frames_global = [int(a + k + 1) for k in transition_starts_local]
    print(f'  transitions detected: {len(transition_frames_global)}')
    print(f'  global event frames : {transition_frames_global}')

    # ---- Build forecasting samples within this session ----
    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))
    valid_centers = [t for t in range(a + HISTORY - 1, b - HORIZON)
                     if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()]
    valid_centers = np.asarray(valid_centers)
    n_total = len(valid_centers)
    print(f'  valid sample centers: {n_total}')

    # Train/test chronological 70/30 (matches all forecasting scripts)
    n_train = int(0.7 * n_total)
    train_centers = valid_centers[:n_train]
    test_centers  = valid_centers[n_train:]
    print(f'  train centers: {len(train_centers)}, test centers: {len(test_centers)}')

    # ---- Labels: y(t) = 1 iff a transition starts in [t+1, t+HORIZON] ----
    event_set = set(transition_frames_global)
    def label(t):
        for k in range(1, HORIZON + 1):
            if (t + k) in event_set:
                return 1
        return 0
    y_train = np.asarray([label(int(t)) for t in train_centers], dtype=np.float32)
    y_test  = np.asarray([label(int(t)) for t in test_centers],  dtype=np.float32)
    n_pos_train = int(y_train.sum()); n_pos_test = int(y_test.sum())
    print(f'  labels: train pos={n_pos_train}/{len(y_train)} ({100*n_pos_train/len(y_train):.1f}%), '
          f'test pos={n_pos_test}/{len(y_test)} ({100*n_pos_test/len(y_test):.1f}%)')

    if n_pos_train < 2 or n_pos_test < 1:
        print(f'WARN: insufficient positive labels; results will be noisy.')

    # ---- Load epsilon encoder + GRU (frozen by default) ----
    from model_epsilon import DynamicsModel
    dyn = DynamicsModel().to(device)
    ck = torch.load(args.epsilon_checkpoint, map_location=device, weights_only=False)
    dyn.load_state_dict(ck['dynamics'])
    if not args.finetune:
        for p in dyn.parameters():
            p.requires_grad_(False)
        dyn.eval()
        print(f'loaded epsilon dynamics_model.pt (FROZEN; epoch={ck.get("epoch", "?")}, '
              f'best_val={ck.get("best_val", float("nan")):.5f})')
    else:
        dyn.train()
        print(f'loaded epsilon dynamics_model.pt (FINETUNED; epoch={ck.get("epoch", "?")})')

    # ---- Encoder forward to get hidden states (B, 128) ----
    def encode_centers(centers, training_mode=False):
        if not training_mode:
            dyn.eval()
        H = np.zeros((len(centers), 128), dtype=np.float32)
        with torch.no_grad() if not training_mode else torch.enable_grad():
            for i0 in range(0, len(centers), BATCH):
                i1 = min(i0 + BATCH, len(centers))
                windows = np.stack([
                    (np.asarray(tactile_all[t - HISTORY + 1 : t + 1]) - tactile_mean) / tactile_std
                    for t in centers[i0:i1]
                ], axis=0).astype(np.float32)
                x = torch.from_numpy(windows).to(device)
                H[i0:i1] = dyn.encode_history(x).cpu().numpy()
        return H

    print('encoding hidden states (frozen encoder + GRU)...')
    t0 = time.time()
    H_train = encode_centers(train_centers)
    H_test  = encode_centers(test_centers)
    print(f'  encoded {len(H_train)} train + {len(H_test)} test in {time.time() - t0:.1f}s')

    # ---- Within-train val split ----
    perm = np.random.permutation(len(H_train))
    n_val = max(1, int(len(H_train) * VAL_FRAC))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    H_tr = H_train[tr_idx]; y_tr = y_train[tr_idx]
    H_val = H_train[val_idx]; y_val = y_train[val_idx]
    print(f'  tr / val: {len(H_tr)} / {len(H_val)}  (val pos: {int(y_val.sum())})')

    # ---- Classifier head ----
    head = nn.Linear(128, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    pos_weight = max(1.0, (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    print(f'  class-imbalance pos_weight = {pos_weight:.1f}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    H_tr_t  = torch.from_numpy(H_tr).to(device)
    y_tr_t  = torch.from_numpy(y_tr).to(device)
    H_val_t = torch.from_numpy(H_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    train_curve, val_curve = [], []
    best_val = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(H_tr_t), device=device)
        total = 0.0; nb = 0
        for i0 in range(0, len(H_tr_t), BATCH):
            idx = perm[i0:i0 + BATCH]
            hb = H_tr_t[idx]; yb = y_tr_t[idx]
            logit = head(hb).squeeze(-1)
            loss = criterion(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); nb += 1
        avg_train = total / nb
        head.eval()
        with torch.no_grad():
            v_logit = head(H_val_t).squeeze(-1)
            v_loss = float(criterion(v_logit, y_val_t).item())
        train_curve.append(avg_train); val_curve.append(v_loss)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  epoch {epoch:3d}/{args.epochs - 1}  '
                  f'train={avg_train:.4f}  val={v_loss:.4f}  best_val={best_val:.4f}',
                  flush=True)
    head.load_state_dict(best_state)

    # ---- Evaluate on test ----
    H_test_t = torch.from_numpy(H_test).to(device)
    with torch.no_grad():
        logits_test = head(H_test_t).squeeze(-1).cpu().numpy()
    probs_test = 1.0 / (1.0 + np.exp(-logits_test))

    # Metrics
    def auroc(y, scores):
        # Compute AUROC by ranking; handles ties.
        order = np.argsort(scores)
        y = y[order]
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float('nan')
        # rank from 1; sum of ranks of positives
        ranks = np.arange(1, len(y) + 1)
        sum_pos_ranks = float(ranks[y == 1].sum())
        return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    auroc_test = auroc(y_test, probs_test)
    pred_at_05 = (probs_test >= 0.5).astype(int)
    tp = int(((pred_at_05 == 1) & (y_test == 1)).sum())
    fp = int(((pred_at_05 == 1) & (y_test == 0)).sum())
    tn = int(((pred_at_05 == 0) & (y_test == 0)).sum())
    fn = int(((pred_at_05 == 0) & (y_test == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    acc       = (tp + tn) / max(1, len(y_test))

    # Persistence baseline: always predict 0 ("no event")
    acc_persist = float((y_test == 0).sum() / len(y_test))               # = 1 - positive rate
    auroc_persist = 0.5                                                   # by construction

    bar = '=' * 78
    print(f'\n{bar}\nPATH B RESTRICTED — YiyueLuo session {TARGET_SESSION_INDEX} event detection\n{bar}')
    print(f'  events in train : {n_pos_train}/{len(y_train)} samples'
          f'  ({100*n_pos_train/len(y_train):.2f}%)')
    print(f'  events in test  : {n_pos_test}/{len(y_test)} samples'
          f'  ({100*n_pos_test/len(y_test):.2f}%)')
    print(f'\n  Test set (n={len(y_test)}):')
    print(f'    AUROC               : {auroc_test:.3f}   (baseline = 0.500)')
    print(f'    accuracy @ 0.5      : {acc:.3f}        (always-0 baseline = {acc_persist:.3f})')
    print(f'    precision @ 0.5     : {precision:.3f}')
    print(f'    recall @ 0.5        : {recall:.3f}')
    print(f'    confusion matrix    : TP={tp}, FP={fp}, FN={fn}, TN={tn}')
    print()
    if np.isnan(auroc_test):
        print('  Reading: AUROC undefined (zero positives or zero negatives in test).')
    elif auroc_test > 0.75:
        print('  Reading: AUROC > 0.75 -> architecture DOES detect events when they exist.')
        print('           Validates Path D direction (switch to event-rich dataset).')
    elif auroc_test > 0.6:
        print('  Reading: AUROC 0.6-0.75 -> mild signal, but noisy. Possibly architectural OK.')
    else:
        print('  Reading: AUROC near 0.5 -> architecture does NOT detect events here.')
        print('           Either the encoder representation is uninformative, or 7 train events are too few.')

    # ---- Save metrics ----
    out = {
        'session_index':        TARGET_SESSION_INDEX,
        'subject':              subj,
        'date':                 dt,
        'round':                rd,
        'session_frames':       int(b - a),
        'n_transitions':        len(transition_frames_global),
        'transition_frames_global': transition_frames_global,
        'n_train_samples':      int(len(train_centers)),
        'n_test_samples':       int(len(test_centers)),
        'n_pos_train':          n_pos_train,
        'n_pos_test':           n_pos_test,
        'pos_weight':           float(pos_weight),
        'finetuned':            bool(args.finetune),
        'epsilon_epoch':        int(ck.get('epoch', -1)),
        'epsilon_best_val_mse': float(ck.get('best_val', float('nan'))),
        'test_metrics': {
            'auroc':          float(auroc_test) if not np.isnan(auroc_test) else None,
            'accuracy_at_05': float(acc),
            'precision_at_05': float(precision),
            'recall_at_05':    float(recall),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'baseline_auroc': 0.5,
            'baseline_accuracy': float(acc_persist),
        },
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved {os.path.join(args.output_dir, "metrics.json")}')

    # ---- Plots ----
    # 1. Training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_curve, label='train', color='tab:blue')
    ax.plot(val_curve, label='val', color='tab:orange', linestyle='--')
    ax.set(xlabel='epoch', ylabel='BCE loss (with pos_weight)',
           title=f'Path B classifier head training (session {TARGET_SESSION_INDEX})')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curve.png'), dpi=100)
    plt.close()

    # 2. Event-score timeline: model score vs sample frame, true events overlaid
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_centers, np.zeros(len(train_centers)) - 0.05,
            'o', color='lightgray', markersize=1, label='train sample')
    train_pos = train_centers[y_train == 1]
    ax.plot(train_pos, np.zeros(len(train_pos)) - 0.05, 'o', color='tab:blue', markersize=3,
            label='train: y=1 (event in next 1s)')
    ax.plot(test_centers, probs_test, 'o-', color='tab:red', markersize=3, alpha=0.7,
            label='test: model score')
    for ev in transition_frames_global:
        ax.axvline(ev, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='threshold 0.5')
    ax.set(xlabel='global frame index', ylabel='event probability / label',
           title=f'Event-score timeline — YiyueLuo session {TARGET_SESSION_INDEX}  '
                 f'(AUROC test = {auroc_test:.3f}; events as vertical dashed lines)')
    ax.legend(fontsize='small', loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'event_score_timeline.png'), dpi=100)
    plt.close()
    print(f'saved plots to {args.output_dir}')


if __name__ == '__main__':
    main()

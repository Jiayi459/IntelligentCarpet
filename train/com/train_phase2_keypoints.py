"""train_phase2_keypoints.py — Phase 2 (delta variant): predict next-second CoM
from the *full* 21-joint pose history.

Phase 1 question (history-only):
    Can simple history-only models forecast CoM 1 s ahead?
    Answer: no — persistence (86 mm median at 1 s) is unbeatable in the
    dataset's low-motion regime; the tiny 3-dim-input GRU underperformed.

Phase 2 delta question:
    Does giving the forecaster access to the *full* pose history
    (21 keypoints, not just the derived CoM) close the gap to persistence?

Models compared:
    1. Persistence         (no-training Phase 1 baseline, predict no change)
    2. Phase 1 GRU         (loaded from train/com/output/phase1/gru_model.pt
                            if present; input dim 3 = CoM only)
    3. Phase 2 GRU (this script)  input dim 63 = (21 keypoints x 3)

Inputs / target (both from com_results.p):
    X_history : kp_gt_mm[t-HISTORY+1 : t+1]  reshaped to (HISTORY, 63)
    Y_future  : com_gt   [t+1 : t+1+HORIZON]                (HORIZON, 3)
    The target stays in CoM space — we forecast where the *body's CoM* will be,
    given pose-history input. Mirrors Phase 1 metric definition exactly.

Train / test split:
    Same per-session chronological 70/30 split as Phase 1, with the same
    GT-outlier window filter. By construction, the training and test sample
    indices match Phase 1's, so the GRU's headline numbers are directly
    comparable to Phase 1's.

Outputs (under train/com/output/phase2_keypoints/):
    metrics.json
    training_curve.png
    error_vs_horizon.png
    error_vs_horizon_z.png
    comparison_phase1_vs_phase2.png
    gru_model.pt

Module-level definitions (importable without side effects):
    TrajForecaster        Phase 2 v1 GRU model class
    HISTORY, HORIZON, SEED, GRU_*, KP_DIM    constants
All heavy execution (data load, training, eval, plotting) lives inside main()
so `from train_phase2_keypoints import TrajForecaster` does not trigger training.
"""

import os
import json
import pickle
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE   = os.path.dirname(os.path.abspath(__file__))   # .../train/com
_TRAIN  = os.path.dirname(_HERE)                       # .../train
_OUT    = os.path.join(_HERE, 'output')                # .../train/com/output
_PHASE1 = os.path.join(_OUT, 'phase1')
_PHASE2 = os.path.join(_OUT, 'phase2_keypoints')


# ---------------------------------------------------------------------------
# Hyperparameters — kept identical to Phase 1 for an apples-to-apples comparison
# ---------------------------------------------------------------------------

HISTORY    = 100        # 10 s at 10 Hz
HORIZON    = 10         # 1 s at 10 Hz
SEED       = 42

GRU_HIDDEN = 64
GRU_LAYERS = 1
GRU_LR     = 1e-3
GRU_EPOCHS = 50
GRU_BATCH  = 256
VAL_FRAC   = 0.10

LINEAR_LOOKBACK = 10    # for the "linear extrap on CoM history" reference

KP_DIM     = 21 * 3     # 63 — flat pose vector per frame


# ---------------------------------------------------------------------------
# Model class (module level — importable without triggering training)
# ---------------------------------------------------------------------------

class TrajForecaster(nn.Module):
    """GRU encoder over (HISTORY, KP_DIM) → linear projection → (HORIZON, 3).

    Identical architecture to Phase 1 except for input_size, so the only
    additional capacity comes from the wider input projection (~12k extra weights).
    """
    def __init__(self, input_dim=KP_DIM, hidden=GRU_HIDDEN, layers=GRU_LAYERS, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, horizon * 3)

    def forward(self, x):                                  # x: (B, HISTORY, KP_DIM)
        _, h = self.gru(x)                                 # h: (layers, B, hidden)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


# ---------------------------------------------------------------------------
# Main execution — only runs when invoked as a script, not on import
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_PHASE2, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load CoM results + session metadata
    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)

    com_gt    = com_results['com_gt']        # (T, 3) mm
    kp_gt_mm  = com_results['kp_gt_mm']      # (T, 21, 3) mm
    T         = len(com_gt)
    assert kp_gt_mm.shape == (T, 21, 3), f'unexpected kp_gt_mm shape {kp_gt_mm.shape}'

    with open(os.path.join(_TRAIN, 'singlePerson_test', 'log.p'), 'rb') as f:
        log = pickle.load(f)
    with open(os.path.join(_TRAIN, 'singlePerson_test', 'fileNames.p'), 'rb') as f:
        file_names = pickle.load(f)

    n_sessions = len(log) - 1
    _SUBJECT_RE = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
    subjects_per_sess = [_SUBJECT_RE.match(n).group(3) for n in file_names]

    # GT-outlier mask (matches Phase 1 exactly)
    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))

    # Build (X_history, Y_future) — same windowing + split logic as Phase 1
    def build_samples():
        """Per-session, enumerate valid centers t and emit (kp_history, com_future) pairs.

        Window validity rule is identical to Phase 1, so the resulting train/test
        sample indices align with Phase 1's. The only difference is what we put in X.
        """
        X, Y = [], []
        meta = {'subject': [], 'session': [], 'frame': [], 'split': []}
        for s in range(n_sessions):
            a, b = log[s], log[s + 1]
            valid_t = [
                t for t in range(a + HISTORY - 1, b - HORIZON)
                if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
            ]
            n_train_s = int(0.7 * len(valid_t))
            for i, t in enumerate(valid_t):
                X.append(kp_gt_mm[t - HISTORY + 1 : t + 1].reshape(HISTORY, KP_DIM))  # (HISTORY, 63)
                Y.append(com_gt[t + 1            : t + 1 + HORIZON])                  # (HORIZON, 3)
                meta['subject'].append(subjects_per_sess[s])
                meta['session'].append(s)
                meta['frame'].append(t)
                meta['split'].append('train' if i < n_train_s else 'test')
        return (np.asarray(X, dtype=np.float64),
                np.asarray(Y, dtype=np.float64),
                {k: np.asarray(v) for k, v in meta.items()})

    X, Y, meta = build_samples()
    train_mask = meta['split'] == 'train'
    test_mask  = meta['split'] == 'test'

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]

    print(f'samples generated   : {len(X):>6d}   (HISTORY={HISTORY}, HORIZON={HORIZON}, KP_DIM={KP_DIM})')
    print(f'  train             : {train_mask.sum():>6d}')
    print(f'  test              : {test_mask.sum():>6d}')

    assert len(X) == 17218, f'expected 17218 total samples (matching Phase 1), got {len(X)}'

    # Persistence baseline — CoM at t (the last frame of each sample's history)
    # We need to look it up from com_gt using the sample's frame index in meta.
    test_frame_centers = meta['frame'][test_mask]      # (N_test,)
    com_at_t           = com_gt[test_frame_centers]    # (N_test, 3)
    pred_persistence   = np.broadcast_to(com_at_t[:, None, :],
                                         (len(test_frame_centers), HORIZON, 3)).copy()

    # Standardize X and Y *separately* (different feature distributions)
    # Compute stats on train set only — never touch test for fit.
    _train_pool_X = X_train.reshape(-1, KP_DIM)            # (N_train * HISTORY, 63)
    _train_pool_Y = Y_train.reshape(-1, 3)                 # (N_train * HORIZON, 3)
    mean_X = _train_pool_X.mean(axis=0)
    std_X  = _train_pool_X.std(axis=0)
    mean_Y = _train_pool_Y.mean(axis=0)
    std_Y  = _train_pool_Y.std(axis=0)

    # Guard against any zero-variance feature (would NaN the normalize).
    std_X = np.where(std_X < 1e-6, 1.0, std_X)
    std_Y = np.where(std_Y < 1e-6, 1.0, std_Y)

    print(f'  X (kp)  mean range: {mean_X.min():.1f} .. {mean_X.max():.1f}')
    print(f'  X (kp)  std  range: {std_X.min():.1f} .. {std_X.max():.1f}')
    print(f'  Y (CoM) mean      : {mean_Y}')
    print(f'  Y (CoM) std       : {std_Y}')

    def norm_X(arr):    return (arr - mean_X) / std_X
    def norm_Y(arr):    return (arr - mean_Y) / std_Y
    def denorm_Y(arr):  return arr * std_Y + mean_Y

    # Split train -> {tr, val}
    n_train = X_train.shape[0]
    perm    = np.random.permutation(n_train)
    n_val   = int(n_train * VAL_FRAC)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  device            : {device}')

    Xt = torch.tensor(norm_X(X_train), dtype=torch.float32)
    Yt = torch.tensor(norm_Y(Y_train), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xt[tr_idx], Yt[tr_idx]),
                              batch_size=GRU_BATCH, shuffle=True)
    X_val = Xt[val_idx].to(device)
    Y_val = Yt[val_idx].to(device)

    model     = TrajForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nTraining Phase 2 GRU ({n_params} params, {GRU_EPOCHS} epochs)...')

    train_losses, val_losses = [], []
    for epoch in range(GRU_EPOCHS):
        model.train()
        total = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        avg_train = total / n_batches

        model.eval()
        with torch.no_grad():
            avg_val = criterion(model(X_val), Y_val).item()

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        if epoch % 5 == 0 or epoch == GRU_EPOCHS - 1:
            print(f'  epoch {epoch:3d}/{GRU_EPOCHS - 1}  '
                  f'train_mse={avg_train:.5f}  val_mse={avg_val:.5f}')

    torch.save(model.state_dict(), os.path.join(_PHASE2, 'gru_model.pt'))

    # Evaluate Phase 2 GRU + persistence on the test set
    print('\nEvaluating on the test set...')
    model.eval()
    with torch.no_grad():
        pred_phase2_norm = model(torch.tensor(norm_X(X_test), dtype=torch.float32)
                                  .to(device)).cpu().numpy()
    pred_phase2 = denorm_Y(pred_phase2_norm)

    def euc(pred, gt):
        """Per-sample per-horizon Euclidean error.  pred, gt: (N, H, 3) -> (N, H)."""
        return np.linalg.norm(pred - gt, axis=2)

    methods = {
        'persistence':       pred_persistence,
        'phase2_gru_kp':     pred_phase2,
    }

    # Optional: try loading the Phase 1 GRU output for direct comparison.
    # Phase 1 GRU is a different model on CoM history; we re-run it through
    # its own architecture if its checkpoint is present.
    _phase1_model_path = os.path.join(_PHASE1, 'gru_model.pt')
    try:
        from train_phase1 import TrajForecaster as P1Forecaster
        if os.path.exists(_phase1_model_path):
            print('  loading Phase 1 GRU for comparison...')
            p1 = P1Forecaster().to(device)
            p1.load_state_dict(torch.load(_phase1_model_path, map_location=device))
            p1.eval()

            # Phase 1's input is CoM history (3-dim) standardized by *its* stats.
            # Rebuild the CoM-history input for the same test samples we used here.
            com_history_test = np.stack(
                [com_gt[t - HISTORY + 1 : t + 1] for t in test_frame_centers],
                axis=0
            )  # (N_test, HISTORY, 3)

            # Phase 1 used train-pool standardization. Recompute the stats here
            # using the same train indices used in this script (the splits are aligned).
            train_frame_centers = meta['frame'][train_mask]
            com_history_train = np.stack(
                [com_gt[t - HISTORY + 1 : t + 1] for t in train_frame_centers],
                axis=0
            )
            com_future_train = np.stack(
                [com_gt[t + 1 : t + 1 + HORIZON] for t in train_frame_centers],
                axis=0
            )
            p1_pool = np.concatenate([com_history_train.reshape(-1, 3),
                                      com_future_train.reshape(-1, 3)], axis=0)
            p1_mean = p1_pool.mean(axis=0)
            p1_std  = p1_pool.std(axis=0)
            p1_std  = np.where(p1_std < 1e-6, 1.0, p1_std)

            with torch.no_grad():
                ph1_input = torch.tensor((com_history_test - p1_mean) / p1_std,
                                         dtype=torch.float32).to(device)
                ph1_pred_norm = p1(ph1_input).cpu().numpy()
            pred_phase1 = ph1_pred_norm * p1_std + p1_mean
            methods['phase1_gru_com'] = pred_phase1
            print('  Phase 1 GRU comparison: ready')
    except Exception as exc:
        print(f'  WARNING: could not load Phase 1 GRU for comparison ({exc})')

    # Metrics
    results = {}
    e_persist_all  = euc(pred_persistence, Y_test)
    median_persist = float(np.median(e_persist_all))

    for name, pred in methods.items():
        e_3d = euc(pred, Y_test)
        e_ax = np.abs(pred - Y_test)
        results[name] = {
            'overall_median_3d_mm':         float(np.median(e_3d)),
            'overall_mean_3d_mm':           float(np.mean(e_3d)),
            'overall_p95_3d_mm':            float(np.percentile(e_3d, 95)),
            'per_horizon_median_3d_mm':     [float(np.median(e_3d[:, h]))    for h in range(HORIZON)],
            'per_horizon_p95_3d_mm':        [float(np.percentile(e_3d[:, h], 95)) for h in range(HORIZON)],
            'per_axis_median_mm':           {ax: float(np.median(e_ax[:, :, i])) for i, ax in enumerate('xyz')},
            'per_axis_per_horizon_median_mm': {
                ax: [float(np.median(e_ax[:, h, i])) for h in range(HORIZON)]
                for i, ax in enumerate('xyz')
            },
            'skill_score_vs_persistence':   float(np.median(e_3d) / median_persist),
        }

    # Per-subject for the Phase 2 GRU specifically
    test_subj = meta['subject'][test_mask]
    e_phase2  = euc(pred_phase2, Y_test)
    per_subject = {}
    for subj in sorted(np.unique(test_subj)):
        m = test_subj == subj
        per_subject[subj] = {
            'n':      int(m.sum()),
            'median': float(np.median(e_phase2[m])),
            'p95':    float(np.percentile(e_phase2[m], 95)),
        }
    results['phase2_gru_kp']['per_subject_median_mm'] = per_subject

    with open(os.path.join(_PHASE2, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Console summary
    bar = '=' * 78
    print(f'\n{bar}\nPHASE 2 (delta keypoints) RESULTS  '
          f'(n_test = {Y_test.shape[0]}, horizon = {HORIZON} frames = 1.0 s)\n{bar}')

    ordered = ['persistence']
    if 'phase1_gru_com' in results:  ordered.append('phase1_gru_com')
    ordered.append('phase2_gru_kp')

    print(f'{"method":<20} {"median 3D":>12} {"mean 3D":>10} {"p95 3D":>10} {"skill":>8}')
    for name in ordered:
        r = results[name]
        print(f'  {name:<18}  {r["overall_median_3d_mm"]:>10.1f}   '
              f'{r["overall_mean_3d_mm"]:>8.1f}   '
              f'{r["overall_p95_3d_mm"]:>8.1f}   '
              f'{r["skill_score_vs_persistence"]:>6.3f}')

    print('\nPer-horizon median 3D error (mm):')
    print(f'{"horizon (frame)":<20}' + ''.join(f'{h + 1:>7d}' for h in range(HORIZON)))
    for name in ordered:
        vals = results[name]['per_horizon_median_3d_mm']
        print(f'  {name:<18}' + ''.join(f'{v:>7.1f}' for v in vals))

    print('\nPer-axis median error (mm, averaged across horizons):')
    for name in ordered:
        by_ax = results[name]['per_axis_median_mm']
        print(f'  {name:<20}  x={by_ax["x"]:>5.1f}   y={by_ax["y"]:>5.1f}   z={by_ax["z"]:>5.1f}')

    print('\nPhase 2 GRU per-subject median 3D error (mm):')
    for subj, st in sorted(per_subject.items(), key=lambda kv: kv[1]['median']):
        print(f'  {subj:<14}  n={st["n"]:>5d}   median={st["median"]:>6.1f}   p95={st["p95"]:>6.1f}')

    # Plots
    # Training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='train')
    ax.plot(val_losses, label='val', linestyle='--')
    ax.set(xlabel='epoch', ylabel='MSE (normalized units)',
           title=f'Phase 2 GRU training   hidden={GRU_HIDDEN}, layers={GRU_LAYERS}, input_dim={KP_DIM}')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(_PHASE2, 'training_curve.png'), dpi=100); plt.close()

    # Error vs horizon — overall 3D
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons_sec = np.arange(1, HORIZON + 1) / 10.0
    for name in ordered:
        ax.plot(horizons_sec, results[name]['per_horizon_median_3d_mm'], marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median 3D Euclidean error (mm)',
           title='Phase 2 — forecasting accuracy vs horizon')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(_PHASE2, 'error_vs_horizon.png'), dpi=100); plt.close()

    # Error vs horizon — z axis only (where the action is)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ordered:
        vals = results[name]['per_axis_per_horizon_median_mm']['z']
        ax.plot(horizons_sec, vals, marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median |z| error (mm)',
           title='Phase 2 — z-axis error vs horizon')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(_PHASE2, 'error_vs_horizon_z.png'), dpi=100); plt.close()

    # Comparison bar plot — overall medians + skill scores
    if 'phase1_gru_com' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        names = ['persistence', 'phase1_gru_com', 'phase2_gru_kp']
        medians = [results[n]['overall_median_3d_mm']      for n in names]
        p95s    = [results[n]['overall_p95_3d_mm']         for n in names]
        skills  = [results[n]['skill_score_vs_persistence'] for n in names]
        axes[0].bar(names, medians, color=['gray', 'tab:blue', 'tab:red'])
        axes[0].set(ylabel='median 3D error (mm)', title='Median 3D error (1-s horizon)')
        axes[0].grid(alpha=0.3, axis='y')
        axes[1].bar(names, skills, color=['gray', 'tab:blue', 'tab:red'])
        axes[1].axhline(1.0, color='black', linestyle='--', alpha=0.5)
        axes[1].set(ylabel='skill score (lower = better than persistence)',
                    title='Skill score vs persistence')
        axes[1].grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(_PHASE2, 'comparison_phase1_vs_phase2.png'), dpi=100)
        plt.close()

    print(f'\nSaved metrics + plots to {_PHASE2}')


if __name__ == '__main__':
    main()

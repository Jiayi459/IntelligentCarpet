"""train_phase1.py — Phase 1 forecasting: history-only CoM prediction at 1-s horizon.

Question:
    How well can simple, history-only models predict CoM(t+1 : t+10)
    given the past HISTORY frames? Establishes the noise floor every
    Phase 2 / Phase 3 model must beat.

Models (all output a (B, HORIZON, 3) tensor):
    1. Persistence:        pred[t+i] = CoM[t]                       (no training)
    2. Linear extrapolation: line fit through last `lookback` frames (no training)
    3. Constant-velocity:  pred[t+i] = CoM[t] + i * (CoM[t] - CoM[t-1])  (no training)
    4. Small GRU:          learned dynamics on standardized CoM history (trained)

Inputs / target:
    Both come from `com_gt` (camera-derived) — pure trajectory predictability.
    Edge filtering (CNN window clamping) doesn't apply since we never use com_pred here.

Split:
    Per-session 70 % train / 30 % test, chronological (no shuffling across the split).
    Same subjects appear in both halves (generalize-to-new-time, per user's Q5).

Outputs (under train/com/output/phase1/):
    metrics.json                          full per-method × per-axis × per-horizon stats
    training_curve.png                    GRU train + val loss vs epoch
    error_vs_horizon.png                  median error vs horizon for all 4 methods
    error_vs_horizon_z.png                same but z axis only (where the action is)
    example_trajectories.png              5 random test samples, all axes, all methods
    gru_model.pt                          trained GRU state_dict
"""

import os
import json
import pickle
import re
from collections import defaultdict

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
_PHASE1 = os.path.join(_OUT, 'phase1')                 # .../train/com/output/phase1
os.makedirs(_PHASE1, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters (all in one place — easy to sweep)
# ---------------------------------------------------------------------------

HISTORY     = 100        # 10 s of past CoM at 10 Hz
HORIZON     = 10         # 1 s of future CoM at 10 Hz
SEED        = 42

# GRU
GRU_HIDDEN  = 64
GRU_LAYERS  = 1
GRU_LR      = 1e-3
GRU_EPOCHS  = 50
GRU_BATCH   = 256
VAL_FRAC    = 0.10       # fraction of train set held out for validation curve

# Linear-extrapolation baseline
LINEAR_LOOKBACK = 10     # use the last 10 history frames to fit the line

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Load CoM data and session metadata
# ---------------------------------------------------------------------------

with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
    com_results = pickle.load(f)

com_gt = com_results['com_gt']    # (T, 3) mm — GT CoM from OpenPose keypoints
T      = len(com_gt)

with open(os.path.join(_TRAIN, 'singlePerson_test', 'log.p'), 'rb') as f:
    log = pickle.load(f)
with open(os.path.join(_TRAIN, 'singlePerson_test', 'fileNames.p'), 'rb') as f:
    file_names = pickle.load(f)

n_sessions = len(log) - 1

# Parse subject name out of each session filename
_SUBJECT_RE = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
subjects_per_sess = [_SUBJECT_RE.match(n).group(3) for n in file_names]

# GT-outlier mask (physically impossible CoM — OpenPose triangulation failures)
_in_carpet = lambda v: (v >= -100) & (v <= 1800)
gt_outliers = (~_in_carpet(com_gt[:, 0])
               | ~_in_carpet(com_gt[:, 1])
               | (com_gt[:, 2] > 0))    # z > 0 ⇒ below floor


# ---------------------------------------------------------------------------
# Build (X_history, Y_future) samples with per-session time-split
# ---------------------------------------------------------------------------

def build_samples():
    """For every session, enumerate valid centers t and emit (history, future) pairs.

    A center t is valid iff:
        - t-HISTORY+1 >= session_start
        - t+HORIZON   <= session_end - 1
        - no GT outlier in [t-HISTORY+1, t+HORIZON]
    """
    X, Y = [], []
    meta = {'subject': [], 'session': [], 'frame': [], 'split': []}
    for s in range(n_sessions):
        a, b = log[s], log[s + 1]                  # frame range for session s: [a, b)
        valid_t = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t))
        for i, t in enumerate(valid_t):
            X.append(com_gt[t - HISTORY + 1 : t + 1])        # (HISTORY, 3)
            Y.append(com_gt[t + 1         : t + 1 + HORIZON])# (HORIZON, 3)
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

print(f'samples generated  : {len(X):>6d}   (HISTORY={HISTORY}, HORIZON={HORIZON})')
print(f'  train            : {train_mask.sum():>6d}')
print(f'  test             : {test_mask.sum():>6d}')
print(f'  filtered (outliers): {(~train_mask & ~test_mask).sum():>6d}')


# ---------------------------------------------------------------------------
# Closed-form baselines (vectorized over the whole test set)
# ---------------------------------------------------------------------------

def persistence(X):
    """pred[t+i] = CoM[t]  for all i in [1, HORIZON]."""
    return np.broadcast_to(X[:, -1:, :], (X.shape[0], HORIZON, 3)).copy()


def linear_extrap(X, lookback=LINEAR_LOOKBACK):
    """Per-sample, per-axis OLS line fit on the last `lookback` frames; project HORIZON steps.

    Uses closed-form regression coefficients (no numpy.polyfit loop) for speed.
    """
    block = X[:, -lookback:, :]                           # (N, lookback, 3)
    t = np.arange(lookback, dtype=np.float64)[None, :, None]  # (1, lookback, 1)
    t_mean = t.mean(axis=1, keepdims=True)                # (1, 1, 1)
    y_mean = block.mean(axis=1, keepdims=True)            # (N, 1, 3)
    t_dev  = t - t_mean
    y_dev  = block - y_mean
    slope  = (t_dev * y_dev).sum(axis=1, keepdims=True) / (t_dev ** 2).sum(axis=1, keepdims=True)
    intercept = y_mean - slope * t_mean
    # Future indices: lookback, lookback+1, ..., lookback + HORIZON - 1
    t_future = np.arange(lookback, lookback + HORIZON, dtype=np.float64)[None, :, None]
    return slope * t_future + intercept                   # (N, HORIZON, 3)


def constant_velocity(X):
    """v = CoM[t] - CoM[t-1] (per-frame velocity); pred[t+i] = CoM[t] + i * v."""
    last = X[:, -1:, :]                                   # (N, 1, 3)
    v    = X[:, -1:, :] - X[:, -2:-1, :]                  # (N, 1, 3)
    steps = np.arange(1, HORIZON + 1, dtype=np.float64)[None, :, None]
    return last + steps * v                               # (N, HORIZON, 3)


# ---------------------------------------------------------------------------
# Small GRU forecaster
# ---------------------------------------------------------------------------

class TrajForecaster(nn.Module):
    """GRU encoder → linear projection → 10-step trajectory.

    The projection-then-reshape decoder is simpler than autoregressive decoding
    and works well for short horizons. It does NOT share weights across horizons,
    so each future frame gets its own learned mapping from the final hidden state.
    """
    def __init__(self, hidden=GRU_HIDDEN, layers=GRU_LAYERS, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(3, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, horizon * 3)

    def forward(self, x):                                  # x: (B, HISTORY, 3)
        _, h = self.gru(x)                                 # h: (layers, B, hidden)
        return self.proj(h[-1]).view(-1, self.horizon, 3)  # (B, HORIZON, 3)


# Standardize using *train-set* statistics (over both history and future combined).
# Using train stats only avoids leaking test distribution into the normalization.
_train_pool = np.concatenate([X_train.reshape(-1, 3), Y_train.reshape(-1, 3)], axis=0)
mean = _train_pool.mean(axis=0)                            # (3,)
std  = _train_pool.std(axis=0)                             # (3,)
print(f'  norm mean        : {mean}')
print(f'  norm std         : {std}')


def normalize(arr):    return (arr - mean) / std
def denormalize(arr):  return arr * std + mean


# Split train -> {tr, val}
n_train     = X_train.shape[0]
perm        = np.random.permutation(n_train)
n_val       = int(n_train * VAL_FRAC)
val_idx     = perm[:n_val]
tr_idx      = perm[n_val:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  device           : {device}')

Xt = torch.tensor(normalize(X_train), dtype=torch.float32)
Yt = torch.tensor(normalize(Y_train), dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xt[tr_idx], Yt[tr_idx]),
                          batch_size=GRU_BATCH, shuffle=True)
X_val = Xt[val_idx].to(device)
Y_val = Yt[val_idx].to(device)

model     = TrajForecaster().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
print(f'\nTraining GRU ({sum(p.numel() for p in model.parameters())} params, {GRU_EPOCHS} epochs)...')
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

torch.save(model.state_dict(), os.path.join(_PHASE1, 'gru_model.pt'))


# ---------------------------------------------------------------------------
# Evaluate all 4 methods on the test set
# ---------------------------------------------------------------------------

print('\nEvaluating all methods on the test set...')
pred_persist  = persistence(X_test)
pred_linear   = linear_extrap(X_test)
pred_const_v  = constant_velocity(X_test)

model.eval()
with torch.no_grad():
    pred_gru_norm = model(torch.tensor(normalize(X_test), dtype=torch.float32)
                           .to(device)).cpu().numpy()
pred_gru = denormalize(pred_gru_norm)

methods = {
    'persistence':    pred_persist,
    'linear':         pred_linear,
    'const_velocity': pred_const_v,
    'gru':            pred_gru,
}


def euc(pred, gt):
    """Per-sample per-horizon Euclidean error.  pred, gt: (N, H, 3) -> (N, H)."""
    return np.linalg.norm(pred - gt, axis=2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

results = {}
e_persist_all = euc(pred_persist, Y_test)                # (N, H)
median_persist = float(np.median(e_persist_all))         # for skill score

for name, pred in methods.items():
    e_3d  = euc(pred, Y_test)                            # (N, H)
    e_ax  = np.abs(pred - Y_test)                        # (N, H, 3)

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

# GRU-only: per-subject median 3D error
test_subj = meta['subject'][test_mask]
e_gru     = euc(pred_gru, Y_test)
per_subject = {}
for subj in sorted(np.unique(test_subj)):
    m = test_subj == subj
    per_subject[subj] = {
        'n':      int(m.sum()),
        'median': float(np.median(e_gru[m])),
        'p95':    float(np.percentile(e_gru[m], 95)),
    }
results['gru']['per_subject_median_mm'] = per_subject

with open(os.path.join(_PHASE1, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

bar = '=' * 78
print(f'\n{bar}\nPHASE 1 RESULTS  (n_test = {Y_test.shape[0]}, horizon = {HORIZON} frames = 1.0 s)\n{bar}')

print(f'{"method":<18} {"median 3D":>12} {"mean 3D":>10} {"p95 3D":>10} {"skill":>8}')
for name in ['persistence', 'linear', 'const_velocity', 'gru']:
    r = results[name]
    print(f'  {name:<16}  {r["overall_median_3d_mm"]:>10.1f}   '
          f'{r["overall_mean_3d_mm"]:>8.1f}   '
          f'{r["overall_p95_3d_mm"]:>8.1f}   '
          f'{r["skill_score_vs_persistence"]:>6.3f}')

print('\nPer-horizon median 3D error (mm) — how accuracy degrades with lookahead:')
print(f'{"horizon (frame)":<18}' + ''.join(f'{h + 1:>7d}' for h in range(HORIZON)))
for name in ['persistence', 'linear', 'const_velocity', 'gru']:
    vals = results[name]['per_horizon_median_3d_mm']
    print(f'  {name:<16}' + ''.join(f'{v:>7.1f}' for v in vals))

print('\nPer-axis median error (mm, averaged across all horizons):')
for name in ['persistence', 'linear', 'const_velocity', 'gru']:
    by_ax = results[name]['per_axis_median_mm']
    print(f'  {name:<18}  x={by_ax["x"]:>5.1f}   y={by_ax["y"]:>5.1f}   z={by_ax["z"]:>5.1f}')

print('\nGRU per-subject median 3D error (mm):')
for subj, st in sorted(per_subject.items(), key=lambda kv: kv[1]['median']):
    print(f'  {subj:<14}  n={st["n"]:>5d}   median={st["median"]:>6.1f}   p95={st["p95"]:>6.1f}')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# 1. GRU training curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label='train')
ax.plot(val_losses, label='val', linestyle='--')
ax.set(xlabel='epoch', ylabel='MSE (normalized units)',
       title=f'GRU training curve   hidden={GRU_HIDDEN}, layers={GRU_LAYERS}')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(_PHASE1, 'training_curve.png'), dpi=100); plt.close()

# 2. Error vs horizon (3D)
fig, ax = plt.subplots(figsize=(8, 5))
horizons_sec = np.arange(1, HORIZON + 1) / 10.0
for name in ['persistence', 'linear', 'const_velocity', 'gru']:
    ax.plot(horizons_sec, results[name]['per_horizon_median_3d_mm'], marker='o', label=name)
ax.set(xlabel='forecast horizon (seconds)',
       ylabel='median 3D Euclidean error (mm)',
       title='Phase 1 — forecasting accuracy vs horizon')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(_PHASE1, 'error_vs_horizon.png'), dpi=100); plt.close()

# 3. Error vs horizon — z axis only (where the action is)
fig, ax = plt.subplots(figsize=(8, 5))
for name in ['persistence', 'linear', 'const_velocity', 'gru']:
    vals = results[name]['per_axis_per_horizon_median_mm']['z']
    ax.plot(horizons_sec, vals, marker='o', label=name)
ax.set(xlabel='forecast horizon (seconds)',
       ylabel='median |z| error (mm)',
       title='Phase 1 — z-axis error vs horizon  (gait-cadence regime)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(_PHASE1, 'error_vs_horizon_z.png'), dpi=100); plt.close()

# 4. Example trajectories (5 random test samples)
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(Y_test.shape[0], size=5, replace=False)
t_h = np.arange(-HISTORY + 1, 1)
t_f = np.arange(1, HORIZON + 1)
fig, axes = plt.subplots(5, 3, figsize=(15, 18))
for r, i in enumerate(sample_idx):
    hist = X_test[i]; truth = Y_test[i]
    for c, ax_name in enumerate('xyz'):
        ax = axes[r, c]
        ax.plot(t_h, hist[:, c], color='gray', alpha=0.6, label='history')
        ax.plot(t_f, truth[:, c], 'k-', linewidth=2, label='truth')
        ax.plot(t_f, pred_persist[i, :, c],  '--', alpha=0.8, label='persistence')
        ax.plot(t_f, pred_linear[i, :, c],   '--', alpha=0.8, label='linear')
        ax.plot(t_f, pred_const_v[i, :, c],  '--', alpha=0.8, label='const_velocity')
        ax.plot(t_f, pred_gru[i, :, c],      '-',  linewidth=1.5, label='gru')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set(xlabel='frame (relative to t)', ylabel=f'{ax_name} (mm)')
        ax.set_title(f'subj {test_subj[i]}   axis {ax_name}', fontsize=10)
        if r == 0 and c == 0:
            ax.legend(fontsize='small', loc='best')
plt.tight_layout()
plt.savefig(os.path.join(_PHASE1, 'example_trajectories.png'), dpi=100)
plt.close()

print(f'\nSaved metrics + plots to {_PHASE1}')

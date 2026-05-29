"""train_phase2_keypoints_v2.py — Phase 2 (delta keypoints, strengthened).

Phase 2 v1 (`train_phase2_keypoints.py`) showed that a small 64-hidden GRU
fed raw 21-joint pose history does *worse* than the same architecture fed
3-dim CoM history (66.3 vs 61.1 mm median 3D at 1-s horizon).

This v2 script tests whether the negative result holds under:
    - a stronger architecture (hidden=256, layers=2 vs 64, 1)
    - longer training (200 epochs vs 50)
    - delta-prediction targets (future CoM minus CoM(t), not absolute future)

The delta framing is critical: target magnitudes drop from O(1000 mm)
absolute positions to O(100 mm) displacements, making standardization
informative and removing a translation degree of freedom.

If THIS still loses to persistence, history-only is genuinely capped on
this dataset and tactile is the only remaining lever.

Same train/test split, outlier filter, and 17,218 / 5,255 sample counts
as Phase 1 and Phase 2 v1, so all numbers are directly comparable.
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

_HERE   = os.path.dirname(os.path.abspath(__file__))
_TRAIN  = os.path.dirname(_HERE)
_OUT    = os.path.join(_HERE, 'output')
_PHASE1 = os.path.join(_OUT, 'phase1')
_PHASE2 = os.path.join(_OUT, 'phase2_keypoints')
_PHASE2V2 = os.path.join(_OUT, 'phase2_keypoints_v2')
os.makedirs(_PHASE2V2, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HISTORY    = 100
HORIZON    = 10
SEED       = 42

GRU_HIDDEN = 256        # v1 was 64
GRU_LAYERS = 2          # v1 was 1
GRU_LR     = 1e-3
GRU_EPOCHS = 200        # v1 was 50
GRU_BATCH  = 256
VAL_FRAC   = 0.10

KP_DIM     = 21 * 3     # 63

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Load CoM results + session metadata (same as v1)
# ---------------------------------------------------------------------------

with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
    com_results = pickle.load(f)

com_gt    = com_results['com_gt']       # (T, 3) mm
kp_gt_mm  = com_results['kp_gt_mm']     # (T, 21, 3) mm
T         = len(com_gt)
assert kp_gt_mm.shape == (T, 21, 3)

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


# ---------------------------------------------------------------------------
# Build samples — identical windowing to Phase 1 / Phase 2 v1
# ---------------------------------------------------------------------------

def build_samples():
    X, Y, ref = [], [], []
    meta = {'subject': [], 'session': [], 'frame': [], 'split': []}
    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        valid_t = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t))
        for i, t in enumerate(valid_t):
            X.append(kp_gt_mm[t - HISTORY + 1 : t + 1].reshape(HISTORY, KP_DIM))
            Y.append(com_gt[t + 1            : t + 1 + HORIZON])
            ref.append(com_gt[t])           # CoM at the prediction anchor
            meta['subject'].append(subjects_per_sess[s])
            meta['session'].append(s)
            meta['frame'].append(t)
            meta['split'].append('train' if i < n_train_s else 'test')
    return (np.asarray(X, dtype=np.float64),
            np.asarray(Y, dtype=np.float64),
            np.asarray(ref, dtype=np.float64),
            {k: np.asarray(v) for k, v in meta.items()})


X, Y_abs, ref, meta = build_samples()
train_mask = meta['split'] == 'train'
test_mask  = meta['split'] == 'test'

X_train, Y_train_abs, ref_train = X[train_mask], Y_abs[train_mask], ref[train_mask]
X_test,  Y_test_abs,  ref_test  = X[test_mask],  Y_abs[test_mask],  ref[test_mask]

# Delta target: future_CoM - CoM(t)
Y_train_delta = Y_train_abs - ref_train[:, None, :]
Y_test_delta  = Y_test_abs  - ref_test[:, None, :]

print(f'samples generated   : {len(X):>6d}')
print(f'  train             : {train_mask.sum():>6d}')
print(f'  test              : {test_mask.sum():>6d}')
print(f'  Y_delta range     : {Y_train_delta.min():.1f} .. {Y_train_delta.max():.1f} mm (vs Y_abs '
      f'{Y_train_abs.min():.1f} .. {Y_train_abs.max():.1f})')

assert len(X) == 17218, f'expected 17218 samples, got {len(X)}'


# ---------------------------------------------------------------------------
# Persistence baseline (predict zero delta = no change)
# ---------------------------------------------------------------------------

pred_persistence = np.broadcast_to(ref_test[:, None, :],
                                   (len(ref_test), HORIZON, 3)).copy()


# ---------------------------------------------------------------------------
# Model — same scaffold, bigger
# ---------------------------------------------------------------------------

class DeltaForecaster(nn.Module):
    def __init__(self, input_dim=KP_DIM, hidden=GRU_HIDDEN, layers=GRU_LAYERS, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=0.1 if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, horizon * 3)

    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


# Standardize X (input) and Y_delta (target) using train stats
mean_X = X_train.reshape(-1, KP_DIM).mean(axis=0)
std_X  = X_train.reshape(-1, KP_DIM).std(axis=0)
std_X  = np.where(std_X < 1e-6, 1.0, std_X)
mean_Y = Y_train_delta.reshape(-1, 3).mean(axis=0)
std_Y  = Y_train_delta.reshape(-1, 3).std(axis=0)
std_Y  = np.where(std_Y < 1e-6, 1.0, std_Y)

def norm_X(arr):    return (arr - mean_X) / std_X
def norm_Y(arr):    return (arr - mean_Y) / std_Y
def denorm_Y(arr):  return arr * std_Y + mean_Y

print(f'  Y_delta mean      : {mean_Y}')
print(f'  Y_delta std       : {std_Y}')


# Split train -> tr / val
n_train = X_train.shape[0]
perm    = np.random.permutation(n_train)
n_val   = int(n_train * VAL_FRAC)
val_idx = perm[:n_val]
tr_idx  = perm[n_val:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  device            : {device}')

Xt = torch.tensor(norm_X(X_train),       dtype=torch.float32)
Yt = torch.tensor(norm_Y(Y_train_delta), dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xt[tr_idx], Yt[tr_idx]),
                          batch_size=GRU_BATCH, shuffle=True)
X_val = Xt[val_idx].to(device)
Y_val = Yt[val_idx].to(device)

model     = DeltaForecaster().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
criterion = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
print(f'\nTraining Phase 2 v2 GRU ({n_params} params, {GRU_EPOCHS} epochs)...')

train_losses, val_losses = [], []
best_val = float('inf')
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
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), os.path.join(_PHASE2V2, 'gru_model_best.pt'))

    if epoch % 20 == 0 or epoch == GRU_EPOCHS - 1:
        print(f'  epoch {epoch:3d}/{GRU_EPOCHS - 1}  '
              f'train={avg_train:.5f}  val={avg_val:.5f}  best_val={best_val:.5f}')

# Reload best model for eval
model.load_state_dict(torch.load(os.path.join(_PHASE2V2, 'gru_model_best.pt'),
                                 map_location=device, weights_only=False))
torch.save(model.state_dict(), os.path.join(_PHASE2V2, 'gru_model.pt'))


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

print('\nEvaluating on the test set...')
model.eval()
with torch.no_grad():
    pred_delta_norm = model(torch.tensor(norm_X(X_test), dtype=torch.float32)
                             .to(device)).cpu().numpy()
pred_delta = denorm_Y(pred_delta_norm)
pred_phase2_v2 = pred_delta + ref_test[:, None, :]      # back to absolute CoM space

methods = {
    'persistence':         pred_persistence,
    'phase2_v2_gru_kp':    pred_phase2_v2,
}


# Try loading Phase 1 GRU (CoM input) and Phase 2 v1 GRU (kp input, absolute) for direct comparison
def safe_load_phase1():
    pt = os.path.join(_PHASE1, 'gru_model.pt')
    if not os.path.exists(pt):
        return None
    from train_phase1 import TrajForecaster as P1
    m = P1().to(device)
    m.load_state_dict(torch.load(pt, map_location=device, weights_only=False))
    m.eval()
    train_frames = meta['frame'][train_mask]
    com_hist_train = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in train_frames], axis=0)
    com_fut_train  = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_frames], axis=0)
    pool = np.concatenate([com_hist_train.reshape(-1, 3), com_fut_train.reshape(-1, 3)], axis=0)
    p1_mean = pool.mean(axis=0)
    p1_std  = pool.std(axis=0); p1_std = np.where(p1_std < 1e-6, 1.0, p1_std)
    test_frames = meta['frame'][test_mask]
    com_hist_test = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in test_frames], axis=0)
    with torch.no_grad():
        ph1_in = torch.tensor((com_hist_test - p1_mean) / p1_std, dtype=torch.float32).to(device)
        ph1_pred = m(ph1_in).cpu().numpy() * p1_std + p1_mean
    return ph1_pred

def safe_load_phase2_v1():
    pt = os.path.join(_PHASE2, 'gru_model.pt')
    if not os.path.exists(pt):
        return None
    from train_phase2_keypoints import TrajForecaster as P2
    m = P2().to(device)
    m.load_state_dict(torch.load(pt, map_location=device, weights_only=False))
    m.eval()
    # v1 used absolute target with X+Y co-standardization — reproduce here
    p2_pool_X = X_train.reshape(-1, KP_DIM)
    p2_pool_Y_abs = Y_train_abs.reshape(-1, 3)
    p2_mean_X = p2_pool_X.mean(axis=0); p2_std_X = p2_pool_X.std(axis=0)
    p2_std_X = np.where(p2_std_X < 1e-6, 1.0, p2_std_X)
    p2_mean_Y = p2_pool_Y_abs.mean(axis=0); p2_std_Y = p2_pool_Y_abs.std(axis=0)
    p2_std_Y = np.where(p2_std_Y < 1e-6, 1.0, p2_std_Y)
    with torch.no_grad():
        p2_in = torch.tensor((X_test - p2_mean_X) / p2_std_X, dtype=torch.float32).to(device)
        p2_pred = m(p2_in).cpu().numpy() * p2_std_Y + p2_mean_Y
    return p2_pred

try:
    p1 = safe_load_phase1()
    if p1 is not None:
        methods['phase1_gru_com'] = p1
        print('  loaded Phase 1 GRU (CoM input) for comparison')
except Exception as e:
    print(f'  skip Phase 1 comparison: {e}')

try:
    p2v1 = safe_load_phase2_v1()
    if p2v1 is not None:
        methods['phase2_v1_gru_kp'] = p2v1
        print('  loaded Phase 2 v1 GRU (kp input, absolute) for comparison')
except Exception as e:
    print(f'  skip Phase 2 v1 comparison: {e}')


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def euc(pred, gt):
    return np.linalg.norm(pred - gt, axis=2)


results = {}
e_persist_all  = euc(pred_persistence, Y_test_abs)
median_persist = float(np.median(e_persist_all))

for name, pred in methods.items():
    e_3d = euc(pred, Y_test_abs)
    e_ax = np.abs(pred - Y_test_abs)
    results[name] = {
        'overall_median_3d_mm':       float(np.median(e_3d)),
        'overall_mean_3d_mm':         float(np.mean(e_3d)),
        'overall_p95_3d_mm':          float(np.percentile(e_3d, 95)),
        'per_horizon_median_3d_mm':   [float(np.median(e_3d[:, h]))    for h in range(HORIZON)],
        'per_horizon_p95_3d_mm':      [float(np.percentile(e_3d[:, h], 95)) for h in range(HORIZON)],
        'per_axis_median_mm':         {ax: float(np.median(e_ax[:, :, i])) for i, ax in enumerate('xyz')},
        'per_axis_per_horizon_median_mm': {
            ax: [float(np.median(e_ax[:, h, i])) for h in range(HORIZON)]
            for i, ax in enumerate('xyz')
        },
        'skill_score_vs_persistence': float(np.median(e_3d) / median_persist),
    }

test_subj = meta['subject'][test_mask]
e_p2v2 = euc(pred_phase2_v2, Y_test_abs)
per_subject = {}
for subj in sorted(np.unique(test_subj)):
    m = test_subj == subj
    per_subject[subj] = {
        'n':      int(m.sum()),
        'median': float(np.median(e_p2v2[m])),
        'p95':    float(np.percentile(e_p2v2[m], 95)),
    }
results['phase2_v2_gru_kp']['per_subject_median_mm'] = per_subject

with open(os.path.join(_PHASE2V2, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

bar = '=' * 78
print(f'\n{bar}\nPHASE 2 v2 (delta keypoints, hidden=256, 2L, 200ep, delta target) RESULTS\n{bar}')
print(f'(n_test = {Y_test_abs.shape[0]}, horizon = {HORIZON} frames = 1.0 s)')

ordered = ['persistence']
if 'phase1_gru_com'    in results: ordered.append('phase1_gru_com')
if 'phase2_v1_gru_kp'  in results: ordered.append('phase2_v1_gru_kp')
ordered.append('phase2_v2_gru_kp')

print(f'\n{"method":<22} {"median 3D":>12} {"mean 3D":>10} {"p95 3D":>10} {"skill":>8}')
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

print('\nPhase 2 v2 GRU per-subject median 3D error (mm):')
for subj, st in sorted(per_subject.items(), key=lambda kv: kv[1]['median']):
    print(f'  {subj:<14}  n={st["n"]:>5d}   median={st["median"]:>6.1f}   p95={st["p95"]:>6.1f}')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label='train')
ax.plot(val_losses, label='val', linestyle='--')
ax.axvline(int(np.argmin(val_losses)), color='gray', linestyle=':', alpha=0.5,
           label=f'best val @ ep {int(np.argmin(val_losses))}')
ax.set(xlabel='epoch', ylabel='MSE on standardized delta',
       title=f'Phase 2 v2 GRU   hidden={GRU_HIDDEN}, layers={GRU_LAYERS}, delta target')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(_PHASE2V2, 'training_curve.png'), dpi=100); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
hs = np.arange(1, HORIZON + 1) / 10.0
for name in ordered:
    ax.plot(hs, results[name]['per_horizon_median_3d_mm'], marker='o', label=name)
ax.set(xlabel='forecast horizon (seconds)',
       ylabel='median 3D Euclidean error (mm)',
       title='Phase 2 v2 — forecasting accuracy vs horizon')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(_PHASE2V2, 'error_vs_horizon.png'), dpi=100); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for name in ordered:
    vals = results[name]['per_axis_per_horizon_median_mm']['z']
    ax.plot(hs, vals, marker='o', label=name)
ax.set(xlabel='forecast horizon (seconds)',
       ylabel='median |z| error (mm)',
       title='Phase 2 v2 — z-axis error vs horizon')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(_PHASE2V2, 'error_vs_horizon_z.png'), dpi=100); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
medians = [results[n]['overall_median_3d_mm']       for n in ordered]
skills  = [results[n]['skill_score_vs_persistence'] for n in ordered]
colors  = ['gray', 'tab:blue', 'tab:orange', 'tab:red'][:len(ordered)]
axes[0].bar(ordered, medians, color=colors)
axes[0].set(ylabel='median 3D error (mm)', title='Median 3D error (1-s horizon)')
axes[0].grid(alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=30)
axes[1].bar(ordered, skills, color=colors)
axes[1].axhline(1.0, color='black', linestyle='--', alpha=0.5, label='persistence floor')
axes[1].set(ylabel='skill score (lower = better)', title='Skill vs persistence')
axes[1].grid(alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(_PHASE2V2, 'comparison_all_models.png'), dpi=100)
plt.close()

print(f'\nSaved metrics + plots to {_PHASE2V2}')

"""train_phase2_tactile.py — Phase 2 (beta): tactile-only CoM forecasting.

The central scientific question of the project: does tactile carry
forecastable signal beyond what CoM / pose history alone provides?

Pipeline:
    tactile(t-99 : t) -> small 2D CNN encoder (per frame) -> 128-dim feature
                     -> GRU(128) over the 100-frame feature sequence
                     -> linear head -> 10 future CoM deltas relative to CoM(t)

The model never sees CoM directly. Tactile is the only input modality.
Targets are deltas (future - CoM(t)) for translation invariance; at eval
we add CoM(t) back to recover absolute predictions. CoM(t) is itself
obtainable from the *current* tactile frame via the existing compute_com
pipeline, so this remains a deployment-realistic setup.

Sample-index alignment matches Phase 1 / Phase 2 v1 / v2 exactly
(17,218 total / 11,963 train / 5,255 test), so all numbers are directly
comparable to the prior runs.

Cache: pre-extracts all 32,600 tactile frames into a single (T, 96, 96)
float32 numpy array on disk (~1.2 GB). One-time cost (~5 min I/O).

Module-level definitions (importable without side effects):
    TactileEncoder, TactileForecaster   model classes
    extract_tactile_cache               (function defined here, called in main())
    HISTORY, HORIZON, SEED, ...         constants
All heavy execution lives inside main(), so import does not build the cache
or trigger training.
"""

import os
import sys
import json
import pickle
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE   = os.path.dirname(os.path.abspath(__file__))   # .../train/com
_TRAIN  = os.path.dirname(_HERE)                       # .../train
sys.path.insert(0, _TRAIN)                             # so threeD_dataLoader_batch imports

_OUT       = os.path.join(_HERE, 'output')
_TACTILE   = os.path.join(_OUT, 'phase2_tactile')
_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HISTORY      = 100
HORIZON      = 10
SEED         = 42

FEATURE_DIM  = 128       # tactile-encoder output per frame
GRU_HIDDEN   = 128
GRU_LAYERS   = 1
LR           = 1e-3
EPOCHS       = 50
BATCH        = 64        # smaller than Phase 1/2 because tactile inputs are heavier
VAL_FRAC     = 0.10


# ---------------------------------------------------------------------------
# Cache builder (function definition only — call happens inside main())
# ---------------------------------------------------------------------------

def extract_tactile_cache():
    """Iterate every per-frame .p file, stack tactile into (T, 96, 96), cache."""
    from threeD_dataLoader_batch import sample_data

    if os.path.exists(_CACHE_NPY):
        arr = np.load(_CACHE_NPY)
        print(f'tactile cache exists: {_CACHE_NPY}  shape={arr.shape}')
        return arr

    test_dir = os.path.join(_TRAIN, 'singlePerson_test') + os.sep
    ds = sample_data(test_dir, 0, [], 1)           # window=0 returns the single tactile frame
    n = len(ds)
    print(f'extracting tactile from {n} per-frame .p files into {_CACHE_NPY}...')
    arr = np.zeros((n, 96, 96), dtype=np.float32)
    start = time.time()
    for i in range(n):
        tactile, _, _, _, _ = ds[i]                # tactile shape (1, 96, 96)
        arr[i] = tactile[0]
        if i % 2000 == 0 or i == n - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f'  {i+1}/{n}  ({100*(i+1)/n:.1f}%)  {elapsed:.0f}s elapsed  ETA {eta:.0f}s', flush=True)
    np.save(_CACHE_NPY, arr)
    print(f'cache written: {_CACHE_NPY}  ({arr.nbytes / 1e9:.2f} GB)')
    return arr


# ---------------------------------------------------------------------------
# Model classes (module level — importable without triggering training)
# ---------------------------------------------------------------------------

class TactileEncoder(nn.Module):
    """Per-frame 2D CNN encoder. (96, 96) -> FEATURE_DIM."""
    def __init__(self, out_dim=FEATURE_DIM):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1,   32, kernel_size=5, stride=2, padding=2),   # 96 -> 48
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  64, kernel_size=5, stride=2, padding=2),   # 48 -> 24
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 24 -> 12
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                  # -> (B, 128, 1, 1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):                                             # x: (B, 1, 96, 96)
        h = self.body(x).flatten(1)
        return self.proj(h)


class TactileForecaster(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden=GRU_HIDDEN, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.encoder = TactileEncoder(feature_dim)
        self.gru     = nn.GRU(feature_dim, hidden, batch_first=True)
        self.proj    = nn.Linear(hidden, horizon * 3)

    def forward(self, x):                                             # x: (B, HISTORY, 96, 96)
        B, Tlen, H, W = x.shape
        flat = x.reshape(B * Tlen, 1, H, W)
        feat = self.encoder(flat).reshape(B, Tlen, -1)
        _, h = self.gru(feat)
        return self.proj(h[-1]).view(B, self.horizon, 3)


# ---------------------------------------------------------------------------
# Main execution — only runs when invoked as a script, not on import
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_TACTILE, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Pre-extract all tactile frames (one-time cache)
    tactile_all = extract_tactile_cache()
    T = tactile_all.shape[0]
    assert tactile_all.shape == (T, 96, 96)

    # Load com_gt + session metadata
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

    # Build sample indices — same per-session 70/30 split as Phase 1 / 2
    # We store the *frame centers* and read tactile slices lazily at training time
    # to keep memory bounded.
    def build_indices():
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
        return np.asarray(centers), {k: np.asarray(v) for k, v in sources.items()}

    centers, meta = build_indices()
    train_mask = meta['split'] == 'train'
    test_mask  = meta['split'] == 'test'
    N = len(centers)
    print(f'\nsamples generated   : {N}')
    print(f'  train             : {train_mask.sum()}')
    print(f'  test              : {test_mask.sum()}')
    assert N == 17218

    # Precompute the reference CoM(t) and target deltas (small — fits easily in memory)
    ref_all      = com_gt[centers]                                        # (N, 3)
    Y_abs_all    = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in centers], axis=0)
    Y_delta_all  = Y_abs_all - ref_all[:, None, :]                        # (N, HORIZON, 3)

    ref_train, ref_test = ref_all[train_mask], ref_all[test_mask]
    Y_abs_train, Y_abs_test = Y_abs_all[train_mask], Y_abs_all[test_mask]
    Y_delta_train, Y_delta_test = Y_delta_all[train_mask], Y_delta_all[test_mask]

    # Persistence baseline
    pred_persistence = np.broadcast_to(ref_test[:, None, :], (len(ref_test), HORIZON, 3)).copy()

    # Standardization stats — on tactile (normalize once globally) and on Y_delta
    # Tactile is already roughly in [0, 1] (normalized by upstream preprocessing), but
    # the actual range differs per subject. Use train-data global mean/std.
    print('\ncomputing tactile train-set statistics (over a sample of windows for memory)...')
    # Sample a subset of train centers to estimate tactile mean/std without loading
    # all 11k * 100 = 1.1M tactile frames into RAM.
    sample_centers = centers[train_mask][np.random.permutation(train_mask.sum())[:1000]]
    sample_tactile = np.concatenate([tactile_all[t - HISTORY + 1 : t + 1] for t in sample_centers], axis=0)
    tactile_mean = float(sample_tactile.mean())
    tactile_std  = float(sample_tactile.std())
    print(f'  tactile mean={tactile_mean:.4f}  std={tactile_std:.4f}')

    mean_Y = Y_delta_train.reshape(-1, 3).mean(axis=0)
    std_Y  = Y_delta_train.reshape(-1, 3).std(axis=0)
    std_Y  = np.where(std_Y < 1e-6, 1.0, std_Y)
    print(f'  Y_delta mean: {mean_Y}')
    print(f'  Y_delta std : {std_Y}')

    # Dataset that lazily slices tactile windows
    class TactileDataset(Dataset):
        def __init__(self, centers, Y_delta):
            self.centers = centers
            self.Y_delta = Y_delta

        def __len__(self):
            return len(self.centers)

        def __getitem__(self, idx):
            t = int(self.centers[idx])
            window = tactile_all[t - HISTORY + 1 : t + 1]                # (HISTORY, 96, 96)
            # Standardize on the fly
            window = (window - tactile_mean) / tactile_std
            target = (self.Y_delta[idx] - mean_Y) / std_Y                # (HORIZON, 3)
            return (torch.from_numpy(window.astype(np.float32)),
                    torch.from_numpy(target.astype(np.float32)))

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  device            : {device}')

    train_centers = centers[train_mask]
    n_train_tot = len(train_centers)
    perm = np.random.permutation(n_train_tot)
    n_val = int(n_train_tot * VAL_FRAC)
    val_idx_local = perm[:n_val]
    tr_idx_local  = perm[n_val:]

    train_ds = TactileDataset(train_centers[tr_idx_local],  Y_delta_train[tr_idx_local])
    val_ds   = TactileDataset(train_centers[val_idx_local], Y_delta_train[val_idx_local])
    test_ds  = TactileDataset(centers[test_mask],           Y_delta_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model     = TactileForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    n_params  = sum(p.numel() for p in model.parameters())
    print(f'\nTraining tactile forecaster ({n_params} params, {EPOCHS} epochs)...')

    train_losses, val_losses = [], []
    best_val = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        n_b = 0
        epoch_start = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_b += 1
        avg_train = total / n_b

        model.eval()
        with torch.no_grad():
            vtot = 0.0; vb = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtot += criterion(model(xb), yb).item()
                vb += 1
        avg_val = vtot / vb

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(_TACTILE, 'tactile_model_best.pt'))

        print(f'  epoch {epoch:3d}/{EPOCHS - 1}  '
              f'train={avg_train:.5f}  val={avg_val:.5f}  best_val={best_val:.5f}  '
              f'({time.time() - epoch_start:.1f}s)', flush=True)

    # Reload best
    model.load_state_dict(torch.load(os.path.join(_TACTILE, 'tactile_model_best.pt'),
                                     map_location=device, weights_only=False))
    torch.save(model.state_dict(), os.path.join(_TACTILE, 'tactile_model.pt'))

    # Evaluate
    print('\nEvaluating on the test set...')
    model.eval()
    preds_delta_norm = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds_delta_norm.append(model(xb.to(device)).cpu().numpy())
    pred_delta_norm = np.concatenate(preds_delta_norm, axis=0)
    pred_delta = pred_delta_norm * std_Y + mean_Y
    pred_tactile = pred_delta + ref_test[:, None, :]

    methods = {'persistence': pred_persistence, 'phase2_tactile': pred_tactile}

    def euc(pred, gt):
        return np.linalg.norm(pred - gt, axis=2)

    results = {}
    e_persist = euc(pred_persistence, Y_abs_test)
    median_persist = float(np.median(e_persist))

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

    with open(os.path.join(_TACTILE, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Console summary
    bar = '=' * 78
    print(f'\n{bar}\nPHASE 2 (beta) TACTILE-ONLY RESULTS  '
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

    print('\nTactile-only per-subject median 3D error (mm):')
    for subj, st in sorted(per_subject.items(), key=lambda kv: kv[1]['median']):
        print(f'  {subj:<14}  n={st["n"]:>5d}   median={st["median"]:>6.1f}   p95={st["p95"]:>6.1f}')

    # Plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='train')
    ax.plot(val_losses, label='val', linestyle='--')
    ax.set(xlabel='epoch', ylabel='MSE on standardized delta',
           title=f'Phase 2 tactile-only — encoder + GRU({GRU_HIDDEN})')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(_TACTILE, 'training_curve.png'), dpi=100); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    hs = np.arange(1, HORIZON + 1) / 10.0
    for name in ordered:
        ax.plot(hs, results[name]['per_horizon_median_3d_mm'], marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median 3D Euclidean error (mm)',
           title='Phase 2 (tactile) — error vs horizon')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(_TACTILE, 'error_vs_horizon.png'), dpi=100); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ordered:
        vals = results[name]['per_axis_per_horizon_median_mm']['z']
        ax.plot(hs, vals, marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median |z| error (mm)',
           title='Phase 2 (tactile) — z-axis error vs horizon')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(_TACTILE, 'error_vs_horizon_z.png'), dpi=100); plt.close()

    print(f'\nSaved metrics + plots to {_TACTILE}')


if __name__ == '__main__':
    main()

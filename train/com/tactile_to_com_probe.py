"""tactile_to_com_probe.py — diagnostic #2: tactile(t) → CoM(t).

Scientific question:
    Does tactile carry the information to INFER current CoM (same-frame),
    even if it can't PREDICT future CoM?

    If a simple linear probe from a single tactile frame to its own-frame CoM
    achieves near-zero error, then tactile *does* contain CoM information --
    just not the kind that's useful for forecasting. That would settle the
    'tactile is contemporaneous, not leading' hypothesis from diagnostic #1
    from a different angle, and would establish tactile-based 'instantaneous
    CoM inference' as a real positive result for the project.

Method:
    Linear probe   : Linear(9216, 3)               on standardized tactile
    Tiny MLP probe : Linear(9216, 128) -> ReLU -> Linear(128, 3)
    Baselines      : predict mean(CoM)  (constant, ~ session-grand-mean error)

Standardization:
    Tactile : scalar mean/std from the train-pool 1000-window subset (same as
              beta / gamma / epsilon, loaded from tactile_stats.json).
    Target  : per-axis mean/std of CoM on the train pool.

Train/test split:
    Per-session 70/30 chronological, same outlier filter as forecasting
    scripts. NOT the forecasting windows -- here every individual frame
    (subject to outlier filter) is a sample. Train ~22.8k frames, test ~9.8k.

Outputs (under train/com/output/tactile_to_com_probe/):
    metrics.json
    predicted_vs_true.png
    training_curve.png
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
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_OUT   = os.path.join(_HERE, 'output')
_PROBE = os.path.join(_OUT, 'tactile_to_com_probe')

_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')
_STATS     = os.path.join(_OUT, 'tactile_stats.json')

SEED         = 42
BATCH        = 256
EPOCHS_LIN   = 30
EPOCHS_MLP   = 30
LR           = 1e-3
VAL_FRAC     = 0.10
MLP_HIDDEN   = 128
FLAT_DIM     = 96 * 96                                   # 9216


class LinearProbe(nn.Module):
    def __init__(self, in_dim=FLAT_DIM, out_dim=3):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.head(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim=FLAT_DIM, hidden=MLP_HIDDEN, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description='Diagnostic #2: tactile(t) -> CoM(t) probe')
    parser.add_argument('--output-dir', type=str, default=_PROBE)
    parser.add_argument('--epochs-linear', type=int, default=EPOCHS_LIN)
    parser.add_argument('--epochs-mlp',    type=int, default=EPOCHS_MLP)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device     : {device}')
    print(f'output_dir : {args.output_dir}')

    # ---- Load inputs ----
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY} -- build with train_phase2_tactile.py')
    if not os.path.exists(_STATS):
        raise SystemExit(f'no tactile stats at {_STATS} -- run seed_tactile_stats.py')

    print(f'loading tactile cache: {_CACHE_NPY}')
    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    T = tactile_all.shape[0]
    assert tactile_all.shape == (T, 96, 96)

    with open(_STATS) as f:
        s = json.load(f)
    tactile_mean = float(s['tactile_mean'])
    tactile_std  = float(s['tactile_std'])
    print(f'tactile mean={tactile_mean:.4f}  std={tactile_std:.4f}')

    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)
    com_gt = com_results['com_gt']
    assert len(com_gt) == T

    with open(os.path.join(_TRAIN, 'singlePerson_test', 'log.p'), 'rb') as f:
        log = pickle.load(f)
    n_sessions = len(log) - 1
    print(f'n_sessions = {n_sessions}')

    # ---- Build per-frame train/test split (same 70/30 chronological per session) ----
    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))

    frame_idx, splits = [], []
    for sess in range(n_sessions):
        a, b = log[sess], log[sess + 1]
        valid = [t for t in range(a, b) if not gt_outliers[t]]
        n_train_s = int(0.7 * len(valid))
        for i, t in enumerate(valid):
            frame_idx.append(t)
            splits.append('train' if i < n_train_s else 'test')
    frame_idx = np.asarray(frame_idx)
    splits = np.asarray(splits)
    train_mask = splits == 'train'
    test_mask  = splits == 'test'
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    print(f'frames: total={len(frame_idx)}, train={n_train}, test={n_test}')

    train_frames = frame_idx[train_mask]
    test_frames  = frame_idx[test_mask]

    # ---- Target standardization (train pool only) ----
    Y_train_pool = com_gt[train_frames]                              # (n_train, 3)
    mean_Y = Y_train_pool.mean(axis=0)
    std_Y  = Y_train_pool.std(axis=0)
    std_Y  = np.where(std_Y < 1e-6, 1.0, std_Y)
    print(f'CoM train mean={mean_Y}')
    print(f'CoM train std ={std_Y}')

    # ---- Dataset (lazy tactile flatten + standardize) ----
    class TactileFrameDataset(Dataset):
        def __init__(self, frames):
            self.frames = frames
        def __len__(self):
            return len(self.frames)
        def __getitem__(self, idx):
            t = int(self.frames[idx])
            x = np.asarray(tactile_all[t]).astype(np.float32)         # (96, 96)
            x = (x - tactile_mean) / tactile_std
            y = (com_gt[t] - mean_Y) / std_Y
            return (torch.from_numpy(x.reshape(-1)),                  # (9216,)
                    torch.from_numpy(y.astype(np.float32)))

    # Train / val split
    perm = np.random.permutation(n_train)
    n_val = int(n_train * VAL_FRAC)
    val_local = perm[:n_val]
    tr_local  = perm[n_val:]

    train_ds = TactileFrameDataset(train_frames[tr_local])
    val_ds   = TactileFrameDataset(train_frames[val_local])
    test_ds  = TactileFrameDataset(test_frames)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    # ---- Train helper ----
    def train_probe(model, name, epochs):
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'\n[{name}] params = {n_params}')
        train_curve, val_curve = [], []
        best_val = float('inf'); best_state = None
        for epoch in range(epochs):
            model.train()
            total = 0.0; nb = 0
            t0 = time.time()
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = ((pred - yb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item()); nb += 1
            tr = total / nb
            model.eval()
            with torch.no_grad():
                vt = 0.0; vb = 0
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    vt += float(((model(xb) - yb) ** 2).mean().item()); vb += 1
            vl = vt / vb
            train_curve.append(tr); val_curve.append(vl)
            if vl < best_val:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f'  [{name}] epoch {epoch:3d}/{epochs - 1}  '
                  f'train={tr:.5f}  val={vl:.5f}  best_val={best_val:.5f}  ({time.time() - t0:.1f}s)',
                  flush=True)
        model.load_state_dict(best_state)
        return model, train_curve, val_curve, best_val

    # ---- Train linear probe ----
    linear_model, lin_tc, lin_vc, lin_bv = train_probe(LinearProbe(), 'linear', args.epochs_linear)
    torch.save({'state_dict': linear_model.state_dict(),
                'train_curve': lin_tc, 'val_curve': lin_vc, 'best_val_mse': lin_bv},
               os.path.join(args.output_dir, 'linear_probe.pt'))

    # ---- Train MLP probe ----
    mlp_model, mlp_tc, mlp_vc, mlp_bv = train_probe(MLPProbe(), 'mlp', args.epochs_mlp)
    torch.save({'state_dict': mlp_model.state_dict(),
                'train_curve': mlp_tc, 'val_curve': mlp_vc, 'best_val_mse': mlp_bv},
               os.path.join(args.output_dir, 'mlp_probe.pt'))

    # ---- Eval on test set ----
    def predict(model):
        model.eval()
        out = np.zeros((n_test, 3), dtype=np.float64)
        i = 0
        with torch.no_grad():
            for xb, _ in test_loader:
                p = model(xb.to(device)).cpu().numpy()
                p = p * std_Y + mean_Y                                # de-standardize
                out[i:i + len(p)] = p
                i += len(p)
        return out

    pred_lin = predict(linear_model)
    pred_mlp = predict(mlp_model)
    true_test = com_gt[test_frames]                                   # (n_test, 3)

    # Baseline: predict mean
    pred_mean = np.broadcast_to(mean_Y[None, :], (n_test, 3)).copy()

    def metrics(pred, name):
        err_axis = np.abs(pred - true_test)                           # (n_test, 3)
        err_3d   = np.linalg.norm(pred - true_test, axis=1)            # (n_test,)
        return {
            'name':                  name,
            'median_3d_mm':          float(np.median(err_3d)),
            'mean_3d_mm':            float(np.mean(err_3d)),
            'p95_3d_mm':             float(np.percentile(err_3d, 95)),
            'median_x_mm':           float(np.median(err_axis[:, 0])),
            'median_y_mm':           float(np.median(err_axis[:, 1])),
            'median_z_mm':           float(np.median(err_axis[:, 2])),
            'rms_3d_mm':             float(np.sqrt((err_3d ** 2).mean())),
        }

    results = {
        'mean_baseline': metrics(pred_mean, 'mean_baseline'),
        'linear_probe':  metrics(pred_lin,  'linear_probe'),
        'mlp_probe':     metrics(pred_mlp,  'mlp_probe'),
    }

    # ---- Console summary ----
    bar = '=' * 78
    print(f'\n{bar}\nTACTILE(t) -> CoM(t) RESULTS  (n_test = {n_test})\n{bar}')
    print(f'{"method":<18} {"med 3D":>9} {"mean 3D":>9} {"p95 3D":>9} {"med x":>8} {"med y":>8} {"med z":>8}')
    for name in ['mean_baseline', 'linear_probe', 'mlp_probe']:
        r = results[name]
        print(f'  {name:<16}  '
              f'{r["median_3d_mm"]:>7.1f}  {r["mean_3d_mm"]:>7.1f}  {r["p95_3d_mm"]:>7.1f}  '
              f'{r["median_x_mm"]:>6.1f}  {r["median_y_mm"]:>6.1f}  {r["median_z_mm"]:>6.1f}')

    # Decision hint
    print(f'\nReading:')
    if results['linear_probe']['median_3d_mm'] < 30:
        print('  Linear probe error < 30 mm 3D -> tactile carries STRONG instantaneous-CoM information.')
        print('  Together with diagnostic #1 (lag), this would confirm: tactile knows current state,')
        print('  not future state. Project framing should pivot to instantaneous tactile->state tasks.')
    elif results['linear_probe']['median_3d_mm'] < 100:
        print('  Linear probe error 30-100 mm 3D -> tactile carries some CoM info but is noisy.')
        print('  Possibly: non-linear probe (MLP) closes the gap. Check MLP result above.')
    else:
        print('  Linear probe error > 100 mm 3D -> tactile alone does NOT pin down current CoM.')
        print('  This would be a surprising result that needs a follow-up (check data alignment).')

    # ---- Save ----
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'n_train':       int(train_mask.sum()),
            'n_val':         int(n_val),
            'n_test':        int(n_test),
            'tactile_mean':  tactile_mean,
            'tactile_std':   tactile_std,
            'com_train_mean': mean_Y.tolist(),
            'com_train_std':  std_Y.tolist(),
            'best_val_mse':  {'linear': lin_bv, 'mlp': mlp_bv},
            'methods':       results,
        }, f, indent=2)
    print(f'\nsaved {os.path.join(args.output_dir, "metrics.json")}')

    # ---- Plots ----
    # Training curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lin_tc, label='linear train', color='tab:blue')
    ax.plot(lin_vc, label='linear val',   color='tab:blue', linestyle='--')
    ax.plot(mlp_tc, label='MLP train',    color='tab:orange')
    ax.plot(mlp_vc, label='MLP val',      color='tab:orange', linestyle='--')
    ax.set(xlabel='epoch', ylabel='MSE (standardized CoM)',
           title='tactile(t) -> CoM(t) probes')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curve.png'), dpi=100)
    plt.close()

    # Predicted vs true scatter, per axis (use MLP since it's at least as good)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, ax_name in enumerate('xyz'):
        ax = axes[i]
        ax.scatter(true_test[:, i], pred_mlp[:, i], s=1, alpha=0.3, color='tab:orange', label='MLP')
        ax.scatter(true_test[:, i], pred_lin[:, i], s=1, alpha=0.3, color='tab:blue',   label='linear')
        lo = min(true_test[:, i].min(), pred_mlp[:, i].min())
        hi = max(true_test[:, i].max(), pred_mlp[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'k-', alpha=0.5, linewidth=1, label='y=x')
        ax.set(xlabel=f'true CoM {ax_name} (mm)', ylabel=f'predicted {ax_name} (mm)',
               title=f'{ax_name}-axis (test set, n={n_test})')
        ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'predicted_vs_true.png'), dpi=100)
    plt.close()


if __name__ == '__main__':
    main()

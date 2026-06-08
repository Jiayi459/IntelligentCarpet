"""train_phase2_gamma.py — Phase 2 (gamma): tactile + CoM-history fused forecaster.

Scientific question:
    "Conditional on already having access to the CoM history, does the raw
     tactile signal add forecasting-relevant information?"

Architecture (two-branch late fusion):

    tactile_history (B, 100, 96, 96)        com_history (B, 100, 3)
        │                                       │
        ▼                                       ▼
    TactileEncoder (per-frame)               (identity)
        │                                       │
    (B, 100, 128)                          (B, 100, 3)
        │                                       │
    GRU(128, 128)                           GRU(3, 64)         <- same as Phase 1
        │                                       │
      h_T : (B, 128)                       h_C : (B, 64)
        │                                       │
        └─────────────┬─────────────────────────┘
                      ▼
              concat -> (B, 192)
                      │
              Linear(192 -> 128) -> ReLU -> Linear(128 -> 30)
                      │
              delta_pred : (B, 10, 3)
                      │
                      ▼
            future_CoM = CoM(t) + denorm(delta_pred)

Total params ~190k (vs ~15k Phase 1, ~650k v2, ~150k phase2_tactile).

Training:
    Same per-session 70/30 split as Phase 1/2 (17,218 / 11,963 / 5,255 samples
    by construction). Same SEED, same outlier filter. 200 epochs, Adam lr=1e-3,
    batch 64 (tactile dominates memory), best-val checkpointing on val MSE in
    standardized-delta units.

Standardization (all from train pool only):
    - Tactile     : scalar mean/std from 1000-window subset (replayed RNG
                    matches phase2_tactile bit-for-bit so the tactile encoder
                    sees the same input distribution as in beta).
    - CoM history : 3-dim mean/std on (history + future) train pool (matches
                    Phase 1 exactly so the CoM branch's GRU sees the same
                    inputs as Phase 1's GRU would).
    - Delta target: 3-dim mean/std on Y_delta train pool (matches v2 and
                    phase2_tactile so the loss is directly comparable).

Evaluation outputs (under train/com/output/phase2_gamma/):
    metrics.json                            full + static + moving subsets
    gamma_model.pt, gamma_model_best.pt     end-of-training + best-val weights
    training_curve.png                      train/val MSE vs epoch
    error_vs_horizon.png                    per-horizon median for full / static / moving
    comparison_bars.png                     gamma + reference methods on each subset
    gamma_vs_phase1_per_horizon.png         the decisive plot

Run:
    python train/com/train_phase2_gamma.py                       # 200 ep, default output dir
    python train/com/train_phase2_gamma.py --epochs 50           # quick test
    python train/com/train_phase2_gamma.py --output-dir <other>  # alternate output dir

Module-level definitions (importable without side effects):
    GammaForecaster        the two-branch model class
    HISTORY, HORIZON, ...  constants
All heavy execution lives inside main(); `import train_phase2_gamma` is safe.
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
sys.path.insert(0, _TRAIN)
sys.path.insert(0, _HERE)

_OUT       = os.path.join(_HERE, 'output')
_GAMMA     = os.path.join(_OUT, 'phase2_gamma')
_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')


# ---------------------------------------------------------------------------
# Constants (match Phase 1 / Phase 2 conventions exactly)
# ---------------------------------------------------------------------------

HISTORY            = 100
HORIZON            = 10
SEED               = 42

TACTILE_FEATURE_DIM = 128       # per-frame tactile feature output (same as beta)
TACTILE_HIDDEN      = 128       # tactile GRU hidden state
COM_HIDDEN          = 64        # CoM GRU hidden state (same as Phase 1)
MLP_HIDDEN          = 128       # fusion-head MLP hidden
LR                  = 1e-3
EPOCHS              = 200
BATCH               = 64        # tactile branch dominates memory; same as beta
VAL_FRAC            = 0.10
MOVING_FRAC         = 0.30      # for the eval-time motion-subset split


# ---------------------------------------------------------------------------
# Model classes (module level — importable without triggering training)
# ---------------------------------------------------------------------------

class TactileEncoder(nn.Module):
    """Per-frame 2D CNN encoder. (96, 96) -> TACTILE_FEATURE_DIM.

    Identical to phase2_tactile's encoder so the encoder bottleneck issue
    carries over deliberately — gamma tests fusion, not encoder quality.
    """
    def __init__(self, out_dim=TACTILE_FEATURE_DIM):
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

    def forward(self, x):                                             # (B, 1, 96, 96)
        return self.proj(self.body(x).flatten(1))


class GammaForecaster(nn.Module):
    """Two-branch late-fusion forecaster: tactile + CoM history -> delta CoM."""
    def __init__(self,
                 tactile_feature_dim=TACTILE_FEATURE_DIM,
                 tactile_hidden=TACTILE_HIDDEN,
                 com_hidden=COM_HIDDEN,
                 mlp_hidden=MLP_HIDDEN,
                 horizon=HORIZON):
        super().__init__()
        self.horizon = horizon

        # Tactile branch
        self.tactile_encoder = TactileEncoder(tactile_feature_dim)
        self.tactile_gru     = nn.GRU(tactile_feature_dim, tactile_hidden, batch_first=True)

        # CoM branch (architecture matches Phase 1's GRU exactly)
        self.com_gru         = nn.GRU(3, com_hidden, batch_first=True)

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(tactile_hidden + com_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, horizon * 3),
        )

    def forward(self, tactile, com_history):
        """
        tactile     : (B, HISTORY, 96, 96)   already standardized (scalar)
        com_history : (B, HISTORY, 3)        already standardized (3-vector)
        returns delta_pred : (B, HORIZON, 3) in standardized-delta units
        """
        B, Tlen, H, W = tactile.shape
        flat = tactile.reshape(B * Tlen, 1, H, W)
        feat = self.tactile_encoder(flat).reshape(B, Tlen, -1)       # (B, T, 128)
        _, h_t = self.tactile_gru(feat)                              # (1, B, 128)
        h_t = h_t[-1]                                                 # (B, 128)

        _, h_c = self.com_gru(com_history)                            # (1, B, 64)
        h_c = h_c[-1]                                                 # (B, 64)

        fused = torch.cat([h_t, h_c], dim=1)                          # (B, 192)
        out = self.fusion(fused)                                      # (B, 30)
        return out.view(B, self.horizon, 3)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Phase 2 gamma: tactile + CoM-history fused forecaster.')
    parser.add_argument('--epochs',     type=int, default=EPOCHS,
                        help=f'training epochs (default {EPOCHS})')
    parser.add_argument('--output-dir', type=str, default=_GAMMA,
                        help=f'output directory (default {_GAMMA})')
    args = parser.parse_args()
    epochs     = args.epochs
    output_dir = args.output_dir
    print(f'epochs     : {epochs}')
    print(f'output_dir : {output_dir}')

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device     : {device}')

    # ---- Load tactile cache + CoM data ----
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY} -- build it with phase2_tactile.py first')
    print(f'loading tactile cache: {_CACHE_NPY}')
    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    T = tactile_all.shape[0]
    assert tactile_all.shape == (T, 96, 96)

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

    # ---- Build sample indices + 70/30 chronological split ----
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
    centers = np.asarray(centers)
    meta = {k: np.asarray(v) for k, v in sources.items()}
    train_mask = meta['split'] == 'train'
    test_mask  = meta['split'] == 'test'
    train_centers = centers[train_mask]
    test_centers  = centers[test_mask]
    test_subj     = meta['subject'][test_mask]
    n_train_tot   = len(train_centers)
    n_test        = len(test_centers)
    assert len(centers) == 17218
    print(f'samples: total={len(centers)}, train={n_train_tot}, test={n_test}')

    # ---- Targets (precomputed: small enough to fit in RAM) ----
    ref_all      = com_gt[centers]
    Y_abs_all    = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in centers], axis=0)
    Y_delta_all  = Y_abs_all - ref_all[:, None, :]
    Y_delta_train = Y_delta_all[train_mask]
    Y_delta_test  = Y_delta_all[test_mask]
    ref_test      = ref_all[test_mask]
    Y_abs_test    = Y_abs_all[test_mask]

    # Persistence baseline (no model)
    pred_persistence = np.broadcast_to(ref_test[:, None, :],
                                       (n_test, HORIZON, 3)).copy()

    # ---- Standardization stats (all from TRAIN POOL only) ----
    # 1. Tactile scalar mean/std -- REPRODUCE phase2_tactile's RNG-replayed sample
    #    so the tactile encoder sees identical input distribution as in beta.
    print('computing tactile train-set statistics...')
    sample_centers = train_centers[np.random.permutation(n_train_tot)[:1000]]
    sample_tactile = np.concatenate([tactile_all[t - HISTORY + 1 : t + 1] for t in sample_centers], axis=0)
    tactile_mean = float(sample_tactile.mean())
    tactile_std  = float(sample_tactile.std())
    print(f'  tactile mean={tactile_mean:.4f}  std={tactile_std:.4f}')

    # 2. CoM history mean/std -- match PHASE 1's standardization (train pool = history + future combined).
    com_hist_train = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in train_centers], axis=0)
    com_fut_train  = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_centers], axis=0)
    com_pool       = np.concatenate([com_hist_train.reshape(-1, 3), com_fut_train.reshape(-1, 3)], axis=0)
    com_mean = com_pool.mean(axis=0)
    com_std  = com_pool.std(axis=0); com_std = np.where(com_std < 1e-6, 1.0, com_std)
    print(f'  CoM history mean={com_mean}  std={com_std}')

    # 3. Delta target mean/std -- match v2 and phase2_tactile.
    mean_Y = Y_delta_train.reshape(-1, 3).mean(axis=0)
    std_Y  = Y_delta_train.reshape(-1, 3).std(axis=0); std_Y = np.where(std_Y < 1e-6, 1.0, std_Y)
    print(f'  Y_delta mean={mean_Y}  std={std_Y}')

    # Save the stats so eval-only restart (after a wall-time disconnect) can reload them exactly.
    np.savez(os.path.join(output_dir, 'stats.npz'),
             tactile_mean=tactile_mean, tactile_std=tactile_std,
             com_mean=com_mean, com_std=com_std,
             mean_Y=mean_Y, std_Y=std_Y)

    # Precompute standardized CoM history for every sample (small: 17218 * 100 * 3 * 8 bytes ~= 41 MB)
    com_hist_all      = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in centers], axis=0)
    com_hist_norm_all = (com_hist_all - com_mean) / com_std

    # ---- Dataset (lazy tactile slicing) ----
    class GammaDataset(Dataset):
        def __init__(self, indices_local):
            self.indices_local = indices_local                       # indices into the full `centers` array

        def __len__(self):
            return len(self.indices_local)

        def __getitem__(self, idx):
            i_global = int(self.indices_local[idx])
            t = int(centers[i_global])
            window = tactile_all[t - HISTORY + 1 : t + 1]
            window = (window - tactile_mean) / tactile_std
            com_h  = com_hist_norm_all[i_global]                     # (HISTORY, 3) already std
            target = (Y_delta_all[i_global] - mean_Y) / std_Y        # (HORIZON, 3)
            return (torch.from_numpy(window.astype(np.float32)),
                    torch.from_numpy(com_h.astype(np.float32)),
                    torch.from_numpy(target.astype(np.float32)))

    # Train / val split within the train indices (uses next np.random call -- same scheme as v2)
    train_global_idx = np.where(train_mask)[0]
    perm             = np.random.permutation(n_train_tot)
    n_val            = int(n_train_tot * VAL_FRAC)
    val_local        = train_global_idx[perm[:n_val]]
    tr_local         = train_global_idx[perm[n_val:]]
    test_global_idx  = np.where(test_mask)[0]

    train_ds = GammaDataset(tr_local)
    val_ds   = GammaDataset(val_local)
    test_ds  = GammaDataset(test_global_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model     = GammaForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    n_params  = sum(p.numel() for p in model.parameters())
    print(f'\nGamma model: {n_params} params')

    # ---- Train ----
    print(f'training for {epochs} epochs...')
    train_losses, val_losses = [], []
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n_b = 0
        start = time.time()
        for tact, com_h, target in train_loader:
            tact, com_h, target = tact.to(device), com_h.to(device), target.to(device)
            pred = model(tact, com_h)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_b   += 1
        avg_train = total / n_b

        model.eval()
        with torch.no_grad():
            vtot = 0.0; vb = 0
            for tact, com_h, target in val_loader:
                tact, com_h, target = tact.to(device), com_h.to(device), target.to(device)
                vtot += criterion(model(tact, com_h), target).item()
                vb   += 1
        avg_val = vtot / vb

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, 'gamma_model_best.pt'))

        print(f'  epoch {epoch:3d}/{epochs - 1}  '
              f'train={avg_train:.5f}  val={avg_val:.5f}  best_val={best_val:.5f}  '
              f'({time.time() - start:.1f}s)', flush=True)

    # ---- Reload best-val for eval ----
    model.load_state_dict(torch.load(os.path.join(output_dir, 'gamma_model_best.pt'),
                                     map_location=device, weights_only=False))
    torch.save(model.state_dict(), os.path.join(output_dir, 'gamma_model.pt'))

    # ---- Eval on test set ----
    print('\nevaluating on test set...')
    model.eval()
    preds = []
    with torch.no_grad():
        for tact, com_h, _ in test_loader:
            tact, com_h = tact.to(device), com_h.to(device)
            preds.append(model(tact, com_h).cpu().numpy())
    pred_delta_norm = np.concatenate(preds, axis=0)                  # (n_test, HORIZON, 3)
    pred_delta      = pred_delta_norm * std_Y + mean_Y
    pred_gamma      = pred_delta + ref_test[:, None, :]              # (n_test, HORIZON, 3) absolute CoM

    # ---- Motion criterion + subset masks (same as high_motion_subset.py) ----
    future_xy   = Y_abs_test[:, :, :2]
    step_diffs  = np.diff(future_xy, axis=1)
    step_speeds = np.linalg.norm(step_diffs, axis=2)
    v_future    = step_speeds.max(axis=1)
    threshold   = float(np.quantile(v_future, 1.0 - MOVING_FRAC))
    moving_mask = v_future > threshold
    static_mask = ~moving_mask
    subsets = {'full': np.ones(n_test, dtype=bool), 'static': static_mask, 'moving': moving_mask}
    print(f'subset partition (threshold {threshold:.2f} mm/frame): '
          f'static={static_mask.sum()}, moving={moving_mask.sum()}')

    # ---- Metrics (gamma + persistence) ----
    def euc(pred, gt):
        return np.linalg.norm(pred - gt, axis=2)

    methods   = {'persistence': pred_persistence, 'phase2_gamma': pred_gamma}
    persist_e = euc(pred_persistence, Y_abs_test)
    persist_med_per_sub = {sub: float(np.median(persist_e[mask])) for sub, mask in subsets.items()}

    results = {}
    for name, pred in methods.items():
        e_3d = euc(pred, Y_abs_test)
        e_ax = np.abs(pred - Y_abs_test)
        rec = {}
        for sub, mask in subsets.items():
            rec[sub] = {
                'n':                  int(mask.sum()),
                'median_3d_mm':       float(np.median(e_3d[mask])),
                'mean_3d_mm':         float(np.mean(e_3d[mask])),
                'p95_3d_mm':          float(np.percentile(e_3d[mask], 95)),
                'per_horizon_median': [float(np.median(e_3d[mask, h])) for h in range(HORIZON)],
                'per_axis_median':    {ax: float(np.median(e_ax[mask, :, i])) for i, ax in enumerate('xyz')},
                'skill_vs_persistence': float(np.median(e_3d[mask]) / persist_med_per_sub[sub])
                                        if persist_med_per_sub[sub] > 1e-6 else float('nan'),
            }
        results[name] = rec

    # Per-subject medians for gamma on full + moving
    per_subject = {}
    e_gamma = euc(pred_gamma, Y_abs_test)
    for subj in sorted(np.unique(test_subj)):
        sm = test_subj == subj
        per_subject[subj] = {
            'n_full':        int(sm.sum()),
            'median_full':   float(np.median(e_gamma[sm])),
            'n_moving':      int((sm & moving_mask).sum()),
            'median_moving': float(np.median(e_gamma[sm & moving_mask]))
                             if (sm & moving_mask).any() else float('nan'),
        }
    results['phase2_gamma']['per_subject'] = per_subject

    out = {
        'n_test':                  n_test,
        'threshold_mm_per_frame':  threshold,
        'n_static':                int(static_mask.sum()),
        'n_moving':                int(moving_mask.sum()),
        'epochs_trained':          epochs,
        'best_val_mse':            best_val,
        'n_params':                n_params,
        'methods':                 results,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # ---- Console summary ----
    bar = '=' * 78
    print(f'\n{bar}\nGAMMA FUSION RESULTS  (n_test = {n_test}, horizon = 1.0 s)\n{bar}')
    print(f'subsets: full ({n_test}), static ({static_mask.sum()}), moving ({moving_mask.sum()})')

    print(f'\n{"method":<22} {"FULL median":>12} {"STATIC median":>14} {"MOVING median":>14} {"skill@MOVING":>14}')
    for name in ['persistence', 'phase2_gamma']:
        rec = results[name]
        print(f'  {name:<20}  '
              f'{rec["full"]["median_3d_mm"]:>10.1f}    '
              f'{rec["static"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["skill_vs_persistence"]:>12.3f}')

    print(f'\nMOVING-subset per-axis medians (mm):')
    for name in ['persistence', 'phase2_gamma']:
        a = results[name]['moving']['per_axis_median']
        print(f'  {name:<22}  x={a["x"]:>6.1f}  y={a["y"]:>6.1f}  z={a["z"]:>6.1f}')

    print(f'\nGamma per-subject median (full / moving) mm:')
    for subj, st in sorted(per_subject.items(), key=lambda kv: kv[1]['median_moving']):
        print(f'  {subj:<14}  '
              f'full: n={st["n_full"]:>4d} median={st["median_full"]:>6.1f}    '
              f'moving: n={st["n_moving"]:>4d} median={st["median_moving"]:>6.1f}')

    # ---- Plots ----
    # 1. Training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='train')
    ax.plot(val_losses, label='val', linestyle='--')
    ax.axvline(int(np.argmin(val_losses)), color='gray', linestyle=':', alpha=0.5,
               label=f'best val @ ep {int(np.argmin(val_losses))}')
    ax.set(xlabel='epoch', ylabel='MSE (standardized delta)',
           title=f'Gamma fusion training  ({n_params} params)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=100)
    plt.close()

    # 2. Per-horizon (gamma vs persistence) across subsets
    hs = np.arange(1, HORIZON + 1) / 10.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, sub in zip(axes, ['full', 'static', 'moving']):
        for name in ['persistence', 'phase2_gamma']:
            ax.plot(hs, results[name][sub]['per_horizon_median'], marker='o', label=name)
        ax.set(xlabel='horizon (s)', title=f'{sub} (n={results["phase2_gamma"][sub]["n"]})')
        ax.grid(alpha=0.3)
        if sub == 'full':
            ax.set_ylabel('median 3D Euclidean error (mm)')
            ax.legend()
    plt.suptitle('Gamma vs persistence: error vs horizon')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_horizon.png'), dpi=100)
    plt.close()

    # 3. Comparison bars: gamma side-by-side with reference methods, on each subset.
    # Load reference numbers from existing metrics.json files in sibling output dirs.
    reference_methods = {}
    def try_load(name, json_path, key=None, source='full'):
        if not os.path.exists(json_path):
            return
        with open(json_path) as f:
            data = json.load(f)
        # high_motion_subset.py's format has {'methods': {name: {full, static, moving}}}
        if 'methods' in data and key in data['methods']:
            reference_methods[name] = {
                'full':   data['methods'][key]['full']['median_3d_mm'],
                'static': data['methods'][key]['static']['median_3d_mm'],
                'moving': data['methods'][key]['moving']['median_3d_mm'],
            }
            return
        # otherwise, only the full-set number is available; mark static / moving as None
        if key is None:
            return
        if key in data:
            reference_methods[name] = {
                'full':   data[key].get('overall_median_3d_mm') or data[key].get('median_3d_mm'),
                'static': None, 'moving': None,
            }

    # Prefer the high_motion_subset re-eval file because it has all three subsets already.
    hms_metrics = os.path.join(_OUT, 'high_motion', 'metrics.json')
    if os.path.exists(hms_metrics):
        with open(hms_metrics) as f:
            hms_data = json.load(f)
        for name in ['persistence', 'phase1_gru_com', 'phase2_v1_gru_kp',
                     'phase2_v2_gru_kp', 'phase2_tactile_50ep', 'phase2_tactile_200ep']:
            if name in hms_data['methods']:
                reference_methods[name] = {
                    sub: hms_data['methods'][name][sub]['median_3d_mm']
                    for sub in ['full', 'static', 'moving']
                }

    # Add gamma's numbers from this run
    reference_methods['phase2_gamma'] = {sub: results['phase2_gamma'][sub]['median_3d_mm']
                                          for sub in ['full', 'static', 'moving']}

    plot_order = ['persistence', 'phase1_gru_com', 'phase2_v1_gru_kp',
                  'phase2_tactile_50ep', 'phase2_tactile_200ep',
                  'phase2_gamma',                  # highlighted
                  'phase2_v2_gru_kp']
    plot_order = [m for m in plot_order if m in reference_methods]

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.25
    x = np.arange(len(plot_order))
    sub_colors = {'full': 'tab:gray', 'static': 'tab:blue', 'moving': 'tab:red'}
    for i, sub in enumerate(['full', 'static', 'moving']):
        vals = [reference_methods[m][sub] if reference_methods[m][sub] is not None else 0
                for m in plot_order]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=sub,
                       color=sub_colors[sub], alpha=0.9)
        # Highlight gamma's bars
        gamma_i = plot_order.index('phase2_gamma') if 'phase2_gamma' in plot_order else -1
        if gamma_i >= 0:
            bars[gamma_i].set_edgecolor('black')
            bars[gamma_i].set_linewidth(2.0)
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(x[j] + (i - 1) * width, v + 1, f'{v:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_order, rotation=25, ha='right')
    ax.set_ylabel('median 3D Euclidean error (mm) at 1-s horizon')
    ax.set_title('Phase 2 gamma vs all prior methods  '
                 f'(motion threshold {threshold:.2f} mm/frame, moving = {moving_mask.sum()}/{n_test})')
    ax.legend(title='subset', loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bars.png'), dpi=100)
    plt.close()

    # 4. The decisive plot: gamma vs phase1_gru_com on the moving subset, per horizon.
    if os.path.exists(hms_metrics):
        phase1_per_h = hms_data['methods']['phase1_gru_com']['moving']['per_horizon_median'] \
                       if 'phase1_gru_com' in hms_data['methods'] else None
        v2_per_h     = hms_data['methods']['phase2_v2_gru_kp']['moving']['per_horizon_median'] \
                       if 'phase2_v2_gru_kp' in hms_data['methods'] else None
        persist_per_h = hms_data['methods']['persistence']['moving']['per_horizon_median']
    else:
        phase1_per_h = v2_per_h = persist_per_h = None
    gamma_per_h = results['phase2_gamma']['moving']['per_horizon_median']

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if persist_per_h: ax.plot(hs, persist_per_h, marker='o', linestyle='--', label='persistence', color='gray')
    if phase1_per_h:  ax.plot(hs, phase1_per_h,  marker='o', linestyle='--', label='Phase 1 (CoM only)', color='tab:blue')
    if v2_per_h:      ax.plot(hs, v2_per_h,      marker='o', linestyle='--', label='Phase 2 v2 (kp)',   color='tab:orange')
    ax.plot(hs, gamma_per_h, marker='s', linestyle='-', linewidth=2.0,
            label='Phase 2 gamma (tactile + CoM)', color='tab:green')
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median 3D Euclidean error (mm), MOVING subset',
           title='Gamma vs reference methods on MOVING subset')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gamma_vs_phase1_per_horizon.png'), dpi=100)
    plt.close()

    # ---- Final headline interpretation hint ----
    g_moving  = results['phase2_gamma']['moving']['median_3d_mm']
    if os.path.exists(hms_metrics):
        p1_moving = hms_data['methods']['phase1_gru_com']['moving']['median_3d_mm'] \
                    if 'phase1_gru_com' in hms_data['methods'] else None
        v2_moving = hms_data['methods']['phase2_v2_gru_kp']['moving']['median_3d_mm'] \
                    if 'phase2_v2_gru_kp' in hms_data['methods'] else None
    else:
        p1_moving = v2_moving = None

    print(f'\n{bar}\nDECISION HINT (MOVING subset, 1-s horizon)\n{bar}')
    print(f'gamma:    {g_moving:.1f} mm')
    if p1_moving is not None:
        print(f'Phase 1:  {p1_moving:.1f} mm  (CoM history only)')
    if v2_moving is not None:
        print(f'v2:       {v2_moving:.1f} mm  (camera 21-keypoint history)')
    if p1_moving is not None and v2_moving is not None:
        if g_moving < v2_moving:
            print('-> gamma BEATS v2 -- tactile adds info beyond camera-pose history. STRONG positive.')
        elif g_moving < p1_moving:
            print('-> gamma BEATS Phase 1 -- tactile adds info on top of CoM history. POSITIVE (but less than camera-pose).')
        else:
            print('-> gamma does NOT beat Phase 1 -- tactile contributes no info beyond CoM history. NEGATIVE.')
    print(f'\nSaved metrics + plots to {output_dir}')


if __name__ == '__main__':
    main()

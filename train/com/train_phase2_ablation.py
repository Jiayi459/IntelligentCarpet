"""train_phase2_ablation.py — isolate which v1->v2 change drove the persistence-beating gain.

Phase 2 v1 (small, abs target, 50ep)  -> 66.3 mm median  (LOST to persistence by 22 %)
Phase 2 v2 (big, delta target, 200ep) -> 46.8 mm median  (BEAT persistence by 13.6 %)

The v1 -> v2 jump bundles three simultaneous changes. This script runs three
single-change variants on top of v1 to attribute the 19.5 mm gain:

    Variant            target   hidden / layers / dropout   epochs   best-val?
    --------------------------------------------------------------------------
    v1  (reference)    abs      64 / 1 / 0.0                50       no
    A1  delta_only     delta    64 / 1 / 0.0                50       no
    A2  bigger_only    abs      256 / 2 / 0.1               50       no
    A3  longer_only    abs      64 / 1 / 0.0                200      no
    v2  (reference)    delta    256 / 2 / 0.1               200      yes

Each variant changes EXACTLY ONE thing from v1. End-of-training model is used
(no best-val checkpointing) so the only difference between A3 and v1 is the
epoch count — purely isolated. If A3 overfits at late epochs, that is itself
a finding ("longer training without checkpointing hurts").

Same per-session 70/30 split, same outlier filter, same 17,218/5,255 sample
counts, same SEED-driven train/val split as Phase 1 / 2 v1 / 2 v2 — numbers
are directly comparable. Each variant re-seeds before training so all three
start from the same RNG state.

Outputs (under train/com/output/phase2_ablation/):
    A1_delta_only/  gru_model.pt, metrics.json, training_curve.png
    A2_bigger_only/ ...
    A3_longer_only/ ...
    ablation_summary.png       bar + line chart vs persistence/v1/v2
    ablation_metrics.json      combined metrics for all 3 variants

Run (on CRC GPU node):
    python train/com/train_phase2_ablation.py

Module-level definitions (importable without side effects):
    AblationForecaster      configurable GRU forecaster
    VARIANTS                dict of {name: config}
    HISTORY, HORIZON, SEED, KP_DIM   constants
"""

import os
import json
import pickle
import re
import time

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

_HERE     = os.path.dirname(os.path.abspath(__file__))
_TRAIN    = os.path.dirname(_HERE)
_OUT      = os.path.join(_HERE, 'output')
_PHASE1   = os.path.join(_OUT, 'phase1')
_PHASE2   = os.path.join(_OUT, 'phase2_keypoints')
_PHASE2V2 = os.path.join(_OUT, 'phase2_keypoints_v2')
_ABLATION = os.path.join(_OUT, 'phase2_ablation')


# ---------------------------------------------------------------------------
# Constants (must match Phase 1 / 2 / 2v2)
# ---------------------------------------------------------------------------

HISTORY   = 100
HORIZON   = 10
SEED      = 42
KP_DIM    = 21 * 3

GRU_LR    = 1e-3
GRU_BATCH = 256
VAL_FRAC  = 0.10


# Each variant changes exactly ONE thing from v1.
# v1 baseline (for reference, NOT trained here — uses checkpoint at _PHASE2):
#     target='abs', hidden=64, layers=1, dropout=0.0, epochs=50
VARIANTS = {
    'A1_delta_only':  {'hidden': 64,  'layers': 1, 'dropout': 0.0, 'target': 'delta', 'epochs': 50},
    'A2_bigger_only': {'hidden': 256, 'layers': 2, 'dropout': 0.1, 'target': 'abs',   'epochs': 50},
    'A3_longer_only': {'hidden': 64,  'layers': 1, 'dropout': 0.0, 'target': 'abs',   'epochs': 200},
}


# ---------------------------------------------------------------------------
# Configurable model (module level — importable without triggering training)
# ---------------------------------------------------------------------------

class AblationForecaster(nn.Module):
    """GRU + linear head. Parameterized so a single class covers all variants."""
    def __init__(self, hidden, layers, dropout, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(KP_DIM, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, horizon * 3)

    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_ABLATION, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # Load CoM results + session metadata (same as v1/v2)
    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)
    com_gt    = com_results['com_gt']
    kp_gt_mm  = com_results['kp_gt_mm']
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

    # Build samples (identical to v1/v2)
    def build_samples():
        X, Y_abs, ref = [], [], []
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
                Y_abs.append(com_gt[t + 1            : t + 1 + HORIZON])
                ref.append(com_gt[t])
                meta['subject'].append(subjects_per_sess[s])
                meta['session'].append(s)
                meta['frame'].append(t)
                meta['split'].append('train' if i < n_train_s else 'test')
        return (np.asarray(X, dtype=np.float64),
                np.asarray(Y_abs, dtype=np.float64),
                np.asarray(ref, dtype=np.float64),
                {k: np.asarray(v) for k, v in meta.items()})

    X, Y_abs, ref, meta = build_samples()
    train_mask = meta['split'] == 'train'
    test_mask  = meta['split'] == 'test'

    X_train, X_test = X[train_mask], X[test_mask]
    Y_train_abs, Y_test_abs = Y_abs[train_mask], Y_abs[test_mask]
    ref_train, ref_test = ref[train_mask], ref[test_mask]

    Y_train_delta = Y_train_abs - ref_train[:, None, :]
    Y_test_delta  = Y_test_abs  - ref_test[:, None, :]

    print(f'samples: total={len(X)}, train={train_mask.sum()}, test={test_mask.sum()}')
    assert len(X) == 17218

    # Persistence reference (no model)
    pred_persistence = np.broadcast_to(ref_test[:, None, :],
                                       (len(ref_test), HORIZON, 3)).copy()


    # Helper functions (closures over com_gt, X_*, etc.)
    def euc(pred, gt):
        return np.linalg.norm(pred - gt, axis=2)

    def compute_metrics(pred_abs):
        """Return the full metrics dict matching Phase 2 v2's schema."""
        e_3d = euc(pred_abs, Y_test_abs)
        e_ax = np.abs(pred_abs - Y_test_abs)
        e_persist_3d = euc(pred_persistence, Y_test_abs)
        median_persist = float(np.median(e_persist_3d))
        return {
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


    # ----- Train each variant in sequence -----
    all_results = {}
    all_training_curves = {}

    for variant_name, config in VARIANTS.items():
        bar = '=' * 78
        print(f'\n{bar}\nTRAINING {variant_name}\n{bar}')
        print(f'  config: {config}')

        out_dir = os.path.join(_ABLATION, variant_name)
        os.makedirs(out_dir, exist_ok=True)

        # Re-seed before each variant so they all start from the same RNG state.
        # This isolates the effect of the config change from any RNG drift between
        # sequential runs.
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Pick target tensors based on variant's target framing
        if config['target'] == 'delta':
            Y_train_used = Y_train_delta
            Y_test_used  = Y_test_delta
        else:
            Y_train_used = Y_train_abs
            Y_test_used  = Y_test_abs

        # Standardize: X always uses kp pool; Y uses whichever framing this variant chose
        mean_X = X_train.reshape(-1, KP_DIM).mean(axis=0)
        std_X  = X_train.reshape(-1, KP_DIM).std(axis=0)
        std_X  = np.where(std_X < 1e-6, 1.0, std_X)
        mean_Y = Y_train_used.reshape(-1, 3).mean(axis=0)
        std_Y  = Y_train_used.reshape(-1, 3).std(axis=0)
        std_Y  = np.where(std_Y < 1e-6, 1.0, std_Y)

        def norm_X(arr):    return (arr - mean_X) / std_X
        def norm_Y(arr):    return (arr - mean_Y) / std_Y
        def denorm_Y(arr):  return arr * std_Y + mean_Y

        # Train / val split (uses the just-set seed -> same across variants)
        n_train = X_train.shape[0]
        perm    = np.random.permutation(n_train)
        n_val   = int(n_train * VAL_FRAC)
        val_idx = perm[:n_val]
        tr_idx  = perm[n_val:]

        Xt = torch.tensor(norm_X(X_train),    dtype=torch.float32)
        Yt = torch.tensor(norm_Y(Y_train_used), dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(Xt[tr_idx], Yt[tr_idx]),
                                  batch_size=GRU_BATCH, shuffle=True)
        X_val = Xt[val_idx].to(device)
        Y_val = Yt[val_idx].to(device)

        model     = AblationForecaster(hidden=config['hidden'],
                                        layers=config['layers'],
                                        dropout=config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
        criterion = nn.MSELoss()
        n_params  = sum(p.numel() for p in model.parameters())
        print(f'  params: {n_params}')

        train_losses, val_losses = [], []
        start = time.time()
        for epoch in range(config['epochs']):
            model.train()
            total = 0.0
            n_b = 0
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
                avg_val = criterion(model(X_val), Y_val).item()
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            log_every = 5 if config['epochs'] <= 50 else 20
            if epoch % log_every == 0 or epoch == config['epochs'] - 1:
                print(f'  epoch {epoch:3d}/{config["epochs"] - 1}  '
                      f'train={avg_train:.5f}  val={avg_val:.5f}  '
                      f'({time.time() - start:.0f}s elapsed)', flush=True)

        # Save end-of-training model (NOT best-val, to keep ablation clean)
        torch.save(model.state_dict(), os.path.join(out_dir, 'gru_model.pt'))

        # Eval on test set
        model.eval()
        with torch.no_grad():
            pred_norm = model(torch.tensor(norm_X(X_test), dtype=torch.float32)
                              .to(device)).cpu().numpy()
        pred_used = denorm_Y(pred_norm)

        # Convert delta -> absolute if needed for fair comparison
        if config['target'] == 'delta':
            pred_abs = pred_used + ref_test[:, None, :]
        else:
            pred_abs = pred_used

        metrics = compute_metrics(pred_abs)
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump({'config': config, 'n_params': n_params, **metrics}, f, indent=2)

        print(f'  RESULT: median 3D = {metrics["overall_median_3d_mm"]:.1f} mm, '
              f'mean = {metrics["overall_mean_3d_mm"]:.1f}, '
              f'p95 = {metrics["overall_p95_3d_mm"]:.1f}, '
              f'skill = {metrics["skill_score_vs_persistence"]:.3f}')

        # Per-variant training curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, label='train')
        ax.plot(val_losses, label='val', linestyle='--')
        ax.set(xlabel='epoch', ylabel='MSE (normalized)',
               title=f'{variant_name}   {config}')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_curve.png'), dpi=100)
        plt.close()

        all_results[variant_name] = {'config': config, 'n_params': n_params, **metrics}
        all_training_curves[variant_name] = {'train': train_losses, 'val': val_losses}

    # ----- Final summary plot vs persistence / v1 / v2 -----
    # Try to load v1 and v2 reference metrics from their existing dirs.
    reference_results = {'persistence': compute_metrics(pred_persistence)}

    def load_ref(path, name):
        if not os.path.exists(path):
            print(f'WARNING: no {name} metrics at {path}, skipping in summary')
            return None
        with open(path) as f:
            data = json.load(f)
        # v1's metrics.json has multiple methods nested; v2's has multiple methods nested too.
        # We need the specific entry for this method.
        if 'phase2_gru_kp' in data:        # v1's file (phase2_keypoints)
            return data['phase2_gru_kp']
        if 'phase2_v2_gru_kp' in data:     # v2's file (phase2_keypoints_v2)
            return data['phase2_v2_gru_kp']
        return data                          # fall back to whole dict

    v1_ref = load_ref(os.path.join(_PHASE2,   'metrics.json'), 'v1')
    v2_ref = load_ref(os.path.join(_PHASE2V2, 'metrics.json'), 'v2')
    if v1_ref: reference_results['v1_small_abs_50ep'] = v1_ref
    if v2_ref: reference_results['v2_big_delta_200ep'] = v2_ref

    # Bar chart of overall medians
    ordered = ['persistence']
    if 'v1_small_abs_50ep'   in reference_results: ordered.append('v1_small_abs_50ep')
    ordered += list(VARIANTS.keys())
    if 'v2_big_delta_200ep'  in reference_results: ordered.append('v2_big_delta_200ep')

    medians = []
    skills  = []
    persist_median = reference_results['persistence']['overall_median_3d_mm']
    for name in ordered:
        if name in all_results:
            r = all_results[name]
        else:
            r = reference_results[name]
        medians.append(r['overall_median_3d_mm'])
        skills.append(r['overall_median_3d_mm'] / persist_median)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['gray', 'tab:blue'] + ['tab:purple', 'tab:olive', 'tab:cyan'] + ['tab:red']
    colors = colors[:len(ordered)]
    axes[0].bar(ordered, medians, color=colors)
    axes[0].axhline(persist_median, color='black', linestyle='--', alpha=0.4, label='persistence')
    axes[0].set(ylabel='median 3D error (mm)', title='Phase 2 ablation: median 3D at 1-s horizon')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=30)
    for i, v in enumerate(medians):
        axes[0].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)

    axes[1].bar(ordered, skills, color=colors)
    axes[1].axhline(1.0, color='black', linestyle='--', alpha=0.4, label='persistence floor')
    axes[1].set(ylabel='skill (lower=better)', title='Skill score vs persistence')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend()
    for i, v in enumerate(skills):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(_ABLATION, 'ablation_summary.png'), dpi=100)
    plt.close()

    # Per-horizon comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    hs = np.arange(1, HORIZON + 1) / 10.0
    for name in ordered:
        r = all_results.get(name) or reference_results.get(name)
        if r and 'per_horizon_median_3d_mm' in r:
            ax.plot(hs, r['per_horizon_median_3d_mm'], marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median 3D Euclidean error (mm)',
           title='Phase 2 ablation: error vs horizon')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_ABLATION, 'ablation_error_vs_horizon.png'), dpi=100)
    plt.close()

    # Combined metrics dump
    out = {
        'variants': all_results,
        'references': reference_results,
        'ordered_for_plot': ordered,
    }
    with open(os.path.join(_ABLATION, 'ablation_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Console summary
    bar = '=' * 78
    print(f'\n{bar}\nPHASE 2 ABLATION SUMMARY  (1-s horizon)\n{bar}')
    print(f'{"variant":<25} {"median 3D":>12} {"mean 3D":>10} {"p95 3D":>10} {"skill":>8}')
    for name in ordered:
        r = all_results.get(name) or reference_results.get(name)
        if r:
            print(f'  {name:<23}  {r["overall_median_3d_mm"]:>10.1f}   '
                  f'{r["overall_mean_3d_mm"]:>8.1f}   '
                  f'{r["overall_p95_3d_mm"]:>8.1f}   '
                  f'{r.get("skill_score_vs_persistence", r["overall_median_3d_mm"] / persist_median):>6.3f}')
    print(f'\nSaved to {_ABLATION}')


if __name__ == '__main__':
    main()

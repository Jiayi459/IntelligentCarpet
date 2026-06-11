"""generate_future_tactile.py — diagnostic #3: visualize what ε's Stage 2 learned.

Scientific question:
    ε's DynamicsModel was trained (Stage 2) to predict future delta-tactile
    from past tactile. Its dynamics_best_val_mse landed at 0.937 (vs 1.0
    tactile-persistence baseline) — barely above the floor. We never looked
    at what its predictions actually look like.

    Three possible visual signatures:
        (A) predictions = current tactile copied 10 times
            -> model defaulted to tactile-persistence
            -> consistent with the negative CoM result
        (B) predictions show plausible (even wrong) motion
            -> SSL learned dynamics but they didn't transfer to CoM
            -> tells us the SSL succeeded structurally
        (C) predictions = static noise / mean tactile
            -> mode collapse during SSL
            -> different failure mode than persistence

What this script does:
    Load ε's dynamics_model.pt; pick representative test samples (static,
    moving, transition); for each, feed the 100-frame history through the
    model and render a comparison grid:

        Row 1: tactile(t-2), tactile(t-1), tactile(t)               (history tail)
        Row 2: true tactile(t+1) ... tactile(t+10)                  (ground truth future)
        Row 3: predicted tactile(t+1) ... tactile(t+10)             (model's output)
        Row 4: difference (predicted - true), diverging colormap    (error map)

    Plus a quantitative summary:
        - per-sample MSE (all cells, active cells)
        - tactile-persistence baseline MSE for comparison
        - active-region MSE = MSE on cells where current tactile > 0.05

Outputs (under train/com/output/generate_future_tactile/):
    example_static.png        4-row grid for a single static sample
    example_moving.png        same for a moving sample
    example_transition.png    same for a transition (mid-motion) sample
    metrics.json              per-sample + aggregated quantitative summary

Run (needs ε dynamics_model.pt + tactile_all.npy):
    python train/com/generate_future_tactile.py
    python train/com/generate_future_tactile.py --n-per-regime 3   # multiple per regime
"""

import os
import sys
import json
import pickle
import re
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_OUT   = os.path.join(_HERE, 'output')
_GEN   = os.path.join(_OUT, 'generate_future_tactile')

_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')
_STATS     = os.path.join(_OUT, 'tactile_stats.json')

# epsilon's model classes
sys.path.insert(0, os.path.join(_TRAIN, 'tactile_direct'))

HISTORY = 100
HORIZON = 10
SEED    = 42
ACTIVE_THRESHOLD = 0.05         # raw-tactile (pre-standardization) cell-activation threshold


def main():
    parser = argparse.ArgumentParser(description='Visualize what epsilon Stage 2 learned.')
    parser.add_argument('--epsilon-checkpoint',
                        default=os.path.join(_OUT, 'phase2_epsilon', 'dynamics_model.pt'),
                        help='path to dynamics_model.pt')
    parser.add_argument('--n-per-regime', type=int, default=1,
                        help='how many samples to render per regime (static/moving/transition)')
    parser.add_argument('--output-dir',   type=str, default=_GEN)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device     : {device}')
    print(f'output_dir : {args.output_dir}')

    # ---- Load inputs ----
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY}')
    if not os.path.exists(_STATS):
        raise SystemExit(f'no tactile stats at {_STATS} -- run seed_tactile_stats.py')
    if not os.path.exists(args.epsilon_checkpoint):
        raise SystemExit(f'no epsilon checkpoint at {args.epsilon_checkpoint}')

    tactile_all = np.load(_CACHE_NPY, mmap_mode='r')
    T = tactile_all.shape[0]
    assert tactile_all.shape == (T, 96, 96)
    print(f'tactile_all: shape={tactile_all.shape}')

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

    # ---- Load epsilon dynamics model ----
    from model_epsilon import DynamicsModel
    dyn = DynamicsModel().to(device)
    ck = torch.load(args.epsilon_checkpoint, map_location=device, weights_only=False)
    dyn.load_state_dict(ck['dynamics'])
    dyn.eval()
    print(f'loaded dynamics_model.pt  (best_val={ck.get("best_val", float("nan")):.5f}, '
          f'epoch={ck.get("epoch", "?")})')

    # ---- Build test-set sample indices using the standard 1-s rule ----
    _in_carpet = lambda v: (v >= -100) & (v <= 1800)
    gt_outliers = (~_in_carpet(com_gt[:, 0])
                   | ~_in_carpet(com_gt[:, 1])
                   | (com_gt[:, 2] > 0))

    centers, splits = [], []
    for sess in range(n_sessions):
        a, b = log[sess], log[sess + 1]
        valid_t = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t))
        for i, t in enumerate(valid_t):
            centers.append(t)
            splits.append('train' if i < n_train_s else 'test')
    centers = np.asarray(centers)
    splits  = np.asarray(splits)
    test_centers = centers[splits == 'test']
    n_test = len(test_centers)
    print(f'test samples: {n_test}')

    # ---- Motion classification ----
    future_xy = np.stack([com_gt[t + 1 : t + 1 + HORIZON, :2] for t in test_centers], axis=0)
    step_speeds = np.linalg.norm(np.diff(future_xy, axis=1), axis=2)        # (n_test, HORIZON-1)
    v_future = step_speeds.max(axis=1)                                       # (n_test,)
    q25, q50, q75 = np.quantile(v_future, [0.25, 0.50, 0.75])
    print(f'v_future quantiles: q25={q25:.2f}  q50={q50:.2f}  q75={q75:.2f} mm/frame')

    sel_rng = np.random.default_rng(SEED)
    static_pool     = np.where(v_future < q25)[0]
    moving_pool     = np.where(v_future > q75)[0]
    transition_pool = np.where((v_future >= q25) & (v_future <= q75))[0]

    def pick(pool, k):
        if len(pool) <= k:
            return pool
        return pool[sel_rng.choice(len(pool), size=k, replace=False)]

    sel = {
        'static':     pick(static_pool,     args.n_per_regime),
        'transition': pick(transition_pool, args.n_per_regime),
        'moving':     pick(moving_pool,     args.n_per_regime),
    }
    for regime, idx in sel.items():
        speeds = v_future[idx]
        starts = test_centers[idx]
        print(f'  {regime:<12}: indices={idx.tolist()}  v_future={speeds.round(1).tolist()}  '
              f'starts={starts.tolist()}')

    # ---- Inference + metrics ----
    results_per_sample = []

    def infer_one(t0):
        """Forward through epsilon dynamics. Return (pred_future_raw, true_future_raw, history_raw)."""
        hist_raw = np.asarray(tactile_all[t0 - HISTORY + 1 : t0 + 1]).astype(np.float32)   # (100, 96, 96)
        fut_raw  = np.asarray(tactile_all[t0 + 1 : t0 + 1 + HORIZON]).astype(np.float32)    # (10, 96, 96)
        hist_std = (hist_raw - tactile_mean) / tactile_std
        with torch.no_grad():
            x = torch.from_numpy(hist_std[None]).to(device)             # (1, 100, 96, 96)
            delta_pred_std = dyn(x).cpu().numpy()[0]                    # (10, 96, 96) in standardized delta units
        delta_pred_raw = delta_pred_std * tactile_std                   # de-standardize the delta
        pred_future    = hist_raw[-1][None] + delta_pred_raw            # (10, 96, 96)
        return hist_raw, fut_raw, pred_future

    def sample_metrics(hist_raw, fut_raw, pred_future):
        current = hist_raw[-1]                                          # (96, 96) tactile(t)
        active  = (current > ACTIVE_THRESHOLD)                          # (96, 96)
        active_b = np.broadcast_to(active[None], pred_future.shape)     # (10, 96, 96)
        err_eps    = (pred_future - fut_raw) ** 2
        err_persist = (current[None] - fut_raw) ** 2
        return {
            'epsilon_mse_all':         float(err_eps.mean()),
            'epsilon_mse_active':      float(err_eps[active_b].mean()) if active_b.any() else float('nan'),
            'persistence_mse_all':     float(err_persist.mean()),
            'persistence_mse_active':  float(err_persist[active_b].mean()) if active_b.any() else float('nan'),
            'n_active_cells_at_t':     int(active.sum()),
            'mean_pressure_at_t':      float(current.mean()),
            'max_pressure_at_t':       float(current.max()),
        }

    def render_grid(regime, sample_i, t0, hist_raw, fut_raw, pred_future, m):
        """Render the 4-row × 10-col grid for one sample."""
        last_3_hist = hist_raw[-3:]                                     # (3, 96, 96)
        diff = pred_future - fut_raw                                    # (10, 96, 96)
        vmax_pressure = max(0.3, float(fut_raw.max()), float(pred_future.max()),
                            float(last_3_hist.max()))
        vmax_diff = max(0.05, float(np.abs(diff).max()))

        n_cols = max(10, 3)
        fig, axes = plt.subplots(4, 10, figsize=(20, 9))

        # Row 0: history tail (3 frames -- pad the other 7 columns with blanks)
        for c in range(10):
            ax = axes[0, c]
            if c < 3:
                im = ax.imshow(last_3_hist[c], vmin=0, vmax=vmax_pressure, cmap='viridis')
                ax.set_title(f't{c - 3:+d}', fontsize=9)
            else:
                ax.set_facecolor('white')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, 0].set_ylabel('history\n(last 3)', fontsize=10)

        # Row 1: true future
        for c in range(10):
            ax = axes[1, c]
            im = ax.imshow(fut_raw[c], vmin=0, vmax=vmax_pressure, cmap='viridis')
            ax.set_title(f't+{c + 1}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[1, 0].set_ylabel('true future', fontsize=10)

        # Row 2: predicted future
        for c in range(10):
            ax = axes[2, c]
            im = ax.imshow(pred_future[c], vmin=0, vmax=vmax_pressure, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
        axes[2, 0].set_ylabel('ε predicted', fontsize=10)

        # Row 3: difference (pred - true)
        for c in range(10):
            ax = axes[3, c]
            im_d = ax.imshow(diff[c], vmin=-vmax_diff, vmax=vmax_diff, cmap='RdBu_r')
            ax.set_xticks([]); ax.set_yticks([])
        axes[3, 0].set_ylabel('ε − true\n(diff)', fontsize=10)

        # Shared colorbars
        cbar_ax_p = fig.add_axes([0.92, 0.42, 0.012, 0.4])
        plt.colorbar(axes[1, -1].images[0], cax=cbar_ax_p, label='pressure (raw)')
        cbar_ax_d = fig.add_axes([0.92, 0.10, 0.012, 0.18])
        plt.colorbar(axes[3, -1].images[0], cax=cbar_ax_d, label='diff')

        title = (f'[{regime} #{sample_i}] start t₀={t0}  |  '
                 f'ε all-MSE={m["epsilon_mse_all"]:.4f}  '
                 f'(persistence={m["persistence_mse_all"]:.4f})  |  '
                 f'ε active-MSE={m["epsilon_mse_active"]:.4f}  '
                 f'(persistence={m["persistence_mse_active"]:.4f})  |  '
                 f'n_active={m["n_active_cells_at_t"]}')
        fig.suptitle(title, fontsize=10)
        plt.subplots_adjust(left=0.04, right=0.91, top=0.93, bottom=0.05,
                            wspace=0.05, hspace=0.15)
        out_png = os.path.join(args.output_dir, f'example_{regime}_sample{sample_i}.png')
        plt.savefig(out_png, dpi=100)
        plt.close()
        print(f'  saved {out_png}')

    for regime in ('static', 'transition', 'moving'):
        for k, idx_in_test in enumerate(sel[regime]):
            t0 = int(test_centers[int(idx_in_test)])
            hist_raw, fut_raw, pred_future = infer_one(t0)
            m = sample_metrics(hist_raw, fut_raw, pred_future)
            m.update({
                'regime':           regime,
                'sample_index':     int(k),
                'idx_in_test_pool': int(idx_in_test),
                't0_frame':         t0,
                'v_future_at_t0':   float(v_future[int(idx_in_test)]),
            })
            results_per_sample.append(m)
            render_grid(regime, k, t0, hist_raw, fut_raw, pred_future, m)

    # ---- Aggregate over the whole test set (cheap — model is fast) ----
    print('\ncomputing aggregate ε vs tactile-persistence MSE over the whole test set...')
    BATCH = 32
    total_eps_all = 0.0; total_eps_active = 0.0
    total_per_all = 0.0; total_per_active = 0.0
    n_all = 0; n_active = 0.0
    for i0 in range(0, n_test, BATCH):
        i1 = min(i0 + BATCH, n_test)
        windows_std = np.stack([
            (np.asarray(tactile_all[t - HISTORY + 1 : t + 1]) - tactile_mean) / tactile_std
            for t in test_centers[i0:i1]
        ], axis=0).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(windows_std).to(device)
            delta_pred_std = dyn(x).cpu().numpy()                       # (B, 10, 96, 96)
        delta_pred_raw = delta_pred_std * tactile_std
        currents = np.stack([np.asarray(tactile_all[t]) for t in test_centers[i0:i1]], axis=0).astype(np.float32)
        futures  = np.stack([np.asarray(tactile_all[t + 1 : t + 1 + HORIZON])
                              for t in test_centers[i0:i1]], axis=0).astype(np.float32)
        preds = currents[:, None, :, :] + delta_pred_raw                # (B, 10, 96, 96)
        err_eps = (preds - futures) ** 2
        err_per = (currents[:, None, :, :] - futures) ** 2
        active = (currents > ACTIVE_THRESHOLD)[:, None, :, :]            # (B, 1, 96, 96)
        active_b = np.broadcast_to(active, err_eps.shape)
        total_eps_all    += float(err_eps.sum())
        total_per_all    += float(err_per.sum())
        n_all            += err_eps.size
        if active_b.any():
            total_eps_active += float(err_eps[active_b].sum())
            total_per_active += float(err_per[active_b].sum())
            n_active         += float(active_b.sum())
    agg = {
        'epsilon_mse_all':        total_eps_all / n_all,
        'persistence_mse_all':    total_per_all / n_all,
        'epsilon_mse_active':     total_eps_active / n_active if n_active > 0 else float('nan'),
        'persistence_mse_active': total_per_active / n_active if n_active > 0 else float('nan'),
        'n_test_samples':         n_test,
        'n_active_cells_total':   int(n_active),
        'n_all_cells_total':      int(n_all),
    }

    # ---- Console summary ----
    bar = '=' * 78
    print(f'\n{bar}\nFUTURE TACTILE PREDICTION — aggregate over test set ({n_test} samples)\n{bar}')
    print(f'  ε      MSE (all cells)    = {agg["epsilon_mse_all"]:.5f}')
    print(f'  persist MSE (all cells)   = {agg["persistence_mse_all"]:.5f}')
    print(f'  ε      MSE (active cells) = {agg["epsilon_mse_active"]:.5f}')
    print(f'  persist MSE (active cells)= {agg["persistence_mse_active"]:.5f}')
    skill_all    = agg["epsilon_mse_all"]    / agg["persistence_mse_all"]
    skill_active = agg["epsilon_mse_active"] / agg["persistence_mse_active"]
    print(f'  ε skill vs persistence (all)    = {skill_all:.3f}   ({"BEATS" if skill_all < 1 else "loses to"} persistence)')
    print(f'  ε skill vs persistence (active) = {skill_active:.3f}  ({"BEATS" if skill_active < 1 else "loses to"} persistence)')
    print()
    if abs(skill_active - 1.0) < 0.02:
        print('  Reading: ε predictions are ESSENTIALLY tactile-persistence -> case (A) confirmed.')
        print('           Stage 2 SSL defaulted to copying the current frame forward.')
    elif skill_active < 0.95:
        print('  Reading: ε meaningfully BEATS tactile-persistence on active cells -> case (B).')
        print('           SSL learned dynamics, but the representation did not transfer to CoM forecasting.')
    else:
        print('  Reading: marginal improvement only. Closer to (A) than (B).')

    # ---- Save metrics.json ----
    out = {
        'epsilon_checkpoint':     args.epsilon_checkpoint,
        'epsilon_best_val_mse':   float(ck.get('best_val', float('nan'))),
        'epsilon_epoch':          int(ck.get('epoch', -1)),
        'active_threshold':       ACTIVE_THRESHOLD,
        'aggregate':              agg,
        'per_sample':             results_per_sample,
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved {os.path.join(args.output_dir, "metrics.json")}')


if __name__ == '__main__':
    main()

"""high_motion_subset.py — re-evaluate all 6 methods on static vs moving subsets.

The CoP persistence result (compute_cop.py) showed CoP is essentially flat at
~2.5 mm at 1-s horizon — confirming the dataset is dominated by frames where
the person is barely moving. On a static-dominated test set, persistence wins
by construction and every learned method looks weak.

This script partitions the 5,255 test samples by a principled motion criterion
(peak xy speed during the 10-frame target window) and re-evaluates all 6
existing methods on `full`, `static`, and `moving` subsets — exposing how each
method's lead vs persistence changes when the prediction problem actually
requires forecasting motion.

Methods (each plotted if its checkpoint is present):
    persistence
    phase1_gru_com        train/com/output/phase1/gru_model.pt
    phase2_v1_gru_kp      train/com/output/phase2_keypoints/gru_model.pt
    phase2_v2_gru_kp      train/com/output/phase2_keypoints_v2/gru_model_best.pt
    phase2_tactile_50ep   train/com/output/phase2_tactile/tactile_model_best.pt
    phase2_tactile_200ep  train/com/output/phase2_tactile_200ep/tactile_model_best.pt

Model classes are inlined (decoupled from training scripts).

Outputs (under train/com/output/high_motion/):
    metrics.json
    comparison_bars.png
    error_vs_horizon_moving.png
    motion_threshold_histogram.png

Run:
    python train/com/high_motion_subset.py                # default 30% moving
    python train/com/high_motion_subset.py --moving-frac 0.20
    python train/com/high_motion_subset.py --moving-frac 0.50
"""

import os
import sys
import json
import pickle
import re
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_OUT   = os.path.join(_HERE, 'output')
_HM    = os.path.join(_OUT, 'high_motion')

# So `import model_epsilon` works when we re-run the epsilon forecaster below.
# epsilon source lives at train/tactile_direct/ (parallel to train/com/).
sys.path.insert(0, os.path.join(_TRAIN, 'tactile_direct'))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORY = 100
HORIZON = 10
SEED    = 42
KP_DIM  = 21 * 3


# ---------------------------------------------------------------------------
# Inlined model classes (same as compare_trajectories.py)
# ---------------------------------------------------------------------------

class P1Forecaster(nn.Module):
    def __init__(self, hidden=64, layers=1, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(3, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, horizon * 3)
    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


class P2v1Forecaster(nn.Module):
    def __init__(self, hidden=64, layers=1, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(KP_DIM, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, horizon * 3)
    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


class P2v2Forecaster(nn.Module):
    def __init__(self, hidden=256, layers=2, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(KP_DIM, hidden, num_layers=layers, batch_first=True,
                           dropout=0.1 if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, horizon * 3)
    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


class TactileEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1,   32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)
    def forward(self, x):
        return self.proj(self.body(x).flatten(1))


class TactileForecaster(nn.Module):
    def __init__(self, feature_dim=128, hidden=128, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.encoder = TactileEncoder(feature_dim)
        self.gru     = nn.GRU(feature_dim, hidden, batch_first=True)
        self.proj    = nn.Linear(hidden, horizon * 3)
    def forward(self, x):
        B, Tlen, H, W = x.shape
        flat = x.reshape(B * Tlen, 1, H, W)
        feat = self.encoder(flat).reshape(B, Tlen, -1)
        _, h = self.gru(feat)
        return self.proj(h[-1]).view(B, self.horizon, 3)


class GammaForecaster(nn.Module):
    """Two-branch (tactile + CoM history) -> delta CoM. Mirrors train_phase2_gamma.py."""
    def __init__(self, tactile_feature_dim=128, tactile_hidden=128, com_hidden=64,
                 mlp_hidden=128, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.tactile_encoder = TactileEncoder(tactile_feature_dim)
        self.tactile_gru     = nn.GRU(tactile_feature_dim, tactile_hidden, batch_first=True)
        self.com_gru         = nn.GRU(3, com_hidden, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(tactile_hidden + com_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, horizon * 3),
        )

    def forward(self, tactile, com_history):
        B, Tlen, H, W = tactile.shape
        flat = tactile.reshape(B * Tlen, 1, H, W)
        feat = self.tactile_encoder(flat).reshape(B, Tlen, -1)
        _, h_t = self.tactile_gru(feat); h_t = h_t[-1]
        _, h_c = self.com_gru(com_history); h_c = h_c[-1]
        fused = torch.cat([h_t, h_c], dim=1)
        out = self.fusion(fused)
        return out.view(B, self.horizon, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Re-evaluate methods on static vs moving subsets.')
    parser.add_argument('--moving-frac', type=float, default=0.30,
                        help='fraction of test samples to classify as "moving" '
                             '(threshold chosen so the top fraction by future-window '
                             'peak speed is the moving subset). Default 0.30.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for tactile inference (default 64).')
    parser.add_argument('--no-tactile', action='store_true',
                        help='skip tactile evaluations (use if tactile cache missing or for speed).')
    args = parser.parse_args()

    os.makedirs(_HM, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    print(f'moving fraction: {args.moving_frac:.2f}')

    # ---- Load CoM + metadata ----
    with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
        com_results = pickle.load(f)
    com_gt   = com_results['com_gt']
    kp_gt_mm = com_results['kp_gt_mm']
    T = len(com_gt)

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

    # ---- Build split (same as Phase 1/2) ----
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
    n_test = len(test_centers)
    print(f'samples: total={len(centers)}, train={train_mask.sum()}, test={n_test}')
    assert len(centers) == 17218

    # ---- Build inputs for each test sample ----
    hist_com    = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in test_centers], axis=0)         # (N, 100, 3)
    hist_kp     = np.stack([kp_gt_mm[t - HISTORY + 1 : t + 1].reshape(HISTORY, KP_DIM)
                            for t in test_centers], axis=0)                                          # (N, 100, 63)
    future_true = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in test_centers], axis=0)         # (N, 10, 3)
    ref_now     = com_gt[test_centers]                                                               # (N, 3)

    # ---- Motion criterion: peak xy speed during the 10-frame future window ----
    future_xy   = future_true[:, :, :2]                                # (N, 10, 2)
    step_diffs  = np.diff(future_xy, axis=1)                           # (N, 9, 2)
    step_speeds = np.linalg.norm(step_diffs, axis=2)                   # (N, 9)
    v_future    = step_speeds.max(axis=1)                              # (N,) peak xy speed mm/frame
    threshold   = float(np.quantile(v_future, 1.0 - args.moving_frac))
    moving_mask = v_future > threshold
    static_mask = ~moving_mask
    print(f'\nmotion criterion: peak xy speed in future window (mm/frame)')
    print(f'  v_future distribution: '
          f'min={v_future.min():.2f}  median={np.median(v_future):.2f}  '
          f'p90={np.percentile(v_future, 90):.2f}  max={v_future.max():.2f}')
    print(f'  threshold (top {args.moving_frac:.0%} above this): {threshold:.2f} mm/frame')
    print(f'  partition: moving={moving_mask.sum()}, static={static_mask.sum()}')

    # ---- Stats helpers ----
    def stats_phase1():
        hist = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in train_centers], axis=0).reshape(-1, 3)
        fut  = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_centers], axis=0).reshape(-1, 3)
        pool = np.concatenate([hist, fut], axis=0)
        m = pool.mean(axis=0); s = pool.std(axis=0); s = np.where(s < 1e-6, 1.0, s)
        return m, s

    def stats_kp_xy(delta_target):
        X = np.stack([kp_gt_mm[t - HISTORY + 1 : t + 1].reshape(HISTORY, KP_DIM)
                      for t in train_centers], axis=0).reshape(-1, KP_DIM)
        Y_abs = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_centers], axis=0)
        if delta_target:
            ref = com_gt[train_centers]
            Y = (Y_abs - ref[:, None, :]).reshape(-1, 3)
        else:
            Y = Y_abs.reshape(-1, 3)
        mX = X.mean(axis=0); sX = X.std(axis=0); sX = np.where(sX < 1e-6, 1.0, sX)
        mY = Y.mean(axis=0); sY = Y.std(axis=0); sY = np.where(sY < 1e-6, 1.0, sY)
        return mX, sX, mY, sY

    def stats_tactile(tactile_all):
        np.random.seed(SEED)
        n_tr = train_mask.sum()
        sub = train_centers[np.random.permutation(n_tr)[:1000]]
        chunk = np.concatenate([tactile_all[t - HISTORY + 1 : t + 1] for t in sub], axis=0)
        tac_mean = float(chunk.mean()); tac_std = float(chunk.std())
        Y_abs = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_centers], axis=0)
        ref   = com_gt[train_centers]
        Y_d   = (Y_abs - ref[:, None, :]).reshape(-1, 3)
        mY = Y_d.mean(axis=0); sY = Y_d.std(axis=0); sY = np.where(sY < 1e-6, 1.0, sY)
        return tac_mean, tac_std, mY, sY

    # ---- Run each method on the full test set ----
    predictions = {}

    # 1. Persistence
    predictions['persistence'] = np.broadcast_to(ref_now[:, None, :], (n_test, HORIZON, 3)).copy()
    print('persistence:           ready')

    # 2. Phase 1 GRU
    p1_pt = os.path.join(_OUT, 'phase1', 'gru_model.pt')
    if os.path.exists(p1_pt):
        mean_p1, std_p1 = stats_phase1()
        m1 = P1Forecaster().to(device)
        m1.load_state_dict(torch.load(p1_pt, map_location=device, weights_only=False))
        m1.eval()
        with torch.no_grad():
            x = torch.tensor((hist_com - mean_p1) / std_p1, dtype=torch.float32).to(device)
            y = m1(x).cpu().numpy()
        predictions['phase1_gru_com'] = y * std_p1 + mean_p1
        print('phase1_gru_com:        ready')
    else:
        print(f'phase1_gru_com:        SKIP   ({p1_pt} missing)')

    # 3. Phase 2 v1
    p2v1_pt = os.path.join(_OUT, 'phase2_keypoints', 'gru_model.pt')
    if os.path.exists(p2v1_pt):
        mX, sX, mY, sY = stats_kp_xy(delta_target=False)
        m2 = P2v1Forecaster().to(device)
        m2.load_state_dict(torch.load(p2v1_pt, map_location=device, weights_only=False))
        m2.eval()
        with torch.no_grad():
            x = torch.tensor((hist_kp - mX) / sX, dtype=torch.float32).to(device)
            y = m2(x).cpu().numpy()
        predictions['phase2_v1_gru_kp'] = y * sY + mY
        print('phase2_v1_gru_kp:      ready')
    else:
        print(f'phase2_v1_gru_kp:      SKIP   ({p2v1_pt} missing)')

    # 4. Phase 2 v2
    p2v2_pt_best = os.path.join(_OUT, 'phase2_keypoints_v2', 'gru_model_best.pt')
    p2v2_pt      = p2v2_pt_best if os.path.exists(p2v2_pt_best) else os.path.join(
        _OUT, 'phase2_keypoints_v2', 'gru_model.pt')
    if os.path.exists(p2v2_pt):
        mX, sX, mY, sY = stats_kp_xy(delta_target=True)
        m3 = P2v2Forecaster().to(device)
        m3.load_state_dict(torch.load(p2v2_pt, map_location=device, weights_only=False))
        m3.eval()
        with torch.no_grad():
            x = torch.tensor((hist_kp - mX) / sX, dtype=torch.float32).to(device)
            y = m3(x).cpu().numpy()
        delta = y * sY + mY
        predictions['phase2_v2_gru_kp'] = delta + ref_now[:, None, :]
        print('phase2_v2_gru_kp:      ready')
    else:
        print(f'phase2_v2_gru_kp:      SKIP   ({p2v2_pt} missing)')

    # 5/6/7. Tactile variants + gamma (need the cache; batch through to control memory)
    tac_cache = os.path.join(_OUT, 'tactile_all.npy')
    if args.no_tactile:
        print('tactile_* / gamma:     SKIP (per --no-tactile)')
    elif not os.path.exists(tac_cache):
        print(f'tactile_* / gamma:     SKIP   ({tac_cache} missing)')
    else:
        tactile_all = np.load(tac_cache, mmap_mode='r')
        tac_mean, tac_std, mY, sY = stats_tactile(tactile_all)

        TACTILE_VARIANTS = [
            ('phase2_tactile_50ep',  os.path.join(_OUT, 'phase2_tactile')),
            ('phase2_tactile_200ep', os.path.join(_OUT, 'phase2_tactile_200ep')),
        ]
        for name, ddir in TACTILE_VARIANTS:
            pt_best = os.path.join(ddir, 'tactile_model_best.pt')
            pt      = pt_best if os.path.exists(pt_best) else os.path.join(ddir, 'tactile_model.pt')
            if not os.path.exists(pt):
                print(f'{name:<22}: SKIP   ({pt} missing)')
                continue
            m = TactileForecaster().to(device)
            m.load_state_dict(torch.load(pt, map_location=device, weights_only=False))
            m.eval()
            preds = np.zeros((n_test, HORIZON, 3), dtype=np.float64)
            with torch.no_grad():
                for i0 in range(0, n_test, args.batch_size):
                    i1 = min(i0 + args.batch_size, n_test)
                    batch_centers = test_centers[i0:i1]
                    windows = np.stack(
                        [(tactile_all[t - HISTORY + 1 : t + 1] - tac_mean) / tac_std
                         for t in batch_centers],
                        axis=0
                    ).astype(np.float32)
                    x = torch.from_numpy(windows).to(device)
                    y = m(x).cpu().numpy()
                    delta = y * sY + mY
                    preds[i0:i1] = delta + ref_now[i0:i1][:, None, :]
            predictions[name] = preds
            print(f'{name:<22}: ready')

        # 7. Phase 2 gamma — same tactile cache + CoM-history input. Phase 1 stats for CoM.
        gamma_pt_best = os.path.join(_OUT, 'phase2_gamma', 'gamma_model_best.pt')
        gamma_pt      = gamma_pt_best if os.path.exists(gamma_pt_best) else os.path.join(
            _OUT, 'phase2_gamma', 'gamma_model.pt')
        if os.path.exists(gamma_pt):
            mean_p1, std_p1 = stats_phase1()
            gm = GammaForecaster().to(device)
            gm.load_state_dict(torch.load(gamma_pt, map_location=device, weights_only=False))
            gm.eval()
            preds = np.zeros((n_test, HORIZON, 3), dtype=np.float64)
            hist_com_norm = (hist_com - mean_p1) / std_p1                # (N, 100, 3) standardized
            with torch.no_grad():
                for i0 in range(0, n_test, args.batch_size):
                    i1 = min(i0 + args.batch_size, n_test)
                    batch_centers = test_centers[i0:i1]
                    windows = np.stack(
                        [(tactile_all[t - HISTORY + 1 : t + 1] - tac_mean) / tac_std
                         for t in batch_centers],
                        axis=0
                    ).astype(np.float32)
                    tact = torch.from_numpy(windows).to(device)
                    com_h = torch.from_numpy(hist_com_norm[i0:i1].astype(np.float32)).to(device)
                    y = gm(tact, com_h).cpu().numpy()
                    delta = y * sY + mY
                    preds[i0:i1] = delta + ref_now[i0:i1][:, None, :]
            predictions['phase2_gamma'] = preds
            print(f'phase2_gamma:          ready   ({gamma_pt})')
        else:
            print(f'phase2_gamma:          SKIP    ({gamma_pt} missing)')

        # 8. Phase 2 epsilon — CNN-free ViT + GRU + probe head. Uses tactile only.
        #    Reuses the same tac_mean/tac_std already computed above (= beta's stats,
        #    seeded by seed_tactile_stats.py).
        eps_dyn  = os.path.join(_OUT, 'phase2_epsilon', 'dynamics_model.pt')
        eps_lin  = os.path.join(_OUT, 'phase2_epsilon', 'probe_linear.pt')
        eps_mlp  = os.path.join(_OUT, 'phase2_epsilon', 'probe_mlp.pt')
        if os.path.exists(eps_dyn) and (os.path.exists(eps_lin) or os.path.exists(eps_mlp)):
            from model_epsilon import DynamicsModel, LinearProbe, MLPProbe
            dyn = DynamicsModel().to(device)
            dyn.load_state_dict(torch.load(eps_dyn, map_location=device, weights_only=False)['dynamics'])
            dyn.eval()

            # Encode test hidden states once (shared across both probes)
            H_test = np.zeros((n_test, 128), dtype=np.float32)  # GRU_HIDDEN = 128
            with torch.no_grad():
                for i0 in range(0, n_test, args.batch_size):
                    i1 = min(i0 + args.batch_size, n_test)
                    batch_centers = test_centers[i0:i1]
                    windows = np.stack(
                        [(tactile_all[t - HISTORY + 1 : t + 1] - tac_mean) / tac_std
                         for t in batch_centers],
                        axis=0
                    ).astype(np.float32)
                    x = torch.from_numpy(windows).to(device)
                    H_test[i0:i1] = dyn.encode_history(x).cpu().numpy()
            H_test_t = torch.from_numpy(H_test).to(device)

            def run_probe(probe_cls, probe_ckpt, name):
                probe = probe_cls().to(device)
                probe.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=False)['probe'])
                probe.eval()
                with torch.no_grad():
                    y_norm = probe(H_test_t).cpu().numpy()
                delta = y_norm * sY + mY
                predictions[name] = delta + ref_now[:, None, :]
                print(f'{name:<22}: ready   ({probe_ckpt})')

            if os.path.exists(eps_lin):
                run_probe(LinearProbe, eps_lin, 'phase2_epsilon_linear')
            else:
                print(f'phase2_epsilon_linear: SKIP   ({eps_lin} missing)')
            if os.path.exists(eps_mlp):
                run_probe(MLPProbe, eps_mlp, 'phase2_epsilon_mlp')
            else:
                print(f'phase2_epsilon_mlp:    SKIP   ({eps_mlp} missing)')
        else:
            print(f'phase2_epsilon_*:      SKIP   ({eps_dyn} or probes missing)')

    # ---- Compute metrics per subset ----
    def euc(pred, gt):
        return np.linalg.norm(pred - gt, axis=2)             # (N, H)

    subsets = {
        'full':   np.ones(n_test, dtype=bool),
        'static': static_mask,
        'moving': moving_mask,
    }

    results = {}
    persist_e = euc(predictions['persistence'], future_true)
    persist_medians = {sub: float(np.median(persist_e[mask])) for sub, mask in subsets.items()}

    for name, pred in predictions.items():
        e_3d = euc(pred, future_true)
        e_ax = np.abs(pred - future_true)
        rec = {}
        for sub, mask in subsets.items():
            n_sub = int(mask.sum())
            rec[sub] = {
                'n':                 n_sub,
                'median_3d_mm':      float(np.median(e_3d[mask])),
                'mean_3d_mm':        float(np.mean(e_3d[mask])),
                'p95_3d_mm':         float(np.percentile(e_3d[mask], 95)),
                'per_horizon_median':[float(np.median(e_3d[mask, h])) for h in range(HORIZON)],
                'per_axis_median':   {ax: float(np.median(e_ax[mask, :, i])) for i, ax in enumerate('xyz')},
                'skill_vs_persistence': float(np.median(e_3d[mask]) / persist_medians[sub]) if persist_medians[sub] > 1e-6 else float('nan'),
            }
        results[name] = rec

    # ---- Console summary ----
    bar = '=' * 78
    print(f'\n{bar}\nHIGH-MOTION SUBSET RE-EVALUATION  (n_test = {n_test}, horizon = 1.0 s)\n{bar}')
    print(f'subsets: full ({n_test}), static ({static_mask.sum()}), moving ({moving_mask.sum()})')
    print(f'moving threshold: future-window peak xy speed > {threshold:.2f} mm/frame')

    print(f'\n{"method":<22} {"FULL median":>12} {"STATIC median":>14} {"MOVING median":>14} {"skill@MOVING":>14}')
    for name, rec in results.items():
        print(f'  {name:<20}  '
              f'{rec["full"]["median_3d_mm"]:>10.1f}    '
              f'{rec["static"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["skill_vs_persistence"]:>12.3f}')

    print(f'\nMOVING-subset per-axis medians (mm):')
    for name, rec in results.items():
        a = rec['moving']['per_axis_median']
        print(f'  {name:<22}  x={a["x"]:>6.1f}  y={a["y"]:>6.1f}  z={a["z"]:>6.1f}')

    # ---- Save metrics ----
    out_dump = {
        'threshold_mm_per_frame': threshold,
        'moving_fraction_target': args.moving_frac,
        'n_test':                  n_test,
        'n_static':                int(static_mask.sum()),
        'n_moving':                int(moving_mask.sum()),
        'persistence_medians':     persist_medians,
        'methods':                 results,
    }
    with open(os.path.join(_HM, 'metrics.json'), 'w') as f:
        json.dump(out_dump, f, indent=2)

    # ---- Plots ----
    # 1. Grouped bar chart: full / static / moving for each method
    method_order = list(predictions.keys())
    sub_order    = ['full', 'static', 'moving']
    sub_colors   = {'full': 'tab:gray', 'static': 'tab:blue', 'moving': 'tab:red'}
    width = 0.25
    x = np.arange(len(method_order))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, sub in enumerate(sub_order):
        vals = [results[m][sub]['median_3d_mm'] for m in method_order]
        ax.bar(x + (i - 1) * width, vals, width, label=sub, color=sub_colors[sub], alpha=0.9)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * width, v + 1, f'{v:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(method_order, rotation=25, ha='right')
    ax.set_ylabel('median 3D Euclidean error (mm) at 1-s horizon')
    ax.set_title(f'High-motion subset re-evaluation  '
                 f'(threshold = {threshold:.2f} mm/frame, moving = {moving_mask.sum()}/{n_test})')
    ax.legend(title='subset', loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(_HM, 'comparison_bars.png'), dpi=100)
    plt.close()

    # 2. Per-horizon line plot on the moving subset
    hs = np.arange(1, HORIZON + 1) / 10.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in method_order:
        ax.plot(hs, results[name]['moving']['per_horizon_median'], marker='o', label=name)
    ax.set(xlabel='forecast horizon (seconds)',
           ylabel='median 3D Euclidean error (mm)',
           title='Moving subset — per-horizon error')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_HM, 'error_vs_horizon_moving.png'), dpi=100)
    plt.close()

    # 3. Histogram of motion criterion with threshold marker
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(v_future, bins=80, color='tab:gray', alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.axvline(threshold, color='tab:red', linestyle='--',
               label=f'threshold (top {args.moving_frac:.0%}): {threshold:.2f} mm/frame')
    ax.set(xlabel='peak xy speed during future window (mm/frame)',
           ylabel='# test samples',
           title='Motion-criterion distribution across test samples')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(_HM, 'motion_threshold_histogram.png'), dpi=100)
    plt.close()

    print(f'\nSaved metrics + plots to {_HM}')


if __name__ == '__main__':
    main()

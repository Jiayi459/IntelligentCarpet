"""eval_phase2_epsilon.py — re-evaluate epsilon from saved checkpoints.

Use this when a qsub job died mid-Stage-3 (probes already trained but eval/plots
incomplete), or to re-run the eval with a different moving-fraction without
retraining the probes.

Loads:
    output-dir/dynamics_model.pt
    output-dir/probe_linear.pt
    output-dir/probe_mlp.pt
Writes:
    output-dir/metrics.json  (overwritten)
    output-dir/error_vs_horizon.png
    output-dir/comparison_bars.png

Run:
    python train/com/eval_phase2_epsilon.py
    python train/com/eval_phase2_epsilon.py --moving-frac 0.20
    python train/com/eval_phase2_epsilon.py --output-dir train/com/output/phase2_epsilon_run2
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_HERE  = os.path.dirname(os.path.abspath(__file__))      # train/tactile_direct/
_TRAIN = os.path.dirname(_HERE)                           # train/
sys.path.insert(0, _HERE)

from model_epsilon import DynamicsModel, LinearProbe, MLPProbe, GRU_HIDDEN, HORIZON
from train_phase2_epsilon import (
    _load_dataset_layout, _load_tactile_stats, _encode_all_hidden_states,
    HISTORY, SEED, VAL_FRAC, PROBE_BATCH, MOVING_FRAC, _OUT, _EPSILON,
)


def main():
    parser = argparse.ArgumentParser(description='Re-eval epsilon from saved checkpoints.')
    parser.add_argument('--output-dir',  type=str, default=_EPSILON)
    parser.add_argument('--moving-frac', type=float, default=MOVING_FRAC)
    args = parser.parse_args()

    output_dir = args.output_dir
    moving_frac = args.moving_frac

    if not os.path.isdir(output_dir):
        raise SystemExit(f'no output dir {output_dir}')
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'output_dir  : {output_dir}')
    print(f'moving_frac : {moving_frac:.2f}')
    print(f'device      : {device}')

    tactile_all, com_gt, centers, meta = _load_dataset_layout()
    tactile_mean, tactile_std = _load_tactile_stats()

    test_mask  = meta['split'] == 'test'
    train_mask = meta['split'] == 'train'
    test_subj  = meta['subject'][test_mask]
    n_test = int(test_mask.sum())

    # Targets
    ref_all     = com_gt[centers]
    Y_abs_all   = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in centers], axis=0)
    Y_delta_all = Y_abs_all - ref_all[:, None, :]
    Y_delta_train = Y_delta_all[train_mask]
    Y_abs_test  = Y_abs_all[test_mask]
    ref_test    = ref_all[test_mask]

    mean_Y = Y_delta_train.reshape(-1, 3).mean(axis=0)
    std_Y  = Y_delta_train.reshape(-1, 3).std(axis=0)
    std_Y  = np.where(std_Y < 1e-6, 1.0, std_Y)

    # Load dynamics + probes
    dyn_ckpt = os.path.join(output_dir, 'dynamics_model.pt')
    lin_ckpt = os.path.join(output_dir, 'probe_linear.pt')
    mlp_ckpt = os.path.join(output_dir, 'probe_mlp.pt')
    for p in (dyn_ckpt, lin_ckpt, mlp_ckpt):
        if not os.path.exists(p):
            raise SystemExit(f'missing checkpoint: {p}')
    dyn_model = DynamicsModel().to(device)
    dyn_model.load_state_dict(torch.load(dyn_ckpt, map_location=device, weights_only=False)['dynamics'])
    print(f'loaded dynamics_model.pt')

    linear_probe = LinearProbe().to(device)
    linear_probe.load_state_dict(torch.load(lin_ckpt, map_location=device, weights_only=False)['probe'])
    mlp_probe = MLPProbe().to(device)
    mlp_probe.load_state_dict(torch.load(mlp_ckpt, map_location=device, weights_only=False)['probe'])
    print(f'loaded probe_linear.pt and probe_mlp.pt')

    # Encode test hidden states
    test_global = np.where(test_mask)[0]
    print('encoding test hidden states...')
    H_test = _encode_all_hidden_states(dyn_model, centers, test_global,
                                        tactile_all, tactile_mean, tactile_std,
                                        batch_size=PROBE_BATCH, device=device)
    H_test_t = torch.from_numpy(H_test).to(device)

    def predict(probe):
        probe.eval()
        with torch.no_grad():
            preds_norm = probe(H_test_t).cpu().numpy()
        preds_delta = preds_norm * std_Y + mean_Y
        return preds_delta + ref_test[:, None, :]

    pred_linear      = predict(linear_probe)
    pred_mlp         = predict(mlp_probe)
    pred_persistence = np.broadcast_to(ref_test[:, None, :], (n_test, HORIZON, 3)).copy()

    # Motion partition
    future_xy   = Y_abs_test[:, :, :2]
    step_speeds = np.linalg.norm(np.diff(future_xy, axis=1), axis=2)
    v_future    = step_speeds.max(axis=1)
    threshold   = float(np.quantile(v_future, 1.0 - moving_frac))
    moving_mask = v_future > threshold
    static_mask = ~moving_mask
    subsets = {'full': np.ones(n_test, dtype=bool),
               'static': static_mask, 'moving': moving_mask}

    def euc(p, g): return np.linalg.norm(p - g, axis=2)

    methods = {
        'persistence': pred_persistence,
        'phase2_epsilon_linear': pred_linear,
        'phase2_epsilon_mlp':    pred_mlp,
    }
    persist_e = euc(pred_persistence, Y_abs_test)
    persist_med = {sub: float(np.median(persist_e[m])) for sub, m in subsets.items()}

    results = {}
    for name, pred in methods.items():
        e_3d = euc(pred, Y_abs_test); e_ax = np.abs(pred - Y_abs_test)
        rec = {}
        for sub, mask in subsets.items():
            rec[sub] = {
                'n':                int(mask.sum()),
                'median_3d_mm':     float(np.median(e_3d[mask])),
                'mean_3d_mm':       float(np.mean(e_3d[mask])),
                'p95_3d_mm':        float(np.percentile(e_3d[mask], 95)),
                'per_horizon_median': [float(np.median(e_3d[mask, h])) for h in range(HORIZON)],
                'per_axis_median':  {ax: float(np.median(e_ax[mask, :, i])) for i, ax in enumerate('xyz')},
                'skill_vs_persistence': (float(np.median(e_3d[mask]) / persist_med[sub])
                                          if persist_med[sub] > 1e-6 else float('nan')),
            }
        results[name] = rec

    per_subject_mlp = {}
    e_mlp = euc(pred_mlp, Y_abs_test)
    for subj in sorted(np.unique(test_subj)):
        sm = test_subj == subj
        per_subject_mlp[subj] = {
            'n_full':        int(sm.sum()),
            'median_full':   float(np.median(e_mlp[sm])),
            'n_moving':      int((sm & moving_mask).sum()),
            'median_moving': (float(np.median(e_mlp[sm & moving_mask]))
                               if (sm & moving_mask).any() else float('nan')),
        }
    results['phase2_epsilon_mlp']['per_subject'] = per_subject_mlp

    out_dump = {
        'n_test':                 n_test,
        'threshold_mm_per_frame': threshold,
        'n_static':               int(static_mask.sum()),
        'n_moving':               int(moving_mask.sum()),
        'methods':                results,
        'eval_only_rerun':        True,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(out_dump, f, indent=2)
    print(f'wrote {os.path.join(output_dir, "metrics.json")}')

    # Console summary
    bar = '=' * 78
    print(f'\n{bar}\nEPSILON EVAL-ONLY RESULTS  (n_test={n_test}, '
          f'static={static_mask.sum()}, moving={moving_mask.sum()})\n{bar}')
    print(f'{"method":<26} {"FULL":>10} {"STATIC":>10} {"MOVING":>10} {"skill@MOV":>12}')
    for name, rec in results.items():
        print(f'  {name:<24}  {rec["full"]["median_3d_mm"]:>8.1f}  '
              f'{rec["static"]["median_3d_mm"]:>8.1f}  '
              f'{rec["moving"]["median_3d_mm"]:>8.1f}  '
              f'{rec["moving"]["skill_vs_persistence"]:>10.3f}')


if __name__ == '__main__':
    main()

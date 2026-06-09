"""train_phase2_epsilon.py — CNN-free tactile forecaster (epsilon).

Three-stage self-supervised recipe for tactile-only CoM forecasting:

    Stage 1 (mae)      : MAE-style masked patch reconstruction on individual
                         tactile frames. Trains the ViT encoder only.
                         Frames source: train-window-only.
    Stage 2 (dynamics) : Encoder + GRU pretraining on delta-tactile forecasting
                         (predict next 1 s of tactile change). Initializes
                         encoder from mae checkpoint. Trains encoder + GRU
                         + a factored patch-latent decoder (~1.5M params)
                         which is discarded afterwards.
    Stage 3 (probe)    : Freeze encoder + GRU; train a tiny linear probe AND
                         a tiny MLP probe mapping the GRU's final hidden state
                         to future CoM delta. Evaluate both on the test set
                         (full / static / moving subsets, same protocol as
                         high_motion_subset.py and phase2_gamma).

Run (full pipeline; default budget ~ < 2 h on A10):
    python train/com/train_phase2_epsilon.py --stage all

Run individual stages:
    python train/com/train_phase2_epsilon.py --stage mae       --epochs-mae 30
    python train/com/train_phase2_epsilon.py --stage dynamics  --epochs-dynamics 20
    python train/com/train_phase2_epsilon.py --stage probe     --epochs-probe 30

All heavy execution lives inside main(); importing this file is side-effect-free.
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

_HERE  = os.path.dirname(os.path.abspath(__file__))      # train/tactile_direct/
_TRAIN = os.path.dirname(_HERE)                           # train/
sys.path.insert(0, _HERE)

from model_epsilon import (
    ViTEncoder, MAEDecoder, DynamicsModel,
    LinearProbe, MLPProbe, EpsilonForecaster,
    IMG_SIZE, PATCH_SIZE, N_PATCHES, PATCH_DIM,
    EMBED_DIM, GRU_HIDDEN, HORIZON as MODEL_HORIZON,
    MAE_MASK_RATIO,
)

# Outputs (tactile cache, com_results.p, tactile_stats.json, phase2_epsilon/) all
# live under train/com/output/ to stay grouped with beta/gamma/phase1/v1/v2's
# artifacts. The output relocation to a sibling train/output/ is tracked as a
# separate cleanup (see SESSION_LOG 2026-06-09).
_OUT       = os.path.join(_TRAIN, 'com', 'output')
_EPSILON   = os.path.join(_OUT, 'phase2_epsilon')
_CACHE_NPY = os.path.join(_OUT, 'tactile_all.npy')
_STATS     = os.path.join(_OUT, 'tactile_stats.json')


# ---------------------------------------------------------------------------
# Run-time constants (history/horizon/seed match everything else in the project)
# ---------------------------------------------------------------------------

HISTORY            = 100
HORIZON            = MODEL_HORIZON     # 10
SEED               = 42

# Stage 1 (MAE)
MAE_EPOCHS_DEFAULT = 30
MAE_BATCH          = 128
MAE_LR             = 3e-4

# Stage 2 (dynamics)
DYN_EPOCHS_DEFAULT = 20
DYN_BATCH          = 32
DYN_LR             = 1e-3
VAL_FRAC           = 0.10
ACTIVE_THRESHOLD   = 0.05              # raw (pre-standardization) tactile threshold

# Stage 3 (probe)
PROBE_EPOCHS_DEFAULT = 30
PROBE_BATCH          = 64
PROBE_LR             = 1e-3

# Eval (motion-subset split — same as high_motion_subset.py and gamma)
MOVING_FRAC        = 0.30


# ===========================================================================
# Shared setup helpers
# ===========================================================================

def _load_dataset_layout():
    """Build the (centers, masks, com_gt, sessions, subjects) tuple shared by
    all stages. Mirrors phase2_gamma.main() exactly so sample indices are
    bit-identical across phases."""
    if not os.path.exists(_CACHE_NPY):
        raise SystemExit(f'no tactile cache at {_CACHE_NPY} -- build it with phase2_tactile.py first')
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
    assert len(centers) == 17218, f'expected 17218 samples, got {len(centers)}'
    return tactile_all, com_gt, centers, meta


def _load_tactile_stats():
    if not os.path.exists(_STATS):
        raise SystemExit(
            f'no tactile stats cache at {_STATS} -- run seed_tactile_stats.py first.'
        )
    with open(_STATS) as f:
        s = json.load(f)
    return float(s['tactile_mean']), float(s['tactile_std'])


# ===========================================================================
# Stage 1 -- MAE-style spatial pretraining of the ViT encoder
# ===========================================================================

def _sample_mae_mask(B, device):
    """Sample a per-sample random subset of N_PATCHES patches to KEEP (the rest
    are masked). Returns keep_indices (B, K) and mask_indices (B, M) where
    K + M = N_PATCHES.

    Uses MAE_MASK_RATIO = 0.75 -> M = 27 masked, K = 9 visible.
    """
    K = int(round(N_PATCHES * (1.0 - MAE_MASK_RATIO)))     # 9
    M = N_PATCHES - K                                       # 27
    # Random permutation per sample
    rand = torch.rand(B, N_PATCHES, device=device)
    perm = rand.argsort(dim=1)
    return perm[:, :K], perm[:, K:]                         # (B, 9), (B, 27)


def run_stage_mae(args, output_dir, device):
    print('\n' + '=' * 78)
    print('STAGE 1 (MAE) — masked patch reconstruction, train-window-only frames')
    print('=' * 78)

    tactile_all, com_gt, centers, meta = _load_dataset_layout()
    tactile_mean, tactile_std = _load_tactile_stats()
    print(f'tactile_mean={tactile_mean:.4f}  tactile_std={tactile_std:.4f}')

    # Build the train-window-only frame index (union of [t-99..t] for train centers)
    T = tactile_all.shape[0]
    train_mask = meta['split'] == 'train'
    is_train_frame = np.zeros(T, dtype=bool)
    for t in centers[train_mask]:
        is_train_frame[t - HISTORY + 1 : t + 1] = True
    train_frame_idx = np.where(is_train_frame)[0]
    print(f'train-window frames: {len(train_frame_idx)} / {T}  '
          f'({100 * len(train_frame_idx) / T:.1f} %)')

    encoder     = ViTEncoder().to(device)
    mae_decoder = MAEDecoder().to(device)
    n_params    = sum(p.numel() for p in encoder.parameters()) + \
                  sum(p.numel() for p in mae_decoder.parameters())
    enc_params  = sum(p.numel() for p in encoder.parameters())
    print(f'encoder params  : {enc_params}')
    print(f'mae decoder pars: {n_params - enc_params}')
    print(f'mask ratio      : {MAE_MASK_RATIO:.2f}  (9 visible / 27 masked patches)')

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(mae_decoder.parameters()),
        lr=MAE_LR,
    )
    n_batches_per_epoch = (len(train_frame_idx) + MAE_BATCH - 1) // MAE_BATCH

    train_losses = []
    epochs = args.epochs_mae
    print(f'training {epochs} epochs of {n_batches_per_epoch} batches each')
    for epoch in range(epochs):
        encoder.train(); mae_decoder.train()
        np.random.shuffle(train_frame_idx)
        total = 0.0; n_b = 0
        ep_start = time.time()
        for bi in range(0, len(train_frame_idx), MAE_BATCH):
            batch_idx = train_frame_idx[bi : bi + MAE_BATCH]
            frames    = np.asarray(tactile_all[batch_idx])               # (B, 96, 96) float32
            frames    = (frames - tactile_mean) / tactile_std
            x         = torch.from_numpy(frames.astype(np.float32)).to(device)
            B         = x.shape[0]

            keep_idx, mask_idx = _sample_mae_mask(B, device)
            from model_epsilon import patchify                            # local import (cheap)
            true_patches = patchify(x)                                    # (B, 36, 256) standardized

            z_keep       = encoder.encode_tokens(x, keep_indices=keep_idx)
            pred_patches = mae_decoder(z_keep, keep_idx)                  # (B, 36, 256)

            # Loss only on masked patches
            gather = mask_idx.unsqueeze(-1).expand(-1, -1, PATCH_DIM)
            loss = ((pred_patches.gather(1, gather) - true_patches.gather(1, gather)) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()); n_b += 1
        avg_train = total / n_b
        train_losses.append(avg_train)
        print(f'  epoch {epoch:3d}/{epochs - 1}  '
              f'train_mae_mse={avg_train:.5f}  ({time.time() - ep_start:.1f}s)',
              flush=True)

    # Save encoder + mae decoder
    ckpt_path = os.path.join(output_dir, 'mae_encoder.pt')
    torch.save(
        {'encoder': encoder.state_dict(),
         'mae_decoder': mae_decoder.state_dict(),
         'train_losses': train_losses,
         'config': {'mask_ratio': MAE_MASK_RATIO, 'epochs': epochs,
                    'batch': MAE_BATCH, 'lr': MAE_LR}},
        ckpt_path,
    )
    print(f'saved {ckpt_path}')

    # Training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='train', color='tab:blue')
    ax.set(xlabel='epoch', ylabel='masked-patch MSE (standardized)',
           title=f'Stage 1 (MAE) — mask {MAE_MASK_RATIO:.0%}, '
                 f'{len(train_frame_idx)} train frames')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_training_curve.png'), dpi=100)
    plt.close()

    # Save a qualitative reconstruction grid for sanity (last batch)
    _save_mae_recon_examples(encoder, mae_decoder, x, keep_idx, mask_idx,
                              tactile_mean, tactile_std,
                              os.path.join(output_dir, 'mae_recon_examples.png'))


def _save_mae_recon_examples(encoder, mae_decoder, x, keep_idx, mask_idx,
                              tac_mean, tac_std, save_path, n=6):
    """Plot a grid of (original | masked-input | reconstructed) examples."""
    from model_epsilon import patchify, unpatchify
    encoder.eval(); mae_decoder.eval()
    with torch.no_grad():
        true_patches = patchify(x)
        z_keep = encoder.encode_tokens(x, keep_indices=keep_idx)
        pred_patches = mae_decoder(z_keep, keep_idx)

    n = min(n, x.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        # Original
        orig = (x[i].cpu().numpy() * tac_std + tac_mean)
        axes[i, 0].imshow(orig, vmin=0, vmax=1, cmap='viridis')
        axes[i, 0].set_title('original' if i == 0 else '')
        axes[i, 0].axis('off')

        # Masked input (set masked patches to gray=0.5)
        masked = true_patches[i].clone()
        masked[mask_idx[i]] = 0.0
        masked_img = unpatchify(masked.unsqueeze(0))[0].cpu().numpy()
        axes[i, 1].imshow(masked_img * tac_std + tac_mean, vmin=0, vmax=1, cmap='viridis')
        axes[i, 1].set_title('encoder input' if i == 0 else '')
        axes[i, 1].axis('off')

        # Reconstruction (use predicted patches at masked positions, true elsewhere)
        merged = true_patches[i].clone()
        merged[mask_idx[i]] = pred_patches[i, mask_idx[i]]
        merged_img = unpatchify(merged.unsqueeze(0))[0].cpu().numpy()
        axes[i, 2].imshow(merged_img * tac_std + tac_mean, vmin=0, vmax=1, cmap='viridis')
        axes[i, 2].set_title('reconstruction' if i == 0 else '')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'saved {save_path}')


# ===========================================================================
# Stage 2 -- delta-tactile dynamics pretraining
# ===========================================================================

class DynamicsDataset(Dataset):
    """Yields (tactile_history_std, delta_tactile_future_std, active_mask_current).

    active_mask_current is the per-cell mask used for the active-region MSE
    diagnostic; it is computed from the *raw* (pre-standardization) tactile
    at the current frame.
    """
    def __init__(self, centers, indices_local, tactile_all, tactile_mean, tactile_std):
        self.centers       = centers
        self.indices_local = indices_local
        self.tactile_all   = tactile_all
        self.tactile_mean  = tactile_mean
        self.tactile_std   = tactile_std

    def __len__(self):
        return len(self.indices_local)

    def __getitem__(self, idx):
        i = int(self.indices_local[idx])
        t = int(self.centers[i])
        hist_raw  = np.asarray(self.tactile_all[t - HISTORY + 1 : t + 1])     # (100, 96, 96)
        fut_raw   = np.asarray(self.tactile_all[t + 1 : t + 1 + HORIZON])     # (10, 96, 96)
        current   = hist_raw[-1]
        # Targets: delta in RAW space (still in [-1, 1]-ish), then standardize by dividing by tactile_std
        # We standardize the input window but predict the delta in standardized units too,
        # which is just (fut_raw - current) / tactile_std.
        hist_std    = (hist_raw - self.tactile_mean) / self.tactile_std
        delta_std   = (fut_raw - current[None]) / self.tactile_std
        active_mask = (current > ACTIVE_THRESHOLD).astype(np.float32)         # (96, 96)
        return (
            torch.from_numpy(hist_std.astype(np.float32)),
            torch.from_numpy(delta_std.astype(np.float32)),
            torch.from_numpy(active_mask),
        )


def run_stage_dynamics(args, output_dir, device):
    print('\n' + '=' * 78)
    print('STAGE 2 (dynamics) — delta-tactile forecasting pretrain')
    print('=' * 78)

    tactile_all, com_gt, centers, meta = _load_dataset_layout()
    tactile_mean, tactile_std = _load_tactile_stats()
    print(f'tactile_mean={tactile_mean:.4f}  tactile_std={tactile_std:.4f}')

    train_mask = meta['split'] == 'train'
    test_mask  = meta['split'] == 'test'
    n_train_tot = train_mask.sum()
    print(f'samples: total={len(centers)}, train={n_train_tot}, test={test_mask.sum()}')

    # Same train/val split as gamma (replayed RNG order matches phase2_tactile / gamma)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_global_idx = np.where(train_mask)[0]
    perm  = np.random.permutation(n_train_tot)
    n_val = int(n_train_tot * VAL_FRAC)
    val_local = train_global_idx[perm[:n_val]]
    tr_local  = train_global_idx[perm[n_val:]]

    train_ds = DynamicsDataset(centers, tr_local,  tactile_all, tactile_mean, tactile_std)
    val_ds   = DynamicsDataset(centers, val_local, tactile_all, tactile_mean, tactile_std)
    train_loader = DataLoader(train_ds, batch_size=DYN_BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=DYN_BATCH, shuffle=False, num_workers=0)

    model = DynamicsModel().to(device)
    n_params_total  = sum(p.numel() for p in model.parameters())
    n_params_enc    = sum(p.numel() for p in model.encoder.parameters())
    n_params_gru    = sum(p.numel() for p in model.gru.parameters())
    n_params_dec    = n_params_total - n_params_enc - n_params_gru
    print(f'dynamics params: encoder={n_params_enc}, gru={n_params_gru}, '
          f'decoder={n_params_dec}, total={n_params_total}')

    # Init encoder from Stage 1
    mae_ckpt = os.path.join(output_dir, 'mae_encoder.pt')
    if os.path.exists(mae_ckpt):
        ck = torch.load(mae_ckpt, map_location=device, weights_only=False)
        model.encoder.load_state_dict(ck['encoder'])
        print(f'loaded encoder weights from {mae_ckpt}')
    else:
        print(f'NOTE: no MAE checkpoint at {mae_ckpt} -- training encoder from scratch.')

    optimizer = torch.optim.Adam(model.parameters(), lr=DYN_LR)

    # Sanity baseline: delta = 0 MSE (= mean of delta**2). Computed on val once.
    print('computing tactile-persistence baseline (delta=0 MSE) on val set...')
    with torch.no_grad():
        zb_total = 0.0; zb_total_active = 0.0; zb_n = 0; zb_n_active = 0
        for _, delta_std, active in val_loader:
            zb_total       += float((delta_std ** 2).sum().item())
            zb_n           += delta_std.numel()
            am             = active.unsqueeze(1).expand_as(delta_std)      # (B, 10, 96, 96)
            zb_total_active += float(((delta_std ** 2) * am).sum().item())
            zb_n_active     += float(am.sum().item())
    val_persistence_mse        = zb_total / zb_n
    val_persistence_mse_active = (zb_total_active / zb_n_active) if zb_n_active > 0 else float('nan')
    print(f'val tactile-persistence MSE        : {val_persistence_mse:.5f}')
    print(f'val tactile-persistence MSE (active): {val_persistence_mse_active:.5f}')

    epochs = args.epochs_dynamics
    print(f'training {epochs} epochs')
    train_losses, val_losses, val_active_losses = [], [], []
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        total = 0.0; n_b = 0
        ep_start = time.time()
        for hist, delta, _active in train_loader:
            hist  = hist.to(device); delta = delta.to(device)
            pred  = model(hist)
            loss  = ((pred - delta) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.item()); n_b += 1
        avg_train = total / n_b

        model.eval()
        with torch.no_grad():
            vtot = 0.0; vb = 0
            vtot_active = 0.0; vn_active = 0.0
            for hist, delta, active in val_loader:
                hist  = hist.to(device); delta = delta.to(device); active = active.to(device)
                pred  = model(hist)
                err2  = (pred - delta) ** 2
                vtot += float(err2.mean().item()); vb += 1
                am   = active.unsqueeze(1).expand_as(err2)
                vtot_active += float((err2 * am).sum().item())
                vn_active   += float(am.sum().item())
        avg_val        = vtot / vb
        avg_val_active = (vtot_active / vn_active) if vn_active > 0 else float('nan')

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        val_active_losses.append(avg_val_active)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'dynamics': model.state_dict(),
                        'epoch': epoch,
                        'best_val': best_val},
                       os.path.join(output_dir, 'dynamics_model_best.pt'))

        skill_all    = avg_val        / val_persistence_mse        if val_persistence_mse        > 0 else float('nan')
        skill_active = avg_val_active / val_persistence_mse_active if val_persistence_mse_active > 0 else float('nan')
        print(f'  epoch {epoch:3d}/{epochs - 1}  '
              f'train={avg_train:.5f}  val={avg_val:.5f}  '
              f'val_active={avg_val_active:.5f}  '
              f'skill(all|active)={skill_all:.3f}|{skill_active:.3f}  '
              f'best_val={best_val:.5f}  '
              f'({time.time() - ep_start:.1f}s)', flush=True)

    # Reload best-val and save final dynamics_model.pt (= encoder + GRU + decoder)
    best_ck = torch.load(os.path.join(output_dir, 'dynamics_model_best.pt'),
                          map_location=device, weights_only=False)
    model.load_state_dict(best_ck['dynamics'])
    torch.save({'dynamics': model.state_dict(),
                'epoch': best_ck['epoch'],
                'best_val': best_ck['best_val'],
                'val_persistence_mse': val_persistence_mse,
                'val_persistence_mse_active': val_persistence_mse_active,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_active_losses': val_active_losses,
                'config': {'epochs': epochs, 'batch': DYN_BATCH, 'lr': DYN_LR,
                           'active_threshold': ACTIVE_THRESHOLD}},
               os.path.join(output_dir, 'dynamics_model.pt'))
    print(f'saved dynamics_model.pt (best @ epoch {best_ck["epoch"]}, '
          f'val_mse={best_ck["best_val"]:.5f})')

    # Stage 2 training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    epoch_x = np.arange(len(train_losses))
    ax.plot(epoch_x, train_losses, label='train', color='tab:blue')
    ax.plot(epoch_x, val_losses,   label='val',   color='tab:orange', linestyle='--')
    ax.plot(epoch_x, val_active_losses, label='val (active region)',
             color='tab:red', linestyle=':')
    ax.axhline(val_persistence_mse, color='gray', linestyle=':', alpha=0.6,
                label=f'persistence floor (all) = {val_persistence_mse:.4f}')
    ax.axhline(val_persistence_mse_active, color='tab:red', linestyle=':',
                alpha=0.4,
                label=f'persistence floor (active) = {val_persistence_mse_active:.4f}')
    ax.set(xlabel='epoch', ylabel='MSE (standardized delta-tactile)',
           title='Stage 2 (dynamics) — delta-tactile forecasting')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage2_val_loss_curve.png'), dpi=100)
    plt.close()


# ===========================================================================
# Stage 3 -- CoM probe (linear + MLP)
# ===========================================================================

def _encode_all_hidden_states(dyn_model, centers, indices_global, tactile_all,
                              tac_mean, tac_std, batch_size, device):
    """Run encoder + GRU on each sample to get its final hidden state (B, GRU_HIDDEN).
    Returns numpy array (N, GRU_HIDDEN) cached in memory."""
    dyn_model.eval()
    N = len(indices_global)
    out = np.zeros((N, GRU_HIDDEN), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
            batch_centers = centers[indices_global[i0:i1]]
            windows = np.stack([
                (np.asarray(tactile_all[t - HISTORY + 1 : t + 1]) - tac_mean) / tac_std
                for t in batch_centers
            ], axis=0).astype(np.float32)
            x = torch.from_numpy(windows).to(device)
            h = dyn_model.encode_history(x).cpu().numpy()
            out[i0:i1] = h
    return out


def run_stage_probe(args, output_dir, device):
    print('\n' + '=' * 78)
    print('STAGE 3 (probe) — train linear + MLP probes; evaluate on full/static/moving')
    print('=' * 78)

    tactile_all, com_gt, centers, meta = _load_dataset_layout()
    tactile_mean, tactile_std = _load_tactile_stats()

    train_mask = meta['split'] == 'train'
    test_mask  = meta['split'] == 'test'
    test_subj  = meta['subject'][test_mask]
    n_train_tot = int(train_mask.sum())
    n_test      = int(test_mask.sum())

    # Targets (delta CoM, same as v2 / beta / gamma)
    ref_all      = com_gt[centers]
    Y_abs_all    = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in centers], axis=0)
    Y_delta_all  = Y_abs_all - ref_all[:, None, :]
    Y_delta_train = Y_delta_all[train_mask]
    Y_delta_test  = Y_delta_all[test_mask]
    ref_test     = ref_all[test_mask]
    Y_abs_test   = Y_abs_all[test_mask]

    # Delta-target standardization (same as v2 / beta / gamma)
    mean_Y = Y_delta_train.reshape(-1, 3).mean(axis=0)
    std_Y  = Y_delta_train.reshape(-1, 3).std(axis=0)
    std_Y  = np.where(std_Y < 1e-6, 1.0, std_Y)
    print(f'Y_delta mean={mean_Y}  std={std_Y}')

    # Load frozen encoder + GRU from Stage 2
    dyn_ckpt_path = os.path.join(output_dir, 'dynamics_model.pt')
    if not os.path.exists(dyn_ckpt_path):
        raise SystemExit(
            f'no dynamics checkpoint at {dyn_ckpt_path}. Run --stage dynamics first.'
        )
    dyn_model = DynamicsModel().to(device)
    ck = torch.load(dyn_ckpt_path, map_location=device, weights_only=False)
    dyn_model.load_state_dict(ck['dynamics'])
    print(f'loaded dynamics_model.pt (epoch {ck.get("epoch", "?")}, '
          f'best_val={ck.get("best_val", float("nan")):.5f})')

    # Same train/val split as Stage 2
    np.random.seed(SEED); torch.manual_seed(SEED)
    train_global_idx = np.where(train_mask)[0]
    perm  = np.random.permutation(n_train_tot)
    n_val = int(n_train_tot * VAL_FRAC)
    val_global  = train_global_idx[perm[:n_val]]
    tr_global   = train_global_idx[perm[n_val:]]
    test_global = np.where(test_mask)[0]

    # Cache hidden states (forward through encoder + GRU once)
    print(f'caching hidden states: train={len(tr_global)}, val={len(val_global)}, test={len(test_global)}...')
    t0 = time.time()
    H_train = _encode_all_hidden_states(dyn_model, centers, tr_global,
                                         tactile_all, tactile_mean, tactile_std,
                                         batch_size=PROBE_BATCH, device=device)
    H_val   = _encode_all_hidden_states(dyn_model, centers, val_global,
                                         tactile_all, tactile_mean, tactile_std,
                                         batch_size=PROBE_BATCH, device=device)
    H_test  = _encode_all_hidden_states(dyn_model, centers, test_global,
                                         tactile_all, tactile_mean, tactile_std,
                                         batch_size=PROBE_BATCH, device=device)
    print(f'hidden states cached in {time.time() - t0:.0f}s')

    # Targets in standardized-delta units, aligned with hidden states
    Y_tr_norm  = (Y_delta_all[tr_global]  - mean_Y) / std_Y
    Y_val_norm = (Y_delta_all[val_global] - mean_Y) / std_Y

    H_train_t  = torch.from_numpy(H_train).to(device)
    Y_tr_t     = torch.from_numpy(Y_tr_norm.astype(np.float32)).to(device)
    H_val_t    = torch.from_numpy(H_val).to(device)
    Y_val_t    = torch.from_numpy(Y_val_norm.astype(np.float32)).to(device)
    H_test_t   = torch.from_numpy(H_test).to(device)

    def train_probe(probe, name):
        probe = probe.to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
        N = H_train_t.shape[0]
        train_curve, val_curve = [], []
        best_val_mse = float('inf')
        best_state = None
        epochs = args.epochs_probe
        for epoch in range(epochs):
            probe.train()
            perm = torch.randperm(N, device=device)
            total = 0.0; n_b = 0
            for i0 in range(0, N, PROBE_BATCH):
                idx = perm[i0:i0 + PROBE_BATCH]
                hb = H_train_t[idx]; yb = Y_tr_t[idx]
                pred = probe(hb)
                loss = ((pred - yb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item()); n_b += 1
            avg_train = total / n_b

            probe.eval()
            with torch.no_grad():
                v_pred = probe(H_val_t)
                v_loss = float(((v_pred - Y_val_t) ** 2).mean().item())
            train_curve.append(avg_train); val_curve.append(v_loss)
            if v_loss < best_val_mse:
                best_val_mse = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            print(f'    [{name}] epoch {epoch:3d}/{epochs - 1}  '
                  f'train={avg_train:.5f}  val={v_loss:.5f}  best_val={best_val_mse:.5f}',
                  flush=True)
        probe.load_state_dict(best_state)
        return probe, train_curve, val_curve, best_val_mse

    print('\n--- linear probe ---')
    linear_probe, lin_tc, lin_vc, lin_bv = train_probe(LinearProbe(), 'linear')
    torch.save({'probe': linear_probe.state_dict(),
                'best_val_mse': lin_bv,
                'train_curve': lin_tc, 'val_curve': lin_vc},
               os.path.join(output_dir, 'probe_linear.pt'))

    print('\n--- MLP probe ---')
    mlp_probe, mlp_tc, mlp_vc, mlp_bv = train_probe(MLPProbe(), 'MLP')
    torch.save({'probe': mlp_probe.state_dict(),
                'best_val_mse': mlp_bv,
                'train_curve': mlp_tc, 'val_curve': mlp_vc},
               os.path.join(output_dir, 'probe_mlp.pt'))

    # Probe training curve (one figure with both)
    fig, ax = plt.subplots(figsize=(8, 5))
    ex = np.arange(len(lin_tc))
    ax.plot(ex, lin_tc, label='linear train', color='tab:blue')
    ax.plot(ex, lin_vc, label='linear val',   color='tab:blue', linestyle='--')
    ax.plot(ex, mlp_tc, label='MLP train',    color='tab:orange')
    ax.plot(ex, mlp_vc, label='MLP val',      color='tab:orange', linestyle='--')
    ax.set(xlabel='epoch', ylabel='MSE (standardized delta CoM)',
           title='Stage 3 probes — linear vs MLP')
    ax.legend(fontsize='small'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probe_training_curve.png'), dpi=100)
    plt.close()

    # ---- Evaluate both probes on the test set ----
    print('\n--- evaluating on test set (full / static / moving) ---')

    def predict_with_probe(probe):
        probe.eval()
        with torch.no_grad():
            preds_norm = probe(H_test_t).cpu().numpy()
        preds_delta = preds_norm * std_Y + mean_Y                          # de-standardize
        return preds_delta + ref_test[:, None, :]                          # absolute CoM

    pred_linear = predict_with_probe(linear_probe)
    pred_mlp    = predict_with_probe(mlp_probe)
    pred_persistence = np.broadcast_to(ref_test[:, None, :], (n_test, HORIZON, 3)).copy()

    # Motion criterion (same as high_motion_subset.py / gamma)
    future_xy = Y_abs_test[:, :, :2]
    step_speeds = np.linalg.norm(np.diff(future_xy, axis=1), axis=2)
    v_future = step_speeds.max(axis=1)
    threshold = float(np.quantile(v_future, 1.0 - MOVING_FRAC))
    moving_mask = v_future > threshold
    static_mask = ~moving_mask
    subsets = {'full': np.ones(n_test, dtype=bool),
               'static': static_mask, 'moving': moving_mask}

    def euc(pred, gt):
        return np.linalg.norm(pred - gt, axis=2)

    methods = {
        'persistence': pred_persistence,
        'phase2_epsilon_linear': pred_linear,
        'phase2_epsilon_mlp':    pred_mlp,
    }
    persist_e = euc(pred_persistence, Y_abs_test)
    persist_med_per_sub = {sub: float(np.median(persist_e[m])) for sub, m in subsets.items()}

    results = {}
    for name, pred in methods.items():
        e_3d = euc(pred, Y_abs_test)
        e_ax = np.abs(pred - Y_abs_test)
        rec = {}
        for sub, mask in subsets.items():
            rec[sub] = {
                'n':                int(mask.sum()),
                'median_3d_mm':     float(np.median(e_3d[mask])),
                'mean_3d_mm':       float(np.mean(e_3d[mask])),
                'p95_3d_mm':        float(np.percentile(e_3d[mask], 95)),
                'per_horizon_median': [float(np.median(e_3d[mask, h])) for h in range(HORIZON)],
                'per_axis_median':  {ax: float(np.median(e_ax[mask, :, i])) for i, ax in enumerate('xyz')},
                'skill_vs_persistence': (float(np.median(e_3d[mask]) / persist_med_per_sub[sub])
                                          if persist_med_per_sub[sub] > 1e-6 else float('nan')),
            }
        results[name] = rec

    # Per-subject medians for the MLP probe (the more-likely-to-be-headline result)
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
        'linear_best_val_mse':    lin_bv,
        'mlp_best_val_mse':       mlp_bv,
        'dynamics_epoch':         int(ck.get('epoch', -1)),
        'dynamics_best_val_mse':  float(ck.get('best_val', float('nan'))),
        'methods':                results,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(out_dump, f, indent=2)

    # ---- Console summary ----
    bar = '=' * 78
    print(f'\n{bar}\nEPSILON RESULTS  (n_test = {n_test}, horizon = 1.0 s)\n{bar}')
    print(f'subsets: full ({n_test}), static ({static_mask.sum()}), moving ({moving_mask.sum()})')
    print(f'moving threshold: future-window peak xy speed > {threshold:.2f} mm/frame')

    print(f'\n{"method":<26} {"FULL median":>12} {"STATIC median":>14} {"MOVING median":>14} {"skill@MOVING":>14}')
    for name in ['persistence', 'phase2_epsilon_linear', 'phase2_epsilon_mlp']:
        rec = results[name]
        print(f'  {name:<24}  '
              f'{rec["full"]["median_3d_mm"]:>10.1f}    '
              f'{rec["static"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["median_3d_mm"]:>12.1f}    '
              f'{rec["moving"]["skill_vs_persistence"]:>12.3f}')

    print('\nMOVING-subset per-axis medians (mm):')
    for name in ['persistence', 'phase2_epsilon_linear', 'phase2_epsilon_mlp']:
        a = results[name]['moving']['per_axis_median']
        print(f'  {name:<26}  x={a["x"]:>6.1f}  y={a["y"]:>6.1f}  z={a["z"]:>6.1f}')

    print('\nMLP probe per-subject median (full / moving) mm:')
    for subj, st in sorted(per_subject_mlp.items(), key=lambda kv: kv[1]['median_moving']):
        print(f'  {subj:<14}  '
              f'full: n={st["n_full"]:>4d} median={st["median_full"]:>6.1f}    '
              f'moving: n={st["n_moving"]:>4d} median={st["median_moving"]:>6.1f}')

    # ---- Decision hint ----
    mlp_moving = results['phase2_epsilon_mlp']['moving']['median_3d_mm']
    print(f'\n{bar}\nDECISION HINT (MOVING subset, 1-s horizon)\n{bar}')
    print(f'epsilon MLP probe: {mlp_moving:.1f} mm')
    hms_metrics = os.path.join(_OUT, 'high_motion', 'metrics.json')
    if os.path.exists(hms_metrics):
        with open(hms_metrics) as f:
            hms = json.load(f)
        beta200 = hms['methods'].get('phase2_tactile_200ep', {}).get('moving', {}).get('median_3d_mm')
        p1      = hms['methods'].get('phase1_gru_com',       {}).get('moving', {}).get('median_3d_mm')
        v2      = hms['methods'].get('phase2_v2_gru_kp',     {}).get('moving', {}).get('median_3d_mm')
        if beta200 is not None: print(f'beta-200ep tactile : {beta200:.1f} mm')
        if p1      is not None: print(f'Phase 1 (CoM hist) : {p1:.1f} mm')
        if v2      is not None: print(f'v2 (camera kp hist): {v2:.1f} mm')
        if beta200 is not None and mlp_moving < beta200:
            print('-> epsilon BEATS beta-200ep -- CNN-free encoder is an improvement. WEAK POSITIVE.')
        if p1 is not None and mlp_moving < p1:
            print('-> epsilon BEATS Phase 1 -- tactile-alone matches CoM-history-alone. STRONG POSITIVE.')

    # ---- Plots ----
    hs = np.arange(1, HORIZON + 1) / 10.0
    # Per-horizon vs persistence on moving
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, sub in zip(axes, ['full', 'static', 'moving']):
        for name in ['persistence', 'phase2_epsilon_linear', 'phase2_epsilon_mlp']:
            ax.plot(hs, results[name][sub]['per_horizon_median'], marker='o', label=name)
        ax.set(xlabel='horizon (s)', title=f'{sub} (n={results["phase2_epsilon_mlp"][sub]["n"]})')
        ax.grid(alpha=0.3)
        if sub == 'full':
            ax.set_ylabel('median 3D Euclidean error (mm)')
            ax.legend(fontsize='small')
    plt.suptitle('Epsilon vs persistence — error vs horizon')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_horizon.png'), dpi=100)
    plt.close()

    # Comparison bars — fold in reference methods from high_motion if available
    reference = {n: results[n] for n in
                 ['persistence', 'phase2_epsilon_linear', 'phase2_epsilon_mlp']}
    extras = {}
    if os.path.exists(hms_metrics):
        with open(hms_metrics) as f:
            hms = json.load(f)
        for name in ['phase1_gru_com', 'phase2_v1_gru_kp', 'phase2_v2_gru_kp',
                     'phase2_tactile_50ep', 'phase2_tactile_200ep']:
            if name in hms.get('methods', {}):
                extras[name] = {sub: {'median_3d_mm': hms['methods'][name][sub]['median_3d_mm']}
                                 for sub in ['full', 'static', 'moving']}

    order = ['persistence', 'phase1_gru_com', 'phase2_v1_gru_kp',
             'phase2_tactile_50ep', 'phase2_tactile_200ep',
             'phase2_epsilon_linear', 'phase2_epsilon_mlp',
             'phase2_v2_gru_kp']
    plot_data = {**{n: {sub: reference[n][sub]['median_3d_mm']
                         for sub in ['full', 'static', 'moving']}
                    for n in reference},
                 **{n: {sub: extras[n][sub]['median_3d_mm']
                         for sub in ['full', 'static', 'moving']}
                    for n in extras}}
    order = [m for m in order if m in plot_data]
    x = np.arange(len(order))
    width = 0.25
    sub_colors = {'full': 'tab:gray', 'static': 'tab:blue', 'moving': 'tab:red'}
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, sub in enumerate(['full', 'static', 'moving']):
        vals = [plot_data[m][sub] for m in order]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=sub,
                      color=sub_colors[sub], alpha=0.9)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * width, v + 1, f'{v:.0f}', ha='center', fontsize=8)
        # Highlight epsilon bars
        for hl in ('phase2_epsilon_linear', 'phase2_epsilon_mlp'):
            if hl in order:
                bars[order.index(hl)].set_edgecolor('black')
                bars[order.index(hl)].set_linewidth(2.0)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha='right')
    ax.set_ylabel('median 3D Euclidean error (mm) at 1-s horizon')
    ax.set_title('Phase 2 epsilon vs prior methods  '
                 f'(motion threshold {threshold:.2f} mm/frame, moving = {moving_mask.sum()}/{n_test})')
    ax.legend(title='subset', loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bars.png'), dpi=100)
    plt.close()

    print(f'\nsaved metrics + plots to {output_dir}')


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 2 epsilon — CNN-free tactile forecaster.')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['mae', 'dynamics', 'probe', 'all'],
                        help='which stage(s) to run (default: all)')
    parser.add_argument('--epochs-mae',      type=int, default=MAE_EPOCHS_DEFAULT)
    parser.add_argument('--epochs-dynamics', type=int, default=DYN_EPOCHS_DEFAULT)
    parser.add_argument('--epochs-probe',    type=int, default=PROBE_EPOCHS_DEFAULT)
    parser.add_argument('--output-dir',      type=str, default=_EPSILON)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'stage      : {args.stage}')
    print(f'output_dir : {args.output_dir}')
    print(f'device     : {device}')

    if args.stage in ('mae', 'all'):
        run_stage_mae(args, args.output_dir, device)
    if args.stage in ('dynamics', 'all'):
        run_stage_dynamics(args, args.output_dir, device)
    if args.stage in ('probe', 'all'):
        run_stage_probe(args, args.output_dir, device)


if __name__ == '__main__':
    main()

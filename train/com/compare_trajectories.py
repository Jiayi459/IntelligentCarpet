"""compare_trajectories.py — unified prediction-vs-truth visualization.

For 5 fixed random test samples (SEED=42), overlay on a 5x3 grid (samples x {x,y,z}):
    - 100-frame CoM history                       (gray, dotted line at t=0)
    - 10-frame future truth                       (black, thick)
    - persistence                                 (gray dashed)
    - Phase 1 GRU (CoM history -> abs CoM)        if  output/phase1/gru_model.pt
    - Phase 2 v1 GRU (kp history -> abs CoM)      if  output/phase2_keypoints/gru_model.pt
    - Phase 2 v2 GRU (kp history -> delta CoM)    if  output/phase2_keypoints_v2/gru_model_best.pt
    - Phase 2 tactile (tactile -> delta CoM)      if  output/phase2_tactile/tactile_model_best.pt
                                                  AND output/tactile_all.npy cache present

Re-runnable any time without retraining. Skips methods whose checkpoints are missing
with a warning. Outputs:
    output/compare_trajectories/example_trajectories.png
    output/compare_trajectories/metadata.json     (which sample indices, subjects, frames)

Run:
    python train/com/compare_trajectories.py

Model class definitions are inlined here so that import-side-effects from the training
scripts (which do not have `if __name__ == '__main__':` guards) don't trigger full
re-training runs when this script imports them.
"""

import os
import json
import pickle
import re

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE    = os.path.dirname(os.path.abspath(__file__))
_TRAIN   = os.path.dirname(_HERE)
_OUT     = os.path.join(_HERE, 'output')
_COMPARE = os.path.join(_OUT, 'compare_trajectories')
os.makedirs(_COMPARE, exist_ok=True)


# ---------------------------------------------------------------------------
# Constants — must match all training scripts
# ---------------------------------------------------------------------------

HISTORY            = 100
HORIZON            = 10
SEED               = 42
KP_DIM             = 21 * 3
N_SAMPLES_TO_PLOT  = 5

# Set the global numpy seed once. We deliberately do NOT reseed later, so the
# order of np.random calls below matches what each training script did up to
# the point where it computed standardization stats.
np.random.seed(SEED)
torch.manual_seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')


# ---------------------------------------------------------------------------
# Inlined model classes (copies of the ones in training scripts)
# ---------------------------------------------------------------------------

class P1Forecaster(nn.Module):
    """Phase 1: input (B, 100, 3) CoM history -> (B, 10, 3) absolute CoM future."""
    def __init__(self, hidden=64, layers=1, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(3, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, horizon * 3)

    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


class P2v1Forecaster(nn.Module):
    """Phase 2 v1: input (B, 100, 63) kp history -> (B, 10, 3) absolute CoM future."""
    def __init__(self, hidden=64, layers=1, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.gru  = nn.GRU(KP_DIM, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, horizon * 3)

    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1]).view(-1, self.horizon, 3)


class P2v2Forecaster(nn.Module):
    """Phase 2 v2: input (B, 100, 63) kp history -> (B, 10, 3) delta CoM future."""
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
    """Per-frame 2D CNN: (B, 1, 96, 96) -> (B, 128)."""
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
    """Phase 2 tactile: input (B, 100, 96, 96) -> (B, 10, 3) delta CoM future."""
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


# ---------------------------------------------------------------------------
# Load data + rebuild sample indices (matches all training scripts)
# ---------------------------------------------------------------------------

with open(os.path.join(_OUT, 'com_results.p'), 'rb') as f:
    com_results = pickle.load(f)
com_gt   = com_results['com_gt']
kp_gt_mm = com_results['kp_gt_mm']
T = len(com_gt)
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


def build_indices():
    centers, splits, subjects = [], [], []
    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        valid_t = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t))
        for i, t in enumerate(valid_t):
            centers.append(t)
            splits.append('train' if i < n_train_s else 'test')
            subjects.append(subjects_per_sess[s])
    return (np.asarray(centers), np.asarray(splits), np.asarray(subjects))


centers, splits, subjects = build_indices()
train_mask = splits == 'train'
test_mask  = splits == 'test'
train_centers = centers[train_mask]
test_centers  = centers[test_mask]
test_subjects = subjects[test_mask]
assert len(centers) == 17218, f'expected 17218 total samples, got {len(centers)}'
print(f'samples: total={len(centers)}, train={train_mask.sum()}, test={test_mask.sum()}')


# ---------------------------------------------------------------------------
# Pick three subset selections: random / moving / static (5 each, SEED=42).
# We build the union of unique indices and run inference once -- the same
# predictions array gets indexed three different ways for the three plots.
# Motion criterion matches high_motion_subset.py exactly: peak xy speed during
# the 10-frame future window, top 30% = moving.
# ---------------------------------------------------------------------------

# Motion criterion for every test sample
full_future_xy = np.stack([com_gt[t + 1 : t + 1 + HORIZON, :2] for t in test_centers], axis=0)  # (n_test, 10, 2)
step_diffs     = np.diff(full_future_xy, axis=1)                                                  # (n_test, 9, 2)
step_speeds    = np.linalg.norm(step_diffs, axis=2)                                               # (n_test, 9)
v_future       = step_speeds.max(axis=1)                                                          # (n_test,)
MOVING_FRAC    = 0.30
motion_thresh  = float(np.quantile(v_future, 1.0 - MOVING_FRAC))
moving_in_test = v_future > motion_thresh
print(f'motion threshold (top {MOVING_FRAC:.0%}): {motion_thresh:.2f} mm/frame  '
      f'-- moving={moving_in_test.sum()}, static={(~moving_in_test).sum()}')

# Three sample selections (SEED=42, separate RNG so we don't touch the global one)
sel_rng = np.random.default_rng(SEED)
def _pick(pool, k):
    return pool[sel_rng.choice(len(pool), size=k, replace=False)] if len(pool) >= k else pool

random_idx_in_test = _pick(np.arange(len(test_centers)), N_SAMPLES_TO_PLOT)
moving_idx_in_test = _pick(np.where(moving_in_test)[0], N_SAMPLES_TO_PLOT)
static_idx_in_test = _pick(np.where(~moving_in_test)[0], N_SAMPLES_TO_PLOT)

# Union (dedup so inference work doesn't repeat on overlapping picks)
all_idx_in_test = np.unique(np.concatenate([random_idx_in_test,
                                             moving_idx_in_test,
                                             static_idx_in_test]))
print(f'sample selections (5 each, SEED={SEED}):')
print(f'  random  in test idx: {random_idx_in_test.tolist()}')
print(f'  moving  in test idx: {moving_idx_in_test.tolist()}')
print(f'  static  in test idx: {static_idx_in_test.tolist()}')
print(f'  union (for inference): {len(all_idx_in_test)} unique samples')

# Build mapping from "test-set index" -> "row in our inference array"
test2row = {int(t): r for r, t in enumerate(all_idx_in_test)}

# Frame centers + subjects for the union, in inference-array order
sel_frame_centers = test_centers[all_idx_in_test]
sel_subjects      = test_subjects[all_idx_in_test]

# Inputs each method might need (for the union)
hist_com    = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in sel_frame_centers], axis=0)        # (n_union, 100, 3)
hist_kp     = np.stack([kp_gt_mm[t - HISTORY + 1 : t + 1].reshape(HISTORY, KP_DIM)
                        for t in sel_frame_centers], axis=0)                                          # (n_union, 100, 63)
future_true = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in sel_frame_centers], axis=0)         # (n_union, 10, 3)
ref_now     = com_gt[sel_frame_centers]                                                               # (n_union, 3)


# ---------------------------------------------------------------------------
# Helpers: rebuild train-pool standardization stats per method
# ---------------------------------------------------------------------------

def stats_phase1():
    """Phase 1: single (mean, std) over com_gt train pool (X + Y combined)."""
    hist = np.stack([com_gt[t - HISTORY + 1 : t + 1] for t in train_centers], axis=0).reshape(-1, 3)
    fut  = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_centers], axis=0).reshape(-1, 3)
    pool = np.concatenate([hist, fut], axis=0)
    m = pool.mean(axis=0); s = pool.std(axis=0); s = np.where(s < 1e-6, 1.0, s)
    return m, s


def stats_kp_xy(delta_target):
    """Phase 2 v1 / v2: separate stats for X (kp pool) and Y (CoM pool, abs or delta)."""
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
    """Phase 2 tactile: tactile mean/std via 1000-sample subset (matches training).

    The training script uses the GLOBAL np.random state after np.random.seed(SEED)
    at module top and one un-permuted call (build_samples doesn't touch np.random).
    We mirror that exactly here by reseeding right before sampling.
    """
    np.random.seed(SEED)  # match training-script order
    n_tr = train_mask.sum()
    sub = train_centers[np.random.permutation(n_tr)[:1000]]
    chunk = np.concatenate([tactile_all[t - HISTORY + 1 : t + 1] for t in sub], axis=0)
    tac_mean = float(chunk.mean())
    tac_std  = float(chunk.std())
    # Y delta stats (always delta target for tactile)
    Y_abs = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_centers], axis=0)
    ref   = com_gt[train_centers]
    Y_d   = (Y_abs - ref[:, None, :]).reshape(-1, 3)
    mY = Y_d.mean(axis=0); sY = Y_d.std(axis=0); sY = np.where(sY < 1e-6, 1.0, sY)
    return tac_mean, tac_std, mY, sY


# ---------------------------------------------------------------------------
# Run each method -> store predictions[name] = (5, 10, 3) absolute CoM
# ---------------------------------------------------------------------------

predictions = {}

# 1. Persistence
n_union = ref_now.shape[0]
predictions['persistence'] = np.broadcast_to(ref_now[:, None, :],
                                             (n_union, HORIZON, 3)).copy()
print('persistence:        ready')

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
    print(f'phase1_gru_com:     ready  ({p1_pt})')
else:
    print(f'phase1_gru_com:     SKIP   (no checkpoint at {p1_pt})')

# 3. Phase 2 v1 GRU
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
    print(f'phase2_v1_gru_kp:   ready  ({p2v1_pt})')
else:
    print(f'phase2_v1_gru_kp:   SKIP   (no checkpoint at {p2v1_pt})')

# 4. Phase 2 v2 GRU
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
    print(f'phase2_v2_gru_kp:   ready  ({p2v2_pt})')
else:
    print(f'phase2_v2_gru_kp:   SKIP   (no checkpoint at {p2v2_pt})')

# 5. Phase 2 tactile variants — try both the 50-epoch and 200-epoch runs.
# Both use the same standardization stats (same SEED, same train pool, same RNG
# replay), so we only build them once outside the loop.
tac_cache = os.path.join(_OUT, 'tactile_all.npy')
TACTILE_VARIANTS = [
    ('phase2_tactile_50ep',  os.path.join(_OUT, 'phase2_tactile')),
    ('phase2_tactile_200ep', os.path.join(_OUT, 'phase2_tactile_200ep')),
]

if not os.path.exists(tac_cache):
    print(f'phase2_tactile_*:   SKIP all  (no tactile cache at {tac_cache} -- 1.2 GB, build with phase2_tactile.py)')
else:
    tactile_all = np.load(tac_cache, mmap_mode='r')
    tac_mean, tac_std, mY, sY = stats_tactile(tactile_all)
    for name, ddir in TACTILE_VARIANTS:
        pt_best = os.path.join(ddir, 'tactile_model_best.pt')
        pt      = pt_best if os.path.exists(pt_best) else os.path.join(ddir, 'tactile_model.pt')
        if not os.path.exists(pt):
            print(f'{name:<22}: SKIP   (no checkpoint at {pt})')
            continue
        m = TactileForecaster().to(device)
        m.load_state_dict(torch.load(pt, map_location=device, weights_only=False))
        m.eval()
        windows = np.stack(
            [(tactile_all[t - HISTORY + 1 : t + 1] - tac_mean) / tac_std
             for t in sel_frame_centers],
            axis=0
        ).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(windows).to(device)
            y = m(x).cpu().numpy()
        delta = y * sY + mY
        predictions[name] = delta + ref_now[:, None, :]
        print(f'{name:<22}: ready  ({pt})')


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

method_styles = {
    'persistence':           {'color': 'tab:gray',   'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9},
    'phase1_gru_com':        {'color': 'tab:blue',   'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.85},
    'phase2_v1_gru_kp':      {'color': 'tab:orange', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.85},
    'phase2_v2_gru_kp':      {'color': 'tab:red',    'linestyle': '-',  'linewidth': 1.8, 'alpha': 0.95},
    'phase2_tactile_50ep':   {'color': 'tab:green',  'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.75},
    'phase2_tactile_200ep':  {'color': 'tab:olive',  'linestyle': '-',  'linewidth': 1.6, 'alpha': 0.95},
}

t_h = np.arange(-HISTORY + 1, 1)
t_f = np.arange(1, HORIZON + 1)


def render_plot(subset_idx_in_test, subset_label, filename, subtitle=''):
    """Render a 5x3 grid of trajectory plots for the given subset selection.

    subset_idx_in_test : indices INTO test_centers (i.e. positions in the full
                          5,255-sample test set). We translate them to rows in
                          our inference array via test2row.
    """
    n_samples = len(subset_idx_in_test)
    inference_rows = [test2row[int(t)] for t in subset_idx_in_test]
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3.6 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]                                    # keep 2-D indexing safe
    for r, row in enumerate(inference_rows):
        for c, ax_name in enumerate('xyz'):
            ax = axes[r, c]
            first = (r == 0 and c == 0)
            ax.plot(t_h, hist_com[row, :, c], color='lightgray', alpha=0.9,
                    label='history' if first else None)
            ax.plot(t_f, future_true[row, :, c], 'k-', linewidth=2.2,
                    label='truth' if first else None)
            for name, pred in predictions.items():
                style = method_styles.get(name, {})
                ax.plot(t_f, pred[row, :, c],
                        label=(name if first else None), **style)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            if r == n_samples - 1:
                ax.set_xlabel('frame (relative to t)')
            if c == 0:
                ax.set_ylabel(f'{ax_name} (mm)')
            i_test = int(subset_idx_in_test[r])
            ax.set_title(
                f'[{subset_label}] test idx {i_test} | subj {test_subjects[i_test]} | '
                f't={int(test_centers[i_test])} | {ax_name}',
                fontsize=9
            )
            if first:
                ax.legend(fontsize='x-small', loc='best')
    if subtitle:
        fig.suptitle(subtitle, fontsize=11, y=1.0)
    plt.tight_layout()
    out_png = os.path.join(_COMPARE, filename)
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_png}')


print('\nRendering trajectory comparison plots...')
render_plot(random_idx_in_test, 'random', 'example_trajectories.png',
            subtitle='5 random test samples (any motion regime)')
render_plot(moving_idx_in_test, 'moving', 'example_trajectories_moving.png',
            subtitle=f'5 random samples from MOVING subset (future-window peak xy speed > {motion_thresh:.1f} mm/frame)')
render_plot(static_idx_in_test, 'static', 'example_trajectories_static.png',
            subtitle=f'5 random samples from STATIC subset (future-window peak xy speed <= {motion_thresh:.1f} mm/frame)')


# ---------------------------------------------------------------------------
# Metadata sidecar so the user can correlate each figure back to its indices
# ---------------------------------------------------------------------------

def _meta_for(idxs):
    return {
        'sample_indices_in_test':  [int(i) for i in idxs],
        'sample_frame_centers':    [int(test_centers[int(i)]) for i in idxs],
        'sample_subjects':         [str(test_subjects[int(i)]) for i in idxs],
        'peak_xy_speed_mm_frame':  [float(v_future[int(i)]) for i in idxs],
    }

meta = {
    'seed':                  SEED,
    'n_samples_per_subset':  N_SAMPLES_TO_PLOT,
    'methods_plotted':       list(predictions.keys()),
    'moving_threshold_mm_per_frame': motion_thresh,
    'subsets': {
        'random': _meta_for(random_idx_in_test),
        'moving': _meta_for(moving_idx_in_test),
        'static': _meta_for(static_idx_in_test),
    },
}
out_meta = os.path.join(_COMPARE, 'metadata.json')
with open(out_meta, 'w') as f:
    json.dump(meta, f, indent=2)
print(f'\nSaved metadata: {out_meta}')

print(f'\nDone. Methods on plots: {len(predictions)}.')

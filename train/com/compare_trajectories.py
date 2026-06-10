"""compare_trajectories.py — unified prediction-vs-truth visualization.

For each of TWO splits (train, test) and THREE subsets (random / moving / static),
plot 5 starting points × {x, y, z}. Each "sample" is a chain of FIVE consecutive
1-second forecasts (5 × 10 frames = 50 frames = 5 s of predicted future).

For each method, each 1-s prediction is an INDEPENDENT inference call using the
ground-truth history available at that moment (i.e., inference at t₀ + 10 sees
true CoM/keypoints/tactile up to t₀ + 10, not the model's own t₀-prediction).
This mirrors how the system would actually be deployed.

Six PNGs are produced under output/compare_trajectories/:

    example_trajectories_test.png         random samples from TEST set
    example_trajectories_moving_test.png  MOVING samples from TEST set
    example_trajectories_static_test.png  STATIC samples from TEST set
    example_trajectories_train.png        random samples from TRAIN set
    example_trajectories_moving_train.png MOVING samples from TRAIN set
    example_trajectories_static_train.png STATIC samples from TRAIN set

Methods plotted (9 — each skipped silently if its checkpoint is missing):
    persistence
    phase1_gru_com           (CoM history -> abs CoM)
    phase2_v1_gru_kp         (kp history -> abs CoM)
    phase2_v2_gru_kp         (kp history -> delta CoM)
    phase2_tactile_50ep      (tactile -> delta CoM)
    phase2_tactile_200ep     (tactile -> delta CoM)
    phase2_gamma             (tactile + CoM history -> delta CoM)
    phase2_epsilon_linear    (SSL tactile -> linear probe -> delta CoM)
    phase2_epsilon_mlp       (SSL tactile -> MLP probe -> delta CoM)

Re-runnable any time without retraining. Outputs metadata.json with the picked
indices, subjects, frames, and peak speeds for traceability.

Model class definitions are inlined (or imported, for ε whose classes live in
train/tactile_direct/) so this script has no training side effects.

Run:
    python train/com/compare_trajectories.py
"""

import os
import sys
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

# ε's source lives at train/tactile_direct/. Make it importable.
sys.path.insert(0, os.path.join(_TRAIN, 'tactile_direct'))


# ---------------------------------------------------------------------------
# Constants — must match all training scripts
# ---------------------------------------------------------------------------

HISTORY            = 100
HORIZON            = 10        # single inference horizon (1 s)
SEED               = 42
KP_DIM             = 21 * 3

SECONDS_PER_SAMPLE = 5         # NEW: each plotted sample shows 5 consecutive 1-s forecasts
TOTAL_HORIZON      = HORIZON * SECONDS_PER_SAMPLE       # 50 frames = 5 s

N_SAMPLES_TO_PLOT  = 5
MOVING_FRAC        = 0.30      # motion threshold = top 30 % by future-window peak xy speed

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


class TactileEncoderCNN(nn.Module):
    """Per-frame 2D CNN (used by β and γ). Distinct from ε's ViT encoder."""
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
    """Phase 2 β: input (B, 100, 96, 96) -> (B, 10, 3) delta CoM future."""
    def __init__(self, feature_dim=128, hidden=128, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.encoder = TactileEncoderCNN(feature_dim)
        self.gru     = nn.GRU(feature_dim, hidden, batch_first=True)
        self.proj    = nn.Linear(hidden, horizon * 3)
    def forward(self, x):
        B, Tlen, H, W = x.shape
        flat = x.reshape(B * Tlen, 1, H, W)
        feat = self.encoder(flat).reshape(B, Tlen, -1)
        _, h = self.gru(feat)
        return self.proj(h[-1]).view(B, self.horizon, 3)


class GammaForecaster(nn.Module):
    """Phase 2 γ: tactile + CoM history -> delta CoM. Mirrors train_phase2_gamma.py."""
    def __init__(self, tactile_feature_dim=128, tactile_hidden=128, com_hidden=64,
                 mlp_hidden=128, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.tactile_encoder = TactileEncoderCNN(tactile_feature_dim)
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


# ε: import from train/tactile_direct/model_epsilon.py (imports are
# side-effect-free per the module's docstring).
try:
    from model_epsilon import DynamicsModel as _EpsDynamics, \
                              LinearProbe   as _EpsLinearProbe, \
                              MLPProbe      as _EpsMLPProbe
    _EPS_OK = True
except Exception as e:
    print(f'WARN: ε model classes unavailable ({e}); ε methods will be skipped.')
    _EPS_OK = False


# ---------------------------------------------------------------------------
# Load data + rebuild sample indices (matches all training scripts at the 1-s
# horizon — so train/test classification matches what models actually saw).
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


def build_indices_1s():
    """Build centers + splits + subjects + sessions using the 1-second-horizon
    rule (matches every training script). The train/test classification
    produced here is what the models actually saw at training time."""
    centers, splits, subjects, sessions = [], [], [], []
    for s in range(n_sessions):
        a, b = log[s], log[s + 1]
        valid_t = [
            t for t in range(a + HISTORY - 1, b - HORIZON)
            if not gt_outliers[t - HISTORY + 1 : t + HORIZON + 1].any()
        ]
        n_train_s = int(0.7 * len(valid_t))
        for i, t in enumerate(valid_t):
            centers.append(t)
            sessions.append(s)
            subjects.append(subjects_per_sess[s])
            splits.append('train' if i < n_train_s else 'test')
    return (np.asarray(centers), np.asarray(splits),
            np.asarray(subjects), np.asarray(sessions))


centers, splits, subjects, sess_per_center = build_indices_1s()
assert len(centers) == 17218, f'expected 17218 total samples, got {len(centers)}'
train_mask = splits == 'train'
test_mask  = splits == 'test'
train_centers = centers[train_mask]
test_centers  = centers[test_mask]
train_sessions = sess_per_center[train_mask]
test_sessions  = sess_per_center[test_mask]
train_subjects = subjects[train_mask]
test_subjects  = subjects[test_mask]
print(f'samples (1-s eligible): total={len(centers)}, train={train_mask.sum()}, test={test_mask.sum()}')


def supports_5s_chain(t, sess):
    """True iff t supports a 5-s prediction chain — i.e., t + 50 stays inside
    the session AND every frame in [t-99, t+50] is outlier-free."""
    if t + TOTAL_HORIZON >= log[sess + 1]:
        return False
    return not gt_outliers[t - HISTORY + 1 : t + TOTAL_HORIZON + 1].any()


def filter_5s(centers_arr, sessions_arr):
    """Boolean mask: which entries support the 5-s chain."""
    mask = np.zeros(len(centers_arr), dtype=bool)
    for i in range(len(centers_arr)):
        mask[i] = supports_5s_chain(int(centers_arr[i]), int(sessions_arr[i]))
    return mask


train_5s_mask = filter_5s(train_centers, train_sessions)
test_5s_mask  = filter_5s(test_centers,  test_sessions)
print(f'samples (5-s eligible): train={train_5s_mask.sum()}, test={test_5s_mask.sum()}  '
      f'(filtered out {len(train_centers) - train_5s_mask.sum()} train + '
      f'{len(test_centers) - test_5s_mask.sum()} test near session boundaries)')


# ---------------------------------------------------------------------------
# Motion classification on the 5-second future window
# ---------------------------------------------------------------------------

def peak_xy_speed_5s(centers_arr):
    """For each starting t, compute the peak xy speed during the 5-s future window."""
    future_xy = np.stack([com_gt[t + 1 : t + 1 + TOTAL_HORIZON, :2] for t in centers_arr], axis=0)
    step_diffs = np.diff(future_xy, axis=1)                          # (N, TOTAL_HORIZON-1, 2)
    step_speeds = np.linalg.norm(step_diffs, axis=2)                 # (N, TOTAL_HORIZON-1)
    return step_speeds.max(axis=1)                                   # (N,)


# Compute v_future on the 5-s-eligible test pool to set the threshold
test_centers_5s   = test_centers[test_5s_mask]
test_subjects_5s  = test_subjects[test_5s_mask]
train_centers_5s  = train_centers[train_5s_mask]
train_subjects_5s = train_subjects[train_5s_mask]

v_future_test  = peak_xy_speed_5s(test_centers_5s)
v_future_train = peak_xy_speed_5s(train_centers_5s)

# Threshold from the TEST pool (matches high_motion_subset.py's choice of
# pool, so "moving" means the same thing in test and train comparisons).
motion_thresh = float(np.quantile(v_future_test, 1.0 - MOVING_FRAC))
print(f'motion threshold (top {MOVING_FRAC:.0%} of TEST pool, 5-s window): {motion_thresh:.2f} mm/frame')

test_moving  = v_future_test  > motion_thresh
train_moving = v_future_train > motion_thresh
print(f'  test  partition: moving={test_moving.sum()}, static={(~test_moving).sum()}')
print(f'  train partition: moving={train_moving.sum()}, static={(~train_moving).sum()}')


# ---------------------------------------------------------------------------
# Sample selection — 5 random + 5 moving + 5 static per split.
# All draws use a *separate* RNG so they don't disturb the global np.random
# state (which is later consumed by stats_tactile's deterministic replay).
# ---------------------------------------------------------------------------

sel_rng = np.random.default_rng(SEED)
def _pick(pool, k):
    return pool[sel_rng.choice(len(pool), size=k, replace=False)] if len(pool) >= k else pool

selections = {}                # selections[(split, subset)] = array of indices into <split>_centers_5s
for split, mov_mask, n in [('test', test_moving, len(test_centers_5s)),
                            ('train', train_moving, len(train_centers_5s))]:
    selections[(split, 'random')] = _pick(np.arange(n),                  N_SAMPLES_TO_PLOT)
    selections[(split, 'moving')] = _pick(np.where(mov_mask)[0],         N_SAMPLES_TO_PLOT)
    selections[(split, 'static')] = _pick(np.where(~mov_mask)[0],        N_SAMPLES_TO_PLOT)

for (split, sub), idx in selections.items():
    print(f'  {split:5s} {sub:6s} idx: {idx.tolist()}')


# Build the union of unique starting-center positions we need to run inference on.
# We translate "row in <split>_centers_5s" -> "frame center t" and dedup.
union_centers = []
union_owners  = []     # for traceability: list of (split, subset) tuples that requested each
for (split, sub), idx in selections.items():
    arr_centers   = train_centers_5s if split == 'train' else test_centers_5s
    arr_subjects  = train_subjects_5s if split == 'train' else test_subjects_5s
    arr_v_future  = v_future_train if split == 'train' else v_future_test
    for i in idx:
        t0 = int(arr_centers[int(i)])
        union_centers.append(t0)
        union_owners.append((split, sub, int(i)))

# Dedup while preserving the first occurrence; we still need to know which
# (split, subset, position) maps to which row in the inference array.
seen = {}
ordered_starts = []
for t0 in union_centers:
    if t0 not in seen:
        seen[t0] = len(ordered_starts)
        ordered_starts.append(t0)
ordered_starts = np.asarray(ordered_starts)
start2row = {int(t0): r for r, t0 in enumerate(ordered_starts)}
n_starts = len(ordered_starts)
print(f'\nunique starting centers needing inference: {n_starts}')

# For each starting center, build the FIVE inference centers (t0, t0+10, ..., t0+40).
inf_centers_flat = np.asarray([t0 + k * HORIZON
                               for t0 in ordered_starts
                               for k in range(SECONDS_PER_SAMPLE)])
n_inf = inf_centers_flat.shape[0]
assert n_inf == n_starts * SECONDS_PER_SAMPLE
print(f'flat inference points (n_starts × {SECONDS_PER_SAMPLE}): {n_inf}')


# Pre-extract the inputs each method might need, at the flat inference centers.
# These are aligned with inf_centers_flat (row k corresponds to inf_centers_flat[k]).
hist_com_flat = np.stack([com_gt[t - HISTORY + 1 : t + 1]     for t in inf_centers_flat], axis=0)        # (n_inf, 100, 3)
hist_kp_flat  = np.stack([kp_gt_mm[t - HISTORY + 1 : t + 1].reshape(HISTORY, KP_DIM)
                          for t in inf_centers_flat], axis=0)                                              # (n_inf, 100, 63)
ref_now_flat  = com_gt[inf_centers_flat]                                                                   # (n_inf, 3)

# The 50-frame ground-truth future per starting center (used by the plot,
# not by inference). Stitched from the truth at t0+1..t0+50.
truth_50_per_start = np.stack(
    [com_gt[t0 + 1 : t0 + 1 + TOTAL_HORIZON] for t0 in ordered_starts], axis=0)                            # (n_starts, 50, 3)
hist_per_start     = np.stack(
    [com_gt[t0 - HISTORY + 1 : t0 + 1]      for t0 in ordered_starts], axis=0)                             # (n_starts, 100, 3)


# ---------------------------------------------------------------------------
# Stats helpers (rebuild train-pool standardization for each method)
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
    """Tactile mean/std via 1000-sample subset (matches training)."""
    np.random.seed(SEED)                                              # reseed to match training-script order
    n_tr = train_mask.sum()
    sub = train_centers[np.random.permutation(n_tr)[:1000]]
    chunk = np.concatenate([tactile_all[t - HISTORY + 1 : t + 1] for t in sub], axis=0)
    tac_mean = float(chunk.mean())
    tac_std  = float(chunk.std())
    # Y delta stats (always delta target for tactile family)
    Y_abs = np.stack([com_gt[t + 1 : t + 1 + HORIZON] for t in train_centers], axis=0)
    ref   = com_gt[train_centers]
    Y_d   = (Y_abs - ref[:, None, :]).reshape(-1, 3)
    mY = Y_d.mean(axis=0); sY = Y_d.std(axis=0); sY = np.where(sY < 1e-6, 1.0, sY)
    return tac_mean, tac_std, mY, sY


# ---------------------------------------------------------------------------
# Inference — each method outputs (n_inf, 10, 3) absolute CoM at flat inference
# centers. We later reshape to (n_starts, 5, 10, 3) and concatenate along the
# 5-second axis to (n_starts, 50, 3).
# ---------------------------------------------------------------------------

predictions_flat = {}        # name -> (n_inf, 10, 3) absolute CoM

# 1. Persistence (no checkpoint needed). Predicts ref_now broadcast over horizon.
predictions_flat['persistence'] = np.broadcast_to(
    ref_now_flat[:, None, :], (n_inf, HORIZON, 3)).copy()
print('persistence:               ready')

# 2. Phase 1 GRU
p1_pt = os.path.join(_OUT, 'phase1', 'gru_model.pt')
if os.path.exists(p1_pt):
    mean_p1, std_p1 = stats_phase1()
    m1 = P1Forecaster().to(device)
    m1.load_state_dict(torch.load(p1_pt, map_location=device, weights_only=False))
    m1.eval()
    with torch.no_grad():
        x = torch.tensor((hist_com_flat - mean_p1) / std_p1, dtype=torch.float32).to(device)
        y = m1(x).cpu().numpy()
    predictions_flat['phase1_gru_com'] = y * std_p1 + mean_p1
    print(f'phase1_gru_com:            ready  ({p1_pt})')
else:
    print(f'phase1_gru_com:            SKIP   (no checkpoint at {p1_pt})')

# 3. Phase 2 v1 GRU
p2v1_pt = os.path.join(_OUT, 'phase2_keypoints', 'gru_model.pt')
if os.path.exists(p2v1_pt):
    mX, sX, mY, sY = stats_kp_xy(delta_target=False)
    m2 = P2v1Forecaster().to(device)
    m2.load_state_dict(torch.load(p2v1_pt, map_location=device, weights_only=False))
    m2.eval()
    with torch.no_grad():
        x = torch.tensor((hist_kp_flat - mX) / sX, dtype=torch.float32).to(device)
        y = m2(x).cpu().numpy()
    predictions_flat['phase2_v1_gru_kp'] = y * sY + mY
    print(f'phase2_v1_gru_kp:          ready  ({p2v1_pt})')
else:
    print(f'phase2_v1_gru_kp:          SKIP   (no checkpoint at {p2v1_pt})')

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
        x = torch.tensor((hist_kp_flat - mX) / sX, dtype=torch.float32).to(device)
        y = m3(x).cpu().numpy()
    delta = y * sY + mY
    predictions_flat['phase2_v2_gru_kp'] = delta + ref_now_flat[:, None, :]
    print(f'phase2_v2_gru_kp:          ready  ({p2v2_pt})')
else:
    print(f'phase2_v2_gru_kp:          SKIP   (no checkpoint at {p2v2_pt})')

# 5/6/7/8/9. Tactile / γ / ε — all need the tactile cache.
tac_cache = os.path.join(_OUT, 'tactile_all.npy')
TACTILE_BATCH = 32         # avoid loading 150 × 100 × 96 × 96 at once on CPU
if not os.path.exists(tac_cache):
    print(f'tactile-family methods:    SKIP all  (no cache at {tac_cache} -- 1.2 GB, build with phase2_tactile.py)')
else:
    tactile_all = np.load(tac_cache, mmap_mode='r')
    tac_mean, tac_std, mY, sY = stats_tactile(tactile_all)

    def _build_tactile_windows(inf_centers):
        """Return (N, 100, 96, 96) standardized tactile windows. Returns the
        raw numpy array; caller batches into the device."""
        return np.stack(
            [(tactile_all[t - HISTORY + 1 : t + 1] - tac_mean) / tac_std
             for t in inf_centers],
            axis=0
        ).astype(np.float32)

    # 5/6. Phase 2 tactile variants
    TACTILE_VARIANTS = [
        ('phase2_tactile_50ep',  os.path.join(_OUT, 'phase2_tactile')),
        ('phase2_tactile_200ep', os.path.join(_OUT, 'phase2_tactile_200ep')),
    ]
    for name, ddir in TACTILE_VARIANTS:
        pt_best = os.path.join(ddir, 'tactile_model_best.pt')
        pt      = pt_best if os.path.exists(pt_best) else os.path.join(ddir, 'tactile_model.pt')
        if not os.path.exists(pt):
            print(f'{name:<26}: SKIP   (no checkpoint at {pt})')
            continue
        m = TactileForecaster().to(device)
        m.load_state_dict(torch.load(pt, map_location=device, weights_only=False))
        m.eval()
        preds_norm = np.zeros((n_inf, HORIZON, 3), dtype=np.float32)
        with torch.no_grad():
            for i0 in range(0, n_inf, TACTILE_BATCH):
                i1 = min(i0 + TACTILE_BATCH, n_inf)
                w = _build_tactile_windows(inf_centers_flat[i0:i1])
                x = torch.from_numpy(w).to(device)
                preds_norm[i0:i1] = m(x).cpu().numpy()
        delta = preds_norm * sY + mY
        predictions_flat[name] = delta + ref_now_flat[:, None, :]
        print(f'{name:<26}: ready  ({pt})')

    # 7. Phase 2 γ — tactile + CoM history
    gamma_pt_best = os.path.join(_OUT, 'phase2_gamma', 'gamma_model_best.pt')
    gamma_pt      = gamma_pt_best if os.path.exists(gamma_pt_best) else os.path.join(
        _OUT, 'phase2_gamma', 'gamma_model.pt')
    if os.path.exists(gamma_pt):
        mean_p1, std_p1 = stats_phase1()
        gm = GammaForecaster().to(device)
        gm.load_state_dict(torch.load(gamma_pt, map_location=device, weights_only=False))
        gm.eval()
        com_hist_norm_flat = (hist_com_flat - mean_p1) / std_p1
        preds_norm = np.zeros((n_inf, HORIZON, 3), dtype=np.float32)
        with torch.no_grad():
            for i0 in range(0, n_inf, TACTILE_BATCH):
                i1 = min(i0 + TACTILE_BATCH, n_inf)
                w = _build_tactile_windows(inf_centers_flat[i0:i1])
                tact = torch.from_numpy(w).to(device)
                com_h = torch.from_numpy(com_hist_norm_flat[i0:i1].astype(np.float32)).to(device)
                preds_norm[i0:i1] = gm(tact, com_h).cpu().numpy()
        delta = preds_norm * sY + mY
        predictions_flat['phase2_gamma'] = delta + ref_now_flat[:, None, :]
        print(f'phase2_gamma:              ready  ({gamma_pt})')
    else:
        print(f'phase2_gamma:              SKIP   (no checkpoint at {gamma_pt})')

    # 8/9. Phase 2 ε — linear + MLP probes on top of a shared frozen DynamicsModel
    if _EPS_OK:
        eps_dyn = os.path.join(_OUT, 'phase2_epsilon', 'dynamics_model.pt')
        eps_lin = os.path.join(_OUT, 'phase2_epsilon', 'probe_linear.pt')
        eps_mlp = os.path.join(_OUT, 'phase2_epsilon', 'probe_mlp.pt')
        if os.path.exists(eps_dyn) and (os.path.exists(eps_lin) or os.path.exists(eps_mlp)):
            dyn = _EpsDynamics().to(device)
            dyn.load_state_dict(torch.load(eps_dyn, map_location=device, weights_only=False)['dynamics'])
            dyn.eval()

            # Encode hidden states once; both probes share them.
            H_flat = np.zeros((n_inf, 128), dtype=np.float32)
            with torch.no_grad():
                for i0 in range(0, n_inf, TACTILE_BATCH):
                    i1 = min(i0 + TACTILE_BATCH, n_inf)
                    w = _build_tactile_windows(inf_centers_flat[i0:i1])
                    x = torch.from_numpy(w).to(device)
                    H_flat[i0:i1] = dyn.encode_history(x).cpu().numpy()
            H_flat_t = torch.from_numpy(H_flat).to(device)

            def _run_eps_probe(probe_cls, probe_ckpt, name):
                p = probe_cls().to(device)
                p.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=False)['probe'])
                p.eval()
                with torch.no_grad():
                    y_norm = p(H_flat_t).cpu().numpy()
                delta = y_norm * sY + mY
                predictions_flat[name] = delta + ref_now_flat[:, None, :]
                print(f'{name:<26}: ready  ({probe_ckpt})')

            if os.path.exists(eps_lin):
                _run_eps_probe(_EpsLinearProbe, eps_lin, 'phase2_epsilon_linear')
            else:
                print(f'phase2_epsilon_linear:     SKIP   (no checkpoint at {eps_lin})')
            if os.path.exists(eps_mlp):
                _run_eps_probe(_EpsMLPProbe, eps_mlp, 'phase2_epsilon_mlp')
            else:
                print(f'phase2_epsilon_mlp:        SKIP   (no checkpoint at {eps_mlp})')
        else:
            print(f'phase2_epsilon_*:          SKIP   ({eps_dyn} or probes missing)')


# ---------------------------------------------------------------------------
# Reshape (n_inf, 10, 3) -> (n_starts, 50, 3) by concatenating each starting
# center's 5 consecutive 1-s predictions into one 5-s trajectory.
# ---------------------------------------------------------------------------

predictions_50 = {}             # name -> (n_starts, 50, 3) absolute CoM
for name, pred in predictions_flat.items():
    # pred is laid out as [start0_sec0, start0_sec1, ..., start0_sec4, start1_sec0, ...]
    rs = pred.reshape(n_starts, SECONDS_PER_SAMPLE, HORIZON, 3)
    predictions_50[name] = rs.reshape(n_starts, TOTAL_HORIZON, 3)


# ---------------------------------------------------------------------------
# Plot — 6 PNGs, one per (split, subset)
# ---------------------------------------------------------------------------

method_styles = {
    'persistence':           {'color': 'tab:gray',    'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9},
    'phase1_gru_com':        {'color': 'tab:blue',    'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.85},
    'phase2_v1_gru_kp':      {'color': 'tab:orange',  'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.85},
    'phase2_v2_gru_kp':      {'color': 'tab:red',     'linestyle': '-',  'linewidth': 1.8, 'alpha': 0.95},
    'phase2_tactile_50ep':   {'color': 'tab:green',   'linestyle': '--', 'linewidth': 1.0, 'alpha': 0.65},
    'phase2_tactile_200ep':  {'color': 'tab:olive',   'linestyle': '-',  'linewidth': 1.4, 'alpha': 0.85},
    'phase2_gamma':          {'color': 'tab:purple',  'linestyle': '-',  'linewidth': 1.6, 'alpha': 0.95},
    'phase2_epsilon_linear': {'color': 'tab:cyan',    'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.85},
    'phase2_epsilon_mlp':    {'color': 'tab:brown',   'linestyle': '-',  'linewidth': 1.4, 'alpha': 0.85},
}

t_h = np.arange(-HISTORY + 1, 1)                                 # -99 .. 0
t_f = np.arange(1, TOTAL_HORIZON + 1)                            # 1 .. 50


def render_plot(split, subset, filename, subtitle=''):
    """Render a 5x3 grid for the given (split, subset) selection."""
    idx_in_pool = selections[(split, subset)]
    arr_centers = train_centers_5s if split == 'train' else test_centers_5s
    arr_subj    = train_subjects_5s if split == 'train' else test_subjects_5s
    arr_v       = v_future_train if split == 'train' else v_future_test

    starts_t = arr_centers[idx_in_pool]                          # (5,) starting frame centers
    rows     = [start2row[int(t)] for t in starts_t]             # rows into the n_starts arrays

    n_samples = len(starts_t)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3.6 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for r, row in enumerate(rows):
        for c, ax_name in enumerate('xyz'):
            ax = axes[r, c]
            first = (r == 0 and c == 0)

            # 100-frame history (gray, dotted line at t=0)
            ax.plot(t_h, hist_per_start[row, :, c], color='lightgray', alpha=0.9,
                    label='history' if first else None)

            # 50-frame truth (black, thick)
            ax.plot(t_f, truth_50_per_start[row, :, c], 'k-', linewidth=2.2,
                    label='truth' if first else None)

            # Each method's 50-frame prediction
            for name, pred in predictions_50.items():
                style = method_styles.get(name, {})
                ax.plot(t_f, pred[row, :, c],
                        label=(name if first else None), **style)

            # Subtle vertical lines at every 10-frame inference boundary
            for k in range(1, SECONDS_PER_SAMPLE):
                ax.axvline(k * HORIZON, color='lightgray', linestyle=':', alpha=0.4, linewidth=0.6)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

            if r == n_samples - 1:
                ax.set_xlabel('frame (relative to t₀)')
            if c == 0:
                ax.set_ylabel(f'{ax_name} (mm)')

            t0 = int(starts_t[r])
            subj_str = str(arr_subj[idx_in_pool[r]])
            v_str = arr_v[idx_in_pool[r]]
            ax.set_title(
                f'[{split}/{subset}] start t₀={t0} | subj {subj_str} | '
                f'peak xy speed (5-s) = {v_str:.1f} mm/frame | {ax_name}',
                fontsize=8.5
            )
            if first:
                ax.legend(fontsize='x-small', loc='best', ncol=2)

    if subtitle:
        fig.suptitle(subtitle, fontsize=11, y=1.0)
    plt.tight_layout()
    out_png = os.path.join(_COMPARE, filename)
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_png}')


print('\nRendering trajectory comparison plots...')

# TEST plots
render_plot('test', 'random', 'example_trajectories_test.png',
            subtitle=f'TEST: 5 random samples × 5-s consecutive forecasts (any motion regime)')
render_plot('test', 'moving', 'example_trajectories_moving_test.png',
            subtitle=f'TEST · MOVING: 5 samples × 5-s, future-window peak xy speed > {motion_thresh:.1f} mm/frame')
render_plot('test', 'static', 'example_trajectories_static_test.png',
            subtitle=f'TEST · STATIC: 5 samples × 5-s, future-window peak xy speed ≤ {motion_thresh:.1f} mm/frame')

# TRAIN plots
render_plot('train', 'random', 'example_trajectories_train.png',
            subtitle=f'TRAIN: 5 random samples × 5-s consecutive forecasts (any motion regime)')
render_plot('train', 'moving', 'example_trajectories_moving_train.png',
            subtitle=f'TRAIN · MOVING: 5 samples × 5-s, future-window peak xy speed > {motion_thresh:.1f} mm/frame')
render_plot('train', 'static', 'example_trajectories_static_train.png',
            subtitle=f'TRAIN · STATIC: 5 samples × 5-s, future-window peak xy speed ≤ {motion_thresh:.1f} mm/frame')


# ---------------------------------------------------------------------------
# Metadata sidecar so each figure can be correlated back to its indices
# ---------------------------------------------------------------------------

def _meta_for(split, subset):
    idx_in_pool = selections[(split, subset)]
    arr_centers = train_centers_5s if split == 'train' else test_centers_5s
    arr_subj    = train_subjects_5s if split == 'train' else test_subjects_5s
    arr_v       = v_future_train if split == 'train' else v_future_test
    return {
        'sample_position_in_pool':    [int(i) for i in idx_in_pool],
        'sample_start_frame_centers': [int(arr_centers[int(i)]) for i in idx_in_pool],
        'sample_subjects':            [str(arr_subj[int(i)]) for i in idx_in_pool],
        'peak_xy_speed_5s_mm_frame':  [float(arr_v[int(i)]) for i in idx_in_pool],
    }

meta = {
    'seed':                          SEED,
    'history_frames':                HISTORY,
    'horizon_frames_per_inference':  HORIZON,
    'seconds_per_sample':            SECONDS_PER_SAMPLE,
    'total_horizon_frames':          TOTAL_HORIZON,
    'n_samples_per_subset':          N_SAMPLES_TO_PLOT,
    'methods_plotted':               list(predictions_50.keys()),
    'moving_threshold_mm_per_frame': motion_thresh,
    'moving_threshold_source_pool':  'test',
    'subsets': {
        'test':  {sub: _meta_for('test',  sub) for sub in ('random', 'moving', 'static')},
        'train': {sub: _meta_for('train', sub) for sub in ('random', 'moving', 'static')},
    },
}
out_meta = os.path.join(_COMPARE, 'metadata.json')
with open(out_meta, 'w') as f:
    json.dump(meta, f, indent=2)
print(f'\nSaved metadata: {out_meta}')

print(f'\nDone. Methods on plots: {len(predictions_50)}.')

"""Post-hoc analysis of com_results.p.

Produces:
  1. Error percentiles (overall + per axis, on valid frames only).
  2. Per-subject error breakdown.
  3. Per-session error — identifies worst sessions.
  4. Outlier GT frame identification (out-of-carpet, below-floor).
  5. CoM trajectory plots for best + worst session.
  6. CoM autocorrelation on the longest session (predictability ceiling).

"Valid frames" = not in the first/last `window` frames of any session
(dataloader edge effect produces clamped predictions there) AND not a GT outlier.
"""
import os, pickle, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Script lives in train/com/. Test data lives in train/, outputs in train/com/output/.
_HERE     = os.path.dirname(os.path.abspath(__file__))   # .../train/com
_TRAIN    = os.path.dirname(_HERE)                       # .../train
_OUT      = os.path.join(_HERE, 'output')                # .../train/com/output
RESULTS   = os.path.join(_OUT, 'com_results.p')
TESTDIR   = os.path.join(_TRAIN, 'singlePerson_test')
PLOTS_DIR = os.path.join(_OUT, 'plots')
WINDOW    = 10   # dataloader context window — first/last 10 frames of each session are clamped

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- load ----
with open(RESULTS, 'rb') as f:
    r = pickle.load(f)
with open(os.path.join(TESTDIR, 'log.p'), 'rb') as f:
    log = pickle.load(f)
with open(os.path.join(TESTDIR, 'fileNames.p'), 'rb') as f:
    fileNames = pickle.load(f)

gt   = r['com_gt']
pred = r['com_pred']
err  = r['error']
ee   = r['euclidean_error']
T    = len(gt)
n_sessions = len(log) - 1

# Parse subjects
pat = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
subjects_per_sess = []
for n in fileNames:
    m = pat.match(n)
    subjects_per_sess.append(m.group(3) if m else 'unknown')

# Frame → session
def f2s(i):
    return int(np.searchsorted(log, i, side='right') - 1)

sessions = np.array([f2s(i) for i in range(T)])
subjects = np.array([subjects_per_sess[s] for s in sessions])

# Edge mask (skip first/last WINDOW frames of each session — they are clamped by the dataloader)
edge_mask = np.ones(T, dtype=bool)
for s in range(n_sessions):
    a, b = log[s], log[s+1]
    edge_mask[a:a+WINDOW]      = False
    edge_mask[max(a, b-WINDOW):b] = False

# GT outliers (physically impossible CoM positions)
outlier_x = (gt[:,0] < -100) | (gt[:,0] > 1800)
outlier_y = (gt[:,1] < -100) | (gt[:,1] > 1800)
outlier_z = gt[:,2] > 0   # z>0 = below floor in this coordinate system
gt_outliers = outlier_x | outlier_y | outlier_z

valid = edge_mask & ~gt_outliers
print(f'frames total              : {T}')
print(f'  excluded by edge mask   : {(~edge_mask).sum():5d}  ({100*(~edge_mask).sum()/T:.1f}%)')
print(f'  excluded as GT outlier  : {(gt_outliers & edge_mask).sum():5d}  ({100*(gt_outliers & edge_mask).sum()/T:.1f}%)')
print(f'  remaining valid         : {valid.sum():5d}  ({100*valid.sum()/T:.1f}%)')

# ===== 1. Percentile error stats =====
print('\n========== 1. Error percentiles on valid frames (mm) ==========')
print(f'  {"percentile":>10s}  {"3D Euclid":>10s}  {"|x|":>8s}  {"|y|":>8s}  {"|z|":>8s}')
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f'  {"p"+str(p):>10s}  {np.percentile(ee[valid], p):>10.1f}  '
          f'{np.percentile(np.abs(err[valid,0]), p):>8.1f}  '
          f'{np.percentile(np.abs(err[valid,1]), p):>8.1f}  '
          f'{np.percentile(np.abs(err[valid,2]), p):>8.1f}')
print(f'  {"mean":>10s}  {np.mean(ee[valid]):>10.1f}  '
      f'{np.mean(np.abs(err[valid,0])):>8.1f}  '
      f'{np.mean(np.abs(err[valid,1])):>8.1f}  '
      f'{np.mean(np.abs(err[valid,2])):>8.1f}')

# ===== 2. Per-subject =====
print('\n========== 2. Per-subject error (valid frames, mm) ==========')
print(f'  {"subject":<14s} {"n":>6s} {"mean":>8s} {"median":>8s} {"p95":>8s} {"max":>8s}')
unique_subjects = sorted(set(subjects_per_sess))
subject_stats = []
for subj in unique_subjects:
    m = (subjects == subj) & valid
    if m.sum() == 0:
        continue
    ee_s = ee[m]
    subject_stats.append((subj, m.sum(), np.mean(ee_s), np.median(ee_s),
                          np.percentile(ee_s, 95), ee_s.max()))
subject_stats.sort(key=lambda x: x[3])  # by median ascending
for subj, n, mn, med, p95, mx in subject_stats:
    print(f'  {subj:<14s} {n:>6d} {mn:>8.1f} {med:>8.1f} {p95:>8.1f} {mx:>8.1f}')

# ===== 3. Per-session: top 10 worst =====
print('\n========== 3. Top 10 worst sessions by median error ==========')
sess_stats = []
for s in range(n_sessions):
    a, b = log[s], log[s+1]
    m = valid[a:b]
    if m.sum() == 0:
        continue
    ee_s = ee[a:b][m]
    sess_stats.append({'sess': s, 'subj': subjects_per_sess[s],
                       'median': np.median(ee_s), 'mean': np.mean(ee_s),
                       'max': ee_s.max(), 'n_valid': m.sum(),
                       'frames': (a, b)})
sess_stats_sorted = sorted(sess_stats, key=lambda x: -x['median'])
print(f'  {"sess":>5s}  {"subject":<14s} {"median":>8s} {"mean":>8s} {"max":>8s} {"n":>6s}')
for s in sess_stats_sorted[:10]:
    print(f'  {s["sess"]:>5d}  {s["subj"]:<14s} {s["median"]:>8.1f} {s["mean"]:>8.1f} {s["max"]:>8.1f} {s["n_valid"]:>6d}')

# ===== 4. Outlier breakdown =====
print('\n========== 4. GT outliers (physically impossible CoM positions) ==========')
print(f'  out-of-carpet x  (< -100 or > 1800)  : {outlier_x.sum()}')
print(f'  out-of-carpet y                       : {outlier_y.sum()}')
print(f'  below-floor    z (> 0)               : {outlier_z.sum()}')
print(f'  union (any outlier)                   : {gt_outliers.sum()}  ({100*gt_outliers.sum()/T:.2f}%)')

# Distribution across sessions
outlier_by_sess = defaultdict(int)
for i in np.where(gt_outliers)[0]:
    outlier_by_sess[sessions[i]] += 1
top_outlier_sess = sorted(outlier_by_sess.items(), key=lambda kv: -kv[1])[:10]
print('\n  Top 10 sessions with most outlier frames:')
print(f'    {"sess":>5s}  {"subject":<14s} {"n_outliers":>10s}  {"sess_len":>8s}')
for s_idx, cnt in top_outlier_sess:
    a, b = log[s_idx], log[s_idx+1]
    print(f'    {s_idx:>5d}  {subjects_per_sess[s_idx]:<14s} {cnt:>10d}  {b-a:>8d}')

# ===== 5. Trajectory plots: best + worst session =====
print('\n========== 5. Saving trajectory plots ==========')
best = min(sess_stats, key=lambda x: x['median'])
worst = sess_stats_sorted[0]

# Pick worst session WITHOUT mass outliers (more representative)
sess_stats_no_outliers = [s for s in sess_stats if outlier_by_sess.get(s['sess'], 0) < 5]
worst_clean = max(sess_stats_no_outliers, key=lambda x: x['median'])

for label, info in [('best', best), ('worst_with_outliers', worst), ('worst_clean', worst_clean)]:
    a, b = info['frames']
    em = edge_mask[a:b]
    com_g = gt[a:b]
    com_p = pred[a:b]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].plot(com_g[em,0], com_g[em,1], 'k-', label='GT', alpha=0.7, linewidth=1)
    axes[0].plot(com_p[em,0], com_p[em,1], 'r-', label='pred', alpha=0.7, linewidth=1)
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)',
                title=f'Top-down  sess {info["sess"]} ({info["subj"]})',
                xlim=(-200, 1900), ylim=(-200, 1900), aspect='equal')
    axes[0].legend()

    t = np.arange(a, b) - a
    for ax_i, axis_name in enumerate(['x', 'y', 'z']):
        ax = axes[ax_i + 1]
        ax.plot(t[em], com_g[em, ax_i], 'k-', label='GT', alpha=0.7, linewidth=1)
        ax.plot(t[em], com_p[em, ax_i], 'r-', label='pred', alpha=0.7, linewidth=1)
        ax.set(xlabel='frame (10 Hz)', ylabel=f'{axis_name} (mm)',
               title=f'{axis_name} over time')
        ax.legend()

    plt.suptitle(f'{label.upper()}  median 3D err = {info["median"]:.1f} mm, '
                 f'mean = {info["mean"]:.1f} mm, n_valid = {info["n_valid"]}')
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'trajectory_{label}.png')
    plt.savefig(out, dpi=100)
    plt.close()
    print(f'  wrote {out}')

# ===== 6. CoM autocorrelation =====
print('\n========== 6. CoM autocorrelation (predictability ceiling) ==========')
sess_lens = np.diff(log)
longest = int(np.argmax(sess_lens))
a, b = log[longest], log[longest+1]
print(f'Using session {longest} (subject={subjects_per_sess[longest]}, length={sess_lens[longest]} frames)')

com_g_long = gt[a:b]
com_p_long = pred[a:b]

def autocorr(x, max_lag):
    x = x - x.mean()
    var = x.var()
    if var == 0:
        return np.zeros(max_lag + 1)
    n = len(x)
    return np.array([np.mean(x[:n-lag] * x[lag:]) / var for lag in range(max_lag + 1)])

max_lag = min(100, sess_lens[longest] // 4)
lags_sec = np.arange(max_lag + 1) / 10.0

fig, ax = plt.subplots(figsize=(8, 5))
for i, axis_name in enumerate(['x', 'y', 'z']):
    ac_g = autocorr(com_g_long[:,i], max_lag)
    ac_p = autocorr(com_p_long[:,i], max_lag)
    line, = ax.plot(lags_sec, ac_g, label=f'GT {axis_name}', linewidth=2)
    ax.plot(lags_sec, ac_p, '--', color=line.get_color(), alpha=0.5,
            label=f'pred {axis_name}')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='r = 0.5')
ax.axhline(0.0, color='black', linestyle=':', alpha=0.3)
ax.set(xlabel='lag (seconds, at 10 Hz)',
       ylabel='autocorrelation',
       title=f'CoM autocorrelation — session {longest} ({subjects_per_sess[longest]})')
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)
out = os.path.join(PLOTS_DIR, 'autocorrelation.png')
plt.tight_layout()
plt.savefig(out, dpi=100)
plt.close()
print(f'  wrote {out}')

# When does r drop below 0.5?
print('\n  Lag at which GT CoM autocorrelation drops below 0.5:')
for i, axis_name in enumerate(['x', 'y', 'z']):
    ac_g = autocorr(com_g_long[:,i], max_lag)
    below = np.where(ac_g < 0.5)[0]
    if len(below):
        lag = below[0]
        print(f'    {axis_name}: {lag} frames ({lag/10:.1f} sec)')
    else:
        print(f'    {axis_name}: > {max_lag} frames (never drops below 0.5 in this window)')

# Persistence baseline at 1 s and 2 s
print('\n  Persistence-baseline error (||CoM(t+H) - CoM(t)||) on the longest session (mm):')
for horizon_sec in [0.5, 1.0, 2.0]:
    H = int(horizon_sec * 10)
    if H >= len(com_g_long):
        continue
    diff = np.linalg.norm(com_g_long[H:] - com_g_long[:-H], axis=1)
    print(f'    horizon {horizon_sec:>3.1f} s ({H} frames):  mean={diff.mean():6.1f}  '
          f'median={np.median(diff):6.1f}  p95={np.percentile(diff,95):6.1f}')

print('\nDone.')

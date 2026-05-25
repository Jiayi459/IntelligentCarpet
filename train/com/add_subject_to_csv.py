"""Enrich com_results.csv with per-frame subject / date / round / session columns.

For each frame index `i` in com_results, look up:
  - session index f via log.p (largest f s.t. log[f] <= i)
  - source filename via fileNames[f]
  - subject / date / round parsed out of the filename
and write an enriched CSV next to the original.
"""
import os, re, pickle, csv
from collections import Counter
import numpy as np

# Script lives in train/com/. Test data lives in train/, outputs in train/com/output/.
_HERE    = os.path.dirname(os.path.abspath(__file__))   # .../train/com
_TRAIN   = os.path.dirname(_HERE)                       # .../train
_OUT     = os.path.join(_HERE, 'output')                # .../train/com/output
RESULTS  = os.path.join(_OUT, 'com_results.p')
TESTDIR  = os.path.join(_TRAIN, 'singlePerson_test')
OUT_CSV  = os.path.join(_OUT, 'com_results_with_subject.csv')

# ---- load ----
with open(RESULTS, 'rb') as f:
    r = pickle.load(f)
with open(os.path.join(TESTDIR, 'log.p'), 'rb') as f:
    log = pickle.load(f)
with open(os.path.join(TESTDIR, 'fileNames.p'), 'rb') as f:
    fileNames = pickle.load(f)

T = len(r['com_gt'])
print(f'frames in com_results : {T}')
print(f'log.p entries         : {len(log)}  log[0]={log[0]}  log[-1]={log[-1]}')
print(f'fileNames entries     : {len(fileNames)}')

# Quick structural check
session_lengths = np.diff(log).tolist()
print(f'unique session lengths (from diff(log)): {sorted(set(session_lengths))}')
print(f'sum of session lengths from diff(log)  : {sum(session_lengths)}  (vs T={T})')

# ---- parse filenames ----
pat = re.compile(r'(?:split_(\d+)_)?rec_(\d{4}-\d{2}-\d{2})_(.+?)_round(.+?)\.p')
parsed = []
unparsed = []
for n in fileNames:
    m = pat.match(n)
    if m:
        parsed.append({'split': m.group(1) or '',
                       'date': m.group(2),
                       'subject': m.group(3),
                       'round': m.group(4),
                       'filename': n})
    else:
        parsed.append(None)
        unparsed.append(n)

print(f'parsed   : {sum(1 for p in parsed if p)}')
print(f'unparsed : {len(unparsed)}')
if unparsed:
    print('  examples:')
    for n in unparsed[:5]:
        print(f'    {n}')

# ---- frame → session lookup ----
# log is the cumulative starting offset for each session.
# Frame i belongs to session f = largest index with log[f] <= i.
def frame_to_session(i):
    return int(np.searchsorted(log, i, side='right') - 1)

# Sanity check the boundaries
print()
print('Session 0 spans frames', log[0], '..', log[1] - 1)
print('Last few session starts:', log[-3:].tolist())

# Per-subject frame count
per_subject = Counter()
per_session = Counter()
for i in range(T):
    f = frame_to_session(i)
    per_session[f] += 1
    if parsed[f] is not None:
        per_subject[parsed[f]['subject']] += 1
    else:
        per_subject['(unparsed)'] += 1

print()
print('=== Per-subject frame count ===')
for subj, cnt in sorted(per_subject.items(), key=lambda kv: -kv[1]):
    pct = 100 * cnt / T
    print(f'  {subj:>16s}: {cnt:6d} frames  ({pct:5.1f}%)')

print()
print(f'Distinct sessions touched: {len(per_session)}')

# ---- write enriched CSV ----
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame', 'session', 'subject', 'date', 'round', 'source_file',
                'gt_x', 'gt_y', 'gt_z',
                'pred_x', 'pred_y', 'pred_z',
                'err_x', 'err_y', 'err_z', 'euclidean_err'])
    for i in range(T):
        f_idx = frame_to_session(i)
        info = parsed[f_idx] or {'subject': '', 'date': '', 'round': '', 'filename': ''}
        gx, gy, gz = r['com_gt'][i]
        px, py, pz = r['com_pred'][i]
        ex, ey, ez = r['error'][i]
        ee = r['euclidean_error'][i]
        w.writerow([i, f_idx, info['subject'], info['date'], info['round'], info['filename'],
                    f'{gx:.2f}', f'{gy:.2f}', f'{gz:.2f}',
                    f'{px:.2f}', f'{py:.2f}', f'{pz:.2f}',
                    f'{ex:.2f}', f'{ey:.2f}', f'{ez:.2f}', f'{ee:.2f}'])

print(f'\nWrote {OUT_CSV}')

# Session Log

## Session 1 — 2026-05-10

### What was done
- Read the full repo and created `REPO_STRUCTURE.md` explaining every file, function, data shape, and the data flow between modules.

### Repo summary
IntelligentCarpet (CVPR 2021) by Yiyue Luo et al. (MIT CSAIL). Infers 3D human body pose (21 keypoints) from a 96×96 pressure-sensing carpet using a CNN.

**Model pipeline:** tactile frames (96×96) → 2D CNN encoder → 3D CNN decoder → 3D heatmap (21×20×20×18) → SpatialSoftmax3D → keypoints (21×3)

### Data access
- Demo data (checkpoints + test set): https://www.dropbox.com/sh/5l0lm4po64xf6jd/AACuMt_oGy99Beyz_IMeknQ6a?dl=0
  - `ckpts.zip` → unzip to `./train/ckpts/`
  - `singlePerson_test.zip` → unzip to `./train/singlePerson_test/`
- Raw dataset: https://www.dropbox.com/sh/g3l4jdablczffj3/AACuFy9E2YonQdNjUu4beClta?dl=0
- User confirmed demo Dropbox link is active and files were found.

### How to run singlePerson test
Must `cd` into `train/` first (modules import each other by relative name):
```bash
cd train
python threeD_train_final.py --exp_image True   # save per-frame JPGs
python threeD_train_final.py --exp_video True   # save .avi video
python threeD_train_final.py --exp_L2 True      # save L2 distances
python threeD_train_final.py --exp_data True    # save raw predictions
```

Output lands in `train/predictions/{image,video,L2,data}/`.

### Known bug — fix before running
`train/threeD_train_final.py` lines 116–117 hardcode `device = 'cuda:1'` (written for a multi-GPU Linux server). Change to:
```python
use_gpu = torch.cuda.is_available()
device = 'cuda:0' if use_gpu else 'cpu'
```

### Status
User has downloaded the demo files and is ready to run inference.

---

## Session 2 — 2026-05-10

### Goal
Compute center of mass (CoM) from the model's predicted 3D keypoints.

---

### 1. Input Data

The model (`threeD_train_final.py`) produces two outputs per forward pass:

| Variable | Shape | Description |
|---|---|---|
| `keypoint_out` | `(B, 21, 3)` | Predicted 3D joint positions, normalized to [0, 1] |
| `keypoint` (GT) | `(B, 21, 3)` | Ground-truth joint positions, same space |

**Converting normalized coords → real-world mm** (from `get_spatial_keypoint` in `threeD_train_final.py:54–59`):
```python
b = np.array([-100, -100, -1800])   # origin offset in mm
resolution = 100                     # mm per voxel bin
max_idx = 19                         # grid is 0–19
pos_mm = keypoint * max_idx * resolution + b
# x, y range: [-100, 1800] mm  (~2m carpet)
# z range:    [-1800, 0] mm    (height, 0 = floor level)
```

**The 21 keypoints** (BODY_25 joints 0–14 + feet 19–24, remapped to indices 15–20):

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | Nose | 11 | RAnkle |
| 1 | Neck | 12 | LHip |
| 2 | RShoulder | 13 | LKnee |
| 3 | RElbow | 14 | LAnkle |
| 4 | RWrist | 15 | LBigToe |
| 5 | LShoulder | 16 | LSmallToe |
| 6 | LElbow | 17 | LHeel |
| 7 | LWrist | 18 | RBigToe |
| 8 | MidHip | 19 | RSmallToe |
| 9 | RHip | 20 | RHeel |
| 10 | RKnee | | |

---

### 2. Method — Segmental CoM

The standard biomechanical approach (De Leva 1996 / Winter 2009) models the body as rigid segments. Each segment has:
- A **proximal** and **distal** endpoint (two known keypoints)
- A **mass fraction** `m_i` (fraction of total body mass)
- A **CoM location fraction** `α_i` (distance from proximal end as fraction of segment length)

**Segment CoM formula:**
```
CoM_i = P_i + α_i * (D_i - P_i)
```
where `P_i` = proximal keypoint position, `D_i` = distal keypoint position.

**Total body CoM formula:**
```
CoM_total = Σ(m_i * CoM_i) / Σ(m_i)
```
We divide by `Σ(m_i)` (not necessarily 1.0) because some segments are missing (see section 3).

---

### 3. Segment Table

Segments definable from the 21 available keypoints, using Winter (2009) Table 4.1:

| Segment | Proximal KP | Distal KP | Mass fraction `m_i` | CoM from proximal `α_i` |
|---------|-------------|-----------|---------------------|------------------------|
| Trunk | 1 Neck | 8 MidHip | 0.497 | 0.495 |
| R Upper Arm | 2 RShoulder | 3 RElbow | 0.028 | 0.436 |
| R Forearm | 3 RElbow | 4 RWrist | 0.016 | 0.430 |
| L Upper Arm | 5 LShoulder | 6 LElbow | 0.028 | 0.436 |
| L Forearm | 6 LElbow | 7 LWrist | 0.016 | 0.430 |
| R Thigh | 9 RHip | 10 RKnee | 0.100 | 0.433 |
| R Shank | 10 RKnee | 11 RAnkle | 0.0465 | 0.433 |
| L Thigh | 12 LHip | 13 LKnee | 0.100 | 0.433 |
| L Shank | 13 LKnee | 14 LAnkle | 0.0465 | 0.433 |
| R Foot | 11 RAnkle | 18 RBigToe | 0.0145 | 0.500 (approx) |
| L Foot | 14 LAnkle | 15 LBigToe | 0.0145 | 0.500 (approx) |

`Σ m_i = 0.497 + 2*(0.028+0.016) + 2*(0.100+0.0465+0.0145) = 0.497 + 0.088 + 0.322 = 0.907`

**Missing segments (0.093 of body mass unaccounted for):**
- **Head** (m = 0.081): only Nose (0) and Neck (1) are available. Nose is anterior, not the top of the head — not a valid proximal/distal pair for head segment.
- **Hands** (m = 0.006 each): only wrists (4, 7) are available; no distal hand point.

> **OPEN QUESTION 1 — Head:** How should we handle the head segment?
> - **Option A (ignore):** omit head; CoM will be biased slightly low (~4 cm for a 170 cm person).
> - **Option B (Nose as proxy):** treat Nose (0) as head CoM directly. Introduces ~10 cm anterior bias.
> - **Option C (extrapolate):** estimate head CoM as `Neck + k * (Neck - MidHip) / |Neck - MidHip|` (extend trunk axis upward by fraction k). Anatomically motivated but adds a free parameter.
>
> **My recommendation:** Option C with k ≈ 0.13 * body height (roughly 22 cm for an average person), which places head CoM above Neck along the vertical axis. But this needs the subject's height or a scale reference.

> **OPEN QUESTION 2 — Hands:** Include or ignore?
> - Hands are only 0.6% of body mass each → combined effect on CoM < 1 cm. Safe to ignore unless you need high precision.
> - **My recommendation:** ignore hands, note the omission.

---

### 4. Output

```python
# Shape: (T, 3) — one CoM per frame, in mm
CoM_trajectory = np.zeros((T, 3))

for t in range(T):
    kp = keypoint[t]  # shape (21, 3), normalized [0,1]
    kp_mm = kp * 19 * 100 + np.array([-100, -100, -1800])  # convert to mm

    segments = [
        (1, 8,  0.497, 0.495),   # Trunk
        (2, 3,  0.028, 0.436),   # R Upper Arm
        (3, 4,  0.016, 0.430),   # R Forearm
        (5, 6,  0.028, 0.436),   # L Upper Arm
        (6, 7,  0.016, 0.430),   # L Forearm
        (9, 10, 0.100, 0.433),   # R Thigh
        (10,11, 0.0465,0.433),   # R Shank
        (12,13, 0.100, 0.433),   # L Thigh
        (13,14, 0.0465,0.433),   # L Shank
        (11,18, 0.0145,0.500),   # R Foot
        (14,15, 0.0145,0.500),   # L Foot
    ]

    weighted_sum = np.zeros(3)
    total_mass = 0.0
    for (p_idx, d_idx, mass, alpha) in segments:
        com_seg = kp_mm[p_idx] + alpha * (kp_mm[d_idx] - kp_mm[p_idx])
        weighted_sum += mass * com_seg
        total_mass += mass

    CoM_trajectory[t] = weighted_sum / total_mass
```

---

### 5. Ground Truth & Validation

This dataset has **no force plate** data (gold standard for CoM). The best available ground truth is the **GT keypoints** (`keypoint` from the dataloader), which were obtained by Openpose triangulation + optimization from two calibrated cameras.

**Validation strategy:**
1. Compute `CoM_pred` from predicted keypoints `keypoint_out`
2. Compute `CoM_GT` from ground-truth keypoints `keypoint`
3. Compare: `error = |CoM_pred - CoM_GT|` per frame, per axis (x, y, z)
4. Report mean ± std error in mm

**Sanity checks on the output:**
- For a standing adult, CoM_z (height) should be roughly 55% of standing height — approximately 900–1000 mm above the floor (z ≈ -800 to -900 mm in this coordinate system where floor is z=0 and body goes negative)
- CoM_x and CoM_y should stay within the carpet bounds [−100, 1800] mm
- CoM trajectory should be smooth — large frame-to-frame jumps (> ~100 mm) indicate prediction errors
- Bilateral symmetry check: for a person standing still, `CoM_x` should be close to the midpoint between left and right hips `(RHip_x + LHip_x) / 2`

---

### 6. Open Questions (need answers before implementation)

1. **Head segment:** which option (ignore / Nose proxy / extrapolate)? Do you have subject height?
2. **Hands:** ignore or include (wrist as hand CoM proxy)?
3. **Coordinate output:** normalized [0,1] or real-world mm?
4. **Per-frame or aggregated:** do you want a `(T, 3)` trajectory, or summary stats (mean CoM, CoM sway range, etc.)?
5. **Purpose of CoM:** what downstream analysis? (balance scoring, fall risk, gait symmetry, something else?) — this affects which axis and which metric matters most.
6. **Subject body mass:** known or unknown? (affects nothing for CoM position since mass fractions are ratios, but matters if you want to compute momentum or GRF later)

---

### Status
Plan written. Awaiting answers to open questions before implementing.

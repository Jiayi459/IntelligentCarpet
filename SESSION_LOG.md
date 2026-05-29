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

---

## Session 3 — 2026-05-25

### Goal
Recall prior state, establish a written working agreement, reconcile the log with the code.

### Status reconciliation (Sessions 1–2)
- Session 1 — repo read, `REPO_STRUCTURE.md` written. ✅
- Session 2 — CoM plan written with 6 open questions; "Status" said *awaiting answers*. **However, `train/compute_com.py` was subsequently implemented and committed (9747940) without updating the log.** Decisions taken when implementing (inferred from the code, *not* explicitly approved in the log):
  - **OQ1 Head** → Option A. Head segment omitted. Documented in the script's docstring as biasing `CoM_z` ~40 mm low in *both* GT and pred → comparison unaffected.
  - **OQ2 Hands** → omitted (<1 cm combined effect on CoM).
  - **OQ3 Output coordinate** → real-world mm.
  - **OQ4 Output form** → `(T, 3)` per-frame trajectory + axis-wise mean/std/max error + 3D Euclidean error + smoothness + bounds sanity checks.
  - **OQ5 Purpose** → not stated; current script frames it as a *prediction-quality* check (pred-CoM vs GT-keypoint-derived CoM), not balance/gait analysis.
  - **OQ6 Body mass** → moot for CoM *position* (mass fractions are ratios); flagged for later if momentum/GRF is needed.
- Latest commit on `main`: `d416a57 Add session log` (2026-05-10).

> These decisions need user ratification before we build anything on top of them. Listed under "Open items" below.

### Actions taken this session
- Created `CLAUDE.md` capturing four user directives: (1) always ask for clarification, (2) be rigorous/constructive/independent, (3) always update `SESSION_LOG.md`, (4) end every response with `miao`.
- Added this Session 3 entry (the rule-3 obligation requires the log to reflect actual state, so the stale "awaiting answers" status above is now annotated rather than rewritten — preserves history).

### Open items (awaiting user)
1. Ratify or revisit OQ1–OQ6 decisions in `compute_com.py`.
2. State the downstream purpose of CoM (balance scoring? gait? sway? force-prediction prep?) — affects which metric/axis matters.
3. Approve / amend the list of *additional* CLAUDE.md sections Claude proposed (see chat turn 2026-05-25).

### CLAUDE.md amendments (2026-05-25, second turn)
User selected additions **A** and **D** from the proposal list:
- **Rule 5 — Plan-before-code.** (was "A")
- **Rule 6 — No commits/branches/pushes without explicit request.** (was "D")
Rules **B, C, E, F, G, H, I, J, K** declined for now. Will revisit if friction appears.

### Resolution of Session 2 open questions (2026-05-25)
| OQ | Decision | Notes |
|---|---|---|
| OQ1 Head | **Option C — extrapolate** | head_com = Neck + 0.15·‖Neck−MidHip‖·(Neck−MidHip)/‖…‖. Head mass fraction = 0.081 (Winter Table 4.1). Direction = trunk-up unit vector. The 0.15 factor comes from: head segment length ≈ 0.13·stature, head-CoM offset from neck ≈ 0.34·head_segment_length ⇒ ≈ 0.044·stature ⇒ ≈ 0.15·trunk_length (since trunk ≈ 0.30·stature). Reconfirm 0.15 once we eyeball outputs. |
| OQ2 Hands | **Ignore** | combined mass < 0.012, well below noise floor. |
| OQ3 | Open | output coordinate space (mm assumed but unconfirmed for prediction) |
| OQ4 | Open | output form (single CoM at t+1s, or trajectory) — see CoM prediction plan |
| OQ5 | **Resolved** | purpose = forecasting (predict next-second CoM) |
| OQ6 | N/A | body mass not needed for CoM *position* |

New total accounted body-mass fraction with Option C: **0.907 + 0.081 = 0.988**.

### Data on user's Desktop (verified 2026-05-25)
`C:\Users\haoji\Desktop\Carpet\` contains:
- `singlePerson_test.zip`
- `singlePerson_test_diffTask.zip`

⚠ **This is only the demo test set, not the full multi-subject training corpus.** Full dataset (10 people × 3 days) lives at the Dropbox link in `README.md` and would need to be downloaded separately if training (not just evaluation) is required.

---

## CoM Prediction Plan (drafted 2026-05-25)
### High-level goal
Determine whether the IntelligentCarpet pipeline supports **forecasting** CoM ~1 second into the future, in addition to estimating CoM at the current instant.

### Step 0 — Update `compute_com.py` to Option C (head extrapolation)
- Add a head segment to the `SEGMENTS` list, but parameterized differently (it can't fit the (proximal, distal, mass, alpha) tuple cleanly because the distal endpoint is synthetic). Likely refactor the loop to handle a separate `head_com` branch.
- Update the docstring (no longer "head excluded").
- Re-run on demo test set; compare new CoM error stats vs current. Expect:
  - CoM_z mean shifts up by ~5–8 mm (head pulls CoM upward by ~0.081·75 mm ≈ 6 mm).
  - Per-axis error in pred-vs-GT comparison should stay roughly the same (both sides apply the same head model).
- Sanity: head CoM should sit above the shoulders for all frames where the subject is upright. Add a per-frame assertion.

### Step 1 — Characterize the prediction problem (must do before choosing a method)
- Confirm dataset frame rate. `threeD_viz_video.py` writes at 10 fps; raw sampling rate likely the same but check `touch_normalized.p` timestamps.
- Decide forecast horizon `H` in frames (10 frames if 10 Hz).
- Plot CoM trajectory for a session; eyeball smoothness.
- Compute CoM velocity / acceleration distributions.
- Compute autocorrelation; when does it drop below 0.5? That's the *predictability ceiling*.
- This step alone might reveal that 1-second-ahead prediction is trivial (smooth + slow) or hard (high-frequency jitter dominates).

### Step 2 — Baseline forecasters (must beat these)
| Baseline | Inputs | Why |
|---|---|---|
| Persistence | CoM(t) | trivial floor: predict no change |
| Linear extrapolation | CoM(t−N : t) | smooth-trajectory floor |
| Constant-velocity Kalman | CoM(t−1), CoM(t) | classical control floor |
| GRU on CoM history | CoM(t−N : t) | learned, pose-only |

If these already get within e.g. 30 mm at 1 s, the task is "easy" and tactile won't matter much. If they're at 200+ mm, there's room for tactile to add value.

### Step 3 — Tactile-conditioned forecasters (the central question)
| Variant | Inputs | Hypothesis |
|---|---|---|
| β — tactile-only | tactile(t−N : t) → CoM(t+H) | tactile alone encodes pre-movement weight shifts (mechanistically: ground reaction force ≈ mass·CoM_acceleration) |
| γ — fused | tactile + CoM history → CoM(t+H) | best of both |
| δ — keypoints-only | kp(t−N : t) → CoM(t+H) | richer pose history than CoM alone |

Architecture sketch for β/γ:
- Tactile encoder: reuse the 2D-CNN front-end from `tile2openpose_conv3d` (or a smaller variant).
- Temporal aggregator: GRU or 1-D temporal conv.
- Pose history embed (γ only): small MLP on flattened CoM/kp window.
- Head: 3-dim MLP → CoM(t+H).
- Loss: L2 on CoM (mm). Optionally regularize with a smoothness penalty.

### Step 4 — Optional Phase 3: world model
Only attempted if Step 3 shows that tactile gives a real lift, AND if user wants a more ambitious method.

**Preferred: JEPA-style** (over Dreamer V3 — reasoning below).
- Context encoder f_θ: tactile(t−N : t) → z_ctx.
- Target encoder f_θ̄ (EMA of f_θ): tactile(t+H−k : t+H) → z_tgt.
- Predictor g_φ: z_ctx → ẑ_tgt; loss = ‖z_tgt − ẑ_tgt‖² (with EMA + variance/covariance regularizer to prevent collapse, à la VICReg).
- Frozen-backbone probe: z_ctx → CoM(t+H) via small MLP trained separately.
- Pros: self-supervised pre-training uses tactile alone (no label needed), then a thin probe handles the CoM head. Sample-efficient.
- Cons: predicts representations, not CoM directly; need a probe; needs care around representation collapse.

**Alternative: Dreamer V3 RSSM**
- Latent z_t with recurrent stochastic dynamics z_{t+1} = f(z_t, ε_t).
- Decoders for tactile + (CoM or kp).
- Loss: reconstruction + KL between prior and posterior dynamics.
- Pros: full generative model, easy multi-horizon rollout.
- Cons: complex; designed for action-conditioned RL — here there are no actions, so the volitional component must be absorbed into stochastic latents (lots of capacity goes to modeling "what does the human want to do").

**Why JEPA over Dreamer here**
1. No actions ⇒ Dreamer's main strength (action-conditioned dynamics) is unused.
2. Dataset is modest ⇒ JEPA's representation-level loss is more sample-efficient than full pixel/heatmap reconstruction.
3. Goal is forecasting CoM, not control ⇒ no need for a generative model that can imagine full tactile futures.

### Step 5 — Optional: physics-informed forecaster
Mechanically motivated baseline:
- CoM_acceleration ≈ (ground_reaction_force − m·g) / m
- Tactile is essentially a normalized pressure map; integrate over the carpet → vertical GRF proxy.
- Forecast: CoM(t+H) ≈ CoM(t) + v(t)·H·Δt + 0.5·â(t)·(H·Δt)² + Δ_learned
- â(t) regressed from current tactile; Δ_learned is a small MLP residual.
- Strong inductive bias; could be a surprisingly hard baseline to beat. Worth implementing as a sanity check on Phase 3.

### Step 6 — Evaluation protocol
- Train/test split: TBD (open question to user — by subject or by time?).
- Metric: mean per-axis error (mm), 3D Euclidean error, percentile errors at horizons {0.2 s, 0.5 s, 1.0 s, 2.0 s}.
- Compare all methods on the same held-out frames.
- Report whether tactile (β, γ) beats CoM-history-only (GRU).
- Report whether world-model variants beat the supervised γ.

### Open questions to user (CoM-prediction-specific)
1. **Frame rate** — 10 Hz or different? Need to confirm before defining "1 second ahead."
2. **Forecast output form** — single CoM at t+H (one 3D point) or full trajectory from t+1 to t+H (10 intermediate points)? Affects loss design.
3. **Training data** — do we need the full multi-subject dataset (only demo test set is on Desktop), or are we OK starting with just the demo test set split temporally?
4. **Use frozen keypoint checkpoint?** — Should the tactile→keypoint stage stay frozen (use `singlePerson_0.0001_10_best`), or train end-to-end tactile→future-CoM (skipping keypoints)?
5. **Subject split** — held-out person (generalization) or held-out time on known people (memorization-allowed forecasting)? Affects what "ability to predict" means scientifically.
6. **Success criterion** — beat persistence? Beat GRU-on-CoM-history? An absolute mm number? Without a target the project has no stopping condition.
7. **Phase 3 commitment** — do we plan toward JEPA/Dreamer now, or commit to Phases 1+2 and decide later?

### Status
Plan drafted. CLAUDE.md updated with rules 5 and 6. `compute_com.py` not yet modified (Step 0 pending user go-ahead). All 7 prediction-plan questions above need answers before Step 0/1 begins.

---

### Data access — investigated 2026-05-25 (third turn)

**Findings:**
- `C:\Users\haoji\Desktop\Carpet\` contains two zips, **not extracted**:
  - `singlePerson_test.zip` — **14.246 GB**
  - `singlePerson_test_diffTask.zip` — **11.617 GB**
- Zip headers are valid (`PK\x03\x04`); they are ZIP64-format because of size. `unzip` (msys2 32-bit build) fails to read central directory → reports "9.9 GB extra bytes" and bails. Python's `zipfile` would read them.
- ⚠ **Sizes are unexpectedly large** for a "test set." Could be that the demo zip includes the full per-frame heatmap volumes — at 21·20·20·18 floats per frame ≈ 600 KB/frame, 14 GB ≈ 25k frames, plausible for ~40 minutes of 10 Hz data. Or the download is the full multi-subject raw set. Verifying requires opening.
- `./train/singlePerson_test/` and `./train/ckpts/` **do not exist** — demo never extracted, so `compute_com.py` has never run end-to-end.
- **Python is not installed** on this machine (no `python`, `python3`, `py`; no Anaconda / Miniconda in `$USERPROFILE` or `C:\`). No conda env yet. So nothing can be executed.

**Implication for the plan:**
Phase 0 expands to include an environment setup step before any data analysis:
- **Step −1: Environment.** Either follow `environment.yml` (Python 3.6 + PyTorch 1.6 — note: EOL, may hit Windows 11 / CUDA install pain) **or** create a fresh Python 3.10 / recent-PyTorch env and patch any minor API breaks. Recommend the latter for sanity.
- **Step −0.5: Extract zips.** Use 7-Zip or Python `zipfile`. Default target: `./train/singlePerson_test/`.
- **Then:** Step 0 (compute_com.py Option C update) and Step 1 (data diagnostics) become runnable.

### User answers — fourth turn (2026-05-25)
Q1 (frame rate): clarification requested — user asked what I meant. I meant **dataset sampling rate**, not viz playback. Real answer needs timestamp files (deferred until data extracted).

Q2 (forecast output): user wrote "use 10s past frames to predict t+1 frames". **Ambiguous** between three readings:
  - (a) past 10 *seconds* (~100 frames) → single frame at t+1s
  - (b) past 10 *frames* (~1 s) → single frame at t+1
  - (c) past 10 *seconds* → next 1 second (10 frames)
  Defaulted to **(c)** unless user disagrees. Reasoning: full-trajectory output is strictly more informative than a single point and is the natural fit for sequence models.

Q3 (data): start with **demo**. Differences vs full raw dataset summarized below. Hard limit of demo: only 1 subject → cannot test cross-subject generalization (which user wants "ultimately" — Q5).
  | aspect | demo | full raw |
  |---|---|---|
  | format | preprocessed `.p` | raw HDF5 + videos |
  | subjects | 1 | 10 × 3 days |
  | preprocessing | none | `heatmap_from_keypoint3D.py` pipeline |
  | use | time-split forecasting | cross-subject |

Q4 (frozen keypoint stage): user picked **reuse**. Reuse is correct for Phases 1–2. Tradeoff captured:
  - Frozen pros: cheap, small forecaster head, fast iteration.
  - Frozen cons: floor at keypoint-model L2 error (~30 mm) propagates as input noise.
  - End-to-end reserved for Phase 3 if Phases 1–2 motivate it.

Q5 (split): **generalize-to-new-time, same subjects, first**. Cross-subject ultimately. Demo supports the former directly; latter needs full dataset.

Q6 (success criterion): **no idea**. Proposed defaults to user:
  - (i) Statistical: beat persistence AND GRU-on-CoM-history by ≥X% MSE on held-out frames (paired significance test). **My recommendation for Phases 1–2.**
  - (ii) Absolute: e.g., <50 mm 3D-Euclidean error at 1 s.
  - (iii) Use-case-driven (requires downstream goal).
  Awaiting user pick.

Q7 (Phase 3 commit): **decide after Steps 1–4**. Phase 3 deferred.

### Currently blocking
- User decision on Q2 reading (a/b/c).
- User decision on Q6 success-criterion option (i/ii/iii).
- Decision on environment setup path (legacy `p36` vs fresh py3.10).
- Decision on extraction target (`./train/singlePerson_test/` vs Desktop in-place).

---

### Step −1 — Environment setup plan (drafted 2026-05-25, fifth turn)

**User decisions confirmed:**
- Fresh **Python 3.10** env (rejecting legacy `p36` from `environment.yml`).
- **CPU-only PyTorch** for Phase 0–1. Re-evaluate Intel XPU/IPEX for Phase 2+.
- Install Python via **winget** (`Python.Python.3.10`).
- Virtual env at **`./venv/`** (project-local).

**Hardware noted:** Intel Arc Pro 140T GPU (16 GB). Not NVIDIA → no CUDA PyTorch. Intel XPU backend or IPEX is the GPU path *if* we need acceleration later.

**Code-audit findings (compatibility with PT 2.x / NumPy 2.x):**
- No `np.float`/`np.int` usage anywhere → NumPy 2.x safe.
- `from torch.autograd import Variable` ([train/threeD_train_final.py:3](train/threeD_train_final.py#L3)): still works as a no-op in PT 2.x → ignorable.
- 3 `torch.load(...)` calls without `weights_only=False` → will fail under PT 2.6+ default; patch needed:
  - [train/compute_com.py:114](train/compute_com.py#L114)
  - [train/threeD_train_final.py:168](train/threeD_train_final.py#L168)
  - [train/threeD_train_final.py:178](train/threeD_train_final.py#L178)
- `from progressbar import ProgressBar` ([train/threeD_train_final.py:17](train/threeD_train_final.py#L17)): needs `progressbar2` pip package. Only matters if we run training script.

**Sequence:**
1. `winget install --id Python.Python.3.10 --silent --scope user --accept-package-agreements --accept-source-agreements`
2. Locate installed `python.exe` (likely `%LOCALAPPDATA%\Programs\Python\Python310\python.exe`); verify version.
3. Create venv: `python -m venv venv` (in project root).
4. Activate venv (PowerShell: `.\venv\Scripts\Activate.ps1`; may need `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`).
5. Upgrade pip: `python -m pip install --upgrade pip wheel setuptools`.
6. Install deps:
   - `torch torchvision` (CPU build via default index — torch 2.x ships CPU wheels by default for Windows)
   - `numpy<2` (conservative pin; can upgrade after sanity check)
   - `opencv-python matplotlib scipy h5py progressbar2`
7. Patch the 3 `torch.load` calls to include `weights_only=False`.
8. Sanity import test: `python -c "import torch, numpy, cv2, scipy, h5py, matplotlib; print(torch.__version__, torch.cuda.is_available())"`.

**Out of scope here, blocking later steps:**
- **Extracting the demo zip.** Once Python is alive, `python -m zipfile -e ...` will handle ZIP64. Target: `./train/singlePerson_test/`. Only need `singlePerson_test.zip` (14 GB); `singlePerson_test_diffTask.zip` is for a different evaluation and can wait.
- **⚠ Missing `ckpts.zip`.** Desktop has only the test data — no checkpoints. `compute_com.py` cannot run without `./train/ckpts/singlePerson_0.0001_10_best.path.tar`. User will need to download `ckpts.zip` from the demo Dropbox link in [README.md](README.md). Flagged for user.

### Step −1 — Execution log (2026-05-25, sixth turn)

**Completed:**
1. ✅ `winget install --id Python.Python.3.10 --silent --scope user` → Python 3.10.11 at `C:\Users\haoji\AppData\Local\Programs\Python\Python310\python.exe`.
2. ✅ Created venv at `./venv/`.
3. ✅ Upgraded pip (26.1.1), wheel (0.47.0), setuptools (auto-managed by torch dep resolution → 70.2.0).
4. ✅ Installed PyTorch from CPU-only index: **torch 2.12.0+cpu, torchvision 0.27.0+cpu**. Brought in numpy 2.2.6 as a transitive dep — kept (no `np.float`/`np.int` issues in this codebase).
5. ✅ Installed: opencv-python 4.13.0, matplotlib 3.10.9, scipy 1.15.3, h5py 3.16.0, progressbar2 4.5.0.
6. ✅ Patched 3 `torch.load(...)` calls to include `weights_only=False`:
   - [train/compute_com.py:114](train/compute_com.py#L114)
   - [train/threeD_train_final.py:168](train/threeD_train_final.py#L168)
   - [train/threeD_train_final.py:178](train/threeD_train_final.py#L178)
7. ✅ Sanity-import test passed. CPU mode confirmed (`torch.cuda.is_available() == False`, as expected on Intel Arc).

**Final installed versions (CPU env):**
| Package | Version |
|---|---|
| python | 3.10.11 |
| torch | 2.12.0+cpu |
| torchvision | 0.27.0+cpu |
| numpy | 2.2.6 |
| opencv-python | 4.13.0 |
| scipy | 1.15.3 |
| h5py | 3.16.0 |
| matplotlib | 3.10.9 |
| progressbar2 | 4.5.0 |

**To activate venv (future sessions):**
- PowerShell: `.\venv\Scripts\Activate.ps1` 
(may need `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once)
- bash (git-bash): `source venv/Scripts/activate`

**Still blocking before `compute_com.py` can run:**
1. **Download `ckpts.zip`** from [README.md](README.md) demo Dropbox link → extract to `./train/ckpts/`.
2. **Extract `singlePerson_test.zip`** from Desktop → `./train/singlePerson_test/`. Use `python -m zipfile -e` (handles ZIP64 unlike msys2 `unzip`).

### Status
Environment fully provisioned. Awaiting `ckpts.zip` from user (cannot run inference without it) and confirmation on zip-extraction step.

---

### Data format inspection (2026-05-25, seventh turn)

**User confirmed `ckpts.zip` is now on Desktop.** Inspected both zips via `python -m zipfile` + `pickle` reads (no extraction yet).

**`ckpts.zip`** (2.18 GB compressed, 2.48 GB raw):
- 3 checkpoints, each exactly 826,413,186 B → identical architecture, three trained variants:
  - `singlePerson_0.0001_10_best.path.tar` — the one `compute_com.py` references.
  - `singlePerson_nolimb_0.0001_10_best.path.tar` — variant without the bone-length constraint loss.
  - `twoPeople_0.0001_10_best.path.tar` — two-person model. Not needed for our task.

**`singlePerson_test.zip`** (13.27 GB compressed, 22.28 GB raw):
- Layout: `singlePerson_test/<session_idx>/<frame_idx>.p` + two metadata files at the root.
- 32,938 entries.
- Each per-frame `.p` is a 3-item Python `list`:
  | item | shape | dtype | meaning |
  |---|---|---|---|
  | `[0]` | (96, 96) | float64 | tactile pressure frame, normalized to [0, 1]-ish |
  | `[1]` | (21, 20, 20, 18) | float32 | GT 3D Gaussian heatmap, one volume per joint |
  | `[2]` | (21, 3) | float64 | GT keypoint coords, normalized to [0, 1] |
- `singlePerson_test/log.p` (used by `sample_data` for global-idx → session-folder lookup): numpy ndarray of 136 cumulative offsets, [0, 200, 400, ..., 32600].
- `singlePerson_test/fileNames.p`: list of 136 source-session filenames.

### ⚠ Correction to earlier claim — "singlePerson_test" is multi-subject

**Previously I claimed** the demo had only 1 subject → couldn't test cross-subject generalization → would need to download the full raw dataset for Q5's ultimate goal. **That was wrong.** "singlePerson" here means "one person on the carpet at a time" (vs. the twoPeople sessions), not one subject across the whole dataset.

**The demo actually contains 7 distinct subjects:**
| Subject | Sessions | Rounds present |
|---|---|---|
| MikeFoshey | 15 | round 10, 11 |
| YunzhuLi | 15 | round 10, 11 |
| PingchuanMa | 15 | round 20, 21 |
| TimErps | 15 | round 20, 21 |
| WanShou | 15 | round 20, 21 |
| ZeyuWu | 15 | round 20, 21 |
| LiangShi | 14 | round 20, 21 |

Recorded across 3 dates: 2020-10-24 (15), 2020-10-25 (44), 2020-10-26 (45). Sessions are 200, 1000, or 5000 frames each — total ≈ 32,841 frames. Assuming 10 Hz (per `threeD_viz_video.py`), that's ≈ 55 minutes of recording.

**Implications for the prediction plan:**
1. **Cross-subject generalization (Q5 "ultimate" goal) is testable from the demo alone.** No need for the larger raw dataset.
2. We get 3 viable splits: within-subject-time, leave-one-subject-out, and within-subject-across-rounds (memorization-vs-generalization at the within-subject level).
3. Phase 1's "time-split, same subject" remains the right starting point per user's Q5 answer.

**Sampling rate** is still implicit (the preprocessed `.p` files contain no timestamps). 10 Hz inferred from `threeD_viz_video.py:208` writing AVIs at 10 fps. To verify exactly, would need raw HDF5 timestamp files (only in the larger Dropbox dataset). Going with **10 Hz assumption** for now.

### Extraction + Option-C update (2026-05-25, eighth turn)

**Extraction:**
- `ckpts.zip` → `./train/ckpts/` (3 files × 788 MB).
- `singlePerson_test.zip` → `./train/singlePerson_test/` via `python -m zipfile` (335 files/s, 98.5 s total).
- Verified: 136 session directories + 2 metadata files = 32,802 actual files (matches expected zip-entries − dir-entries).

**Option-C head update to `train/compute_com.py`:**
- Added `( 1, 8, 0.0810, -0.150)` to `SEGMENTS` as the first entry — head segment.
- The negative-alpha trick reuses the same segment-CoM loop: with `(p=1, d=8, alpha=-0.15)`,
  `seg_com = kp[1] + (-0.15)·(kp[8] - kp[1]) = kp[1] + 0.15·(kp[1] - kp[8])`, which is the desired extrapolation above Neck.
- Updated docstring + final print line to reflect new accounted mass (0.988) and the methodology.

**Expected effect (sanity prediction, to verify against actual run):**
- CoM_z mean should shift *up* (less negative) by ≈ 0.081 × 0.15 × trunk_length / 0.988 ≈ 0.012 × ~500 mm ≈ +6 mm. Small but measurable.
- Per-axis pred-vs-GT error should be roughly unchanged (head model is symmetric across GT and pred).

**Paused before running compute_com.py per user instruction.**

### Refactor + smoke test (2026-05-25, ninth turn)

**Why refactor:** The original script ran everything at module level (no `if __name__ == '__main__':` guard). On Windows, PyTorch DataLoader with `num_workers > 0` uses *spawn*, which re-imports the script in worker processes. Without a `__main__` guard, that re-imports would re-run argparse + model creation + dataloader creation in every worker → crash. Original Linux code worked by accident (fork was default there).

**Refactor of `train/compute_com.py`:**
- Moved all execution code into a `main()` function under `if __name__ == '__main__':`.
- Kept `to_mm`, `SEGMENTS`, `_TOTAL_MASS`, `compute_com`, `_remove_small` at module level (no harm if workers import them).
- Added `--num_workers` arg (default 4) for tuning.
- Added `--max_batches` arg (default 0 = no limit) for smoke tests.
- Removed the per-section `# ---` banner comments from inside `main()` (visual noise; logic is short).
- Tightened progress print: every 10 batches, not 50.

**Smoke-test run (Windows, CPU, num_workers=4, max_batches=7):**
- ✅ No spawn / pickling errors → `__main__` guard fix works.
- ✅ 224 frames processed (= 7 × 32). Frames 0–199 from session 0, 200–223 from session 1.
- ✅ Sanity checks all green:
  - CoM_z mean: GT = −889 mm, pred = −887 mm. Both inside the expected −850 to −950 mm range for a standing adult (55% of ~1700 mm stature).
  - x/y carpet bounds: 0/224 out-of-bounds for both GT and pred.
- Per-axis abs error (mean ± std, max): x = 39 ± 30, max 197; y = 21 ± 20, max 82; z = 42 ± 32, max 130 (mm).
- 3D Euclidean error: mean 70 ± 33, max 212 mm.
- Frame-to-frame jump: pred = 19.5 mm/frame, GT = 29.5 mm/frame.

**Observation worth tracking:** pred trajectory is *smoother* than GT (~33% lower frame-to-frame jump). Likely cause: heatmap-based pose estimators regress to the mean on fast pose changes. Cross-check after full run — if the effect holds across all 32,800 frames, that's regression-to-mean bias of the upstream model. **For Phase 1 forecasting this works in our favor**: smoother pred → easier to forecast.

**Effect of Option C vs prior Option A run:** can't compare directly (no Option A run was completed). The CoM_z values (~-887 mm) sit in the right range either way.

**Decisions confirmed:** keep `num_workers=4` for the full run; keep refactored script structure.

Smoke output saved to `./train/predictions/com/smoke.p`. Will keep it for now as a reference; can delete after the full-run results are in.

### Full run launch (2026-05-25, tenth turn — running in background)

Launched: `python compute_com.py --num_workers 4` (defaults otherwise) from `train/`.
Output piped via `Tee-Object` to `c:\Users\haoji\IntelligentCarpet-1\full_run.log`.

**⚠ Gotcha noted for future long-running scripts:** Python's `print()` block-buffers stdout when the destination is a pipe (here: PowerShell `Tee-Object`), so progress lines never reach the log until the process exits. Next time, launch with `python -u compute_com.py ...` to force unbuffered stdout and get live progress. Not retrofitting this run — will see the full output at completion.

Run will be auto-reported when the background PowerShell call returns. ETA: 30 min – 3 h on CPU.

### Full-run results (2026-05-25, completed)

- **Wall time: 196.25 minutes (3 h 16 min)** on CPU, batch_size 32, num_workers 4.
- Inference rate ≈ 360 ms/frame on CPU — ~10× slower than realtime.
- Output saved to `./train/predictions/com/com_results.p`.

**Aggregate CoM error (pred − GT, mm):**
| axis | mean abs | std | max |
|---|---|---|---|
| x | 45.5 | 66.4 | 1237 |
| y | 44.0 | 67.4 | 1311 |
| z | 53.9 | 70.0 | 778 |
| **3D Euclidean** | **98.4** | **105.3** | **1563** |

**Smoothness (frame-to-frame Euclidean jump):**
| | mean |
|---|---|
| Pred CoM | 23.3 mm/frame |
| GT CoM | 33.5 mm/frame |

**Sanity (CoM_z):** GT mean = −770 mm, range [−1150, +100]; Pred mean = −765 mm, range [−1072, −9].
- GT reaching CoM_z = +100 ⇒ above the coord-system floor ⇒ impossible ⇒ outlier GT keypoint frames (likely OpenPose triangulation failures).

**Sanity (carpet bounds):** GT out-of-bounds: x=14/32600 (0.04%), y=35/32600 (0.11%). Pred: 0/0. The CNN has learned the carpet limits; GT does not respect them because triangulation can drift.

**Key observations:**

1. **Heavy-tailed error distribution.** std (105 mm) ≈ mean (98 mm), and max is 16× the mean. A small number of catastrophically bad frames is contaminating the aggregate. **Median will be much smaller than mean** — need percentile stats before any conclusion about "typical" model behavior.

2. **Pred-smoother-than-GT bias is real** at scale, not a smoke artifact (23.3 vs 33.5 mm/frame). This is regression-to-mean from the heatmap-softmax decoder structure. Implication for forecasting: pred trajectories are *easier* to forecast forward in time (less high-frequency content), so a forecaster trained on pred-CoM will look better than one trained on GT-CoM. We should benchmark both.

3. **GT contamination exists.** ~49 frames (~0.15%) are physically impossible (out of carpet or below floor). These are almost certainly the same frames driving the 1.5-meter max errors. Need to identify and either flag or filter before using as forecasting ground truth.

4. **Comparison to smoke (224 frames, sessions 0-1) vs full (32,600 frames, all 136 sessions):**
   - Mean 3D error: 70 → 98 mm (+40%)
   - y error: 21 → 44 mm (doubled)
   - The smoke set was unusually clean — full set has the outlier tail.

**Encoding gotcha** in `full_run.log`: PowerShell 5.1's `Out-File` and `Tee-Object` default to UTF-16 LE, which shows up as space-padded characters when read in tools expecting UTF-8. Data is fine, just visually annoying. Next long-running script should pipe through `| Out-File -Encoding utf8` or use `> file 2>&1` with the right encoding.

### Next: post-hoc analysis (pending user go-ahead)
Plan: load `com_results.p` and compute:
- Percentile error stats (p25/p50/p75/p90/p95/p99) to see "typical" vs "tail."
- Per-session and per-subject error binning — are bad frames concentrated in specific sessions?
- Identify outlier GT frames (out-of-carpet or below-floor) — propose filtering them.
- One example CoM trajectory plot — sanity-check it looks like human motion.
- CoM autocorrelation curve — establish the **predictability ceiling** for forecasting.

### Post-hoc analysis of `com_results.p` (2026-05-25)

Script: `train/analyze_com_results.py`. Output: stats to stdout + 4 PNG plots in `train/predictions/com/plots/`.

**Frame filtering (drop unreliable frames before any aggregate stat):**
| filter | excludes | reason |
|---|---:|---|
| edge mask (first/last `window=10` frames of each session) | 2,700 (8.3%) | dataloader clamps temporal context at session boundaries → pred is identical for those frames |
| GT outlier (CoM outside carpet or z > 0 / below floor) | 102 (0.3%) | physically impossible — almost certainly OpenPose triangulation failure |
| **valid (kept)** | **29,798 (91.4%)** | |

**1. Error percentiles on valid frames (mm):**
| | 3D Euclid | \|x\| | \|y\| | \|z\| |
|---|---:|---:|---:|---:|
| p10 | 31 | 6 | 5 | 6 |
| p25 | 47 | 14 | 14 | 15 |
| **p50 (median)** | **71** | **31** | **29** | **32** |
| p75 | 105 | 52 | 51 | 61 |
| p90 | 164 | 80 | 80 | 114 |
| p95 | 220 | 102 | 103 | 167 |
| p99 | 407 | 206 | 179 | 378 |
| mean | 90 | 40 | 39 | 52 |

Median 3D error is **71 mm** — the real "typical" performance. Mean was 90 mm even after filtering, still pulled up by the long tail. Axis-wise: x ≈ y ≈ 30 mm median, **z is the worst** at 32 mm median and gets dramatically worse at p99 (378 mm) — fast vertical motion is hard for the CNN.

**2. Per-subject error (mm, median, sorted best→worst):**
| Subject | n | mean | **median** | p95 | max |
|---|---:|---:|---:|---:|---:|
| YiyueLuo | 5960 | 72 | **50** | 210 | 996 |
| LiangShi | 2520 | 67 | **57** | 140 | 388 |
| PingchuanMa | 2685 | 84 | **61** | 210 | 667 |
| TongZhang | 2671 | 97 | **71** | 260 | 1234 |
| YunzhuLi | 2700 | 93 | **75** | 221 | 472 |
| WanShou | 2693 | 113 | **76** | 292 | 1508 |
| ZeyuWu | 2520 | 93 | **77** | 204 | 573 |
| MikeFoshey | 2697 | 88 | **80** | 171 | 463 |
| MantianXue | 2681 | 108 | **89** | 232 | 836 |
| TimErps | 2671 | 107 | **94** | 218 | 628 |

**Best : worst ≈ 50 : 94 mm = 1.9× spread.** The model is not subject-invariant. Worth noting that **YiyueLuo (the easiest)** is also the project's *first author* — quite possibly the subject the model saw most during training, even though this is held-out test data. Worth checking later by looking at the training-set composition (we don't have those split files in the demo).

**3. Worst sessions (top 5):** session 87 (TongZhang), session 73 (TimErps), session 26 (MantianXue), session 98 (WanShou — 1508 mm max!), session 27 (MantianXue). Examined sess 87 — clear sit↔stand transition that the model lags through.

**4. GT outliers — distribution is concentrated, not random:**
| outlier type | count |
|---|---:|
| out-of-carpet x | 14 |
| out-of-carpet y | 35 |
| below-floor (z > 0) | 121 |
| **union** | **121 (0.37%)** |

Top 3 sessions account for **83 of 121** outliers (sess 88 TongZhang: 30, sess 58 PingchuanMa: 28, sess 73 TimErps: 25). These are likely intervals where OpenPose triangulation failed (occlusion, lighting). **Filtering these for forecasting eval is essential** — they would dominate any squared-error loss.

**5. CoM autocorrelation (session 104, YiyueLuo, 5000 frames) — the predictability ceiling:**

![autocorrelation](train/predictions/com/plots/autocorrelation.png)

| axis | lag at which GT autocorr drops below 0.5 |
|---|---|
| x | 50 frames (**5.0 s**) — slow horizontal drift |
| y | 61 frames (**6.1 s**) — slow horizontal drift |
| z | 5 frames (**0.5 s**) — fast oscillation |

The GT z autocorrelation is **periodic with a ~2 s period** — that's walking cadence (each step bobs the head/CoM up-down). Pred z is heavily damped (CNN smooths out the oscillation), so predicting *the smoothed pred trajectory* will be much easier than predicting *the true GT trajectory*. This regression-to-mean bias of the upstream model is a real research caveat.

**Persistence baseline (predict CoM(t+H) = CoM(t)) on session 104:**
| horizon | mean | median | p95 |
|---|---:|---:|---:|
| 0.5 s | 91 | 67 | 250 |
| **1.0 s** | **142** | **92** | **374** |
| 2.0 s | 193 | 100 | 583 |

**This is the target to beat for Phase 1.** At 1 s horizon: persistence gives ~92 mm median Euclidean error. Forecasters must beat this.

Compare to the instantaneous CoM estimation noise floor (71 mm median). **The 1-second persistence error (92) is only ~30% larger than the instantaneous noise floor (71).** Two implications:
- Persistence is a *strong* baseline — CoM moves slowly relative to estimation noise → forecasters need to model real motion dynamics, not just smooth.
- Most of the achievable forecasting gain at 1 s horizon is on **z** (gait cycle), where persistence ignores the periodic structure.

### Revised forecasting plan (Phase 1) — updated 2026-05-25

The analysis sharpens what we should build first:

1. **Inputs**: CoM(t-N : t) where N is to be tuned. Default candidates: N = 10 (1 s) and N = 50 (5 s, matches the x/y autocorr drop-off).
2. **Output**: full 1-second future trajectory, CoM(t+1 : t+10) — 10 frames at 10 Hz. (Tentative — confirm with user before coding.)
3. **Frame filtering**: exclude edge frames + GT outliers (the 8.6% we identified). Skip frames where the next 10 frames cross a session boundary.
4. **Subject split**: held-out *time* within subjects, as user specified (Q5). Split per session: last 30% as test, first 70% as train.
5. **Baselines** (must run all four for honest comparison):
   - Persistence (CoM(t+H) = CoM(t))
   - Linear extrapolation through last 5 frames
   - Constant-velocity Kalman
   - Small GRU on CoM history
6. **Tactile-conditioned model**: fuse CoM history with a tactile-window encoder. Reuse the existing 2D-CNN frontend (frozen) so we don't retrain pose recognition. Phase 3 only if 5 motivates it.
7. **Reporting**: report median, p95, and mean — **median is the right primary metric** given the long tails we just saw. Per-axis (x, y, z) breakdown is mandatory: the z story is fundamentally different from x/y.

**Plots produced (under `train/predictions/com/plots/`):**
- `trajectory_best.png` — LiangShi sess 11, standing still
- `trajectory_worst_clean.png` — TongZhang sess 87, sit↔stand transition
- `trajectory_worst_with_outliers.png` — WanShou sess 98, includes the 1508 mm max outlier
- `autocorrelation.png` — predictability ceiling

### Script reorganization (2026-05-25)

User can't run scripts directly because they don't activate the venv. Two issues addressed:

**1. How to invoke the venv (documented for future):**
- Direct: `.\venv\Scripts\python.exe <script>` — always works.
- Activate-then-run: `.\venv\Scripts\Activate.ps1` then `python <script>`. May need `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once on Win11.
- VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → `.\venv\Scripts\python.exe`.

**2. Folder reorg per user request:**
Moved all CoM-related scripts into a new `train/com/` directory:
- `train/compute_com.py`        → `train/com/compute_com.py`
- `train/add_subject_to_csv.py` → `train/com/add_subject_to_csv.py`
- `train/analyze_com_results.py`→ `train/com/analyze_com_results.py`

**Path-handling changes** (so scripts can run from any cwd):
- Each script computes `_HERE = dirname(__file__)`, `_TRAIN = dirname(_HERE)`.
- `compute_com.py` adds `_TRAIN` to `sys.path` so `from threeD_model_final import ...` and `from threeD_dataLoader_batch import ...` still resolve.
- Default argparse paths for `--exp_dir`, `--test_dir`, `--out` use `_TRAIN` rather than `./` relative-cwd defaults.
- `add_subject_to_csv.py` and `analyze_com_results.py` use `_TRAIN` for their data/output paths.

**Outputs stay** under `train/predictions/com/` (unchanged). Only the *scripts* moved.

**Verified** by running `compute_com.py --help` and the full `add_subject_to_csv.py` from cwd `C:\Users\haoji` (deliberately not inside the project). Both worked unchanged.

**How to run now (from any shell, any cwd):**
```powershell
.\venv\Scripts\python.exe train\com\compute_com.py            # full inference (~3 h CPU)
.\venv\Scripts\python.exe train\com\compute_com.py --max_batches 7   # smoke test (~30 s)
.\venv\Scripts\python.exe train\com\add_subject_to_csv.py     # adds subject column to CSV
.\venv\Scripts\python.exe train\com\analyze_com_results.py    # percentiles + plots
```

### Outputs relocation (2026-05-25)

User asked for outputs to also live inside the new com/ folder, named `output/`.

- Moved everything from `train/predictions/com/*` → `train/com/output/`.
- Removed the empty `train/predictions/com/` and `train/predictions/` directories.
- Updated default paths in the 3 scripts:
  - `compute_com.py` — `--out` default now `_HERE/output/com_results.p` (where `_HERE = train/com/`).
  - `add_subject_to_csv.py` — `RESULTS`, `OUT_CSV` use `_HERE/output/`.
  - `analyze_com_results.py` — `RESULTS`, `PLOTS_DIR` use `_HERE/output/`.
- `TESTDIR` (the demo dataset path) still anchors to `_TRAIN` (= `train/`) since the dataset lives there.

Verified: `add_subject_to_csv.py` reads from new paths and writes to new paths cleanly.

Final layout:
```
train/com/
├── compute_com.py
├── add_subject_to_csv.py
├── analyze_com_results.py
└── output/
    ├── com_results.p
    ├── com_results.csv
    ├── com_results_with_subject.csv
    ├── smoke.p
    └── plots/
        ├── autocorrelation.png
        ├── trajectory_best.png
        ├── trajectory_worst_clean.png
        └── trajectory_worst_with_outliers.png
```

### GPU planning discussion (2026-05-25)

User is considering university research-computing cluster for the forecasting work. Recommendation logged:

| Phase | Training surface | CPU sufficient? | GPU value |
|---|---|---|---|
| 1 — Baselines (persistence / linear / Kalman) | none | ✅ trivial | none |
| 1 — Small GRU on CoM history | ~10k params, ~30k examples | ✅ seconds–minutes on CPU | minor (~5×) |
| 2 — Tactile-conditioned forecaster, frozen CNN backbone | forecaster head only | ✅ if features cached; ⚠ if re-extracting features (~3 h CPU pass) | moderate |
| 3 — World models (JEPA / Dreamer) | full encoder + dynamics, self-sup losses | ❌ impractical | major (10–100×) |

**Plan**: start Phase 1 on CPU immediately; apply for cluster access in parallel (handles any queue/setup delay); by Phase 2/3 access will be ready. **Alternative on-machine**: Intel Arc Pro 140T (16 GB) supports PyTorch XPU backend in PT 2.5+. Setup is moderate (Intel oneAPI runtime + minor code changes). Could be enough for Phase 1–2 without leaving the laptop; Phase 3 still benefits from a cluster.

**Decision deferred** until user requests access status. Not blocking Phase 1.

### Forecasting plan — Phases 1 / 2 / 3 (consolidated 2026-05-25)

User asked for an explicit phase breakdown. **This supersedes** the scattered Step 0–6 / "Revised forecasting plan" descriptions earlier in this log.

#### Phase 0 — DONE
Pipeline `tactile → CNN (frozen ckpt) → 21 keypoints → segmental weighting → 1 CoM point` is in place.
- Output: `train/com/output/com_results.p` — 32,600 frame pairs of (pred CoM, GT CoM).
- 71 mm median instantaneous 3D error on valid frames.
- Autocorrelation: x / y stay > 0.5 for 5–6 s; z drops below 0.5 in 0.5 s (gait cadence).
- Persistence baseline at 1-s horizon: **92 mm median** — the bar to beat.

#### Phase 1 — Establish the forecasting floor
**Question:** can simple history-only models predict CoM(t+1 : t+10) (1-second horizon, 10 frames at 10 Hz)?

| Model | Input | What it learns | Training compute |
|---|---|---|---|
| Persistence | CoM(t) | nothing (predicts no change) | 0 — already have it (92 mm median) |
| Linear extrapolation | last 5 CoM frames | line fit, project H steps | 0 (closed-form) |
| Constant-velocity Kalman | last 2 CoM frames | position + velocity filter | 0 (closed-form) |
| Small GRU on CoM history | CoM(t−10:t) or (t−50:t) | learned trajectory dynamics | seconds–minutes |

**Success criterion:** at 1 s horizon, must beat persistence's 92 mm median. Most of the achievable gain is on z (gait cadence) where persistence fails; x / y persistence is strong.

**Compute:** closed-form baselines need nothing; GRU is trivially small. **User opted to defer training until GPU access is available.**

#### Phase 2 — The central question: does tactile add value?
**Question:** does tactile data improve forecasting beyond CoM history alone?

| Variant | Input | Hypothesis |
|---|---|---|
| **β tactile-only** | tactile(t−N:t) → CoM(t+H) | tactile encodes pre-movement weight shifts; ground reaction force *precedes* CoM displacement |
| **γ fused** | tactile + CoM history → CoM(t+H) | best of both |
| **δ keypoints-only** | all 21 keypoints' history → CoM(t+H) | richer pose history may be enough — no tactile needed |

**Architecture:** reuse the *frozen* 2D-CNN front-end from `tile2openpose_conv3d` for tactile encoding (no retraining the heavy 3D-conv decoder), bolt a small GRU + 3-dim MLP head on top.

**Success criterion:** at least one tactile variant (β or γ) beats Phase 1 GRU **on the z axis specifically** — the axis where tactile (sensing step rhythm) is most likely to add value. x / y are already easy via persistence-style methods.

**Compute:** one-time feature-extraction pass (already cached in `com_results.p`'s `kp_pred_mm`). Forecaster training small; GPU gives ~5× iteration speedup. **Will wait for GPU.**

#### Phase 3 — World models (stretch goal)
**Question:** can we learn a general-purpose latent dynamics model that predicts the tactile + CoM future jointly?

| Approach | Pros | Cons |
|---|---|---|
| **JEPA-style** *(recommended)* | sample-efficient (predicts in embedding space, not pixel); no action conditioning needed; modest data requirements | risk of representation collapse — needs EMA + variance / covariance regularizer |
| Dreamer V3 RSSM | full generative; multi-horizon rollout natural | designed for action-conditioned RL; wasted capacity here since human volition is unobserved |

**Success criterion:** meaningfully outperform Phase 2 on **multi-second horizons (≥ 2 s)** OR transfer well to **cross-subject** setting (user's Q5 "ultimate" goal).

**Decision point:** only attempt Phase 3 if Phase 2 shows tactile adds real value. No point building a world model if simple history-only baselines already win.

**Compute:** heavy self-supervised pre-training on raw tactile sequences. **GPU mandatory** (10–100× slower on CPU). This is the phase where the university cluster genuinely matters.

### Current status (2026-05-25, after this turn)
- Phase 0: complete.
- Phases 1, 2, 3: designed; **training deferred pending GPU access** (user's call).
- **What can still be done on CPU in the meantime, without training**, if user wants to keep momentum:
  1. **Forecasting dataset prep** — turn `com_results.p` into clean train / val / test splits per session, with the edge-filtering and outlier-filtering criteria already established. Output a single `.npz` with arrays `X_history`, `Y_future`, `subject`, `session_id`, `frame_idx`. ~30 min of work, no training.
  2. **Closed-form Phase 1 baselines** (persistence, linear, constant-velocity) — pure numpy, no training. Gives the Phase 1 floor numbers immediately.
  3. **Skeleton code** for the GRU + tactile-conditioned forecasters, ready to run the moment GPU access lands.
  4. (Optional) Small Phase 1 GRU trial on CPU just to validate the pipeline end-to-end before scaling on GPU. Would take minutes.

**Awaiting:** user obtains GPU access (university cluster, or sets up local Intel XPU). Then start Phase 1 GRU + Phase 2 training.

### Phase 1 training executed (2026-05-25) — UPDATE: ran on CPU after all

User reversed earlier "wait for GPU" decision: "yes start phase 1 training." Script implemented, trained, evaluated.

**Script**: `train/com/train_phase1.py`. Single self-contained script implementing all four forecasters + analysis + plots. Hyperparameters: HISTORY=100 (10 s), HORIZON=10 (1 s), GRU hidden=64, 1 layer, 50 epochs, Adam lr=1e-3, batch 256, seed 42. Standardized inputs/outputs by train-set mean/std. Per-session 70/30 time-split (chronological).

**Dataset built**: 17,218 samples (11,963 train / 5,255 test). 0 samples filtered (outlier window check found no qualifying outliers within any candidate (history + future) span — the 121 outlier frames are concentrated in a few sessions, and the window check skips entire candidate centers near them).

**Headline results** (1-second horizon, n_test = 5255):

| method | median 3D (mm) | mean 3D | p95 3D | skill score |
|---|---:|---:|---:|---:|
| persistence    | **54.1** | 92.2  | 327.2 | **1.000** (baseline) |
| linear         | 89.2     | 135.2 | 423.4 | 1.648 |
| const_velocity | 75.8     | 123.7 | 388.1 | 1.401 |
| gru            | 61.1     | 88.8  | 261.2 | 1.128 |

**Headline finding**: **persistence beats every other method at the dataset median**, including the trained GRU (by 13%). Only the GRU is within shouting distance.

**Per-horizon breakdown** (median 3D error, mm):
- h=0.1 s: const_velocity wins (13.2) — short-range projection is reliable.
- h=0.5 s: persistence and GRU tied (~65 mm).
- h=1.0 s: persistence wins decisively (85.7), GRU degrades gracefully (112), linear/const_vel diverge (160/183).

**Per-axis** (median, mm averaged across horizons):
| method | x | y | z |
|---|---:|---:|---:|
| persistence | 21.2 | 19.7 | **9.4** |
| linear | 34.5 | 33.4 | 16.4 |
| const_vel | 27.1 | 30.7 | 24.4 |
| gru | 25.8 | 24.7 | 20.1 |

**⚠ Correction to earlier autocorrelation interpretation**: I previously claimed z would be the hardest axis (autocorr dropped below 0.5 in 0.5 s). At the dataset level, **z is the easiest** — persistence z error is just 9.4 mm. Reason: the autocorr was computed on a single 5000-frame *walking* session. The full test set is dominated by short low-motion sessions where z barely changes. The walking conclusion still holds *locally*, but doesn't dominate the aggregate.

**GRU details**:
- 15,198 params (tiny).
- Train loss still decreasing at epoch 50 (0.040 in normalized units, no plateau). **Model is undertrained, not overfit.**
- Val curve tracks train curve exactly → no overfitting.
- Per-subject median (GRU): TongZhang 47, YiyueLuo 50, LiangShi 62, ..., MantianXue 76, TimErps 81. 1.7× spread, roughly matches the upstream CoM-estimation spread.
- YiyueLuo contributes 1736 / 5255 (33%) of test samples → dominates aggregate stats.

**Interpretation — what this tells us about Phase 2**:
- The 1-s persistence error (86 mm median) is the **bar Phase 2 must beat**. It is genuinely hard because CoM positions are highly autocorrelated on this dataset (mostly static recordings).
- Three explanations for why GRU lost to persistence:
  1. Most test samples are low-motion (people standing still) where persistence is near-optimal.
  2. The GRU is undertrained — needs more capacity or epochs.
  3. The model regresses to the mean on hard cases — its p95 (261) is *better* than persistence (327), so it actually helps in tail cases, but loses the median where persistence is trivially good.
- **Phase 2 thesis is still valid**: tactile carries ground-reaction-force information that precedes motion. History-only forecasting is information-limited, not architecture-limited. Phase 2 has clear room to add value, especially for transition events (sit↔stand, gait initiation) where history alone misses the cue.

**Outputs** (under `train/com/output/phase1/`):
- `metrics.json` — full structured numbers
- `training_curve.png` — smooth decreasing curve, no overfit
- `error_vs_horizon.png` — main comparison, all axes
- `error_vs_horizon_z.png` — z axis only
- `example_trajectories.png` — 5 random test samples
- `gru_model.pt` — trained state dict for re-use

**Open items raised by this run**:
1. Consider re-evaluating on a *high-motion subset* (frames where CoM velocity > some threshold). Persistence will be much weaker there, and forecasting methods will look better. Probably the more scientifically informative evaluation.
2. Should we train the GRU longer (100+ epochs) before declaring its ceiling? Right now it's not clearly converged.
3. The 86 mm persistence-at-1s number is **the official Phase 1 floor** for the Phase 2 comparison.

### Ranked next-step plan with GPU now available (2026-05-28)

User now has access to Notre Dame's CRC (Center for Research Computing) GPU resources. The 6 candidate next steps, ranked by priority for the project's central question — *does the IntelligentCarpet pipeline forecast next-second CoM beyond what CoM history alone can do?*

| # | Step | Why this rank | Effort | GPU? |
|---|---|---|---:|:-:|
| 1 | **CRC env setup + Phase 0 GPU sanity** — re-run compute_com.py with --max_batches 7 on GPU. Confirm same numbers as CPU and ~30× speedup. | Prerequisite for everything below. Surface blockers (CUDA version, wheels, storage) on day 1, not week 1. | 0.5–2 days | ✅ |
| 2 | **Phase 2 — tactile-conditioned forecaster** (β tactile-only, γ fused, δ keypoints-only). Frozen 2D-CNN front-end + GRU head. | Central scientific question. Without this the project has no headline result. | 2–4 days | ✅ heavy |
| 3 | **High-motion subset re-evaluation** — filter test samples where GT CoM velocity > threshold. Re-compute Phase 1 metrics on this subset. | Test set is 70% low-motion frames where persistence is information-theoretically unbeatable. Without this, Phase 2 gains will look smaller than deserved. | <1 hour | ❌ |
| 4 | **Strengthen Phase 1 GRU floor** — hidden 256–512, 2–3 layers, 200+ epochs, delta-prediction. Confirm true history-only ceiling. | Guards against the trap of "Phase 2 looks good only because Phase 1 was undertrained". Only critical if Phase 2 result is ambiguous. | 1 day | ✅ |
| 5 | **Cross-subject leave-one-out** — re-do Phase 1 + 2 with held-out subjects. Tests Q5 ultimate goal. | Premature without a working in-subject forecaster. YiyueLuo's 33% sample dominance complicates the balance. | 2–3 days | ✅ |
| 6 | **Phase 3 — world models (JEPA)** | Only attempted if Phase 2 establishes that tactile adds real value. Otherwise scaling up a hypothesis that failed its small-model test. | weeks | ✅ heavy |

**Sequencing**: do (1), then (2) and (3) in parallel. After Phase 2 results land:
- clean win for tactile → proceed to (5) for the Q5 ultimate goal
- ambiguous → do (4) and re-evaluate
- clear lift, want a flashier method → consider (6)
- no lift → write up honest negative result

**Pushback on record**: do NOT skip step 3 even with GPU available. Test-set imbalance toward static recordings will mask genuine Phase 2 gains. Half-hour of motion-filtering pays back 10× in interpretive clarity.

### Step 1 in progress (2026-05-28): CRC setup
Following https://docs.crc.nd.edu/new_user/quick_start.html. User is at NYU (jh9141@nyu.edu) but has access to ND CRC (likely via collaboration).

**Step-by-step plan delivered to user. Sequenced so each stage's success gates the next.**

| Stage | Goal | Key commands |
|---|---|---|
| A | Account access. Verify ND NetID, log into okta.nd.edu (sets up DUO), install MobaXterm on Windows. | — |
| B | First SSH login. | `ssh netid@crcfe01.crc.nd.edu`, then `quota`, `whoami` |
| C | One-time conda init. | `module load conda && conda init && source ~/.bashrc` |
| D | Create env + install PyTorch CUDA. | `conda create -n carpet python=3.10 -y`; `module load cuda/12.1`; `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`; `pip install numpy opencv-python matplotlib scipy h5py progressbar2` |
| E | GPU verification (interactive session, no project data). | `qrsh -q gpu -l gpu_card=1 -pe smp 1`; then `python -c "import torch; print(torch.cuda.is_available())"` |
| F | Code transfer + data. | `git clone https://github.com/Jiayi459/IntelligentCarpet.git`; data: re-download from Dropbox links in README.md via `wget` directly on CRC (faster than laptop→cluster). Email `CRCSupport@nd.edu` for `/scratch365` access — 22 GB dataset shouldn't sit in AFS home. |
| G | Smoke test on GPU. | `python train/com/compute_com.py --max_batches 7` (~30 s on GPU vs 20 s on CPU; full Phase 0 reproduces 3 h CPU run in 3–10 min) |
| H | Batch jobs for Phase 2. | UGE template: `#$ -q gpu -l gpu_card=1 -pe smp 4 -l h_rt=04:00:00`; submit with `qsub`, monitor with `qstat -u $USER` |

**Blocking questions returned to user:**
1. Does the user actually have an ND NetID? (NYU email + ND CRC access is unusual — likely via collaboration, but the SSH login needs the ND NetID, not NYU credentials.)
2. Which stage is the user currently on?

**Key CRC facts captured for future reference:**
- Front-end hosts (`crcfe01`, `crcfe02`) have a **1-hour runtime limit** — never run inference / training there directly. Always go through `qrsh` (interactive) or `qsub` (batch).
- GPU queue request: `-q gpu -l gpu_card=1 -pe smp N`. Max runtime 4 days.
- Available CUDA modules: 10.0, 10.2, 11.0, 11.2, 11.6, 11.8, 12.1. **Use 12.1** to match a current PyTorch CUDA build.
- Available cuDNN modules: 7.0, 7.4, 8.0.4, 8.9.3.
- Storage: AFS home (100 GB) for code; `/scratch365/<netid>` (request via email) for datasets; `/tmp` on compute nodes is ephemeral.
- AFS retires May 2027, Panasas (scratch365) retires June 2026 — both well beyond this project's window.
- Windows users: MobaXterm Home Edition handles SSH + SFTP in one tool.

### CSV enrichment + subject-count correction (2026-05-25, eleventh turn)

**Built:** `train/add_subject_to_csv.py` — for each frame in `com_results.p`, maps to its session via `log.p`, then looks up the source filename in `fileNames.p`, parses subject / date / round out of the filename, and writes `train/predictions/com/com_results_with_subject.csv` (the original CSV + 5 metadata columns).

**Quick stat-sanity findings, before the user authorized the full post-hoc:**
- 3D Euclidean error percentiles (mm): **p25=49, p50=73, p75=110, p90=179, p95=249, p99=493**.
  - **Typical (median) error ≈ 73 mm**, not the headline 98 mm mean. Long tail driven by ~1% of frames > 0.5 m.
- **First 5 predicted CoMs are identical** (`891.5, 1070.7, -874.5`) while GT varies → **dataloader edge effect**: when the temporal window goes out-of-bounds at session starts, `window_select` clamps to the session boundary, so the first `window` frames of each session receive an identical clamped context. Need to skip the first/last `window` frames of each session before forecasting eval.

**⚠ Correction to earlier "7 subjects" claim:** The original regex `round(\d+)` only matched numeric rounds; `MantianXue` (rounds like `round1d30`) and the two `rec_…` (no `split_N_` prefix) `YiyueLuo` files fell through. **Actual count: 10 distinct subjects, matching the README's "10 people":**
| Subject | Frames | % |
|---|---:|---:|
| YiyueLuo | 6000 | 18.4 |
| MantianXue | 3000 | 9.2 |
| MikeFoshey | 3000 | 9.2 |
| PingchuanMa | 3000 | 9.2 |
| TimErps | 3000 | 9.2 |
| TongZhang | 3000 | 9.2 |
| WanShou | 3000 | 9.2 |
| YunzhuLi | 3000 | 9.2 |
| LiangShi | 2800 | 8.6 |
| ZeyuWu | 2800 | 8.6 |

**Implication for Phase 5 (leave-one-subject-out):** with 10 subjects, this is now a real possibility from the demo set alone — no full raw-dataset download needed.

**Structural clarification:** `log.p` has 136 entries; `fileNames.p` has 136 entries. They correspond 1:1, but there are actually **135 sessions** because the dataloader uses `log[f]` as the start of session `f` and `log[f+1]` as the end — the 136th log entry (32600) is just the end boundary of the last session. `sum(diff(log))` = 32600 = total frames, confirming.

---

## Session 4 — CRC operational + Phase 2 keypoints completed (2026-05-29)

### CRC environment — fully provisioned

Stages A–G of the CRC checklist (laid out 2026-05-28) all completed. Snapshot of the working setup:
- Host: `crcfe01.crc.nd.edu` (front-end) → `qrsh -q gpu -l gpu_card=1 -pe smp 1` lands on a node like `qa-a10-023` / `qa-a10-033` (NVIDIA A10).
- NetID: `jhao3` (note: distinct from the user's NYU `@nyu.edu` email, as expected).
- Home: `/users/jhao3/` (AFS, 100 GB quota; currently ~25 GB used after extraction + outputs).
- Conda env: `carpet` at `/users/jhao3/.conda/envs/carpet/`, Python 3.10.
- PyTorch: `torch 2.5.1+cu121`, CUDA build 12.1. `cuda available` confirmed True on the GPU node.
- Modules required after every `qrsh`: `conda activate carpet && module load cuda/12.1`.

### Data transfer — one detour, now resolved

User downloaded the *entire* shared Dropbox folder (`?dl=1` on the share link) instead of the demo files only, producing an 84 GB outer zip on AFS. We staged inner zips through `/tmp` (`/tmp/jhao3_extract/`) to avoid AFS quota overflow.

The first extraction run was killed mid-Phase-3 (probably a front-end SSH disconnect, not the 1-hour process limit), leaving only ~50 of 135 sessions on disk and the `log.p` / `fileNames.p` metadata missing. The inner `singlePerson_test.zip` was *still* sitting in `/tmp` (persistent across logout), so we re-ran `python -m zipfile -e` directly and got the full dataset. Final state: `~/IntelligentCarpet/train/{ckpts,singlePerson_test}/` matches the laptop layout exactly.

### Path bug fix during smoke test

First GPU run hit `FileNotFoundError: '/users/jhao3/IntelligentCarpet/train/singlePerson_testlog.p'` (no separator). Cause: when I refactored `compute_com.py` to anchor paths to the script location, I changed the `--test_dir` default from `'./singlePerson_test/'` (trailing slash) to `os.path.join(_TRAIN, 'singlePerson_test')` (no trailing slash), but `threeD_dataLoader_batch.py:69` does string concatenation rather than `os.path.join`. Fixed by appending `os.sep` to the default in commit `8fab75c`.

### Phase 0 (instantaneous CoM) reproduced on GPU — exact match

| metric | Laptop CPU (2026-05-25) | CRC A10 GPU (2026-05-29) |
|---|---:|---:|
| x mean abs err (mm) | 45.5 | 45.5 |
| y mean abs err (mm) | 44.0 | 44.0 |
| z mean abs err (mm) | 53.9 | 53.9 |
| 3D Euclidean mean (mm) | **98.4** | **98.4** |
| GT CoM_z mean | −770 | −770 |
| Pred CoM_z mean | −765 | −765 |
| Carpet OOB x | 14 / 32600 | 14 / 32600 |
| Carpet OOB y | 35 / 32600 | 35 / 32600 |

**Wall time: 3 min 55 s on A10 vs 196 min on laptop CPU → ~50× speedup.** PyTorch + cu121 + A10 stack is healthy.

### Phase 1 reproduced on GPU — exact match

Re-ran `train/com/train_phase1.py` on the GPU. Median 3D errors all match the laptop CPU run byte-for-byte: persistence 54.1, linear 89.2, const_velocity 75.8, gru 61.1. Per-horizon table, per-axis breakdown, per-subject GRU stats — all identical. Run time about 1 minute on GPU.

### Phase 2 (δ keypoints-only) — first result is honest but negative

New script: `train/com/train_phase2_keypoints.py` (commit `8e2d01e`). Identical scaffold to Phase 1; only the GRU input dim changes from 3 (CoM) to 63 (21 keypoints × 3). Same per-session 70/30 chronological split, same outlier filter, same hyperparameters, same 17,218 / 5,255 train/test counts. Phase 1 GRU is auto-loaded and re-evaluated alongside for direct comparison.

**Headline (1-s horizon):**
| method | median 3D | mean 3D | p95 3D | skill |
|---|---:|---:|---:|---:|
| persistence | 54.1 | 92.2 | 327.2 | 1.000 |
| Phase 1 GRU (CoM-only, input 3) | 61.1 | 88.8 | 261.3 | 1.128 |
| **Phase 2 GRU (keypoints, input 63)** | **66.3** | 93.4 | 265.0 | **1.225** |

**Per-axis (mm, averaged over horizons):**
| axis | persistence | P1 GRU (CoM) | P2 GRU (kp) |
|---|---:|---:|---:|
| x | 21.2 | 25.8 | 31.3 ⬆ worse |
| y | 19.7 | 24.7 | 29.4 ⬆ worse |
| z | 9.4 | 20.1 | 20.5 same |

**Interpretation — δ falsified at this architecture:**
The extra 60 input dims (joint positions beyond CoM) **degrade x and y forecasting** without helping z. The 64-hidden GRU can't extract signal from the wider input and the inductive bias is gone (CoM is an aggregate summary that this small model can ride; raw keypoints just add noise it cannot denoise). The pose-history hypothesis fails for the cheap architecture.

Per-subject medians: YiyueLuo 50.4 → 61.9 (worse), TongZhang 47.1 → 47.6 (same), TimErps 81.1 → 72.7 (better) — pattern is messy, no clean subject-level story.

**What this rules out and what it does NOT rule out:**
- **Does** falsify: "naive 64-hidden GRU on raw 63-dim pose history beats the 3-dim CoM equivalent."
- **Does NOT** falsify: "a bigger or delta-parameterized pose-history model beats persistence."
- **Does NOT** address: "tactile carries forecastable information beyond pose history." (That's the Phase 2 β question, still open.)

### Infrastructure — git auth on CRC

CRC ↔ GitHub HTTPS password auth fails (GitHub disabled it in 2021). User to set up an SSH key on CRC (`ssh-keygen -t ed25519`, copy public key to https://github.com/settings/keys, switch remote to `git@github.com:Jiayi459/IntelligentCarpet.git`). One-time setup; after that `git push origin main` from CRC just works. Alternative: GitHub Personal Access Token used as password.

### Next move (proposed)

The δ result didn't kill pose history outright — it killed the cheap version. Two questions to resolve next, in order:

1. **Quick sanity: stronger pose-history floor (δ′)** — bigger GRU (hidden=256, layers=2), 200+ epochs, and try predicting deltas relative to `CoM(t)` rather than absolute positions. ~20 min on GPU. Establishes whether *any* history-only model can clear the persistence bar (54 mm at 1 s). If yes → tactile probably won't add much. If no → moves the burden of proof firmly onto tactile.

2. **Main event: Phase 2 β (tactile-only)** — small CNN encoder over raw tactile windows → GRU → CoM future. The central scientific question: does tactile carry information that position-history (CoM or full pose) does not? Plan:
   - Input: `tactile(t−N : t)` where each frame is (96, 96).
   - Encoder: 2D CNN (a few conv layers + GAP) → ~128-dim per frame. Train from scratch initially; later try reusing the frozen `tile2openpose_conv3d` encoder.
   - Sequence: GRU(hidden=128) over the encoded sequence → 3-dim CoM trajectory output.
   - Same train/test split as Phase 1/2. Same metrics.
   - Compute: feature extraction + training. ~30-60 min on A10.

3. **If β works (beats persistence):** straight on to γ (fused tactile + CoM history) and then to Phase 5 (leave-one-subject-out cross-subject eval).
4. **If β fails:** strong evidence the dataset is dominated by static recordings, persistence is unbeatable on aggregate, and we re-frame around high-motion subsets (which would be option 3 from the earlier ranked plan).

**My recommendation: do (1) and (2) in parallel** — (1) is cheap and clarifies the history-only ceiling; (2) is the main event regardless. Awaiting user go-ahead.

### Outputs to view (after `git pull` lands on laptop)

Under `train/com/output/phase2_keypoints/`:
- `metrics.json` — structured numbers for all three methods (persistence, Phase 1 GRU, Phase 2 GRU).
- `training_curve.png` — Phase 2 GRU train/val MSE.
- `error_vs_horizon.png` — 3D Euclidean error vs forecast horizon, all three methods overlaid.
- `error_vs_horizon_z.png` — same restricted to the z axis.
- `comparison_phase1_vs_phase2.png` — side-by-side bar chart of median error and skill score.
- `gru_model.pt` — trained Phase 2 GRU weights.


# IntelligentCarpet — Repository Structure & Code Guide

## Project Overview

**IntelligentCarpet** (CVPR 2021) infers 3D human body pose from pressure signals captured by a smart carpet. A person stands/moves on a 96×96-sensor tactile carpet; the system predicts 21 body keypoint positions in 3D space using a CNN that maps tactile frames → 3D volumetric heatmaps → (x, y, z) keypoint coordinates.

---

## File Tree

```
intelligentCarpet/
├── README.md                          # Project overview, dataset links, usage guide
├── environment.yml                    # Conda environment (Python 3.6, PyTorch 1.6, CUDA 10.2)
├── heatmap_from_keypoint3D.py         # DATA PREPROCESSING: 3D keypoints → 3D heatmaps
├── img/
│   └── teaser.PNG                     # Teaser figure used in README
└── train/
    ├── utils_func.py                  # SHARED UTILITIES: I/O, math, drawing helpers
    ├── threeD_dataLoader.py           # DATASET: in-memory loader (single-file, diff-task variant)
    ├── threeD_dataLoader_batch.py     # DATASET: on-disk lazy loader (batch/streaming variant)
    ├── threeD_model_final.py          # MODEL: 2D CNN encoder + 3D CNN decoder + SpatialSoftmax3D
    ├── threeD_train_final.py          # MAIN: training loop, evaluation, export logic
    ├── threeD_viz_image.py            # VISUALIZATION: save per-frame images (tactile + skeleton)
    ├── threeD_viz_video.py            # VISUALIZATION: save prediction videos (GT vs pred)
    ├── link_min.p                     # DATA: min Euclidean distance for each of 20 body links
    └── link_max.p                     # DATA: max Euclidean distance for each of 20 body links
```

---

## File-by-File Explanation

### `heatmap_from_keypoint3D.py`  (root)

**Purpose:** Offline data preprocessing — converts raw 3D keypoint coordinates into 3D Gaussian heatmaps stored as pickle files.

**Key functions:**

| Function | What it does |
|---|---|
| `heatmap_from_keypoint(keypoint_path, xyz_range, heatmap_size)` | Loads `keypoint_transform.p`, normalizes coordinates into a 20×20×18 voxel grid, then for each of 21 body joints computes a Gaussian-shaped probability volume centered on that joint. Returns `(keypoint_coords, heatmap)`. |
| `gaussian(dis, mu, sigma)` | Gaussian function used to generate soft heatmap peaks. |
| `remove_keypoint_artifact(data, threshold)` | Clamps outlier keypoint coordinates to valid range. |
| `round_to_1(data, sig)` | Rounds small heatmap values to zero to reduce file size. |
| `plotKeypoint(data)` | Renders a 3D skeleton (21 joints, 20 links) to a numpy image for inspection. |
| `plot3Dheatmap(data, separate)` | Renders the 3D voxel heatmap as a scatter plot for inspection. |

**Output files written:**
- `keypoint_transform.p_coord.p` — normalized keypoint coordinates
- `keypoint_transform.p_heatmap3D.p` (or chunked `_heatmap3D_N.p`) — 3D heatmap arrays shaped `(T, 21, 20, 20, 18)`

**Dependency:** imports `utils_func` from the same root directory (not `train/`).

---

### `train/utils_func.py`

**Purpose:** Shared low-level utility functions imported by almost every other file in `train/`.

| Function | What it does |
|---|---|
| `findFrame(ts_target, ts_set)` | Nearest-neighbor timestamp lookup — finds the index in `ts_set` closest to `ts_target`. |
| `readTs(path)` | Reads a text file of timestamps into a numpy array. |
| `tactile_reading(path)` | Reads raw tactile data from an HDF5 file (`touch#.hdf5`), returning pressure frames and timestamps. |
| `normalize(data)` | Min-max normalization to [0, 1]. |
| `normalize_with_range(data, max, min)` | Min-max normalization with explicit bounds. |
| `softmax(x)` | Numpy softmax (used to convert heatmap distances to probabilities). |
| `sigmoid(x)` | Sigmoid activation. |
| `tactile_to_3channel(tactileFrame)` | Converts a single-channel tactile frame to a 3-channel (white) image. |
| `draw_keypoint2D(...)` | Draws a 2D skeleton on a blank canvas. |
| `draw_channel(...)` | Overlays a heatmap channel in a specified color. |
| `outputImage(inputVideo, outputPath)` | Dumps every frame of a video to individual JPEG files. |

---

### `train/threeD_dataLoader.py`

**Purpose:** PyTorch `Dataset` that loads preprocessed data **entirely into RAM** at startup. Used for the "different-task" test set (`singlePerson_test_diffTask`).

**Class: `sample_data_diffTask(Dataset)`**

- `__init__`: Iterates over all `.p` pickle files in a directory, concatenates touch frames, heatmaps, and keypoints into three large numpy arrays held in memory.
- `__getitem__`: Returns a temporal window of tactile frames around index `idx` (using `window_select`), plus the corresponding heatmap, keypoint, and single tactile frame.
- `window_select(data, timestep, window)`: Slices a symmetric temporal window of `2*window` frames around `timestep` from a pre-loaded array.
- `get_subsample(touch, subsample)`: Spatially downsamples tactile frames by averaging over `subsample×subsample` blocks.

**Imported by:** `threeD_train_final.py`

---

### `train/threeD_dataLoader_batch.py`

**Purpose:** PyTorch `Dataset` that loads data **lazily from disk** one sample at a time. Used for the main train/val/test sets where the full dataset is too large to fit in RAM.

**Class: `sample_data(Dataset)`**

- `__init__`: Reads a `log.p` file that maps dataset indices to on-disk folder offsets. Does not load any data at init time.
- `__getitem__`: Given a global index, determines which sub-folder it belongs to, then calls `window_select` to load the necessary frames from individual `N.p` files.
- `window_select(log, path, f, idx, window)`: Loads tactile frames from disk for a `2*window`-frame window around `idx`. Handles boundary conditions (start/end of a session).
- `get_subsample`: Same spatial downsampling as in `threeD_dataLoader.py`.

**Key difference from `threeD_dataLoader.py`:** Reads each sample from disk on demand (slower but memory-efficient); supports multiple sessions concatenated via the `log.p` index.

**Imported by:** `threeD_train_final.py`

---

### `train/threeD_model_final.py`

**Purpose:** Defines the neural network architecture.

**Class: `tile2openpose_conv3d(nn.Module)`**

The main model. Takes a batch of tactile frames `(B, 2*window, 96, 96)` and produces a 3D heatmap `(B, 21, 20, 20, 18)`.

Architecture:
1. **2D CNN Encoder** (`conv_0` → `conv_6`): Six Conv2D blocks with LeakyReLU + BatchNorm, progressively doubling channels (32→64→128→256→512→1024), with MaxPool at layers 1, 3, 6. Final spatial size: `10×10`.
2. **2D→3D Lifting**: The `10×10` feature map is broadcast into `10×10×9` by repeating along a new z-axis, then a learned depth-position embedding (0..1) is concatenated as a 1025th channel.
3. **3D CNN Decoder** (`convTrans_0` → `convTrans_4`): Three Conv3D blocks + one ConvTranspose3D upsampling (×2) + two more Conv3D blocks. Output: `(B, 21, 20, 20, 18)` with Sigmoid activation — one probability volume per body joint.

**Class: `SpatialSoftmax3D(nn.Module)`**

Converts the 3D heatmap output into 3D keypoint coordinates (differentiable argmax).

- Registers a 3D meshgrid of (x, y, z) positions as buffers.
- `forward(feature)`: Takes `(B*21, H, W, D)` heatmap, computes the softmax-weighted expected position in each dimension, returns `feature_keypoints` of shape `(B, 21, 3)` and the reshaped heatmap.

**Imported by:** `threeD_train_final.py`

---

### `train/threeD_train_final.py`

**Purpose:** Main entry point — orchestrates training, evaluation, and all export modes.

**Command-line arguments:**

| Argument | Default | Description |
|---|---|---|
| `--exp_dir` | `./train` | Root directory for checkpoints, logs, predictions |
| `--exp` | `singlePeople` | Experiment name (used in checkpoint filenames) |
| `--lr` | `1e-4` | Adam learning rate |
| `--batch_size` | `32` | Training batch size |
| `--window` | `10` | Temporal context: ±10 frames around each sample |
| `--epoch` | `500` | Number of training epochs |
| `--ckpt` | `singlePerson_0.0001_10_best` | Checkpoint file to load for eval |
| `--eval` | `True` | Run in evaluation mode |
| `--test_dir` | `./singlePerson_test/` | Path to test dataset |
| `--exp_image` | `False` | Export per-frame prediction images |
| `--exp_video` | `False` | Export prediction video |
| `--exp_data` | `False` | Export raw prediction data as pickle |
| `--exp_L2` | `False` | Export L2 keypoint distance to ground truth |
| `--linkLoss` | `True` | Include bone-length constraint in loss |

**Key functions:**

| Function | What it does |
|---|---|
| `weights_init(m)` | Xavier-style initialization for Conv and BatchNorm layers. |
| `check_link(min, max, keypoint, device)` | Computes bone-length violation loss: penalizes predicted bone lengths outside the `[link_min, link_max]` range loaded from pickle files. Uses BODY_25 skeleton pairs. |
| `remove_small(heatmap, threshold, device)` | Zeros out heatmap values below a threshold to suppress noise. |
| `get_spatial_keypoint(keypoint)` | Converts normalized [0,1] keypoint coordinates back to real-world cm coordinates. |
| `get_keypoint_spatial_dis(GT, pred)` | Computes the per-keypoint spatial displacement (in cm) between ground truth and prediction. |

**Training loop:**
1. Loads `sample_data` (batch loader) for train/val sets.
2. Forward pass: `model(tactile)` → heatmap → `SpatialSoftmax3D` → keypoints.
3. Loss = weighted MSE heatmap loss + optional bone-length constraint loss.
4. Saves best checkpoint when validation loss improves.

**Evaluation loop:**
1. Loads checkpoint, runs model in `eval()` mode.
2. Optionally calls `generateImage`, `generateVideo`, saves raw data or L2 distances.

**Imports from:** `threeD_model_final`, `threeD_dataLoader_batch`, `threeD_dataLoader`, `threeD_viz_video`, `threeD_viz_image`

---

### `train/threeD_viz_image.py`

**Purpose:** Generates and saves per-frame visualization images comparing ground truth vs. predicted pose.

**Key functions:**

| Function | What it does |
|---|---|
| `generateImage(data, path, c, base)` | Main export function. For each frame in a batch: saves tactile scatter plot, GT/pred heatmaps as 3D scatter plots, and GT/pred skeleton renders. Writes 6 images per frame. |
| `plotKeypoint(data, tactile, scale, tile_coord, tactile_frame, topVeiw, keypoint)` | Renders a 3D skeleton on a matplotlib 3D axis. Can optionally overlay the tactile frame as a scatter on the carpet plane. Uses a normalized coordinate system scaled to 0–190 cm. |
| `plot3Dheatmap(data)` | Renders all 21 joint heatmap volumes overlaid in one 3D scatter plot. |
| `plot_touch(touch_frame, save_path)` | Saves a 2D heatmap image of the tactile frame using viridis colormap. |
| `plot_touch2(touch_frame)` | Renders the tactile frame as a bubble plot (point size proportional to pressure). |
| `rotate(touch, heatmap, keypoint, degree)` | Rotates tactile frame, heatmap, and keypoints by a multiple of 90° around z-axis (for canonical orientation). |
| `remove_samll(data)` | Zeros out heatmap values below 0.05 for cleaner visualization. |

**Imported by:** `threeD_train_final.py`

---

### `train/threeD_viz_video.py`

**Purpose:** Generates AVI video files comparing GT and predicted pose frame-by-frame.

**Key functions:**

| Function | What it does |
|---|---|
| `generateVideo(data, path, heatmap, tile_coord)` | Main export function. Renders each frame as a composite image (GT heatmap, pred heatmap, GT skeleton, pred skeleton, side-by-side overlay), writes to an MJPEG AVI at 10 fps. |
| `plotKeypoint(data, tactile, scale, tile_coord, tactile_frame, topVeiw, GT_pred_compare)` | Similar to the image version but uses real-world coordinate ranges (-100 to 1900 mm). Supports `GT_pred_compare` mode which overlays two skeletons (GT in black, pred in color). |
| `plot3Dheatmap(data)` | Same as image version — 3D scatter of all 21 joint heatmaps. |
| `remove_samll(data)` | Zeros heatmap values below 0.01. |

**Imported by:** `threeD_train_final.py`

---

### `train/link_min.p` and `train/link_max.p`

**Purpose:** Precomputed constraint data for the bone-length loss. Each is a list of 20 values (one per body link in BODY_25 pairs) representing the minimum and maximum squared Euclidean distance (in normalized coordinates) observed in training data. Used by `check_link()` in `threeD_train_final.py`.

---

## Data Flow Diagram

```
Raw Data (dataset/)
  touch_normalized.p  ──────────────────────────────────────────────────┐
  keypoint_refined.p  ──► heatmap_from_keypoint3D.py                    │
                              │                                          │
                              ▼                                          │
                    keypoint_transform.p_coord.p                        │
                    keypoint_transform.p_heatmap3D.p                    │
                              │                                          │
                              ▼                                          ▼
                    threeD_dataLoader_batch.py ◄── (per-sample .p files on disk)
                    threeD_dataLoader.py       ◄── (for diffTask test sets)
                              │
                              ▼
                    threeD_train_final.py   ◄── link_min.p, link_max.p
                         │        │
                         ▼        ▼
               threeD_model_final.py     (tile2openpose_conv3d + SpatialSoftmax3D)
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
     threeD_viz_image.py    threeD_viz_video.py
     (saves JPG images)     (saves AVI video)
```

---

## Module Dependency Graph

```
threeD_train_final.py
  ├── threeD_model_final.py      (model architecture)
  ├── threeD_dataLoader_batch.py (primary dataset loader)
  ├── threeD_dataLoader.py       (alternate dataset loader)
  ├── threeD_viz_image.py        (image export)
  │     └── utils_func.py
  └── threeD_viz_video.py        (video export)
        └── utils_func.py

heatmap_from_keypoint3D.py
  └── utils_func.py  (must be in same directory, not train/)
```

---

## Key Data Shapes

| Variable | Shape | Description |
|---|---|---|
| `touch` / `tactile` | `(T, 96, 96)` | Normalized pressure readings from the 96×96 carpet |
| `tactile` (with window) | `(B, 2*window, 96, 96)` | Temporal context window fed to the model |
| `heatmap` | `(T, 21, 20, 20, 18)` | GT 3D Gaussian heatmap, one 20×20×18 volume per joint |
| `keypoint` | `(T, 21, 3)` | 3D joint positions normalized to [0, 1] |
| Model output (heatmap) | `(B, 21, 20, 20, 18)` | Predicted heatmap |
| `keypoint_out` | `(B, 21, 3)` | Predicted keypoints from SpatialSoftmax3D |

The 3D voxel space corresponds to a real-world region of approximately 2000×2000×1800 mm, discretized into 20×20×18 bins (~100 mm/bin in x/y, ~100 mm/bin in z).

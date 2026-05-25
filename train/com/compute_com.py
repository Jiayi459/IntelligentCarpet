"""
Compute Center of Mass (CoM) from the IntelligentCarpet model.

For each test frame:
  - Runs the carpet CNN to get predicted 3D keypoints (21 x 3)
  - Uses the paired GT keypoints (from camera Openpose) already in the dataset
  - Converts both to mm via the dataset's coordinate system
  - Applies De Leva / Winter (2009) segmental CoM model to both
  - Reports per-axis error and saves results

Limitations documented:
  - Head (0.081 body mass) included via Option C — extrapolated along the trunk axis:
      head_com = Neck + 0.15 * (Neck - MidHip)
    Derivation: Winter (2009) Table 4.1 — head segment length ≈ 0.13·stature,
    head-CoM offset from neck ≈ 0.34·head_segment_length ≈ 0.044·stature
    ≈ 0.15·trunk_length (trunk ≈ 0.30·stature). Robust to subject scale.
  - Hands (0.006 each) excluded: wrists only; combined effect < 1 cm on CoM.
  - Accounted mass fraction: 0.988 of total body mass.
  - No force-plate ground truth: comparison is pred-CoM vs GT-keypoint-derived CoM.

Run from anywhere (paths anchored to script location):
    python train/com/compute_com.py
    python train/com/compute_com.py --max_batches 7   # quick smoke test
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

# This script lives in train/com/; sibling modules live in train/.
_HERE  = os.path.dirname(os.path.abspath(__file__))   # .../train/com
_TRAIN = os.path.dirname(_HERE)                       # .../train
sys.path.insert(0, _TRAIN)

from threeD_model_final import SpatialSoftmax3D, tile2openpose_conv3d
from threeD_dataLoader_batch import sample_data


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

_B          = np.array([-100.0, -100.0, -1800.0])   # origin offset in mm
_RESOLUTION = 100.0                                  # mm per voxel bin
_MAX_IDX    = 19.0                                   # grid is 0–19

def to_mm(kp_norm):
    """Normalized [0,1] keypoints → mm.  Input/output: (..., 21, 3)."""
    return kp_norm * _MAX_IDX * _RESOLUTION + _B


# ---------------------------------------------------------------------------
# Segmental CoM  (De Leva 1996 / Winter 2009, Table 4.1)
# (proximal_kp_idx, distal_kp_idx, mass_fraction, alpha_from_proximal)
# ---------------------------------------------------------------------------

SEGMENTS = [
    # Head extrapolated above Neck along trunk axis (Option C).
    # Negative alpha flips the direction so the "segment CoM" lands above Neck:
    #   kp[1] + (-0.15)*(kp[8]-kp[1]) = kp[1] + 0.15*(kp[1]-kp[8])
    ( 1,  8, 0.0810, -0.150),  # Head           (extrapolated above Neck)
    ( 1,  8, 0.4970,  0.495),  # Trunk          (Neck → MidHip)
    ( 2,  3, 0.0280,  0.436),  # R Upper Arm    (RShoulder → RElbow)
    ( 3,  4, 0.0160,  0.430),  # R Forearm      (RElbow → RWrist)
    ( 5,  6, 0.0280,  0.436),  # L Upper Arm    (LShoulder → LElbow)
    ( 6,  7, 0.0160,  0.430),  # L Forearm      (LElbow → LWrist)
    ( 9, 10, 0.1000,  0.433),  # R Thigh        (RHip → RKnee)
    (10, 11, 0.0465,  0.433),  # R Shank        (RKnee → RAnkle)
    (12, 13, 0.1000,  0.433),  # L Thigh        (LHip → LKnee)
    (13, 14, 0.0465,  0.433),  # L Shank        (LKnee → LAnkle)
    (11, 18, 0.0145,  0.500),  # R Foot         (RAnkle → RBigToe)
    (14, 15, 0.0145,  0.500),  # L Foot         (LAnkle → LBigToe)
]
_TOTAL_MASS = sum(s[2] for s in SEGMENTS)   # 0.988


def compute_com(kp_mm):
    """
    kp_mm : (T, 21, 3) in mm
    returns: (T, 3) whole-body CoM trajectory in mm
    """
    weighted = np.zeros((kp_mm.shape[0], 3))
    for p, d, mass, alpha in SEGMENTS:
        seg_com = kp_mm[:, p, :] + alpha * (kp_mm[:, d, :] - kp_mm[:, p, :])
        weighted += mass * seg_com
    return weighted / _TOTAL_MASS


def _remove_small(heatmap, threshold=1e-2):
    return torch.where(heatmap < threshold, torch.zeros_like(heatmap), heatmap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',     type=str, default=_TRAIN,
                        help='Directory containing ckpts/')
    parser.add_argument('--ckpt',        type=str, default='singlePerson_0.0001_10_best',
                        help='Checkpoint filename without .path.tar')
    parser.add_argument('--test_dir',    type=str, default=os.path.join(_TRAIN, 'singlePerson_test'),
                        help='Test data path')
    parser.add_argument('--window',      type=int, default=10)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (Windows: 0 if spawn issues)')
    parser.add_argument('--max_batches', type=int, default=0,
                        help='Stop after N batches (0 = no limit). For smoke tests.')
    parser.add_argument('--out',         type=str,
                        default=os.path.join(_HERE, 'output', 'com_results.p'),
                        help='Output pickle path')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model   = tile2openpose_conv3d(args.window).to(device)
    softmax = SpatialSoftmax3D(20, 20, 18, 21).to(device)

    ckpt_path = os.path.join(args.exp_dir, 'ckpts', args.ckpt + '.path.tar')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Checkpoint loaded: {ckpt_path}')

    test_dataset    = sample_data(args.test_dir, args.window, [], 1)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
    print(f'Test frames: {len(test_dataset)}  '
          f'batches: {len(test_dataloader)}  '
          f'workers: {args.num_workers}'
          + (f'  max_batches: {args.max_batches}' if args.max_batches else ''))

    all_kp_gt   = []
    all_kp_pred = []

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_dataloader):
            if args.max_batches and i_batch >= args.max_batches:
                print(f'  smoke-test stop after {args.max_batches} batches')
                break

            tactile     = sample_batched[0].float().to(device)
            kp_gt_batch = sample_batched[2].float().numpy()       # (B, 21, 3)

            heatmap_out = model(tactile, device).reshape(-1, 21, 20, 20, 18)
            heatmap_t   = _remove_small(heatmap_out.transpose(2, 3))
            kp_pred_batch, _ = softmax(heatmap_t)

            all_kp_gt.append(kp_gt_batch)
            all_kp_pred.append(kp_pred_batch.cpu().numpy())

            if i_batch % 10 == 0:
                print(f'  {i_batch}/{len(test_dataloader)} batches processed')

    kp_gt   = np.concatenate(all_kp_gt,   axis=0)   # (T, 21, 3) normalized [0,1]
    kp_pred = np.concatenate(all_kp_pred, axis=0)   # (T, 21, 3) normalized [0,1]

    T = kp_gt.shape[0]
    print(f'Total frames: {T}')

    kp_gt_mm   = to_mm(kp_gt)      # (T, 21, 3) mm
    kp_pred_mm = to_mm(kp_pred)    # (T, 21, 3) mm

    com_gt   = compute_com(kp_gt_mm)    # (T, 3) mm
    com_pred = compute_com(kp_pred_mm)  # (T, 3) mm

    error            = com_pred - com_gt              # (T, 3)
    abs_error        = np.abs(error)                  # (T, 3)
    euclidean_error  = np.linalg.norm(error, axis=1)  # (T,)

    axes = ['x (carpet length)', 'y (carpet width)', 'z (height)']
    print('\n=== CoM Error: predicted vs GT-keypoint-derived ===')
    for i, ax in enumerate(axes):
        print(f'  {ax}:  mean={np.mean(abs_error[:,i]):.1f} mm  '
              f'std={np.std(abs_error[:,i]):.1f} mm  '
              f'max={np.max(abs_error[:,i]):.1f} mm')
    print(f'  Euclidean 3D:  mean={np.mean(euclidean_error):.1f} mm  '
          f'std={np.std(euclidean_error):.1f} mm  '
          f'max={np.max(euclidean_error):.1f} mm')

    jumps_pred = np.linalg.norm(np.diff(com_pred, axis=0), axis=1)
    jumps_gt   = np.linalg.norm(np.diff(com_gt,   axis=0), axis=1)
    print(f'\n=== Trajectory smoothness (mean frame-to-frame jump) ===')
    print(f'  Predicted CoM: {np.mean(jumps_pred):.1f} mm/frame')
    print(f'  GT CoM:        {np.mean(jumps_gt):.1f} mm/frame')

    # Coordinate system: floor = z=0, body extends negative z.
    # Expected for standing adult: CoM_z ~ -850 to -950 mm  (55% of ~1700 mm standing height)
    print(f'\n=== Sanity check: CoM_z (height) ===')
    print(f'  GT   CoM_z: mean={np.mean(com_gt[:,2]):.0f} mm  '
          f'range=[{com_gt[:,2].min():.0f}, {com_gt[:,2].max():.0f}]')
    print(f'  Pred CoM_z: mean={np.mean(com_pred[:,2]):.0f} mm  '
          f'range=[{com_pred[:,2].min():.0f}, {com_pred[:,2].max():.0f}]')
    print(f'  (floor=0 mm; body goes negative; ~-900 mm expected for standing adult)')
    print(f'  NOTE: head included via trunk-axis extrapolation (Option C); hands (1.2% mass) excluded')

    print(f'\n=== Sanity check: CoM_x,y carpet bounds [-100, 1800 mm] ===')
    for i, ax in enumerate(['x', 'y']):
        out_gt   = np.sum((com_gt[:,i]   < -100) | (com_gt[:,i]   > 1800))
        out_pred = np.sum((com_pred[:,i] < -100) | (com_pred[:,i] > 1800))
        print(f'  {ax}: GT out-of-bounds={out_gt}/{T}  Pred out-of-bounds={out_pred}/{T}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = {
        'com_gt':          com_gt,           # (T, 3) mm — CoM from GT keypoints
        'com_pred':        com_pred,         # (T, 3) mm — CoM from predicted keypoints
        'error':           error,            # (T, 3) mm — signed error (pred - GT)
        'euclidean_error': euclidean_error,  # (T,)  mm — 3D Euclidean error
        'kp_gt_mm':        kp_gt_mm,         # (T, 21, 3) mm — GT keypoints in mm
        'kp_pred_mm':      kp_pred_mm,       # (T, 21, 3) mm — predicted keypoints in mm
    }
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved to: {args.out}')
    print('Keys: com_gt, com_pred, error, euclidean_error, kp_gt_mm, kp_pred_mm')


if __name__ == '__main__':
    main()

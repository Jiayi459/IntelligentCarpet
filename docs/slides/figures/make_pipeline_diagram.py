"""make_pipeline_diagram.py — generate the project pipeline diagram for the slides.

Output: docs/slides/figures/pipeline.png

The diagram shows the end-to-end data flow for the CoM-forecasting project:
    tactile carpet (96x96)
      -> frozen pretrained CNN (`tile2openpose_conv3d`)
      -> 3D heatmap (21 joints x 20 x 20 x 18 voxels)
      -> SpatialSoftmax3D
      -> 21 keypoints (BODY_25 subset) in [0, 1]^3
      -> Winter segmental model
      -> CoM at frame t
    ... and in parallel ...
      -> camera/OpenPose ground-truth 21 keypoints (per-frame .p file)
      -> same Winter segmental model
      -> CoM_gt at frame t   (used as supervision for the forecaster)

    Then for forecasting we take CoM_gt (or kp_gt, or tactile) over a 10-s
    history window and predict the next 1 s.

Designed to be readable as a slide image at ~1400x800 pixels.
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

# Colors (consistent with slide-deck palette)
COLOR_INPUT      = '#dbeafe'   # light blue - raw sensor input
COLOR_FROZEN     = '#fce7f3'   # pink - frozen pretrained module
COLOR_DERIVED    = '#dcfce7'   # light green - intermediate representation
COLOR_TARGET     = '#fef3c7'   # yellow - supervisory signal
COLOR_FORECAST   = '#ddd6fe'   # light purple - forecasting model
COLOR_OUTPUT     = '#fed7aa'   # peach - final output
COLOR_GT         = '#fef3c7'   # yellow - camera-derived ground truth

EDGE_COLOR  = '#374151'
TEXT_COLOR  = '#111827'
ARROW_COLOR = '#374151'


def box(ax, x, y, w, h, text, color, fontsize=10, weight='normal'):
    """Draw a rounded box with text. (x, y) is the bottom-left corner."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.05',
        linewidth=1.2, edgecolor=EDGE_COLOR, facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=TEXT_COLOR, weight=weight, zorder=3, wrap=True)


def arrow(ax, x0, y0, x1, y1, label='', label_offset=(0, 0.05), style='-|>'):
    """Draw an arrow from (x0, y0) to (x1, y1) with an optional label."""
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=14,
        linewidth=1.4, color=ARROW_COLOR, zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=8.5, style='italic', color=TEXT_COLOR, zorder=3)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'pipeline.png')

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # =====================================================================
    # Title
    # =====================================================================
    ax.text(7, 8.55, 'IntelligentCarpet -> CoM forecasting: data flow',
            ha='center', fontsize=15, weight='bold', color=TEXT_COLOR)
    ax.text(7, 8.20, 'tactile and camera both produce 21 keypoints; CoM is derived '
                     'segmentally; the forecaster predicts the next 1 s',
            ha='center', fontsize=10, color='#6b7280', style='italic')

    # =====================================================================
    # Top row: tactile -> CNN -> keypoints (predicted) -> CoM_pred
    # =====================================================================
    y_top = 6.0

    box(ax, 0.2, y_top, 1.8, 1.0,
        'Tactile carpet\n96 x 96 pressure\n(10 Hz)',
        COLOR_INPUT, fontsize=10, weight='bold')

    box(ax, 2.4, y_top, 2.0, 1.0,
        'Frozen CNN\ntile2openpose_conv3d\n3D heatmap output',
        COLOR_FROZEN, fontsize=9.5)

    box(ax, 4.8, y_top, 1.8, 1.0,
        'SpatialSoftmax3D\n-> 21 keypoints\n(BODY_25 subset)',
        COLOR_DERIVED, fontsize=9.5)

    box(ax, 7.0, y_top, 1.8, 1.0,
        'Winter (2009)\nsegmental model\n+ Option C head',
        COLOR_DERIVED, fontsize=9.5)

    box(ax, 9.2, y_top, 1.6, 1.0,
        'CoM_pred(t)\n(x, y, z) mm',
        COLOR_OUTPUT, fontsize=10, weight='bold')

    arrow(ax, 2.0, y_top + 0.5, 2.4, y_top + 0.5)
    arrow(ax, 4.4, y_top + 0.5, 4.8, y_top + 0.5)
    arrow(ax, 6.6, y_top + 0.5, 7.0, y_top + 0.5, label='21 x 3', label_offset=(0, 0.22))
    arrow(ax, 8.8, y_top + 0.5, 9.2, y_top + 0.5, label='per-segment\nweighted mean',
          label_offset=(0, 0.32))

    # =====================================================================
    # Bottom row: camera/OpenPose -> kp_gt -> CoM_gt
    # =====================================================================
    y_bot = 3.6

    box(ax, 0.2, y_bot, 1.8, 1.0,
        'Camera (2 views)\nOpenPose 3D\ntriangulation',
        COLOR_GT, fontsize=10, weight='bold')

    box(ax, 4.8, y_bot, 1.8, 1.0,
        'GT 21 keypoints\nkp_gt_mm\n(precomputed in .p)',
        COLOR_DERIVED, fontsize=9.5)

    box(ax, 7.0, y_bot, 1.8, 1.0,
        'Same Winter\nsegmental model',
        COLOR_DERIVED, fontsize=9.5)

    box(ax, 9.2, y_bot, 1.6, 1.0,
        'CoM_gt(t)\n(x, y, z) mm',
        COLOR_TARGET, fontsize=10, weight='bold')

    arrow(ax, 2.0, y_bot + 0.5, 4.8, y_bot + 0.5,
          label='offline; bundled into dataset', label_offset=(0, 0.22))
    arrow(ax, 6.6, y_bot + 0.5, 7.0, y_bot + 0.5)
    arrow(ax, 8.8, y_bot + 0.5, 9.2, y_bot + 0.5)

    # Connect top "CoM_pred" to nothing for now (it's only used for the 71-mm
    # noise-floor estimate, not in the forecasting pipeline).
    ax.text(9.55, y_top - 0.45,
            '71 mm median 3D estimation error\n(vs camera-derived GT)',
            ha='left', va='top', fontsize=8.5, style='italic', color='#9333ea')

    # =====================================================================
    # Forecaster (right side)
    # =====================================================================
    y_fc = (y_top + y_bot) / 2 - 0.25

    # Bracket showing "100-frame history (10 s)" + arrow into forecaster
    arrow(ax, 10.0, y_fc - 0.4, 11.4, y_fc + 0.0,
          label='10 s history\nCoM_gt(t-99:t)\nor kp_gt_mm(t-99:t)\nor tactile(t-99:t)',
          label_offset=(0.05, -0.6))

    box(ax, 11.4, y_fc - 0.3, 2.2, 1.4,
        'Forecaster\n(GRU-based)\n\nPhase 1 / Phase 2 v1+v2 /\nphase2_tactile / gamma',
        COLOR_FORECAST, fontsize=9.5, weight='bold')

    # Arrow out to "future CoM"
    arrow(ax, 12.5, y_fc - 0.3 - 0.1, 12.5, 1.3,
          label='10 frames =\n1 s ahead',
          label_offset=(0.75, 0))

    box(ax, 11.3, 0.6, 2.4, 0.7,
        'future CoM(t+1 : t+10)\n(10, 3) mm',
        COLOR_OUTPUT, fontsize=10, weight='bold')

    # =====================================================================
    # Bottom note: what the forecasters use as supervision
    # =====================================================================
    ax.text(0.3, 1.9,
            'Supervision signal during training and the evaluation target are both '
            'CoM_gt (camera-derived).\n'
            'Phase 2 keypoints uses kp_gt_mm (oracle pose) as input; phase2_tactile '
            'uses raw tactile only;\n'
            'gamma uses both. Phase 1 uses only the 3-d CoM_gt history.',
            ha='left', va='top', fontsize=10, color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f3f4f6',
                       edgecolor='#9ca3af', linewidth=0.8))

    # =====================================================================
    # Legend
    # =====================================================================
    legend_items = [
        ('raw input',    COLOR_INPUT),
        ('frozen model', COLOR_FROZEN),
        ('derived',      COLOR_DERIVED),
        ('GT / supervision', COLOR_TARGET),
        ('forecaster',   COLOR_FORECAST),
        ('output',       COLOR_OUTPUT),
    ]
    lx = 0.3
    ly = 0.6
    for label, color in legend_items:
        ax.add_patch(Rectangle((lx, ly), 0.25, 0.18,
                                facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.7))
        ax.text(lx + 0.32, ly + 0.09, label, ha='left', va='center',
                fontsize=8.5, color=TEXT_COLOR)
        lx += 1.65

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()

"""model_epsilon.py — model classes for the CNN-free tactile forecaster (epsilon).

Importing this module has no side effects (no training, no I/O). All classes
and constants below are reused by train_phase2_epsilon.py, eval_phase2_epsilon.py,
and any future evaluator that loads epsilon checkpoints.

Design decisions (locked 2026-06-09, see SESSION_LOG.md):
    - encoder family : ViT (2 transformer blocks, 4 heads, D=128, FFN=512)
                       NO 2D convolutions anywhere
    - patches        : 6x6 grid of 16x16 cells (= 36 tokens, each 256-dim)
    - MAE mask ratio : 75 % (Stage 1)
    - Stage 2 dec    : FACTORED patch-latent decoder, ~1.5M params
                       Linear(128, 256) -> ReLU -> Linear(256, 10*36*16) -> Linear(16, 256) per patch
                       Predicts in patch-latent space, then unembeds to pixels
    - Stage 3 probes : Linear probe (Linear(128, 30)) + MLP probe (128 -> 64 -> 30)

The encoder is shared across Stage 1 (MAE), Stage 2 (dynamics pretraining),
and Stage 3 (probe). The MAE decoder and dynamics decoder are throwaway heads
discarded after their respective stages.

Reference: Sparsh (Higuera et al. 2024, arXiv 2410.24090), MAE (He et al. 2022).
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architectural constants (locked)
# ---------------------------------------------------------------------------

IMG_SIZE           = 96       # tactile frame side length (cells)
PATCH_SIZE         = 16       # patch side length (cells)
GRID_SIZE          = IMG_SIZE // PATCH_SIZE          # 6
N_PATCHES          = GRID_SIZE * GRID_SIZE           # 36
PATCH_DIM          = PATCH_SIZE * PATCH_SIZE         # 256 (per-patch pixel count)

EMBED_DIM          = 128
N_HEADS            = 4
N_ENCODER_LAYERS   = 2
FFN_DIM            = 4 * EMBED_DIM                    # 512

N_MAE_DECODER_LAYERS = 1
MAE_MASK_RATIO       = 0.75

GRU_HIDDEN         = 128
GRU_LAYERS         = 1

HORIZON            = 10        # 1 s @ 10 fps

DYN_LATENT_DIM     = 16        # per-patch latent in factored Stage 2 decoder


# ---------------------------------------------------------------------------
# Patch / unpatch helpers
# ---------------------------------------------------------------------------

def patchify(x):
    """(B, 96, 96)  ->  (B, 36, 256)   row-major patch order."""
    B, H, W = x.shape
    assert H == IMG_SIZE and W == IMG_SIZE, f'expected {IMG_SIZE}x{IMG_SIZE}, got {H}x{W}'
    x = x.reshape(B, GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE)   # (B, 6, 16, 6, 16)
    x = x.permute(0, 1, 3, 2, 4).contiguous()                         # (B, 6, 6, 16, 16)
    return x.reshape(B, N_PATCHES, PATCH_DIM)                         # (B, 36, 256)


def unpatchify(x):
    """(B, 36, 256)  ->  (B, 96, 96)   inverse of patchify."""
    B, N, D = x.shape
    assert N == N_PATCHES and D == PATCH_DIM
    x = x.reshape(B, GRID_SIZE, GRID_SIZE, PATCH_SIZE, PATCH_SIZE)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.reshape(B, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# ViT encoder (the encoder; shared across all 3 stages)
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """Per-frame patch-tokenized transformer encoder. NO convolutions.

    Input  : (B, 96, 96)  raw (standardized) tactile frame
    Output : (B, 128)     mean-pooled frame feature

    Also exposes encode_tokens() returning (B, 36, 128) per-patch tokens for
    use by the MAE decoder.

    Param count: ~ 230K (with default constants).
    """
    def __init__(self,
                 embed_dim=EMBED_DIM,
                 n_heads=N_HEADS,
                 n_layers=N_ENCODER_LAYERS,
                 ffn_dim=FFN_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Linear(PATCH_DIM, embed_dim)             # (256 -> 128)
        self.pos_embed   = nn.Parameter(torch.zeros(1, N_PATCHES, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def encode_tokens(self, x, keep_indices=None):
        """Return per-patch tokens.

        x             : (B, 96, 96)  standardized tactile frame
        keep_indices  : optional (B, K) long tensor of patch indices to keep
                        (for MAE; encoder sees only unmasked patches).
                        If None, all 36 patches are encoded.

        Returns: (B, K_or_36, EMBED_DIM)
        """
        patches  = patchify(x)                                          # (B, 36, 256)
        tokens   = self.patch_embed(patches) + self.pos_embed            # (B, 36, 128)
        if keep_indices is not None:
            # gather the kept tokens (positional embedding is already added so order is preserved)
            B, K = keep_indices.shape
            gather_idx = keep_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            tokens = torch.gather(tokens, dim=1, index=gather_idx)       # (B, K, 128)
        z = self.transformer(tokens)                                     # (B, *, 128)
        return self.norm(z)

    def forward(self, x):
        """(B, 96, 96) -> (B, 128) mean-pooled frame feature."""
        z = self.encode_tokens(x)                                        # (B, 36, 128)
        return z.mean(dim=1)                                             # (B, 128)


# ---------------------------------------------------------------------------
# MAE decoder (Stage 1 only; discarded after pretraining)
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """Tiny transformer that reconstructs masked patches given encoded visible
    patches + learned mask tokens.

    Input :
        z_keep       : (B, K, EMBED_DIM)   encoded visible-patch tokens
        keep_indices : (B, K)              their original patch indices
    Output:
        pred_patches : (B, 36, 256)        reconstructed pixel values for ALL
                                            patches; loss is computed only on
                                            masked ones by the caller.

    The decoder also has its own positional embedding (separate from the
    encoder's). Visible tokens get the encoder's z_keep; masked positions get
    a learned mask token. Both then receive the decoder's positional embedding.
    """
    def __init__(self,
                 embed_dim=EMBED_DIM,
                 n_heads=N_HEADS,
                 n_layers=N_MAE_DECODER_LAYERS,
                 ffn_dim=FFN_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES, embed_dim))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, PATCH_DIM)               # 128 -> 256 pixels per patch

    def forward(self, z_keep, keep_indices):
        B, K, D = z_keep.shape
        device  = z_keep.device

        # Start from all-mask-token grid, then scatter the visible tokens in.
        tokens = self.mask_token.expand(B, N_PATCHES, D).clone()         # (B, 36, 128)
        scatter_idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)
        tokens = tokens.scatter(dim=1, index=scatter_idx, src=z_keep)    # (B, 36, 128)

        tokens = tokens + self.dec_pos_embed                              # add decoder pos embed
        z      = self.transformer(tokens)
        z      = self.norm(z)
        return self.head(z)                                              # (B, 36, 256)


# ---------------------------------------------------------------------------
# Dynamics model (Stage 2: encoder + GRU + factored patch-latent decoder)
# ---------------------------------------------------------------------------

class DynamicsModel(nn.Module):
    """Encoder + GRU + factored decoder that predicts future tactile deltas.

    Input :
        tactile_window : (B, 100, 96, 96)  standardized history
    Output:
        delta_pred     : (B, 10, 96, 96)   predicted Delta_tactile(t+1..t+10)
                                            relative to tactile(t).

    Decoder factoring (committed 2026-06-09):
        gru_h -> Linear(128, 256) -> ReLU -> Linear(256, 10*36*16)  : per-patch latents
              -> Linear(16, 256) per patch                          : unembed to pixels
              -> unpatchify                                          : back to (96, 96)

    The encoder + GRU together are what Stage 3 reuses; the decoder is
    discarded after Stage 2.
    """
    def __init__(self,
                 embed_dim=EMBED_DIM,
                 gru_hidden=GRU_HIDDEN,
                 horizon=HORIZON,
                 dyn_latent=DYN_LATENT_DIM):
        super().__init__()
        self.horizon    = horizon
        self.dyn_latent = dyn_latent
        self.encoder    = ViTEncoder(embed_dim=embed_dim)
        self.gru        = nn.GRU(embed_dim, gru_hidden,
                                  num_layers=GRU_LAYERS, batch_first=True)
        self.dyn_decoder = nn.Sequential(
            nn.Linear(gru_hidden, 256), nn.ReLU(inplace=True),
            nn.Linear(256, horizon * N_PATCHES * dyn_latent),
        )
        self.patch_unembed = nn.Linear(dyn_latent, PATCH_DIM)            # 16 -> 256

    def forward(self, x):
        B, Tlen, H, W = x.shape
        flat = x.reshape(B * Tlen, H, W)
        feat = self.encoder(flat).reshape(B, Tlen, -1)                   # (B, 100, 128)
        _, h = self.gru(feat)
        h_final = h[-1]                                                  # (B, 128)

        latent = self.dyn_decoder(h_final).view(
            B, self.horizon, N_PATCHES, self.dyn_latent)                 # (B, 10, 36, 16)
        patches = self.patch_unembed(latent)                             # (B, 10, 36, 256)

        # unpatchify each future frame
        patches = patches.view(B * self.horizon, N_PATCHES, PATCH_DIM)
        frames  = unpatchify(patches).view(B, self.horizon, IMG_SIZE, IMG_SIZE)
        return frames

    def encode_history(self, x):
        """Forward through encoder + GRU only (skip the decoder). Returns
        the GRU's final hidden state (B, gru_hidden) -- the representation
        that Stage 3's probe consumes."""
        B, Tlen, H, W = x.shape
        flat = x.reshape(B * Tlen, H, W)
        feat = self.encoder(flat).reshape(B, Tlen, -1)
        _, h = self.gru(feat)
        return h[-1]


# ---------------------------------------------------------------------------
# Probe (Stage 3): linear OR MLP head on top of FROZEN encoder + GRU
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Single Linear layer mapping GRU final hidden state -> 10x3 delta CoM
    (in standardized-delta units)."""
    def __init__(self, hidden=GRU_HIDDEN, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.head    = nn.Linear(hidden, horizon * 3)

    def forward(self, h):
        return self.head(h).view(-1, self.horizon, 3)


class MLPProbe(nn.Module):
    """Tiny MLP probe: Linear(128 -> 64) -> ReLU -> Linear(64 -> 30)."""
    def __init__(self, hidden=GRU_HIDDEN, mid=64, horizon=HORIZON):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(hidden, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, horizon * 3),
        )

    def forward(self, h):
        return self.net(h).view(-1, self.horizon, 3)


class EpsilonForecaster(nn.Module):
    """Convenience wrapper that bundles (frozen) encoder + GRU + a probe head.

    Used at evaluation time and (optionally) for Stage 4 finetuning. The
    constructor is intentionally permissive: probe can be a LinearProbe, an
    MLPProbe, or any nn.Module that maps (B, GRU_HIDDEN) -> (B, HORIZON, 3).
    """
    def __init__(self, dynamics: 'DynamicsModel', probe: nn.Module):
        super().__init__()
        self.dynamics = dynamics
        self.probe    = probe

    def forward(self, x):
        h = self.dynamics.encode_history(x)
        return self.probe(h)


# ---------------------------------------------------------------------------
# Flatten baseline encoder (DEFERRED to a follow-up run; defined here so the
# class is importable but not used by default)
# ---------------------------------------------------------------------------

class FlattenEncoder(nn.Module):
    """No spatial tokenization at all; pure (96*96) -> EMBED_DIM linear projection.

    Baseline / control for the ViT encoder. Per the plan (2026-06-09), this
    is deferred to a follow-up run after the ViT pass; the class is provided
    here so an alternate DynamicsModel can be assembled trivially when the
    follow-up is scheduled.
    """
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(IMG_SIZE * IMG_SIZE, embed_dim)

    def forward(self, x):
        B, H, W = x.shape
        return self.proj(x.reshape(B, H * W))

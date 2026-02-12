"""
models.py — Model architectures for I-JEPA pretraining and segmentation.

Components:
  1. VisionTransformerEncoder  — ViT backbone (from timm)
  2. IJEPAPredictor            — lightweight transformer for masked prediction
  3. IJEPA                     — full pretraining model (context encoder + target encoder + predictor)
  4. SegmentationDecoder       — upsampling head that converts ViT features → masks
  5. SegmentationModel         — complete segmentation model (encoder + decoder)
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ============================================================
#  1. Patch Embedding (custom, for flexibility)
# ============================================================

class PatchEmbed(nn.Module):
    """Convert image into a sequence of patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)                    # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)     # (B, N, D)
        return x


# ============================================================
#  2. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
#  3. Vision Transformer Encoder
# ============================================================

class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder for I-JEPA.
    Outputs patch-level features (no CLS token by default).
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.depth = depth
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, H, W) images
            mask: optional boolean mask (B, N) — True = keep, False = remove
        Returns:
            features: (B, N, D) patch features (only unmasked if mask provided)
        """
        x = self.patch_embed(x)          # (B, N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if mask is not None:
            # Keep only unmasked patches
            # mask: (B, N) boolean, True = keep
            B, N, D = x.shape
            x = x[mask].reshape(B, -1, D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward_features_multilayer(self, x, layers=None):
        """
        Extract features from multiple transformer layers.
        Used by the segmentation decoder for skip connections.
        """
        if layers is None:
            layers = [3, 6, 9, 11]  # default layers for skip connections

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        features = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layers:
                features[i] = x.clone()

        x = self.norm(x)
        features["final"] = x
        return features


# ============================================================
#  4. I-JEPA Predictor
# ============================================================

class IJEPAPredictor(nn.Module):
    """
    Lightweight transformer that predicts target patch representations
    given context patch representations and target position embeddings.
    """

    def __init__(
        self,
        num_patches,
        context_dim=384,
        predictor_dim=192,
        depth=6,
        num_heads=6,
    ):
        super().__init__()
        self.predictor_dim = predictor_dim

        # Project from encoder dim to predictor dim
        self.input_proj = nn.Linear(context_dim, predictor_dim)

        # Positional embeddings for predictor
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim)
        )

        # Mask token (represents target positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dim
        self.output_proj = nn.Linear(predictor_dim, context_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context_features, context_indices, target_indices):
        """
        Args:
            context_features: (B, N_ctx, D) features from context encoder
            context_indices:  (B, N_ctx) indices of context patches
            target_indices:   (B, N_tgt) indices of target patches
        Returns:
            predictions: (B, N_tgt, D) predicted features for target patches
        """
        B = context_features.shape[0]
        N_ctx = context_features.shape[1]
        N_tgt = target_indices.shape[1]

        # Project context features
        ctx = self.input_proj(context_features)  # (B, N_ctx, pred_dim)

        # Add positional embeddings to context
        ctx_pos = self.predictor_pos_embed[:, :, :].expand(B, -1, -1)
        # Gather pos embeddings for context positions
        ctx_pos_sel = torch.gather(
            ctx_pos, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        )
        ctx = ctx + ctx_pos_sel

        # Create mask tokens for target positions
        tgt = self.mask_token.expand(B, N_tgt, -1)
        tgt_pos = torch.gather(
            ctx_pos, 1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        )
        tgt = tgt + tgt_pos

        # Concatenate context + target tokens
        x = torch.cat([ctx, tgt], dim=1)  # (B, N_ctx + N_tgt, pred_dim)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Extract only target predictions
        predictions = x[:, N_ctx:, :]  # (B, N_tgt, pred_dim)

        # Project back to encoder dim
        predictions = self.output_proj(predictions)

        return predictions


# ============================================================
#  5. Full I-JEPA Model (for pretraining)
# ============================================================

class IJEPA(nn.Module):
    """
    Complete I-JEPA model for self-supervised pretraining.
    
    Architecture:
      - Context encoder: processes visible (unmasked) patches
      - Target encoder:  EMA of context encoder, processes target patches
      - Predictor:       predicts target representations from context
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]

        img_size = cfg["data"].get("crop_size", 224)

        # Context encoder
        self.context_encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=model_cfg["patch_size"],
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
        )

        # Target encoder (EMA copy — no gradients)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        num_patches = self.context_encoder.patch_embed.num_patches
        self.predictor = IJEPAPredictor(
            num_patches=num_patches,
            context_dim=model_cfg["embed_dim"],
            predictor_dim=model_cfg["predictor_embed_dim"],
            depth=model_cfg["predictor_depth"],
            num_heads=model_cfg["predictor_num_heads"],
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """EMA update: target = momentum * target + (1 - momentum) * context"""
        for param_t, param_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            param_t.data.mul_(momentum).add_(param_c.data, alpha=1.0 - momentum)

    def forward(self, x, context_mask, target_masks):
        """
        Args:
            x: (B, C, H, W) images
            context_mask:  (B, N) boolean — True = visible to context encoder
            target_masks:  list of (B, N_tgt_i) index tensors for each target block
        Returns:
            predictions: list of (B, N_tgt_i, D) predicted features
            targets:     list of (B, N_tgt_i, D) target features (from EMA encoder)
        """
        # --- Context encoder ---
        # Get all patch embeddings first, then mask
        ctx_patches = self.context_encoder.patch_embed(x)
        ctx_patches = ctx_patches + self.context_encoder.pos_embed
        ctx_patches = self.context_encoder.pos_drop(ctx_patches)

        # Apply context mask: keep only visible patches
        B, N, D = ctx_patches.shape
        # context_mask: (B, N) True = keep
        # Get indices of context patches
        context_indices = context_mask.nonzero(as_tuple=False)  # (total_kept, 2)

        # Reshape: select masked patches per sample
        n_ctx = context_mask.sum(dim=1)[0].item()  # assume same num per sample
        ctx_indices = torch.where(context_mask)[1].reshape(B, n_ctx)  # (B, n_ctx)

        # Select context patches
        ctx = torch.gather(
            ctx_patches, 1,
            ctx_indices.unsqueeze(-1).expand(-1, -1, D)
        )

        # Pass through transformer blocks
        for blk in self.context_encoder.blocks:
            ctx = blk(ctx)
        ctx = self.context_encoder.norm(ctx)

        # --- Target encoder (no gradients) ---
        with torch.no_grad():
            tgt_all = self.target_encoder.patch_embed(x)
            tgt_all = tgt_all + self.target_encoder.pos_embed
            for blk in self.target_encoder.blocks:
                tgt_all = blk(tgt_all)
            tgt_all = self.target_encoder.norm(tgt_all)  # (B, N, D)

        # --- Predict targets ---
        predictions = []
        targets = []
        for tgt_indices in target_masks:
            # tgt_indices: (B, N_tgt_i) — indices of target patches
            pred = self.predictor(ctx, ctx_indices, tgt_indices)
            predictions.append(pred)

            # Get target features
            tgt = torch.gather(
                tgt_all, 1,
                tgt_indices.unsqueeze(-1).expand(-1, -1, D)
            )
            targets.append(tgt)

        return predictions, targets


# ============================================================
#  6. Segmentation Decoder
# ============================================================

class SegmentationDecoder(nn.Module):
    """
    Progressive upsampling decoder that converts ViT patch features
    into pixel-wise segmentation masks.
    
    Takes (B, N, D) patch tokens → reshapes to (B, D, H_p, W_p)
    → progressive upsampling → (B, num_classes, H, W)
    """

    def __init__(
        self,
        embed_dim=384,
        patch_size=16,
        img_size=224,
        num_classes=1,
        decoder_channels=None,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64]

        self.patch_size = patch_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        # Build progressive upsampling layers
        layers = []
        in_ch = embed_dim
        for out_ch in decoder_channels:
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch

        self.upsample = nn.Sequential(*layers)

        # Final 1×1 conv to get class logits
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, N, D) from ViT encoder
        Returns:
            logits: (B, num_classes, H, W) segmentation logits
        """
        B, N, D = patch_tokens.shape
        H_p, W_p = self.grid_size

        # Reshape to spatial feature map
        x = patch_tokens.transpose(1, 2).reshape(B, D, H_p, W_p)

        # Progressive upsampling
        x = self.upsample(x)

        # Final prediction
        x = self.head(x)

        # Resize to exact image size if needed
        if x.shape[-2:] != tuple(self.img_size):
            x = F.interpolate(
                x, size=self.img_size,
                mode="bilinear", align_corners=False,
            )

        return x


# ============================================================
#  7. Complete Segmentation Model
# ============================================================

class SegmentationModel(nn.Module):
    """
    Full segmentation model: pretrained I-JEPA encoder + decoder head.
    
    Usage:
        model = SegmentationModel.from_pretrained("checkpoints/ijepa_encoder_best.pth")
        logits = model(images)  # (B, 1, H, W)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=1,
        decoder_channels=None,
    ):
        super().__init__()

        self.encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        self.decoder = SegmentationDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            logits: (B, num_classes, H, W) segmentation logits
        """
        features = self.encoder(x)          # (B, N, D)
        logits = self.decoder(features)     # (B, num_classes, H, W)
        return logits

    def freeze_encoder(self):
        """Freeze encoder parameters (for initial decoder-only training)."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for full fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    @classmethod
    def from_pretrained(cls, encoder_path, cfg=None, strict=False):
        """
        Create model with pretrained I-JEPA encoder weights.

        Args:
            encoder_path: path to pretrained encoder checkpoint
            cfg: config dict (optional)
            strict: whether to strictly match state dict keys
        """
        if cfg is None:
            cfg = {
                "model": {
                    "patch_size": 16,
                    "embed_dim": 384,
                    "depth": 12,
                    "num_heads": 6,
                    "num_classes": 1,
                }
            }

        model_cfg = cfg["model"]
        model = cls(
            img_size=cfg.get("data", {}).get("crop_size", 224),
            patch_size=model_cfg["patch_size"],
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg.get("depth", 12),
            num_heads=model_cfg.get("num_heads", 6),
            num_classes=model_cfg.get("num_classes", 1),
            decoder_channels=model_cfg.get("decoder_channels", None),
        )

        # Load pretrained encoder weights
        if encoder_path and encoder_path != "none":
            checkpoint = torch.load(encoder_path, map_location="cpu")
            # Handle different checkpoint formats
            if "context_encoder" in checkpoint:
                state_dict = checkpoint["context_encoder"]
            elif "encoder" in checkpoint:
                state_dict = checkpoint["encoder"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            msg = model.encoder.load_state_dict(state_dict, strict=strict)
            print(f"[SegmentationModel] Loaded encoder from {encoder_path}")
            print(f"  Missing keys:    {len(msg.missing_keys)}")
            print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        return model


# ============================================================
#  8. Masking utilities for I-JEPA pretraining
# ============================================================

def generate_masks(batch_size, num_patches, grid_size, cfg):
    """
    Generate context mask and target block masks for I-JEPA.
    
    Args:
        batch_size: B
        num_patches: total number of patches (H_p * W_p)
        grid_size: (H_p, W_p) grid dimensions
        cfg: masking config dict
    
    Returns:
        context_mask:  (B, N) boolean — True = visible to context
        target_masks:  list of (B, N_tgt) index tensors
    """
    H_p, W_p = grid_size
    num_targets = cfg.get("num_targets", 4)
    target_scale = cfg.get("target_scale", [0.15, 0.2])
    target_aspect_ratio = cfg.get("target_aspect_ratio", [0.75, 1.5])
    context_scale = cfg.get("context_scale", [0.85, 1.0])

    device = "cpu"

    # Start with all patches visible
    all_target_indices = set()
    target_masks = []

    for _ in range(num_targets):
        # Random target block
        scale = torch.empty(1).uniform_(target_scale[0], target_scale[1]).item()
        aspect = torch.empty(1).uniform_(
            target_aspect_ratio[0], target_aspect_ratio[1]
        ).item()

        num_target_patches = int(num_patches * scale)
        height = max(1, int(math.sqrt(num_target_patches / aspect)))
        width = max(1, int(height * aspect))
        height = min(height, H_p)
        width = min(width, W_p)

        # Random position
        top = torch.randint(0, max(1, H_p - height + 1), (1,)).item()
        left = torch.randint(0, max(1, W_p - width + 1), (1,)).item()

        # Collect indices
        indices = []
        for r in range(top, min(top + height, H_p)):
            for c in range(left, min(left + width, W_p)):
                idx = r * W_p + c
                indices.append(idx)
                all_target_indices.add(idx)

        # (B, N_tgt_i)
        indices_tensor = torch.tensor(indices, device=device).unsqueeze(0)
        indices_tensor = indices_tensor.expand(batch_size, -1)
        target_masks.append(indices_tensor)

    # Context mask: remove target patches from visible set
    context_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
    for idx in all_target_indices:
        context_mask[:, idx] = False

    # Optionally further reduce context (context_scale)
    scale = torch.empty(1).uniform_(context_scale[0], context_scale[1]).item()
    n_keep = max(1, int(context_mask[0].sum().item() * scale))
    visible_indices = context_mask[0].nonzero(as_tuple=True)[0]
    if len(visible_indices) > n_keep:
        perm = torch.randperm(len(visible_indices))[:n_keep]
        keep_indices = visible_indices[perm]
        context_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
        context_mask[:, keep_indices] = True

    return context_mask, target_masks


# ============================================================
#  Quick test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing model architectures")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 2, 3, 224, 224

    # Test encoder
    print("\n1. VisionTransformerEncoder")
    encoder = VisionTransformerEncoder(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
    ).to(device)
    x = torch.randn(B, C, H, W, device=device)
    out = encoder(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"   Params: {n_params:.1f}M")

    # Test segmentation model
    print("\n2. SegmentationModel")
    seg_model = SegmentationModel(
        img_size=224, patch_size=16, embed_dim=384,
        depth=12, num_heads=6, num_classes=1,
    ).to(device)
    logits = seg_model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {logits.shape}")
    n_params = sum(p.numel() for p in seg_model.parameters()) / 1e6
    print(f"   Params: {n_params:.1f}M")

    # Test I-JEPA
    print("\n3. IJEPA (pretraining model)")
    cfg = {
        "data": {"crop_size": 224},
        "model": {
            "patch_size": 16, "embed_dim": 384, "depth": 12,
            "num_heads": 6, "predictor_embed_dim": 192,
            "predictor_depth": 6, "predictor_num_heads": 6,
        },
        "masking": {
            "num_targets": 4,
            "target_scale": [0.15, 0.2],
            "target_aspect_ratio": [0.75, 1.5],
            "context_scale": [0.85, 1.0],
        },
    }
    ijepa = IJEPA(cfg).to(device)
    grid_size = ijepa.context_encoder.patch_embed.grid_size
    num_patches = ijepa.context_encoder.patch_embed.num_patches
    ctx_mask, tgt_masks = generate_masks(B, num_patches, grid_size, cfg["masking"])
    ctx_mask = ctx_mask.to(device)
    tgt_masks = [t.to(device) for t in tgt_masks]

    preds, tgts = ijepa(x, ctx_mask, tgt_masks)
    print(f"   Num target blocks: {len(preds)}")
    for i, (p, t) in enumerate(zip(preds, tgts)):
        print(f"   Block {i}: pred={p.shape}, target={t.shape}")

    print("\n✓ All model tests passed!")

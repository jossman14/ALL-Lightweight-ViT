"""MSCFA model factory — wraps timm backbone (features_only) + MSCFAHead.

Supports FastViT-T8 (primary) and MobileViT-XS (secondary) with verified
multi-scale feature extraction.

Spec: Phase 3.5 novelty addendum §1.4.
"""
import torch
import torch.nn as nn
import timm
from .mscfa import MSCFAHead

BACKBONE_CONFIGS = {
    "fastvit_t8.apple_in1k": {
        "out_indices": [1, 2, 3],
        "expected_channels": [96, 192, 384],
    },
    "mobilevit_xs.cvnets_in1k": {
        "out_indices": [2, 3, 4],
        "expected_channels": [64, 80, 384],
    },
}


class MSCFAModel(nn.Module):
    """Backbone + MSCFA head wrapper.

    Args:
        backbone: timm backbone with features_only=True.
        head: MSCFAHead instance.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def create_mscfa_model(base_key, num_classes=2, d=96, num_heads=2, pretrained=True):
    """Create backbone + MSCFA head.

    Args:
        base_key: timm model key (must be in BACKBONE_CONFIGS).
        num_classes: number of output classes.
        d: MSCFA projection dimension.
        num_heads: MSCFA attention heads.
        pretrained: load pretrained backbone weights.

    Returns:
        MSCFAModel instance, dict with metadata (channels, param counts).
    """
    if base_key not in BACKBONE_CONFIGS:
        raise ValueError(
            "Unsupported backbone '%s'. Supported: %s"
            % (base_key, list(BACKBONE_CONFIGS.keys()))
        )

    config = BACKBONE_CONFIGS[base_key]
    backbone = timm.create_model(
        base_key,
        pretrained=pretrained,
        features_only=True,
        out_indices=config["out_indices"],
    )

    actual_channels = backbone.feature_info.channels()
    if actual_channels != config["expected_channels"]:
        print(
            "WARNING: channel mismatch for %s: expected %s, got %s"
            % (base_key, config["expected_channels"], actual_channels)
        )

    head = MSCFAHead(
        channels=actual_channels,
        d=d,
        num_heads=num_heads,
        num_classes=num_classes,
    )

    model = MSCFAModel(backbone, head)

    backbone_params = sum(p.numel() for p in backbone.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    meta = {
        "base_key": base_key,
        "channels": actual_channels,
        "out_indices": config["out_indices"],
        "backbone_params_M": round(backbone_params / 1e6, 3),
        "head_params_M": round(head_params / 1e6, 3),
        "total_params_M": round((backbone_params + head_params) / 1e6, 3),
        "head_overhead_pct": round(100 * head_params / backbone_params, 1),
        "d": d,
        "num_heads": num_heads,
    }

    return model, meta

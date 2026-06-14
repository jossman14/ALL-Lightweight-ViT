"""MSCFA — Multi-Scale Cell Feature Attention head.

Replaces standard GAP+FC classification head. Takes feature maps from 3 backbone
stages (chromatin texture @ 28x28, nuclear shape @ 14x14, N:C ratio @ 7x7),
projects each to a shared dimension d, prepends a learnable CLS token, applies
one transformer encoder layer, and classifies from the CLS output.

Spec: Phase 3.5 novelty addendum §1.4.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCFAHead(nn.Module):
    """Multi-Scale Cell Feature Attention classification head.

    Args:
        channels: list of channel dims per scale (e.g. [96, 192, 384] for FastViT-T8).
        d: projection dimension for all scales and the transformer. Default 96.
        num_heads: attention heads in the MHA block. Default 2.
        num_classes: output logits dimension.
        mlp_ratio: FFN hidden dim multiplier. Default 4.0.
        drop: dropout rate for attention and FFN. Default 0.0.
    """

    def __init__(self, channels, d=96, num_heads=2, num_classes=2, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.num_scales = len(channels)
        self.d = d

        self.projections = nn.ModuleList([
            nn.Linear(c, d) for c in channels
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

        self.attn = nn.MultiheadAttention(d, num_heads, dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        mlp_hidden = int(d * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, d),
            nn.Dropout(drop),
        )

        self.head = nn.Linear(d, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for proj in self.projections:
            nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
            nn.init.zeros_(proj.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, scale_features):
        """
        Args:
            scale_features: list of [B, C_i, H_i, W_i] feature maps (one per scale).

        Returns:
            logits: [B, num_classes]
        """
        B = scale_features[0].shape[0]
        tokens = []

        for i, feat in enumerate(scale_features):
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C_i]
            projected = self.projections[i](pooled)  # [B, d]
            tokens.append(projected.unsqueeze(1))  # [B, 1, d]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d]
        x = torch.cat([cls] + tokens, dim=1)  # [B, 1+num_scales, d]

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        cls_out = x[:, 0]  # [B, d]
        return self.head(cls_out)  # [B, num_classes]

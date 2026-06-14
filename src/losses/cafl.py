"""CaFL — Calibration-Aware Focal Loss.

Focal loss with per-class inverse-frequency weighting (alpha_c) plus
Brier score regularization for improved probability calibration.

Loss = FocalLoss(logits, targets, alpha_c, gamma) + lambda * BrierScore(probs, targets)

Spec: Phase 3.5 novelty addendum §2.4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaFL(nn.Module):
    """Calibration-Aware Focal Loss.

    Args:
        gamma: focal loss focusing parameter. Default 2.0.
        lam: Brier regularization weight. Default 0.5.
        eps: numerical stability epsilon. Default 1e-8.
    """

    def __init__(self, gamma=2.0, lam=0.5, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.lam = lam
        self.eps = eps

    def forward(self, logits, targets, alpha):
        """
        Args:
            logits: [B, C] raw model output (pre-softmax).
            targets: [B] integer class labels.
            alpha: [C] per-class weights (inverse-frequency, recomputed per fold).

        Returns:
            scalar loss.
        """
        C = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)  # [B, C]
        probs = log_probs.exp()  # [B, C]

        # Gather target class probabilities
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        target_probs = target_log_probs.exp()  # [B]

        # Per-class alpha weighting
        alpha_t = alpha.to(logits.device)[targets]  # [B]

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - target_probs).clamp(min=self.eps).pow(self.gamma)

        # Focal loss component (negative log-likelihood with focal + alpha)
        focal_loss = -alpha_t * focal_weight * target_log_probs  # [B]
        focal_loss = focal_loss.mean()

        # Brier score: mean squared error between predicted probs and one-hot targets
        one_hot = F.one_hot(targets, num_classes=C).float()  # [B, C]
        brier = ((probs - one_hot) ** 2).sum(dim=1).mean()

        return focal_loss + self.lam * brier

    @staticmethod
    def compute_alpha(labels, num_classes):
        """Compute inverse-frequency class weights from fold labels.

        Args:
            labels: list or 1D tensor of integer labels.
            num_classes: number of classes.

        Returns:
            alpha: [num_classes] tensor, normalized so mean = 1.
        """
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        counts = torch.bincount(labels, minlength=num_classes).float()
        counts = counts.clamp(min=1)
        inv_freq = 1.0 / counts
        alpha = inv_freq / inv_freq.mean()
        return alpha

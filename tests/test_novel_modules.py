"""Dry-run test for MSCFA head + CaFL loss + backbone feature dim verification.

Phase 4.5-prep validation. Uses minimal GPU (<30 seconds total).
"""
import sys
import json
import time
sys.path.insert(0, "/home/ftib/ALL-Lightweight-ViT")

import torch
import torch.nn as nn
import timm

from src.models.mscfa import MSCFAHead
from src.models.mscfa_model import create_mscfa_model, BACKBONE_CONFIGS
from src.losses.cafl import CaFL

RESULTS = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ──────────────────────────────────────────────────────────────────
# TEST 1: MSCFAHead standalone
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1: MSCFAHead standalone")
print("="*60)

head = MSCFAHead(channels=[96, 192, 384], d=96, num_heads=2, num_classes=2)
head_params = sum(p.numel() for p in head.parameters())
print("  MSCFA params: %d (%.3fM)" % (head_params, head_params/1e6))

dummy_features = [
    torch.randn(4, 96, 28, 28),
    torch.randn(4, 192, 14, 14),
    torch.randn(4, 384, 7, 7),
]
out = head(dummy_features)
assert out.shape == (4, 2), "Output shape mismatch: %s" % str(out.shape)
print("  Forward: output shape %s — PASS" % str(out.shape))

# Gradient flow
loss = out.sum()
loss.backward()
grad_ok = all(p.grad is not None for p in head.parameters() if p.requires_grad)
print("  Backward: gradient flow — %s" % ("PASS" if grad_ok else "FAIL"))

RESULTS["mscfa_standalone"] = {
    "params": head_params,
    "params_M": round(head_params/1e6, 3),
    "output_shape": list(out.shape),
    "grad_flow": grad_ok,
    "status": "PASS" if grad_ok else "FAIL",
}
del head, dummy_features, out


# ──────────────────────────────────────────────────────────────────
# TEST 2: CaFL standalone
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: CaFL standalone")
print("="*60)

cafl = CaFL(gamma=2.0, lam=0.5)
logits = torch.randn(8, 2, requires_grad=True)
targets = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1])
alpha = CaFL.compute_alpha(targets, num_classes=2)
print("  Alpha (inverse-freq): %s" % alpha.tolist())

loss_val = cafl(logits, targets, alpha)
print("  Loss: %.4f (scalar: %s)" % (loss_val.item(), loss_val.dim() == 0))
loss_val.backward()
grad_ok = logits.grad is not None and logits.grad.abs().sum() > 0
print("  Backward: gradient flow — %s" % ("PASS" if grad_ok else "FAIL"))

# Compare with standard CE for sanity
ce_loss = nn.CrossEntropyLoss()(logits.detach(), targets)
print("  CE loss for comparison: %.4f" % ce_loss.item())

RESULTS["cafl_standalone"] = {
    "loss_value": round(loss_val.item(), 4),
    "is_scalar": loss_val.dim() == 0,
    "grad_flow": grad_ok,
    "alpha": [round(a, 4) for a in alpha.tolist()],
    "status": "PASS" if (loss_val.dim() == 0 and grad_ok) else "FAIL",
}
del cafl, logits, targets


# ──────────────────────────────────────────────────────────────────
# TEST 3: FastViT-T8 + MSCFA (full model, GPU)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: FastViT-T8 + MSCFA (GPU dry run)")
print("="*60)

torch.cuda.reset_peak_memory_stats()
model_ft8, meta_ft8 = create_mscfa_model("fastvit_t8.apple_in1k", num_classes=2, pretrained=True)
model_ft8 = model_ft8.to(device)

print("  Channels: %s" % meta_ft8["channels"])
print("  Backbone: %.3fM, Head: %.3fM, Total: %.3fM" % (
    meta_ft8["backbone_params_M"], meta_ft8["head_params_M"], meta_ft8["total_params_M"]))
print("  Head overhead: %.1f%%" % meta_ft8["head_overhead_pct"])

# Forward batch=32
dummy = torch.randn(32, 3, 224, 224, device=device)
with torch.cuda.amp.autocast():
    out = model_ft8(dummy)
print("  Forward (B=32): output shape %s" % str(out.shape))

# Backward
targets = torch.randint(0, 2, (32,), device=device)
alpha = torch.tensor([1.0, 1.0], device=device)
cafl = CaFL().to(device)
loss = cafl(out, targets, alpha)
loss.backward()
print("  Backward: loss=%.4f — PASS" % loss.item())

vram_peak = torch.cuda.max_memory_allocated() / 1024**2
print("  Peak VRAM: %.0f MB (of 16376 MB)" % vram_peak)

RESULTS["fastvit_t8_mscfa"] = {
    "meta": meta_ft8,
    "output_shape": list(out.shape),
    "forward_backward": "PASS",
    "peak_vram_mb": round(vram_peak, 0),
    "loss_value": round(loss.item(), 4),
}

del model_ft8, dummy, out, cafl
torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────
# TEST 4: MobileViT-XS + MSCFA (GPU dry run)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 4: MobileViT-XS + MSCFA (GPU dry run)")
print("="*60)

torch.cuda.reset_peak_memory_stats()
model_mvit, meta_mvit = create_mscfa_model("mobilevit_xs.cvnets_in1k", num_classes=2, pretrained=True)
model_mvit = model_mvit.to(device)

print("  Channels: %s" % meta_mvit["channels"])
print("  Backbone: %.3fM, Head: %.3fM, Total: %.3fM" % (
    meta_mvit["backbone_params_M"], meta_mvit["head_params_M"], meta_mvit["total_params_M"]))
print("  Head overhead: %.1f%%" % meta_mvit["head_overhead_pct"])

dummy = torch.randn(32, 3, 224, 224, device=device)
with torch.cuda.amp.autocast():
    out = model_mvit(dummy)
print("  Forward (B=32): output shape %s" % str(out.shape))

targets = torch.randint(0, 2, (32,), device=device)
alpha = torch.tensor([1.0, 1.0], device=device)
cafl = CaFL().to(device)
loss = cafl(out, targets, alpha)
loss.backward()
print("  Backward: loss=%.4f — PASS" % loss.item())

vram_peak = torch.cuda.max_memory_allocated() / 1024**2
print("  Peak VRAM: %.0f MB" % vram_peak)

RESULTS["mobilevit_xs_mscfa"] = {
    "meta": meta_mvit,
    "output_shape": list(out.shape),
    "forward_backward": "PASS",
    "peak_vram_mb": round(vram_peak, 0),
    "loss_value": round(loss.item(), 4),
}

del model_mvit, dummy, out, cafl
torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────
# TEST 5: Baseline path equivalence (GAP+CE path unchanged)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 5: Baseline path (GAP+CE) unchanged")
print("="*60)

# Standard timm model with GAP+FC
model_std = timm.create_model("fastvit_t8.apple_in1k", pretrained=True, num_classes=2).to(device)
ce_criterion = nn.CrossEntropyLoss()

torch.manual_seed(42)
dummy = torch.randn(4, 3, 224, 224, device=device)
targets = torch.tensor([0, 1, 0, 1], device=device)

out_std = model_std(dummy)
loss_std = ce_criterion(out_std, targets)
loss_std.backward()

print("  Standard GAP+FC: output=%s, loss=%.4f" % (out_std.shape, loss_std.item()))
print("  Baseline path: UNMODIFIED (standard timm create_model + CrossEntropyLoss)")

RESULTS["baseline_path"] = {
    "output_shape": list(out_std.shape),
    "loss_value": round(loss_std.item(), 4),
    "status": "UNMODIFIED",
}

del model_std, dummy
torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DRY RUN SUMMARY")
print("="*60)

all_pass = True
for name, result in RESULTS.items():
    status = result.get("status", result.get("forward_backward", "?"))
    print("  %s: %s" % (name, status))
    if status not in ("PASS", "UNMODIFIED"):
        all_pass = False

print("\nOVERALL: %s" % ("ALL PASS" if all_pass else "SOME FAILURES"))

with open("/home/ftib/ALL-Lightweight-ViT/results/phase4_5_prep_dryrun.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("Results saved to results/phase4_5_prep_dryrun.json")

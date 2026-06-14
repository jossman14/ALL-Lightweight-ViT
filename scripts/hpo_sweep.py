"""Phase 4.1a — HPO sweep for learning-rate selection on C-NMC fold-0.

Protocol per Methodology v1.1 §2.1:
  - Optimizer: AdamW, weight_decay=0.05
  - LR schedule: Cosine annealing + 5-epoch linear warmup
  - Layer-wise LR decay: 0.65 (all pretrained models)
  - Batch size: 32 (auto-fallback to 16 on OOM)
  - Early stopping: patience=15 on validation balanced accuracy
  - Max epochs: 100
  - HPO fold: fold_0 (validate on fold 0, train on folds 1-4 indices)

LR grids:
  - Standard (timm): {5e-4, 1e-4, 5e-5}
  - DinoBloom linear probing: {1e-3, 5e-4, 1e-4}
  - DinoBloom fine-tuning: {5e-5, 1e-5, 5e-6}

Usage: python hpo_sweep.py [--model MODEL_KEY] [--lr LR] [--resume]
  Without args: runs full sweep sequentially.
  With --model/--lr: runs single experiment (for manual retry).
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import timm

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
PROJECT = Path.home() / "ALL-Lightweight-ViT"
CNMC_DIR = PROJECT / "data" / "cnmc_2019"
SPLIT_PATH = PROJECT / "splits" / "cnmc_5fold_seed42.json"
RESULTS_BASE = PROJECT / "results" / "phase4_1a_hpo"
METADATA_CSV = CNMC_DIR / "metadata.csv"

# ──────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────────
TIMM_MODELS = OrderedDict([
    ("mobilevit_xs", "mobilevit_xs.cvnets_in1k"),
    ("mobilevit_s", "mobilevit_s.cvnets_in1k"),
    ("edgenext_xs", "edgenext_x_small.in1k"),
    ("edgenext_s", "edgenext_small.usi_in1k"),
    ("fastvit_t8", "fastvit_t8.apple_in1k"),
    ("fastvit_t12", "fastvit_t12.apple_in1k"),
    ("tinyvit_5m", "tiny_vit_5m_224.dist_in22k_ft_in1k"),
    ("tinyvit_11m", "tiny_vit_11m_224.dist_in22k_ft_in1k"),
    ("efficientformer_l1", "efficientformer_l1.snap_dist_in1k"),
    ("efficientnet_b0", "efficientnet_b0.ra_in1k"),
    ("resnet50", "resnet50.a1_in1k"),
    ("convnext_t", "convnext_tiny.fb_in1k"),
    ("vit_b16", "vit_base_patch16_224.augreg_in21k_ft_in1k"),
])

DINOBLOOM_REPO = "marrlab/DinoBloom"
DINOBLOOM_COMMIT = "e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff"

STANDARD_LRS = [5e-4, 1e-4, 5e-5]
DINOBLOOM_FT_LRS = [5e-5, 1e-5, 5e-6]
DINOBLOOM_LP_LRS = [1e-3, 5e-4, 1e-4]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Determinism
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────
class CNMCDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


_corrupt_cache = None

def _scan_corrupt_files():
    """Scan C-NMC train images once, cache result."""
    global _corrupt_cache
    if _corrupt_cache is not None:
        return _corrupt_cache
    corrupt = set()
    for cls in ["all", "hem"]:
        cls_dir = CNMC_DIR / "train" / cls
        for f in cls_dir.iterdir():
            if f.suffix.lower() == ".bmp":
                with open(f, "rb") as fh:
                    magic = fh.read(2)
                if magic != b"BM":
                    corrupt.add(str(f))
    _corrupt_cache = corrupt
    if corrupt:
        print(f"  [DATA] {len(corrupt)} corrupt BMP files detected (zero-byte from download), excluded")
    return corrupt


def build_datasets(fold_key="fold_0"):
    """Build train/val datasets from C-NMC split, excluding corrupt images."""
    meta = pd.read_csv(METADATA_CSV)
    train_meta = meta[meta["split"] == "train"].reset_index(drop=True)

    with open(SPLIT_PATH) as f:
        splits = json.load(f)

    fold = splits[fold_key]
    train_indices = fold["train"]
    val_indices = fold["val"]

    corrupt = _scan_corrupt_files()

    all_paths = []
    all_labels = []
    for idx in range(len(train_meta)):
        row = train_meta.iloc[idx]
        cls_folder = "all" if row["class_label"] == "all" else "hem"
        fpath = str(CNMC_DIR / "train" / cls_folder / row["original_image_name"])
        label = 1 if row["class_label"] == "all" else 0
        all_paths.append(fpath)
        all_labels.append(label)

    train_paths = [all_paths[i] for i in train_indices if all_paths[i] not in corrupt]
    train_labels = [all_labels[i] for i in train_indices if all_paths[i] not in corrupt]
    val_paths = [all_paths[i] for i in val_indices if all_paths[i] not in corrupt]
    val_labels = [all_labels[i] for i in val_indices if all_paths[i] not in corrupt]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = CNMCDataset(train_paths, train_labels, train_transform)
    val_ds = CNMCDataset(val_paths, val_labels, val_transform)

    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────────────
# MODEL CREATION
# ──────────────────────────────────────────────────────────────────────
def create_timm_model(timm_key, num_classes=2):
    model = timm.create_model(timm_key, pretrained=True, num_classes=num_classes)
    return model


def create_dinobloom(variant="base", num_classes=2, linear_probe=False):
    """Load DinoBloom-B or DinoBloom-S with custom weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import importlib

    hub_dir = os.path.join(torch.hub.get_dir(), "marrlab_DinoBloom_main")
    if not os.path.isdir(hub_dir):
        torch.hub.load(DINOBLOOM_REPO, "dinov2_vits14", source="github",
                       trust_repo=True, pretrained=False)

    sys.path.insert(0, hub_dir)
    vits = importlib.import_module("dinov2.models.vision_transformer")

    if variant == "base":
        backbone = vits.vit_base(img_size=224, patch_size=14, init_values=1.0, block_chunks=0)
        ckpt_file = "pytorch_model_b.bin"
        embed_dim = 768
    else:
        backbone = vits.vit_small(img_size=224, patch_size=14, init_values=1.0, block_chunks=0)
        ckpt_file = "pytorch_model_s.bin"
        embed_dim = 384

    ckpt_path = hf_hub_download(DINOBLOOM_REPO, ckpt_file, revision=DINOBLOOM_COMMIT)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(state_dict, strict=True)

    if linear_probe:
        for p in backbone.parameters():
            p.requires_grad = False

    class DinoBloomClassifier(nn.Module):
        def __init__(self, backbone, embed_dim, num_classes, is_linear_probe):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(embed_dim, num_classes)
            self.is_linear_probe = is_linear_probe

        def forward(self, x):
            if self.is_linear_probe:
                with torch.no_grad():
                    features = self.backbone(x)
            else:
                features = self.backbone(x)
            if isinstance(features, dict):
                features = features.get("x_norm_clstoken", list(features.values())[0])
            if features.dim() == 3:
                features = features[:, 0]
            return self.head(features)

    model = DinoBloomClassifier(backbone, embed_dim, num_classes, linear_probe)
    return model


# ──────────────────────────────────────────────────────────────────────
# LAYER-WISE LR DECAY
# ──────────────────────────────────────────────────────────────────────
def get_layer_groups_timm(model, timm_key):
    """Get parameter groups with layer-wise LR decay for timm models."""
    try:
        num_layers = model.get_num_layers() if hasattr(model, "get_num_layers") else None
    except Exception:
        num_layers = None

    if num_layers is None:
        # Estimate from named parameters
        layer_names = set()
        for name, _ in model.named_parameters():
            parts = name.split(".")
            # Common layer identifiers
            for i, p in enumerate(parts):
                if p.startswith("block") or p.startswith("layer") or p.startswith("stage"):
                    try:
                        layer_names.add(int(parts[i + 1]) if i + 1 < len(parts) and parts[i + 1].isdigit() else p)
                    except (ValueError, IndexError):
                        layer_names.add(p)
                elif p.isdigit() and i > 0:
                    parent = parts[i - 1]
                    if parent in ("blocks", "layers", "stages", "features", "network"):
                        layer_names.add((parent, int(p)))
        num_layers = max(12, len(layer_names))

    return num_layers


def build_param_groups(model, base_lr, weight_decay, layer_decay, model_key, is_dinobloom=False, is_linear_probe=False):
    """Build optimizer parameter groups with layer-wise LR decay."""
    if is_linear_probe:
        # Only train the head
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": base_lr, "weight_decay": weight_decay}]

    # Assign layer depths to parameters
    param_groups = {}
    no_decay_keywords = ["bias", "norm", "bn", "ln", "layernorm", "batchnorm"]

    if is_dinobloom:
        # DINOv2-style ViT: blocks are in backbone.blocks.N
        num_layers = 12  # ViT-B has 12 blocks, ViT-S has 12 blocks
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                depth = num_layers
            elif "backbone.blocks." in name:
                block_num = int(name.split("backbone.blocks.")[1].split(".")[0])
                depth = block_num
            elif "backbone.patch_embed" in name or "backbone.cls_token" in name or "backbone.pos_embed" in name:
                depth = 0
            else:
                depth = 0

            lr_scale = layer_decay ** (num_layers - depth)
            wd = 0.0 if any(kw in name.lower() for kw in no_decay_keywords) else weight_decay

            group_key = (depth, wd > 0)
            if group_key not in param_groups:
                param_groups[group_key] = {"params": [], "lr": base_lr * lr_scale, "weight_decay": wd}
            param_groups[group_key]["params"].append(param)
    else:
        # timm models: use naming conventions to assign depths
        depth_map = {}
        max_depth = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Determine depth from parameter name
            depth = 0
            parts = name.split(".")
            found = False
            for i, p in enumerate(parts):
                if p in ("blocks", "layers", "stages", "features", "network", "levels"):
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        depth = int(parts[i + 1]) + 1
                        found = True
                        break
                elif p.startswith("block") and p[5:].isdigit():
                    depth = int(p[5:]) + 1
                    found = True
                    break
                elif p.startswith("layer") and p[5:].isdigit():
                    depth = int(p[5:]) + 1
                    found = True
                    break
                elif p.startswith("stage") and p[5:].isdigit():
                    depth = int(p[5:]) + 1
                    found = True
                    break

            # Head/classifier gets max depth + 1
            if any(h in name for h in ["head", "classifier", "fc", "last_linear"]):
                depth = 999  # placeholder, will be adjusted

            depth_map[name] = depth
            if depth != 999:
                max_depth = max(max_depth, depth)

        # Normalize depths
        for name in depth_map:
            if depth_map[name] == 999:
                depth_map[name] = max_depth + 1

        total_layers = max_depth + 1

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            depth = depth_map.get(name, 0)
            lr_scale = layer_decay ** (total_layers - depth)
            wd = 0.0 if any(kw in name.lower() for kw in no_decay_keywords) else weight_decay

            group_key = (depth, wd > 0)
            if group_key not in param_groups:
                param_groups[group_key] = {"params": [], "lr": base_lr * lr_scale, "weight_decay": wd}
            param_groups[group_key]["params"].append(param)

    groups = list(param_groups.values())
    return groups


# ──────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / n_samples
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, bal_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / n_samples
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, bal_acc


def run_single_hpo(model_key, lr, batch_size=32, max_epochs=100, patience=15,
                   weight_decay=0.05, layer_decay=0.65, warmup_epochs=5,
                   is_dinobloom=False, dinobloom_variant=None, is_linear_probe=False):
    """Run a single HPO experiment."""
    # Determine run name
    if is_dinobloom:
        mode = "lp" if is_linear_probe else "ft"
        run_name = f"dinobloom_{dinobloom_variant}_{mode}"
    else:
        run_name = model_key

    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    run_dir = RESULTS_BASE / run_name / f"lr_{lr_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        print(f"  [SKIP] {run_name} lr={lr} — already completed")
        with open(metrics_file) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"  MODEL: {run_name}  |  LR: {lr}  |  batch: {batch_size}")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Build datasets
    train_ds, val_ds = build_datasets("fold_0")
    print(f"  Data: train={len(train_ds)}, val={len(val_ds)}")

    # Create model
    try:
        if is_dinobloom:
            model = create_dinobloom(dinobloom_variant, num_classes=2, linear_probe=is_linear_probe)
        else:
            timm_key = TIMM_MODELS[model_key]
            model = create_timm_model(timm_key, num_classes=2)
    except Exception as e:
        print(f"  [FAIL] Model creation failed: {e}")
        result = {"model": run_name, "lr": lr, "status": "FAIL_MODEL_CREATION", "error": str(e)}
        with open(metrics_file, "w") as f:
            json.dump(result, f, indent=2)
        return result

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params/1e6:.2f}M total, {n_trainable/1e6:.2f}M trainable")

    model = model.to(device)

    # Build optimizer with layer-wise LR decay
    param_groups = build_param_groups(
        model, lr, weight_decay, layer_decay, model_key,
        is_dinobloom=is_dinobloom, is_linear_probe=is_linear_probe
    )
    optimizer = AdamW(param_groups)

    # LR schedule: linear warmup + cosine decay
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Class weights for imbalanced C-NMC
    train_labels_arr = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels_arr)
    class_weights = 1.0 / class_counts.astype(float)
    class_weights = class_weights / class_weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Try batch_size, fallback if OOM
    actual_batch = batch_size
    try:
        train_loader = DataLoader(train_ds, batch_size=actual_batch, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=actual_batch * 2, shuffle=False,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True)
        # Test with one batch
        batch = next(iter(train_loader))
        images_test = batch[0].to(device)
        with torch.cuda.amp.autocast():
            _ = model(images_test)
        del images_test, batch
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"  [OOM] batch={actual_batch}, falling back to {actual_batch // 2}")
            torch.cuda.empty_cache()
            gc.collect()
            actual_batch = actual_batch // 2
            train_loader = DataLoader(train_ds, batch_size=actual_batch, shuffle=True,
                                      num_workers=4, pin_memory=True, drop_last=True,
                                      persistent_workers=True)
            val_loader = DataLoader(val_ds, batch_size=actual_batch * 2, shuffle=False,
                                    num_workers=4, pin_memory=True,
                                    persistent_workers=True)
        else:
            raise

    # Training loop
    best_val_bal_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    peak_vram_mb = 0

    print(f"  Training with batch_size={actual_batch}, max_epochs={max_epochs}, patience={patience}")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_bal_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, val_bal_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_vram_mb = max(peak_vram_mb, vram_mb)
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "train_bal_acc": round(train_bal_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
            "val_bal_acc": round(val_bal_acc, 4),
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
        })

        if epoch % 5 == 0 or epoch <= 3:
            print(f"  Ep {epoch:3d}/{max_epochs}: "
                  f"train_loss={train_loss:.4f} train_bacc={train_bal_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_bacc={val_bal_acc:.4f} | "
                  f"lr={current_lr:.2e} | {epoch_time:.1f}s")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - start_time
    gpu_hours = total_time / 3600

    result = {
        "model": run_name,
        "model_key": TIMM_MODELS.get(model_key, f"dinobloom_{dinobloom_variant}"),
        "lr": lr,
        "is_dinobloom": is_dinobloom,
        "is_linear_probe": is_linear_probe,
        "dinobloom_variant": dinobloom_variant,
        "status": "COMPLETE",
        "best_val_bal_acc": round(best_val_bal_acc, 5),
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "batch_size": actual_batch,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "total_time_s": round(total_time, 1),
        "gpu_hours": round(gpu_hours, 4),
        "n_params_M": round(n_params / 1e6, 2),
        "n_trainable_M": round(n_trainable / 1e6, 2),
        "weight_decay": weight_decay,
        "layer_decay": layer_decay,
        "warmup_epochs": warmup_epochs,
        "seed": SEED,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Run metadata
    commit_sha = "N/A"
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    run_meta = {
        "commit_sha": commit_sha,
        "config": result,
        "seeds": {"python": SEED, "numpy": SEED, "torch": SEED, "cuda": SEED},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        },
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "command": f"python hpo_sweep.py --model {model_key or run_name} --lr {lr}",
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"  DONE: best_val_bal_acc={best_val_bal_acc:.4f} @ epoch {best_epoch}, "
          f"GPU-hours={gpu_hours:.3f}, peak_VRAM={peak_vram_mb:.0f}MB")

    # Cleanup
    del model, optimizer, scheduler, criterion, train_loader, val_loader, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ──────────────────────────────────────────────────────────────────────
# SWEEP ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────
def build_sweep_queue():
    """Build ordered list of all HPO runs."""
    queue = []

    # timm models (smallest first for fast iteration)
    for model_key in TIMM_MODELS:
        for lr in STANDARD_LRS:
            queue.append({
                "model_key": model_key, "lr": lr,
                "is_dinobloom": False, "dinobloom_variant": None, "is_linear_probe": False,
            })

    # DinoBloom fine-tuning
    for variant in ["small", "base"]:
        for lr in DINOBLOOM_FT_LRS:
            queue.append({
                "model_key": f"dinobloom_{variant}_ft", "lr": lr,
                "is_dinobloom": True, "dinobloom_variant": variant, "is_linear_probe": False,
            })

    # DinoBloom linear probing
    for variant in ["small", "base"]:
        for lr in DINOBLOOM_LP_LRS:
            queue.append({
                "model_key": f"dinobloom_{variant}_lp", "lr": lr,
                "is_dinobloom": True, "dinobloom_variant": variant, "is_linear_probe": True,
            })

    return queue


def run_sweep():
    """Run full HPO sweep."""
    queue = build_sweep_queue()
    print(f"HPO Sweep: {len(queue)} runs total")
    print(f"Fold: fold_0 (validate on fold 0, train on folds 1-4)")
    print(f"Dataset: C-NMC 2019 ({CNMC_DIR})")
    print()

    all_results = []
    sweep_start = time.time()

    for i, run_spec in enumerate(queue):
        print(f"\n[{i+1}/{len(queue)}] ", end="")
        try:
            result = run_single_hpo(
                model_key=run_spec["model_key"],
                lr=run_spec["lr"],
                is_dinobloom=run_spec["is_dinobloom"],
                dinobloom_variant=run_spec["dinobloom_variant"],
                is_linear_probe=run_spec["is_linear_probe"],
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {run_spec['model_key']} lr={run_spec['lr']}: {e}")
            traceback.print_exc()
            error_result = {
                "model": run_spec["model_key"],
                "lr": run_spec["lr"],
                "status": "FAIL",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            all_results.append(error_result)
            # Try to recover GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    sweep_time = time.time() - sweep_start
    print(f"\n\n{'='*70}")
    print(f"SWEEP COMPLETE: {len(all_results)} runs in {sweep_time/3600:.2f} GPU-hours")
    print(f"{'='*70}")

    # Generate summary
    generate_summary(all_results, sweep_time)
    return all_results


def generate_summary(all_results, total_sweep_time):
    """Generate selected_lrs.json and hpo_summary.md."""
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    # Group results by model
    model_results = {}
    for r in all_results:
        model = r.get("model", r.get("model_key", "unknown"))
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(r)

    # Select best LR per model
    selected_lrs = {}
    for model, results in model_results.items():
        completed = [r for r in results if r.get("status") == "COMPLETE"]
        if not completed:
            selected_lrs[model] = {
                "selected_lr": None,
                "val_bal_acc_at_selected": None,
                "status": "ALL_FAILED",
                "errors": [r.get("error", "unknown") for r in results],
            }
            continue

        best = max(completed, key=lambda r: r.get("best_val_bal_acc", 0))
        selected_lrs[model] = {
            "selected_lr": best["lr"],
            "val_bal_acc_at_selected": best["best_val_bal_acc"],
            "best_epoch": best["best_epoch"],
            "total_epochs": best["total_epochs"],
            "batch_size": best.get("batch_size", 32),
            "peak_vram_mb": best.get("peak_vram_mb", 0),
            "gpu_hours": best.get("gpu_hours", 0),
            "all_lrs": {
                str(r["lr"]): r.get("best_val_bal_acc", None)
                for r in completed
            },
        }

    with open(RESULTS_BASE / "selected_lrs.json", "w") as f:
        json.dump(selected_lrs, f, indent=2)

    # Generate markdown summary
    total_gpu_hours = sum(r.get("gpu_hours", 0) for r in all_results if r.get("status") == "COMPLETE")

    lines = [
        "# Phase 4.1a — HPO Sweep Summary",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Fold:** fold_0 (validate on fold 0, train on folds 1-4)",
        f"**Dataset:** C-NMC 2019 (train={8817}, val={1844})",
        f"**Total runs:** {len(all_results)} ({sum(1 for r in all_results if r.get('status')=='COMPLETE')} complete, "
        f"{sum(1 for r in all_results if r.get('status')!='COMPLETE')} failed)",
        f"**Total GPU-hours:** {total_gpu_hours:.2f}h",
        f"**Wall-clock time:** {total_sweep_time/3600:.2f}h",
        "",
        "## Selected Learning Rates",
        "",
        "| Model | Type | Selected LR | Val BalAcc | Best Ep | Batch | VRAM (MB) | GPU-h |",
        "|-------|------|------------|-----------|---------|-------|-----------|-------|",
    ]

    for model in sorted(selected_lrs.keys()):
        info = selected_lrs[model]
        if info.get("selected_lr") is None:
            lines.append(f"| {model} | — | FAILED | — | — | — | — | — |")
            continue
        mode = "LP" if "lp" in model else ("FT" if "dinobloom" in model else "FT")
        lines.append(
            f"| {model} | {mode} | {info['selected_lr']:.0e} | "
            f"{info['val_bal_acc_at_selected']:.4f} | {info['best_epoch']} | "
            f"{info['batch_size']} | {info['peak_vram_mb']:.0f} | "
            f"{info['gpu_hours']:.3f} |"
        )

    lines.extend([
        "",
        "## LR Comparison (all candidates)",
        "",
        "| Model | LR=5e-4 | LR=1e-4 | LR=5e-5 | Selected |",
        "|-------|---------|---------|---------|----------|",
    ])

    # Standard timm models
    for model_key in TIMM_MODELS:
        info = selected_lrs.get(model_key, {})
        all_lrs = info.get("all_lrs", {})
        vals = []
        for lr_str in ["0.0005", "0.0001", "5e-05"]:
            v = all_lrs.get(lr_str, all_lrs.get(str(float(lr_str)), None))
            vals.append(f"{v:.4f}" if v else "—")
        sel = info.get("selected_lr")
        sel_str = f"{sel:.0e}" if sel else "—"
        lines.append(f"| {model_key} | {vals[0]} | {vals[1]} | {vals[2]} | **{sel_str}** |")

    lines.extend([
        "",
        "### DinoBloom Fine-tuning",
        "",
        "| Model | LR=5e-5 | LR=1e-5 | LR=5e-6 | Selected |",
        "|-------|---------|---------|---------|----------|",
    ])

    for variant in ["small", "base"]:
        model = f"dinobloom_{variant}_ft"
        info = selected_lrs.get(model, {})
        all_lrs = info.get("all_lrs", {})
        vals = []
        for lr_val in [5e-5, 1e-5, 5e-6]:
            v = all_lrs.get(str(lr_val), None)
            vals.append(f"{v:.4f}" if v else "—")
        sel = info.get("selected_lr")
        sel_str = f"{sel:.0e}" if sel else "—"
        lines.append(f"| {model} | {vals[0]} | {vals[1]} | {vals[2]} | **{sel_str}** |")

    lines.extend([
        "",
        "### DinoBloom Linear Probing",
        "",
        "| Model | LR=1e-3 | LR=5e-4 | LR=1e-4 | Selected |",
        "|-------|---------|---------|---------|----------|",
    ])

    for variant in ["small", "base"]:
        model = f"dinobloom_{variant}_lp"
        info = selected_lrs.get(model, {})
        all_lrs = info.get("all_lrs", {})
        vals = []
        for lr_val in [1e-3, 5e-4, 1e-4]:
            v = all_lrs.get(str(lr_val), None)
            vals.append(f"{v:.4f}" if v else "—")
        sel = info.get("selected_lr")
        sel_str = f"{sel:.0e}" if sel else "—"
        lines.append(f"| {model} | {vals[0]} | {vals[1]} | {vals[2]} | **{sel_str}** |")

    md_content = "\n".join(lines) + "\n"
    with open(RESULTS_BASE / "hpo_summary.md", "w") as f:
        f.write(md_content)

    print(f"\nSummary saved to:")
    print(f"  {RESULTS_BASE / 'selected_lrs.json'}")
    print(f"  {RESULTS_BASE / 'hpo_summary.md'}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.1a HPO Sweep")
    parser.add_argument("--model", type=str, default=None, help="Run single model (key name)")
    parser.add_argument("--lr", type=float, default=None, help="Single LR to test")
    parser.add_argument("--resume", action="store_true", help="Resume sweep (skip completed)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"timm: {timm.__version__}")
    print()

    if args.model and args.lr:
        # Single run
        is_db = "dinobloom" in args.model
        is_lp = "lp" in args.model if is_db else False
        db_var = None
        if is_db:
            db_var = "base" if "base" in args.model else "small"
        run_single_hpo(
            model_key=args.model, lr=args.lr,
            is_dinobloom=is_db, dinobloom_variant=db_var, is_linear_probe=is_lp,
        )
    else:
        run_sweep()

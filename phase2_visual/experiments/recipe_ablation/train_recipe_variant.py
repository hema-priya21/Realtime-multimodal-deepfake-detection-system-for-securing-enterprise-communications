"""Controlled recipe-ablation training: isolates which component of the
existing yunet_clean recipe (phase2_visual/train_visual_yunet_clean.py) is
responsible for its sealed-test improvement vs. its live-domain regression
(documented in phase2_visual/experiments/model_calibration_boundary_diagnostic.md
and live_camera_ab_test_report.md).

The existing yunet_clean recipe already combines THREE changes relative to
the pre-existing recipe at once: (1) a class-balanced WeightedRandomSampler,
(2) heavier augmentation (RandomResizedCrop, GaussianBlur, JPEG
re-compression), (3) native-resolution-loss simulation. Per the task's
CRITICAL instruction, this script does not "pretend" any of these are new --
it defines four variants that each differ from the previous by exactly ONE
documented ingredient, built additively so the marginal effect of each
ingredient can be read directly from the B->C->D deltas, with A standing
alone as an independent reproduction of the full existing recipe (a
cross-check: A and D end up with an IDENTICAL effective configuration, built
two different ways, which is intentional -- see the module-level RECIPES
dict and the report this framework produces).

  RECIPE A (baseline reproduction): balanced sampler + the FULL existing
      yunet_clean augmentation stack (RandomResizedCrop + resolution-loss
      simulation + flip + rotation + ColorJitter + GaussianBlur + JPEG
      re-compression). Reproduces train_visual_yunet_clean.py as exactly as
      possible, to verify reproducibility before trusting the B/C/D deltas.

  RECIPE B (balanced sampler only): balanced sampler + ONLY the light
      augmentation that already existed in the codebase before yunet_clean
      (dataset_pipeline/face_transforms.py's train_transform: horizontal
      flip p=0.5, +/-5 deg rotation, ColorJitter brightness/contrast=0.2,
      saturation=0.1). No RandomResizedCrop, no GaussianBlur, no JPEG
      re-compression, no resolution-loss simulation.

  RECIPE C (B + heavier augmentation): B's ingredients, PLUS
      RandomResizedCrop(224, scale=(0.85,1.0), ratio=(0.95,1.05)) -- crops
      stay within 85-100% of the face, never zooming out or cropping away
      the frame boundary where blending artifacts live -- PLUS a light
      Gaussian blur (kernel 3, sigma 0.1-1.0, applied with probability 0.3)
      PLUS JPEG re-compression (quality uniformly sampled 40-100). No
      resolution-loss simulation yet. Every augmentation/probability here is
      copied verbatim from the existing yunet_clean recipe, not invented.

  RECIPE D (C + resolution/compression simulation): C's ingredients PLUS
      simulate_native_resolution_loss (downsample to a random size in
      [96,223]px then upsample back to 224, applied with probability 0.4) --
      copied verbatim from train_visual_yunet_clean.py. This is, by
      construction, the SAME effective configuration as Recipe A.

Everything else (model architecture, ImageNet initialization, optimizer,
learning rate, weight decay, LR schedule, batch size, max epochs, early-stop
patience, checkpoint-selection criterion, AMP, seed) is held identical to
train_visual_yunet_clean.py across all four variants.

Reads data/visual_processed_yunet_full/{train,val} and the existing
subject-disjoint split CSVs (read-only, never modified). Never touches
phase2_visual/visual_model.pth, phase2_visual/checkpoints/visual_model_yunet_clean_best.pth,
or phase4_fusion_alerting/web_app.py. Saves ONLY to
phase2_visual/experiments/recipe_ablation/checkpoints/experiment_<X>_best.pth.

Run as: python phase2_visual/experiments/recipe_ablation/train_recipe_variant.py --variant A
        python phase2_visual/experiments/recipe_ablation/train_recipe_variant.py --variant B
        python phase2_visual/experiments/recipe_ablation/train_recipe_variant.py --variant C
        python phase2_visual/experiments/recipe_ablation/train_recipe_variant.py --variant D
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.face_transforms import eval_transform, IMAGENET_MEAN, IMAGENET_STD

# ============================================================
SEED = 42
DATA_ROOT = PROJECT_ROOT / "data" / "visual_processed_yunet_full"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

ABLATION_DIR = PROJECT_ROOT / "phase2_visual" / "experiments" / "recipe_ablation"
CHECKPOINT_ROOT = ABLATION_DIR / "checkpoints"
TENSORBOARD_ROOT = ABLATION_DIR / "runs"
LOG_DIR = ABLATION_DIR / "logs"

BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
NUM_WORKERS = min(4, os.cpu_count() or 0)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
    USE_AMP = True
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"
    USE_AMP = False


def simulate_native_resolution_loss(img):
    """Verbatim copy of train_visual_yunet_clean.py's augmentation -- with
    probability 0.4, downsample then upsample back to 224x224."""
    if random.random() < 0.4:
        target = random.randint(96, 223)
        img = img.resize((target, target), Image.BILINEAR)
        img = img.resize((224, 224), Image.BILINEAR)
    return img


# --- named ingredients, each documented once, composed per-variant below ---
def ingredient_light_aug():
    """The augmentation that already existed in the codebase before
    yunet_clean (dataset_pipeline/face_transforms.py's train_transform),
    reimplemented with v2 ops so it composes with the rest of this file."""
    return [
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ]


def ingredient_random_resized_crop():
    return [v2.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05))]


def ingredient_heavier_aug_extra():
    """The additional pieces of yunet_clean's augmentation beyond
    RandomResizedCrop + the light augmentation: a light Gaussian blur and
    JPEG re-compression -- verbatim probabilities/ranges from
    train_visual_yunet_clean.py."""
    return [
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    ]


def ingredient_jpeg():
    """JPEG re-compression needs the tensor pipeline (ToImage/ToDtype)
    around it -- kept as its own ingredient since it changes the transform
    plumbing, not just an added op."""
    return [v2.ToImage(), v2.JPEG(quality=(40, 100)), v2.ToDtype(torch.float32, scale=True)]


def ingredient_resolution_loss():
    return [simulate_native_resolution_loss]


def build_transform(components):
    """components: ordered list of ingredient lists to concatenate, matching
    train_visual_yunet_clean.py's exact op order when all are present:
    RandomResizedCrop -> resolution-loss -> flip/rotation/colorjitter ->
    blur -> [ToImage -> JPEG -> ToDtype] -> Normalize."""
    ops = []
    for c in components:
        ops.extend(c)
    if not any(isinstance(o, v2.ToImage) for o in ops):
        # variants without JPEG never convert to a tensor mid-pipeline --
        # do it once at the end instead (same effect, same final dtype).
        ops.append(v2.ToImage())
        ops.append(v2.ToDtype(torch.float32, scale=True))
    ops.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose([v2.Resize((224, 224))] + ops)


RECIPES = {
    # variant: (description, ordered ingredient components)
    "A": ("Baseline reproduction: balanced sampler + FULL existing yunet_clean augmentation "
          "(RandomResizedCrop + resolution-loss-sim + flip/rotation/ColorJitter + blur + JPEG). "
          "Verifies reproducibility of the existing checkpoint's training recipe.",
          [ingredient_random_resized_crop(), ingredient_resolution_loss(), ingredient_light_aug(),
           ingredient_heavier_aug_extra(), ingredient_jpeg()]),
    "B": ("Balanced sampler ONLY: sampler is the sole change vs. no augmentation beyond the "
          "light flip/rotation/ColorJitter that already existed in the codebase "
          "(dataset_pipeline/face_transforms.py). No RandomResizedCrop, no blur, no JPEG, "
          "no resolution-loss simulation.",
          [ingredient_light_aug()]),
    # NOTE: redefined per explicit instruction after reviewing B's results. The originally
    # planned "C = B + heavier augmentation" was never run (B already showed heavy augmentation
    # does not move the real hard tail, so it was deprioritized) and "D" was dropped entirely.
    # This C instead isolates the WeightedRandomSampler itself, holding B's augmentation fixed --
    # see USE_SAMPLER below, which is the only thing that differs from B.
    "C": ("NATURAL CLASS FREQUENCY / NO WeightedRandomSampler: identical to Experiment B in every "
          "respect (same light flip/rotation/ColorJitter augmentation, same everything else) "
          "EXCEPT the sampler is removed -- the train loader uses standard shuffling over the "
          "natural ~13.2% real / ~86.8% fake class distribution instead of WeightedRandomSampler's "
          "roughly-balanced per-batch draw. Isolates the sampler's own contribution to the real "
          "hard-tail behavior, independent of augmentation (already ruled out by B).",
          [ingredient_light_aug()]),
}

# Per-variant sampler switch -- the ONLY axis Experiment C changes relative to B.
USE_SAMPLER = {"A": True, "B": True, "C": False}


def format_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def check_dataset(path, name):
    if not path.exists():
        print(f"\nERROR: {name} folder not found: {path}")
        raise SystemExit(1)
    real_count = sum(1 for _ in (path / "real").glob("*.jpg")) if (path / "real").exists() else 0
    fake_count = sum(1 for _ in (path / "fake").glob("*.jpg")) if (path / "fake").exists() else 0
    print(f"{name:5s} | real: {real_count:,} | fake: {fake_count:,} | total: {real_count + fake_count:,}")
    if real_count == 0 or fake_count == 0:
        print(f"ERROR: {name} set is missing a class.")
        raise SystemExit(1)


def create_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, 2)
    return model.to(DEVICE)


def gpu_memory():
    if not torch.cuda.is_available():
        return "N/A"
    return f"{torch.cuda.memory_allocated(0)/(1024**3):.2f} GB used / {torch.cuda.memory_reserved(0)/(1024**3):.2f} GB reserved"


def run_epoch(model, loader, criterion, optimizer, scaler, epoch, train):
    model.train(mode=train)
    running_loss = 0.0
    correct = 0
    processed = 0
    all_labels, all_preds = [], []
    start = time.time()
    phase = "TRAIN" if train else "VAL"

    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()

        batch_size_actual = images.size(0)
        processed += batch_size_actual
        running_loss += loss.item() * batch_size_actual
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

        if batch_number % 50 == 0:
            print(f"\r  [{phase}] epoch {epoch} | images {processed:,} | loss {loss.item():.4f}",
                  end="", flush=True)

    print()
    epoch_time = time.time() - start
    epoch_loss = running_loss / processed
    epoch_accuracy = correct / processed * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"  [{phase}] epoch {epoch} finished | loss {epoch_loss:.4f} | "
          f"accuracy {epoch_accuracy:.2f}% | macro-F1 {macro_f1:.4f} | "
          f"time {format_time(epoch_time)} | GPU {gpu_memory()}")
    return epoch_loss, epoch_accuracy, macro_f1


def train_model(variant):
    desc, components = RECIPES[variant]
    train_transform = build_transform(components)
    model_path = CHECKPOINT_ROOT / f"experiment_{variant}_best.pth"
    checkpoint_dir = CHECKPOINT_ROOT / variant
    tensorboard_dir = TENSORBOARD_ROOT / f"experiment_{variant}"
    config_path = ABLATION_DIR / "results" / f"experiment_{variant}_config.json"

    print("=" * 80)
    print(f"RECIPE ABLATION -- EXPERIMENT {variant}")
    print("=" * 80)
    print("Description:", desc)
    print("Device:", DEVICE, "|", GPU_NAME, "| AMP:", USE_AMP)
    print("Train dataset:", TRAIN_DIR)
    print("Val dataset  :", VAL_DIR)
    print("Model output :", model_path)
    print("Seed:", SEED, "| Max epochs:", MAX_EPOCHS, "| Early-stop patience:", EARLY_STOP_PATIENCE)
    print("Train transform:", train_transform)
    print("=" * 80)

    check_dataset(TRAIN_DIR, "TRAIN")
    check_dataset(VAL_DIR, "VAL")

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=eval_transform)

    print("\nClass mapping:", train_dataset.class_to_idx)
    assert train_dataset.class_to_idx == {"fake": 0, "real": 1}, "Class mapping must be fake=0, real=1"
    assert train_dataset.class_to_idx == val_dataset.class_to_idx, "Train/val class mapping mismatch"

    class_counts = Counter(train_dataset.targets)
    num_classes = len(train_dataset.classes)
    print("Class counts (train):", {train_dataset.classes[i]: class_counts[i] for i in range(num_classes)})
    total_n = sum(class_counts.values())
    for i in range(num_classes):
        print(f"  natural frequency -- {train_dataset.classes[i]}: {class_counts[i]/total_n*100:.2f}%")

    use_sampler = USE_SAMPLER[variant]
    print(f"WeightedRandomSampler enabled for this variant: {use_sampler}")
    if use_sampler:
        sample_weights = [1.0 / class_counts[t] for t in train_dataset.targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True,
                                         generator=torch.Generator().manual_seed(SEED))
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            persistent_workers=(NUM_WORKERS > 0),
        )
    else:
        # Natural class frequency: standard shuffling, no per-sample reweighting. Each epoch
        # draws every training crop exactly once, in the dataset's true ~13%/87% real/fake ratio.
        gen = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=gen,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            persistent_workers=(NUM_WORKERS > 0),
        )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    (ABLATION_DIR / "results").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    config_path.write_text(json.dumps({
        "variant": variant, "description": desc, "seed": SEED, "use_sampler": use_sampler,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "optimizer": "Adam", "scheduler": "CosineAnnealingLR",
        "amp": USE_AMP, "transform_repr": str(train_transform),
    }, indent=2), encoding="utf-8")

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    training_start = time.time()
    history = []
    best_epoch = None
    best_val_acc = None
    train_f1_at_best = None

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{MAX_EPOCHS} ---")
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, scaler, epoch, True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, scaler, epoch, False)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalars("macro_f1", {"train": train_f1, "val": val_f1}, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)
        scheduler.step()
        print(f"  LR after epoch {epoch}: {current_lr:.6f} -> {optimizer.param_groups[0]['lr']:.6f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "train_f1": train_f1, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})

        torch.save(model.state_dict(), str(checkpoint_dir / "last.pth"))

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            best_val_acc = val_acc
            train_f1_at_best = train_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), str(model_path))
            print(f"  *** New best VAL macro-F1: {val_f1:.4f} (acc {val_acc:.2f}%) -- saved to {model_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{EARLY_STOP_PATIENCE} epoch(s)")
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no val macro-F1 improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    writer.close()
    total_time = time.time() - training_start

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Experiment        : {variant}")
    print(f"Epochs run        : {len(history)}")
    print(f"Best epoch        : {best_epoch}")
    print(f"Best VAL macro-F1 : {best_macro_f1:.4f}")
    print(f"Best VAL accuracy : {best_val_acc:.2f}%")
    print(f"Train/val F1 gap at best epoch: {train_f1_at_best - best_macro_f1:+.4f}")
    print(f"Total time        : {format_time(total_time)}")
    print(f"Model saved to    : {model_path}")
    print("visual_model.pth (production) and the existing yunet_clean checkpoint were NOT touched by this run.")
    print("=" * 80)

    summary_path = ABLATION_DIR / "results" / f"experiment_{variant}_training_summary.json"
    summary_path.write_text(json.dumps({
        "variant": variant, "epochs_run": len(history), "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1, "best_val_accuracy": best_val_acc,
        "train_f1_at_best_epoch": train_f1_at_best,
        "train_val_gap_at_best": train_f1_at_best - best_macro_f1 if train_f1_at_best is not None else None,
        "total_time_seconds": total_time, "total_time_formatted": format_time(total_time),
        "history": history, "model_path": str(model_path),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["A", "B", "C"])
    args = parser.parse_args()
    train_model(args.variant)

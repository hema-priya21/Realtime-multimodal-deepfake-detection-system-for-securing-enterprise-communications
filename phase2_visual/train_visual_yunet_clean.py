"""Isolated "clean" retraining experiment: trains MobileNetV3-Small on the
YuNet-extracted, subject-disjoint dataset (data/visual_processed_yunet_full/),
specifically designed to reduce identity/appearance shortcut learning rather
than just swapping the crop convention (see train_visual_yunet_full.py for
that simpler, earlier experiment).

Deliberate differences from the production recipe (train_visual.py) and from
train_visual_yunet_full.py, each chosen for a stated reason:

  1. Class-balanced SAMPLING (WeightedRandomSampler) on the train loader
     instead of a class-weighted loss. A weighted loss still shows the model
     raw, heavily-imbalanced batches (~87% fake) and only rescales the
     gradient after the fact; a balanced sampler makes every batch already
     close to 50/50, which is a more direct way to stop the model from
     treating "fake" as the default answer. Using both at once would
     double-correct, so the loss is left unweighted here -- this is the
     resolution to "inspect whether the weighting is causing unstable
     decision boundaries" from the task spec.
  2. Moderate augmentation targeted at robustness, not at destroying the
     forensic signal: mild RandomResizedCrop (scale 0.85-1.0 -- never zooms
     out or crops away the face boundary, where blending artifacts usually
     live), horizontal flip, +/-5 deg rotation, brightness/contrast/
     saturation jitter, a light Gaussian blur (kernel 3, applied 30% of the
     time), and JPEG re-compression (quality 40-100) to mimic the live
     pipeline's own re-encoding. No heavy crop/cutout/mixup that could erase
     the boundary region a real-vs-fake decision often depends on.
  2b. Native-resolution-loss simulation (simulate_native_resolution_loss,
     applied 40% of the time): downsamples the 224x224 crop to a random
     size between 96-223px, then upsamples it back to 224x224. Added after
     frame-by-frame analysis of a real screen recording (2026-09-23) showed
     a strong correlation (Pearson r=-0.84 across 797 frames) between a
     detected face's crop area and this exact production checkpoint's
     P(fake): small/distant real faces (~154px native, requiring ~1.46x
     upsampling to reach 224x224) averaged P(fake)=0.82 (100% misread as
     FAKE), while large/near real faces from the SAME live session
     (~240px native, near 1:1, negligible upsampling) averaged P(fake)=0.37
     (24% FAKE). Face area was also strongly confounded with brightness in
     that recording (r=-0.93, only two people appear), so that single video
     cannot prove scale is THE cause rather than brightness or identity --
     but training the model to be invariant to native face resolution is a
     safe, generally-beneficial robustness property regardless, and directly
     targets the specific mismatch documented there.
  3. Checkpoint selection and early stopping by validation MACRO-F1, not raw
     validation accuracy. Raw accuracy on an ~87%-fake validation set can
     reward a checkpoint that is simply fake-biased; macro-F1 weighs the
     real and fake classes equally, which is the actual goal here (fix
     real-class false positives without abandoning fake detection).
  4. Deterministic seeding (Python/NumPy/PyTorch) for run-to-run
     reproducibility of shuffling, augmentation and weight init.

Reads data/visual_processed_yunet_full/{train,val} and the existing
subject-disjoint split CSVs (read-only). Never touches data/visual_processed/,
dataset_pipeline/splits/, or visual_model.pth. Saves ONLY to
phase2_visual/checkpoints/visual_model_yunet_clean_best.pth.

Run as: python phase2_visual/train_visual_yunet_clean.py
"""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.face_transforms import eval_transform, IMAGENET_MEAN, IMAGENET_STD

# ============================================================
SEED = 42
DATA_ROOT = PROJECT_ROOT / "data" / "visual_processed_yunet_full"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

EXPERIMENT_NAME = "yunet_clean"
MODEL_PATH = PROJECT_ROOT / "phase2_visual" / "checkpoints" / "visual_model_yunet_clean_best.pth"
CHECKPOINT_DIR = PROJECT_ROOT / "phase2_visual" / "checkpoints" / EXPERIMENT_NAME
TENSORBOARD_DIR = PROJECT_ROOT / "phase2_visual" / "runs" / EXPERIMENT_NAME

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
    """With probability 0.4, downsample then upsample back to 224x224 --
    see module docstring point 2b for the video evidence motivating this."""
    if random.random() < 0.4:
        target = random.randint(96, 223)
        img = img.resize((target, target), Image.BILINEAR)
        img = img.resize((224, 224), Image.BILINEAR)
    return img


# --- moderate, artifact-preserving augmentation (train only) ---
train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    simulate_native_resolution_loss,
    v2.RandomHorizontalFlip(0.5),
    v2.RandomRotation(5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    v2.ToImage(),
    v2.JPEG(quality=(40, 100)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


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


def run_epoch(model, loader, criterion, optimizer, scaler, epoch, train, fake_idx):
    model.train(mode=train)
    total_images = len(loader.dataset) if train else len(loader.sampler) if loader.sampler is not None else len(loader.dataset)
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


def train_model():
    print("=" * 80)
    print("VISUAL DEEPFAKE DETECTION -- yunet_clean (shortcut-reduction experiment)")
    print("=" * 80)
    print("Device:", DEVICE, "|", GPU_NAME, "| AMP:", USE_AMP)
    print("Train dataset:", TRAIN_DIR)
    print("Val dataset  :", VAL_DIR)
    print("Model output :", MODEL_PATH)
    print("Seed:", SEED, "| Max epochs:", MAX_EPOCHS, "| Early-stop patience:", EARLY_STOP_PATIENCE)
    print("=" * 80)

    check_dataset(TRAIN_DIR, "TRAIN")
    check_dataset(VAL_DIR, "VAL")

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=eval_transform)

    print("\nClass mapping:", train_dataset.class_to_idx)
    assert train_dataset.class_to_idx == {"fake": 0, "real": 1}, "Class mapping must be fake=0, real=1"
    assert train_dataset.class_to_idx == val_dataset.class_to_idx, "Train/val class mapping mismatch"
    fake_idx = train_dataset.class_to_idx["fake"]

    class_counts = Counter(train_dataset.targets)
    num_classes = len(train_dataset.classes)
    print("Class counts (train):", {train_dataset.classes[i]: class_counts[i] for i in range(num_classes)})

    # --- class-balanced SAMPLER (replaces loss-level class weighting -- see
    # module docstring point 1). Per-sample weight = 1 / count of its class,
    # so each epoch draws real and fake with roughly equal frequency
    # regardless of the ~87%/13% raw imbalance. Only the TRAIN loader is
    # balanced; val/test keep their natural distribution unchanged.
    sample_weights = [1.0 / class_counts[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True, generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = create_model()
    criterion = nn.CrossEntropyLoss()  # unweighted -- sampler already balances batches
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    training_start = time.time()

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{MAX_EPOCHS} ---")
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, scaler, epoch, True, fake_idx)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, scaler, epoch, False, fake_idx)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalars("macro_f1", {"train": train_f1, "val": val_f1}, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)
        scheduler.step()
        print(f"  LR after epoch {epoch}: {current_lr:.6f} -> {optimizer.param_groups[0]['lr']:.6f}")

        torch.save(model.state_dict(), str(CHECKPOINT_DIR / "last.pth"))

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), str(MODEL_PATH))
            torch.save(model.state_dict(), str(CHECKPOINT_DIR / f"best_epoch{epoch}_valf1{val_f1:.4f}_valacc{val_acc:.2f}.pth"))
            print(f"  *** New best VAL macro-F1: {val_f1:.4f} (acc {val_acc:.2f}%) -- saved to {MODEL_PATH}")
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
    print(f"Best VAL macro-F1 : {best_macro_f1:.4f}")
    print(f"Total time        : {format_time(total_time)}")
    print(f"Model saved to    : {MODEL_PATH}")
    print("visual_model.pth (production baseline) was NOT touched by this run.")
    print("=" * 80)


if __name__ == "__main__":
    train_model()

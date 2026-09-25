"""Isolated crop-distribution experiment: trains the SAME architecture and
hyperparameters as train_visual.py's current recipe (MobileNetV3-Small,
class-weighted CrossEntropyLoss, Adam, cosine LR schedule, weight decay),
changing ONLY the training data source -- data/visual_processed_yunet_full/
(YuNet, tight/no-margin crops, matching live inference exactly) instead of
data/visual_processed/ (Haar + 15% margin, what the deployed visual_model.pth
was trained on).

This is a deliberate single-variable experiment (crop distribution) to test
whether the train/live crop mismatch documented in the forensic investigation
is a meaningful contributor to live false positives on real faces. It does
NOT touch visual_model.pth, data/visual_processed/, or the subject-disjoint
split CSVs -- those are read-only inputs here.

Run as: python phase2_visual/train_visual_yunet_full.py
"""

import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.face_transforms import train_transform, eval_transform

# ============================================================
# PROJECT PATHS -- the only deliberate difference from train_visual.py
# ============================================================

DATA_ROOT = PROJECT_ROOT / "data" / "visual_processed_yunet_full"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

EXPERIMENT_NAME = "yunet_full"
# Clearly separate from visual_model.pth, per task instructions.
MODEL_PATH = PROJECT_ROOT / "phase2_visual" / "checkpoints" / "visual_model_yunet_full_best.pth"
CHECKPOINT_DIR = PROJECT_ROOT / "phase2_visual" / "checkpoints" / EXPERIMENT_NAME
TENSORBOARD_DIR = PROJECT_ROOT / "phase2_visual" / "runs" / EXPERIMENT_NAME

# ============================================================
# SETTINGS -- identical to the current train_visual.py recipe (Experiment 2:
# lower LR + cosine schedule + weight decay), so the crop distribution is
# the only variable relative to the most recent known-good training config.
# ============================================================

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
NUM_WORKERS = min(4, os.cpu_count() or 0)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
    USE_AMP = True
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"
    USE_AMP = False


def format_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def print_header():
    print()
    print("=" * 80)
    print("  VISUAL DEEPFAKE DETECTION TRAINING -- YuNet-crop experiment (isolated)")
    print("=" * 80)
    print("Training device :", DEVICE)
    if torch.cuda.is_available():
        print("GPU             :", GPU_NAME)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU memory      : {total_memory:.2f} GB")
        print("Mixed Precision : Enabled (FP16 AMP)")
    else:
        print("GPU             : Not detected")
        print("WARNING         : Training on CPU -- this will be slow")
    print()
    print("Experiment      :", EXPERIMENT_NAME)
    print("Train dataset   :", TRAIN_DIR)
    print("Val dataset     :", VAL_DIR)
    print("Model output    :", MODEL_PATH)
    print("Batch size      :", BATCH_SIZE)
    print("Epochs          :", EPOCHS)
    print("Learning rate   :", LEARNING_RATE, "(CosineAnnealingLR, T_max=", EPOCHS, ")")
    print("Weight decay    :", WEIGHT_DECAY)
    print("=" * 80)


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
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)
    return model.to(DEVICE)


def gpu_memory():
    if not torch.cuda.is_available():
        return "N/A"
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    return f"{allocated:.2f} GB used / {reserved:.2f} GB reserved"


def run_epoch(model, loader, criterion, optimizer, scaler, epoch, train):
    model.train(mode=train)
    total_images = len(loader.dataset)
    running_loss = 0.0
    correct = 0
    processed = 0
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

        if batch_number % 50 == 0 or processed == total_images:
            pct = processed / total_images * 100
            print(f"\r  [{phase}] epoch {epoch} | {pct:6.2f}% | "
                  f"images {processed:,}/{total_images:,} | loss {loss.item():.4f}",
                  end="", flush=True)

    print()
    epoch_time = time.time() - start
    epoch_loss = running_loss / processed
    epoch_accuracy = correct / processed * 100
    print(f"  [{phase}] epoch {epoch} finished | loss {epoch_loss:.4f} | "
          f"accuracy {epoch_accuracy:.2f}% | time {format_time(epoch_time)} | GPU {gpu_memory()}")
    return epoch_loss, epoch_accuracy


def train_model():
    print_header()
    check_dataset(TRAIN_DIR, "TRAIN")
    check_dataset(VAL_DIR, "VAL")

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=eval_transform)

    print("\nClass mapping:", train_dataset.class_to_idx)
    assert train_dataset.class_to_idx == {"fake": 0, "real": 1}, \
        "Class mapping must match the production convention (fake=0, real=1)"
    assert train_dataset.class_to_idx == val_dataset.class_to_idx, \
        "Train/val class-to-index mapping mismatch"

    class_counts = Counter(train_dataset.targets)
    num_classes = len(train_dataset.classes)
    total_samples = len(train_dataset)
    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(DEVICE)
    print("Class counts   :", {train_dataset.classes[i]: class_counts[i] for i in range(num_classes)})
    print("Class weights  :", {train_dataset.classes[i]: round(class_weights[i].item(), 4) for i in range(num_classes)})

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = create_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))

    best_val_accuracy = 0.0
    training_start = time.time()

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, epoch, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, epoch, train=False)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)
        scheduler.step()
        print(f"  LR after epoch {epoch}: {current_lr:.6f} -> {optimizer.param_groups[0]['lr']:.6f}")

        torch.save(model.state_dict(), str(CHECKPOINT_DIR / "last.pth"))

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), str(MODEL_PATH))
            torch.save(model.state_dict(), str(CHECKPOINT_DIR / f"best_epoch{epoch}_valacc{val_acc:.2f}.pth"))
            print(f"  *** New best VAL accuracy: {val_acc:.2f}% -- saved to {MODEL_PATH}")

    writer.close()
    total_time = time.time() - training_start

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best VAL accuracy : {best_val_accuracy:.2f}%")
    print(f"Total time        : {format_time(total_time)}")
    print(f"Model saved to    : {MODEL_PATH}")
    print("=" * 80)
    print("visual_model.pth (production baseline) was NOT touched by this run.")
    print("=" * 80)


if __name__ == "__main__":
    train_model()

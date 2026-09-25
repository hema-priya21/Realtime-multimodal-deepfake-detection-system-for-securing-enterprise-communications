"""Trains the first honest audio deepfake CNN baseline on ASVspoof2019 LA
train/dev log-Mel features (see prepare_mels.py). Eval is never opened here.

Class mapping (unchanged, matches phase3_audio/manifests/*.csv):
    0 = bonafide
    1 = spoof
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_DIR = PROJECT_ROOT / "phase3_audio" / "manifests"
FEATURES_ROOT = PROJECT_ROOT / "phase3_audio" / "features"
CHECKPOINT_DIR = PROJECT_ROOT / "phase3_audio" / "checkpoints"

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Class weights already verified during the dataset-preparation phase from
# the official train-split counts (2,580 bonafide / 22,800 spoof). Computed
# dynamically below from the manifest (not hardcoded) and asserted to match
# these reference values, so a dataset change would be caught, not silently
# used with stale weights.
EXPECTED_BONAFIDE_WEIGHT = 4.918605
EXPECTED_SPOOF_WEIGHT = 0.556579

LABEL_TO_ID = {"bonafide": 0, "spoof": 1}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    USE_AMP = True
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False


def fatal(message):
    print(f"\nFATAL: {message}", flush=True)
    sys.exit(1)


class AudioFeatureDataset(Dataset):
    """Lazily loads one (80, 401) log-Mel .npy feature per item, keyed off
    the manifest CSV written by prepare_audio_dataset.py -- the feature
    files are matched by audio_id (the manifest's own filename stem), not a
    separate manifest, so there is a single source of truth for labels."""

    def __init__(self, manifest_csv, features_dir):
        import csv
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.features_dir = features_dir

        missing = []
        for row in self.rows:
            audio_id = Path(row["audio_path"]).stem
            if not (self.features_dir / f"{audio_id}.npy").exists():
                missing.append(audio_id)
        if missing:
            fatal(f"{len(missing)} feature file(s) referenced by {manifest_csv} are missing "
                  f"under {self.features_dir}. First missing: {missing[0]}. "
                  f"Run prepare_mels.py first.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio_id = Path(row["audio_path"]).stem
        spec = np.load(self.features_dir / f"{audio_id}.npy")
        tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, 80, 401)
        label_id = int(row["label_id"])
        return tensor, label_id


class AudioCNN(nn.Module):
    """Lightweight 3-block CNN for 80-mel spectrogram input. Small enough
    to comfortably fit a 4GB GPU alongside the visual model, not intended
    to compete with large pretrained audio models -- this is a baseline."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def compute_class_weights(train_rows):
    counts = Counter(int(r["label_id"]) for r in train_rows)
    total = len(train_rows)
    num_classes = 2
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return weights, counts


def run_epoch(model, loader, criterion, optimizer, scaler, train):
    model.train(mode=train)
    total_loss, correct, processed = 0.0, 0, 0
    all_labels, all_preds, all_spoof_probs = [], [], []

    for images, labels in loader:
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

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        processed += batch_size

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_spoof_probs.extend(probs[:, 1].detach().cpu().tolist())  # P(spoof)

    epoch_loss = total_loss / processed
    epoch_acc = correct / processed * 100
    return epoch_loss, epoch_acc, all_labels, all_preds, all_spoof_probs


def dev_metrics(all_labels, all_preds, all_spoof_probs):
    acc = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_spoof_probs)  # spoof = positive class (1)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "bonafide_precision": precision[0], "bonafide_recall": recall[0], "bonafide_f1": f1[0],
        "spoof_precision": precision[1], "spoof_recall": recall[1], "spoof_f1": f1[1],
        "confusion_matrix": cm.tolist(),
    }


def main():
    print("=" * 80, flush=True)
    print("AUDIO DEEPFAKE CNN BASELINE -- ASVspoof2019 LA (train/dev only)", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | Seed: {RANDOM_SEED}", flush=True)

    train_dataset = AudioFeatureDataset(MANIFEST_DIR / "train.csv", FEATURES_ROOT / "train")
    dev_dataset = AudioFeatureDataset(MANIFEST_DIR / "dev.csv", FEATURES_ROOT / "dev")
    print(f"Train samples: {len(train_dataset):,} | Dev samples: {len(dev_dataset):,}", flush=True)

    weights, counts = compute_class_weights(train_dataset.rows)
    bonafide_w, spoof_w = weights[0], weights[1]
    print(f"Class counts (train): bonafide={counts[0]:,} spoof={counts[1]:,}", flush=True)
    print(f"Class weights (computed): bonafide={bonafide_w:.6f} spoof={spoof_w:.6f}", flush=True)

    if abs(bonafide_w - EXPECTED_BONAFIDE_WEIGHT) > 1e-3 or abs(spoof_w - EXPECTED_SPOOF_WEIGHT) > 1e-3:
        fatal(f"Computed class weights (bonafide={bonafide_w:.6f}, spoof={spoof_w:.6f}) do not match "
              f"the verified reference values (bonafide={EXPECTED_BONAFIDE_WEIGHT}, spoof={EXPECTED_SPOOF_WEIGHT}). "
              f"Dataset may have changed since verification -- stopping rather than training on stale weights.")
    print("Class weights match verified reference values.", flush=True)

    class_weights_tensor = torch.tensor([bonafide_w, spoof_w], dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=torch.cuda.is_available())
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=torch.cuda.is_available())

    model = AudioCNN(num_classes=2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_roc_auc = -1.0
    best_epoch = -1
    history = []

    print("\n" + "=" * 80, flush=True)
    print("STARTING TRAINING", flush=True)
    print("=" * 80, flush=True)

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        dev_loss, dev_acc, dev_labels, dev_preds, dev_probs = run_epoch(
            model, dev_loader, criterion, optimizer, scaler, train=False)

        metrics = dev_metrics(dev_labels, dev_preds, dev_probs)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{EPOCHS} ({epoch_time:.1f}s)", flush=True)
        print(f"  train: loss={train_loss:.4f} acc={train_acc:.2f}%", flush=True)
        print(f"  dev  : loss={dev_loss:.4f} acc={dev_acc:.2f}% roc_auc={metrics['roc_auc']:.4f}", flush=True)
        print(f"  dev  : bonafide P={metrics['bonafide_precision']:.3f} R={metrics['bonafide_recall']:.3f} "
              f"F1={metrics['bonafide_f1']:.3f} | spoof P={metrics['spoof_precision']:.3f} "
              f"R={metrics['spoof_recall']:.3f} F1={metrics['spoof_f1']:.3f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "dev_loss": dev_loss, "dev_acc": dev_acc,
            **metrics,
            "epoch_time_seconds": epoch_time,
        })

        torch.save(model.state_dict(), str(CHECKPOINT_DIR / "audio_baseline_last.pth"))

        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), str(CHECKPOINT_DIR / "audio_baseline_best.pth"))
            print(f"  *** New best dev ROC-AUC: {best_roc_auc:.4f} -- checkpoint saved", flush=True)

    total_time = time.time() - training_start

    print("\n" + "=" * 80, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Best epoch: {best_epoch} | Best dev ROC-AUC: {best_roc_auc:.4f}", flush=True)
    print(f"Total training time: {total_time:.1f}s", flush=True)

    # -------------------------------------------------------------------
    # TASK 9 -- final honest evaluation on dev using the BEST checkpoint
    # -------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("FINAL DEV EVALUATION (best checkpoint, dev split only -- eval untouched)", flush=True)
    print("=" * 80, flush=True)

    best_model = AudioCNN(num_classes=2).to(DEVICE)
    best_model.load_state_dict(torch.load(str(CHECKPOINT_DIR / "audio_baseline_best.pth"), map_location=DEVICE))
    best_model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in dev_loader:
            images = images.to(DEVICE)
            outputs = best_model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    final_metrics = dev_metrics(all_labels, all_preds, all_probs)
    cm = np.array(final_metrics["confusion_matrix"])
    report = classification_report(all_labels, all_preds, target_names=["bonafide", "spoof"],
                                    digits=4, zero_division=0)

    print(f"Dev samples evaluated: {len(all_labels):,}", flush=True)
    print("Confusion matrix (rows=actual, cols=predicted) [bonafide, spoof]:", flush=True)
    print(f"  bonafide {cm[0]}", flush=True)
    print(f"  spoof    {cm[1]}", flush=True)
    print(f"Accuracy: {final_metrics['accuracy']*100:.2f}%", flush=True)
    print(f"ROC-AUC : {final_metrics['roc_auc']:.4f}", flush=True)
    print(report, flush=True)

    macro_f1 = (final_metrics["bonafide_f1"] + final_metrics["spoof_f1"]) / 2
    total = cm.sum()
    weighted_f1 = (final_metrics["bonafide_f1"] * cm[0].sum() + final_metrics["spoof_f1"] * cm[1].sum()) / total
    print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}", flush=True)

    # Bias check: is the model just predicting spoof for (almost) everything?
    bonafide_recall = final_metrics["bonafide_recall"]
    spoof_recall = final_metrics["spoof_recall"]
    print("\nClass-imbalance bias check:", flush=True)
    print(f"  bonafide recall={bonafide_recall:.3f}, spoof recall={spoof_recall:.3f}", flush=True)
    if bonafide_recall < 0.5 and spoof_recall > 0.95:
        print("  -> Evidence of bias toward predicting spoof: bonafide recall is low while "
              "spoof recall is very high, despite class weighting.", flush=True)
    elif bonafide_recall > 0.5:
        print("  -> No strong evidence of spoof bias: bonafide recall is above chance, "
              "suggesting the class weighting is having a real effect.", flush=True)
    else:
        print("  -> Mixed signal: recall values do not clearly indicate either outcome; "
              "report both numbers rather than asserting a conclusion.", flush=True)

    # -------------------------------------------------------------------
    # TASK 10 -- save training history + full reproducibility config
    # -------------------------------------------------------------------
    config = {
        "preprocessing": {
            "sample_rate": 16000, "n_fft": 512, "hop_length": 160, "n_mels": 80,
            "fmin": 20, "fmax": 7600, "target_seconds": 4.0, "target_samples": 64000,
            "spectrogram_shape": [80, 401],
            "normalization": "log-mel dB clipped to [-80,0], rescaled to [0,1]",
        },
        "model": "AudioCNN (3 conv blocks: 16/32/64 channels, BN+ReLU+MaxPool, "
                  "AdaptiveAvgPool, FC 64->32->2)",
        "model_parameters": n_params,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "random_seed": RANDOM_SEED,
        "class_weights": {"bonafide": bonafide_w, "spoof": spoof_w},
        "class_mapping": {"bonafide": 0, "spoof": 1},
        "best_epoch": best_epoch,
        "best_dev_roc_auc": best_roc_auc,
        "final_dev_metrics": final_metrics,
        "final_dev_macro_f1": macro_f1,
        "final_dev_weighted_f1": weighted_f1,
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "eval_used": False,
    }

    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "epochs": history}, f, indent=2)
    print(f"\nSaved training history + config -> {history_path}", flush=True)
    print(f"Best checkpoint -> {CHECKPOINT_DIR / 'audio_baseline_best.pth'}", flush=True)
    print(f"Last checkpoint -> {CHECKPOINT_DIR / 'audio_baseline_last.pth'}", flush=True)


if __name__ == "__main__":
    main()

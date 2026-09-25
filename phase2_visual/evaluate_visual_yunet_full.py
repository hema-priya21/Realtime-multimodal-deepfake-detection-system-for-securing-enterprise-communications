"""Honest evaluation of the yunet_full-crop-trained checkpoint on the SAME
sealed, subject-disjoint TEST split used for the production baseline
(visual_model.pth, 82.52% accuracy). Same protocol as evaluate_visual.py,
plus the identity/video-level false-positive breakdown used in the forensic
report, so the two runs are directly, apples-to-apples comparable.

Never writes to data/visual_processed/, dataset_pipeline/splits/, or
visual_model.pth -- read-only against those; only writes its own report file.

Run as: python phase2_visual/evaluate_visual_yunet_full.py
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.config import SPLITS_DIR
from dataset_pipeline.face_transforms import eval_transform

TEST_DIR = PROJECT_ROOT / "data" / "visual_processed_yunet_full" / "test"
MODEL_PATH = PROJECT_ROOT / "phase2_visual" / "checkpoints" / "visual_model_yunet_full_best.pth"
REPORT_PATH = PROJECT_ROOT / "phase2_visual" / "experiments" / "yunet_full_evaluation.json"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CROP_NAME_RE = re.compile(r"^(?P<stem>.+)_frame(?P<frame>\d+)_face(?P<face>\d+)\.jpg$")


def safe_token(text):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def video_stem_of(row):
    p = Path(row["filepath"])
    return f"{row['dataset']}_{safe_token(p.parent.name)}_{safe_token(p.stem)}"


def load_model():
    if not MODEL_PATH.exists():
        print(f"ERROR: no trained model at {MODEL_PATH}. Run train_visual_yunet_full.py first.")
        raise SystemExit(1)
    model = models.mobilenet_v3_small(weights=None)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, 2)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def main():
    if not TEST_DIR.exists():
        print(f"ERROR: {TEST_DIR} not found. Run the yunet_full extraction first.")
        raise SystemExit(1)

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)
    class_to_idx = test_dataset.class_to_idx
    print("Class mapping   :", class_to_idx)
    assert class_to_idx == {"fake": 0, "real": 1}, "Class mapping drifted from the production convention"
    fake_idx, real_idx = class_to_idx["fake"], class_to_idx["real"]

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print("Test set        :", TEST_DIR)
    print("Test images     :", f"{len(test_dataset):,}")
    print("Device          :", DEVICE)

    model = load_model()

    all_labels, all_preds, all_fake_probs = [], [], []
    with torch.no_grad():
        n_done = 0
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_fake_probs.extend(probs[:, fake_idx].cpu().tolist())
            n_done += len(labels)
            print(f"\r  {n_done}/{len(test_dataset)}", end="", flush=True)
    print()

    accuracy = accuracy_score(all_labels, all_preds)
    prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=[fake_idx, real_idx])
    macro_f1 = f1.mean()
    _, _, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    roc_auc = roc_auc_score([1 if lbl == fake_idx else 0 for lbl in all_labels], all_fake_probs)
    cm = confusion_matrix(all_labels, all_preds, labels=[fake_idx, real_idx])

    print(f"\nAccuracy   : {accuracy*100:.2f}%")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print(f"Fake  P/R/F1: {prec[0]*100:.2f}% / {rec[0]*100:.2f}% / {f1[0]*100:.2f}%")
    print(f"Real  P/R/F1: {prec[1]*100:.2f}% / {rec[1]*100:.2f}% / {f1[1]*100:.2f}%")
    print(f"Macro F1    : {macro_f1*100:.2f}%   Weighted F1: {weighted_f1*100:.2f}%")
    print("Confusion matrix [rows=actual, cols=predicted], order=[fake, real]:")
    print(cm)
    print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))

    # --- identity / video-level false-positive breakdown (same method as
    # the forensic report, applied here to the yunet_full checkpoint) ---
    with open(SPLITS_DIR / "test.csv", newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    stem_to_row = {video_stem_of(r): r for r in test_rows}

    per_video = defaultdict(lambda: {"pfake": [], "true": None})
    paths = [p for p, _ in test_dataset.samples]
    unmatched = 0
    for path, true_c, pf in zip(paths, all_labels, all_fake_probs):
        m = CROP_NAME_RE.match(Path(path).name)
        if not m or m.group("stem") not in stem_to_row:
            unmatched += 1
            continue
        v = per_video[m.group("stem")]
        v["pfake"].append(pf)
        v["true"] = "fake" if true_c == fake_idx else "real"
    print(f"\nCrops matched back to a CSV video row: {len(paths) - unmatched}/{len(paths)}")

    real_videos = [(stem, sum(v["pfake"]) / len(v["pfake"])) for stem, v in per_video.items() if v["true"] == "real"]
    real_video_wrong = sum(1 for _, mean in real_videos if mean >= 0.5)
    real_crop_total = sum(1 for c in all_labels if c == real_idx)
    real_crop_wrong = sum(1 for c, p in zip(all_labels, all_preds) if c == real_idx and p == fake_idx)

    print(f"\nReal-class crop-level false-positive-as-FAKE rate: "
          f"{real_crop_wrong}/{real_crop_total} = {real_crop_wrong/real_crop_total*100:.1f}%")
    print(f"Real identity/video-level false-positive rate: "
          f"{real_video_wrong}/{len(real_videos)} = {real_video_wrong/len(real_videos)*100:.1f}%")

    report = {
        "model_path": str(MODEL_PATH), "test_dir": str(TEST_DIR), "n_test_images": len(test_dataset),
        "accuracy": accuracy, "roc_auc": roc_auc,
        "fake_precision": prec[0], "fake_recall": rec[0], "fake_f1": f1[0],
        "real_precision": prec[1], "real_recall": rec[1], "real_f1": f1[1],
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "confusion_matrix_fake_real_order": cm.tolist(),
        "real_crop_false_positive_rate": real_crop_wrong / real_crop_total,
        "real_crop_wrong": real_crop_wrong, "real_crop_total": real_crop_total,
        "real_identity_false_positive_rate": real_video_wrong / len(real_videos),
        "real_identity_wrong": real_video_wrong, "real_identity_total": len(real_videos),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote", REPORT_PATH)


if __name__ == "__main__":
    main()

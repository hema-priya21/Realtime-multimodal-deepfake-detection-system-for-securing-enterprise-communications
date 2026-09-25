"""Sealed, one-time evaluation of the yunet_clean checkpoint on the held-out
TEST split (data/visual_processed_yunet_full/test) -- never touched during
training. Same protocol as evaluate_visual.py, plus the identity/video-level
false-positive breakdown and a deterministic sanity-check sample table.

Read-only against data/visual_processed_yunet_full/, dataset_pipeline/splits/,
and the checkpoint; only writes its own JSON report.

Run as: python phase2_visual/evaluate_visual_yunet_clean.py
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
from dataset_pipeline.subject_utils import get_label

TEST_DIR = PROJECT_ROOT / "data" / "visual_processed_yunet_full" / "test"
MODEL_PATH = PROJECT_ROOT / "phase2_visual" / "checkpoints" / "visual_model_yunet_clean_best.pth"
REPORT_PATH = PROJECT_ROOT / "phase2_visual" / "experiments" / "yunet_clean_evaluation.json"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_RE = re.compile(r"^(?P<stem>.+)_frame(?P<frame>\d+)_face(?P<face>\d+)\.jpg$")

BASELINE = {
    "accuracy": 0.8252, "roc_auc": 0.8821,
    "fake_precision": 0.9328, "fake_recall": 0.8512, "fake_f1": 0.8901,
    "real_precision": 0.4870, "real_recall": 0.6973, "real_f1": 0.5668,
    "macro_f1": 0.73, "weighted_f1": 0.84,
}


def safe_token(text):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def video_stem_of(row):
    p = Path(row["filepath"])
    return f"{row['dataset']}_{safe_token(p.parent.name)}_{safe_token(p.stem)}"


def load_model():
    if not MODEL_PATH.exists():
        print(f"ERROR: no trained model at {MODEL_PATH}. Run train_visual_yunet_clean.py first.")
        raise SystemExit(1)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
    return model.to(DEVICE).eval()


def main():
    if not TEST_DIR.exists():
        print(f"ERROR: {TEST_DIR} not found.")
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
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[fake_idx, real_idx])
    macro_f1 = f1.mean()
    _, _, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    roc_auc = roc_auc_score([1 if lbl == fake_idx else 0 for lbl in all_labels], all_fake_probs)
    cm = confusion_matrix(all_labels, all_preds, labels=[fake_idx, real_idx])
    pred_dist = Counter(all_preds)

    print(f"\n{'METRIC':22s} {'NEW (yunet_clean)':>18s} {'BASELINE (production)':>22s} {'DIFF':>10s}")
    rows = [
        ("Accuracy", accuracy, BASELINE["accuracy"]),
        ("ROC-AUC", roc_auc, BASELINE["roc_auc"]),
        ("Fake precision", prec[0], BASELINE["fake_precision"]),
        ("Fake recall", rec[0], BASELINE["fake_recall"]),
        ("Fake F1", f1[0], BASELINE["fake_f1"]),
        ("Real precision", prec[1], BASELINE["real_precision"]),
        ("Real recall", rec[1], BASELINE["real_recall"]),
        ("Real F1", f1[1], BASELINE["real_f1"]),
        ("Macro F1", macro_f1, BASELINE["macro_f1"]),
        ("Weighted F1", weighted_f1, BASELINE["weighted_f1"]),
    ]
    for name, new_v, base_v in rows:
        print(f"{name:22s} {new_v*100:17.2f}% {base_v*100:21.2f}% {(new_v-base_v)*100:+9.2f}pp")

    print("\nConfusion matrix [rows=actual, cols=predicted], order=[fake, real]:")
    print(cm)
    print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))
    print(f"Prediction distribution across ALL {len(all_preds)} test crops: "
          f"predicted-fake={pred_dist.get(fake_idx,0)} ({pred_dist.get(fake_idx,0)/len(all_preds)*100:.1f}%), "
          f"predicted-real={pred_dist.get(real_idx,0)} ({pred_dist.get(real_idx,0)/len(all_preds)*100:.1f}%)  "
          f"(true distribution: fake={sum(1 for l in all_labels if l==fake_idx)}, real={sum(1 for l in all_labels if l==real_idx)})")

    # --- identity / video-level false-positive breakdown ---
    with open(SPLITS_DIR / "test.csv", newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    stem_to_row = {video_stem_of(r): r for r in test_rows}

    per_video = defaultdict(lambda: {"pfake": [], "true": None, "method": None, "subject_id": None})
    paths = [p for p, _ in test_dataset.samples]
    unmatched = 0
    for path, true_c, pf in zip(paths, all_labels, all_fake_probs):
        m = CROP_RE.match(Path(path).name)
        row = stem_to_row.get(m.group("stem")) if m else None
        if not m or row is None:
            unmatched += 1
            continue
        v = per_video[m.group("stem")]
        v["pfake"].append(pf)
        v["true"] = "fake" if true_c == fake_idx else "real"
        v["method"] = Path(row["filepath"]).parent.name
        v["subject_id"] = row["subject_id"]
    print(f"\nCrops matched back to a CSV video row: {len(paths) - unmatched}/{len(paths)}")

    real_videos = [(stem, sum(v["pfake"]) / len(v["pfake"]), v["subject_id"])
                   for stem, v in per_video.items() if v["true"] == "real"]
    real_video_wrong = [(stem, mean, sid) for stem, mean, sid in real_videos if mean >= 0.5]
    real_crop_total = sum(1 for c in all_labels if c == real_idx)
    real_crop_wrong = sum(1 for c, p in zip(all_labels, all_preds) if c == real_idx and p == fake_idx)

    print(f"\nReal -> Fake crop-level false-positive rate: "
          f"{real_crop_wrong}/{real_crop_total} = {real_crop_wrong/real_crop_total*100:.1f}%")
    print(f"Number of real test identities classified as predominantly FAKE (video mean P(fake)>=0.5): "
          f"{len(real_video_wrong)} / {len(real_videos)}")
    print(f"Video/identity-level real-person false-positive rate: "
          f"{len(real_video_wrong)/len(real_videos)*100:.1f}%")
    for stem, mean, sid in sorted(real_video_wrong, key=lambda x: -x[1])[:15]:
        print(f"   {stem:35s} subj={sid:15s} mean_P(fake)={mean:.3f}")

    # --- STEP 7: deterministic behavior sanity check ---
    print("\n" + "=" * 78)
    print("SANITY CHECK -- diverse real + fake samples across identities/methods")
    print("=" * 78)
    real_paths = sorted(p for p, c in test_dataset.samples if c == real_idx)
    fake_paths = sorted(p for p, c in test_dataset.samples if c == fake_idx)
    pick_real = real_paths[::max(1, len(real_paths)//8)][:8]
    pick_fake = fake_paths[::max(1, len(fake_paths)//8)][:8]
    print(f"{'file':45s} {'true':5s} {'pred':5s} {'P(fake)':8s} {'P(real)':8s}")
    with torch.no_grad():
        for true_lbl, plist in (("real", pick_real), ("fake", pick_fake)):
            for p in plist:
                img = test_dataset.loader(p)
                x = eval_transform(img).unsqueeze(0).to(DEVICE)
                probs = torch.softmax(model(x)[0], dim=0)
                pred = "fake" if probs[fake_idx] > probs[real_idx] else "real"
                print(f"{Path(p).name:45s} {true_lbl:5s} {pred:5s} "
                      f"{probs[fake_idx].item():8.4f} {probs[real_idx].item():8.4f}")

    report = {
        "model_path": str(MODEL_PATH), "test_dir": str(TEST_DIR), "n_test_images": len(test_dataset),
        "accuracy": accuracy, "roc_auc": roc_auc,
        "fake_precision": prec[0], "fake_recall": rec[0], "fake_f1": f1[0],
        "real_precision": prec[1], "real_recall": rec[1], "real_f1": f1[1],
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "confusion_matrix_fake_real_order": cm.tolist(),
        "real_crop_false_positive_rate": real_crop_wrong / real_crop_total,
        "real_crop_wrong": real_crop_wrong, "real_crop_total": real_crop_total,
        "real_identity_false_positive_count": len(real_video_wrong),
        "real_identity_total": len(real_videos),
        "real_identity_false_positive_rate": len(real_video_wrong) / len(real_videos),
        "baseline": BASELINE,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote", REPORT_PATH)


if __name__ == "__main__":
    main()

"""Honest, one-time evaluation of the trained visual model on the held-out
TEST split -- the only split neither training nor checkpoint-selection ever
saw. Reports accuracy, precision, recall, F1, ROC-AUC, and a confusion
matrix, instead of the single training-accuracy number the old pipeline
reported.

Run as: python phase2_visual/evaluate_visual.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.config import PROCESSED_ROOT
from dataset_pipeline.face_transforms import eval_transform

TEST_DIR = PROCESSED_ROOT / "test"
MODEL_PATH = PROJECT_ROOT / "phase2_visual" / "visual_model.pth"
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    if not MODEL_PATH.exists():
        print(f"ERROR: no trained model at {MODEL_PATH}. Run train_visual.py first.")
        raise SystemExit(1)

    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def main():
    if not TEST_DIR.exists():
        print(f"ERROR: {TEST_DIR} not found. Run dataset_pipeline.extract_frames first.")
        raise SystemExit(1)

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_to_idx = test_dataset.class_to_idx
    fake_class_idx = class_to_idx["fake"]

    print("=" * 80)
    print("HONEST TEST-SET EVALUATION (held out from training and val entirely)")
    print("=" * 80)
    print("Test set        :", TEST_DIR)
    print("Class mapping   :", class_to_idx)
    print("Test images     :", f"{len(test_dataset):,}")
    print("Device          :", DEVICE)
    print("=" * 80)

    model = load_model()

    all_labels = []
    all_preds = []
    all_fake_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_fake_probs.extend(probs[:, fake_class_idx].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=fake_class_idx)
    recall = recall_score(all_labels, all_preds, pos_label=fake_class_idx)
    f1 = f1_score(all_labels, all_preds, pos_label=fake_class_idx)
    roc_auc = roc_auc_score(
        [1 if lbl == fake_class_idx else 0 for lbl in all_labels],
        all_fake_probs,
    )
    cm = confusion_matrix(all_labels, all_preds)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print(f"Precision  : {precision * 100:.2f}%  (positive class = fake)")
    print(f"Recall     : {recall * 100:.2f}%  (positive class = fake)")
    print(f"F1 score   : {f1 * 100:.2f}%")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print()
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"           {list(class_to_idx.keys())}")
    for i, row in enumerate(cm):
        label = list(class_to_idx.keys())[i]
        print(f"  {label:6s} {row}")
    print()
    print(classification_report(all_labels, all_preds, target_names=list(class_to_idx.keys())))
    print("=" * 80)
    print("These numbers are from a subject-disjoint test split the model has")
    print("never seen in any form during training or checkpoint selection.")
    print("=" * 80)


if __name__ == "__main__":
    main()

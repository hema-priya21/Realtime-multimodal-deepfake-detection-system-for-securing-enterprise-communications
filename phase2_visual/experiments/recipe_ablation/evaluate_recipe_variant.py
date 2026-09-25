"""READ-ONLY evaluation of one recipe-ablation checkpoint on the sealed
YuNet test set (data/visual_processed_yunet_full/test, the exact same set
used throughout this project's model comparisons). Computes the full metric
suite requested for the recipe ablation: sealed-test metrics,
calibration (Brier/ECE, same 10-bin method as
model_calibration_boundary_diagnostic.py), and identity-level analysis
including churn against the PRODUCTION model.

Reuses production's per-sample predictions from the already-computed
phase2_visual/experiments/model_calibration_boundary_predictions.csv rather
than re-running the production model each time (identical result, no
redundant compute, and guarantees the exact same production reference
across every experiment in this ablation).

Writes phase2_visual/experiments/recipe_ablation/results/experiment_<X>_metrics.json
and .../experiment_<X>_predictions.csv. Does not modify any production file,
the existing yunet_clean checkpoint, or any split CSV.

Run as: python phase2_visual/experiments/recipe_ablation/evaluate_recipe_variant.py --variant A
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    confusion_matrix, brier_score_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.config import SPLITS_DIR
from dataset_pipeline.face_transforms import eval_transform

TEST_DIR = PROJECT_ROOT / "data" / "visual_processed_yunet_full" / "test"
ABLATION_DIR = PROJECT_ROOT / "phase2_visual" / "experiments" / "recipe_ablation"
CHECKPOINT_ROOT = ABLATION_DIR / "checkpoints"
RESULTS_DIR = ABLATION_DIR / "results"
PRODUCTION_REF_CSV = PROJECT_ROOT / "phase2_visual" / "experiments" / "model_calibration_boundary_predictions.csv"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ECE_BINS = 10
CROP_RE = re.compile(r"^(?P<stem>.+)_frame(?P<frame>\d+)_face(?P<face>\d+)\.jpg$")


def safe_token(text):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def video_stem_of(row):
    p = Path(row["filepath"])
    return f"{row['dataset']}_{safe_token(p.parent.name)}_{safe_token(p.stem)}"


def load_model(ckpt_path):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    return model.to(DEVICE).eval()


def run_inference(model, loader, n_total, tag):
    all_fake_probs = []
    with torch.no_grad():
        n_done = 0
        for images, _ in loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)
            all_fake_probs.extend(probs[:, 0].cpu().tolist())
            n_done += len(images)
            print(f"\r  [{tag}] {n_done}/{n_total}", end="", flush=True)
    print()
    return all_fake_probs


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def expected_calibration_error(y_true_fake, p_fake, n_bins=N_ECE_BINS):
    """Identical definition to model_calibration_boundary_diagnostic.py:
    equal-width bins of the winning-class confidence max(p,1-p)."""
    y_true_fake = np.asarray(y_true_fake)
    p_fake = np.asarray(p_fake)
    confidence = np.maximum(p_fake, 1 - p_fake)
    pred_fake = (p_fake >= 0.5).astype(int)
    correct = (pred_fake == y_true_fake).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p_fake)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi) if i == n_bins - 1 else (confidence >= lo) & (confidence < hi)
        count = mask.sum()
        if count == 0:
            continue
        ece += (count / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def load_production_reference():
    if not PRODUCTION_REF_CSV.exists():
        print(f"ERROR: production reference CSV not found at {PRODUCTION_REF_CSV}")
        raise SystemExit(1)
    with open(PRODUCTION_REF_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["file"]: float(r["production_pfake"]) for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["A", "B", "C", "D"])
    args = parser.parse_args()
    variant = args.variant
    ckpt_path = CHECKPOINT_ROOT / f"experiment_{variant}_best.pth"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    print(f"=== Evaluating Experiment {variant}: {ckpt_path} ===")
    prod_by_file = load_production_reference()
    print(f"Loaded {len(prod_by_file)} production reference predictions from {PRODUCTION_REF_CSV}")

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)
    class_to_idx = test_dataset.class_to_idx
    assert class_to_idx == {"fake": 0, "real": 1}
    fake_idx, real_idx = class_to_idx["fake"], class_to_idx["real"]
    paths = [p for p, _ in test_dataset.samples]
    true_labels_idx = [c for _, c in test_dataset.samples]
    y_true_fake = [1 if c == fake_idx else 0 for c in true_labels_idx]

    with open(SPLITS_DIR / "test.csv", newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    stem_to_row = {video_stem_of(r): r for r in test_rows}

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model = load_model(ckpt_path)
    pfake = run_inference(model, loader, len(test_dataset), variant)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pred_idx = [fake_idx if p >= 0.5 else real_idx for p in pfake]
    real_mask = [c == real_idx for c in true_labels_idx]
    fake_mask = [c == fake_idx for c in true_labels_idx]

    # --- sealed test metrics ---
    accuracy = sum(1 for a, b in zip(true_labels_idx, pred_idx) if a == b) / len(true_labels_idx)
    roc_auc = roc_auc_score(y_true_fake, pfake)
    pr_auc_fake = average_precision_score(y_true_fake, pfake)
    y_true_real = [1 - v for v in y_true_fake]
    preal = [1 - p for p in pfake]
    pr_auc_real = average_precision_score(y_true_real, preal)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels_idx, pred_idx, labels=[fake_idx, real_idx])
    macro_f1 = f1.mean()
    _, _, weighted_f1, _ = precision_recall_fscore_support(true_labels_idx, pred_idx, average="weighted")
    cm = confusion_matrix(true_labels_idx, pred_idx, labels=[fake_idx, real_idx])
    real_total = sum(real_mask)
    real_fp = sum(1 for p, m in zip(pfake, real_mask) if m and p >= 0.5)
    real_fp_rate = real_fp / real_total

    # --- calibration ---
    brier = brier_score_loss(y_true_fake, pfake)
    ece = expected_calibration_error(y_true_fake, pfake)
    prod_real = sorted(pf for pf, m in zip(pfake, real_mask) if m)
    prod_fake = sorted(pf for pf, m in zip(pfake, fake_mask) if m)
    real_stats = {"mean": statistics.mean(prod_real), "median": statistics.median(prod_real),
                  "p90": percentile(prod_real, 0.90), "p95": percentile(prod_real, 0.95)}
    fake_stats = {"mean": statistics.mean(prod_fake), "median": statistics.median(prod_fake),
                  "p10": percentile(prod_fake, 0.10), "p05": percentile(prod_fake, 0.05)}

    # --- per-sample CSV + identity-level analysis (vs. production reference) ---
    csv_path = RESULTS_DIR / f"experiment_{variant}_predictions.csv"
    per_video = defaultdict(lambda: {"pfake": [], "prod_pfake": [], "true": None, "subject_id": None})
    unmatched = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "stem", "method", "subject_id", "true_label", "pfake", "pred",
                    "production_pfake_ref", "delta_vs_production"])
        for path, true_c, pf in zip(paths, true_labels_idx, pfake):
            name = Path(path).name
            m = CROP_RE.match(name)
            stem = m.group("stem") if m else ""
            row = stem_to_row.get(stem)
            method = Path(row["filepath"]).parent.name if row else ""
            subject_id = row["subject_id"] if row else ""
            true_l = "fake" if true_c == fake_idx else "real"
            pred_l = "fake" if pf >= 0.5 else "real"
            prod_ref = prod_by_file.get(name)
            delta = (pf - prod_ref) if prod_ref is not None else None
            w.writerow([name, stem, method, subject_id, true_l, f"{pf:.6f}", pred_l,
                        f"{prod_ref:.6f}" if prod_ref is not None else "", f"{delta:+.6f}" if delta is not None else ""])
            if not m or row is None:
                unmatched += 1
                continue
            v = per_video[stem]
            v["pfake"].append(pf)
            v["true"] = true_l
            v["subject_id"] = subject_id
            if prod_ref is not None:
                v["prod_pfake"].append(prod_ref)

    real_identities = {stem: v for stem, v in per_video.items() if v["true"] == "real"}
    n_identities = len(real_identities)
    id_rows = []
    for stem, v in real_identities.items():
        mean_pf = statistics.mean(v["pfake"])
        prod_mean = statistics.mean(v["prod_pfake"]) if v["prod_pfake"] else None
        id_rows.append({"stem": stem, "subject_id": v["subject_id"], "mean_pfake": mean_pf,
                         "prod_mean_pfake": prod_mean})
    n_id_fp = sum(1 for r in id_rows if r["mean_pfake"] >= 0.5)
    id_fp_rate = n_id_fp / n_identities if n_identities else None
    newly_broken = [r for r in id_rows if r["prod_mean_pfake"] is not None
                    and r["prod_mean_pfake"] < 0.5 <= r["mean_pfake"]]
    fixed = [r for r in id_rows if r["prod_mean_pfake"] is not None
             and r["prod_mean_pfake"] >= 0.5 > r["mean_pfake"]]

    metrics = {
        "variant": variant, "checkpoint": str(ckpt_path),
        "n_test": len(test_dataset), "n_real": real_total, "n_fake": len(test_dataset) - real_total,
        "unmatched_crops": unmatched,
        "accuracy": accuracy, "roc_auc": roc_auc, "pr_auc_fake_positive": pr_auc_fake,
        "pr_auc_real_positive": pr_auc_real,
        "fake_precision": prec[0], "fake_recall": rec[0], "fake_f1": f1[0],
        "real_precision": prec[1], "real_recall": rec[1], "real_f1": f1[1],
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "confusion_matrix_fake_real_order": cm.tolist(),
        "real_false_positive_rate": real_fp_rate, "real_fp_count": real_fp, "real_total": real_total,
        "brier_score": brier, "ece": ece,
        "real_pfake_mean": real_stats["mean"], "real_pfake_median": real_stats["median"],
        "real_pfake_p90": real_stats["p90"], "real_pfake_p95": real_stats["p95"],
        "fake_pfake_mean": fake_stats["mean"], "fake_pfake_median": fake_stats["median"],
        "fake_pfake_p10": fake_stats["p10"], "fake_pfake_p05": fake_stats["p05"],
        "n_real_identities": n_identities,
        "pct_real_identities_fp": (n_id_fp / n_identities * 100) if n_identities else None,
        "identity_fp_count": n_id_fp, "identity_fp_rate": id_fp_rate,
        "identity_newly_broken_vs_production": len(newly_broken),
        "identity_fixed_vs_production": len(fixed),
        "identity_newly_broken_list": [r["stem"] for r in newly_broken],
        "identity_fixed_list": [r["stem"] for r in fixed],
    }

    json_path = RESULTS_DIR / f"experiment_{variant}_metrics.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"EXPERIMENT {variant} -- SEALED TEST SUMMARY")
    print("=" * 90)
    print(f"Accuracy={accuracy*100:.2f}% ROC-AUC={roc_auc:.4f} PR-AUC(real+)={pr_auc_real:.4f} "
          f"MacroF1={macro_f1*100:.2f}%")
    print(f"Real FP rate={real_fp_rate*100:.2f}% Fake recall={rec[0]*100:.2f}% Real recall={rec[1]*100:.2f}%")
    print(f"Brier={brier:.4f} ECE={ece:.4f}")
    print(f"Real P(fake): mean={real_stats['mean']:.4f} p90={real_stats['p90']:.4f} p95={real_stats['p95']:.4f}")
    print(f"Fake P(fake): mean={fake_stats['mean']:.4f} p10={fake_stats['p10']:.4f} p05={fake_stats['p05']:.4f}")
    print(f"Identity FP rate={id_fp_rate*100:.2f}% ({n_id_fp}/{n_identities})  "
          f"newly_broken_vs_prod={len(newly_broken)} fixed_vs_prod={len(fixed)}")
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

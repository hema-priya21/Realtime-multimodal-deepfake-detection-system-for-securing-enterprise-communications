"""READ-ONLY controlled model-difference / calibration diagnostic: production
visual_model.pth vs. the new yunet_clean checkpoint, both evaluated on the
exact same sealed YuNet test set (data/visual_processed_yunet_full/test,
12,009 crops -- the same set used in the earlier same-test-set comparison and
root-cause audit).

Determines whether the new model's extreme live P(fake) behavior (documented
in yunet_clean_root_cause_audit.md and live_camera_ab_test_report.md) is
primarily a calibration issue (probabilities badly scaled but ranking intact),
a decision-boundary/generalization issue (the real/fake distributions overlap
differently), an identity-specific generalization issue, or a combination.

Does not modify phase4_fusion_alerting/web_app.py, does not modify or
overwrite phase2_visual/visual_model.pth, does not integrate the new
checkpoint anywhere, does not change any threshold/fusion weight/hysteresis,
and does not train anything. Only reads the two checkpoints and the sealed
test set; only writes its own report + CSV under phase2_visual/experiments/.

Run as: python phase2_visual/experiments/model_calibration_boundary_diagnostic.py
"""
import csv
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.config import SPLITS_DIR
from dataset_pipeline.face_transforms import eval_transform

TEST_DIR = PROJECT_ROOT / "data" / "visual_processed_yunet_full" / "test"
PROD_CKPT = PROJECT_ROOT / "phase2_visual" / "visual_model.pth"
NEW_CKPT = PROJECT_ROOT / "phase2_visual" / "checkpoints" / "visual_model_yunet_clean_best.pth"
REPORT_PATH = PROJECT_ROOT / "phase2_visual" / "experiments" / "model_calibration_boundary_diagnostic.md"
CSV_PATH = PROJECT_ROOT / "phase2_visual" / "experiments" / "model_calibration_boundary_predictions.csv"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ECE_BINS = 10  # equal-width bins [0,0.1), [0.1,0.2), ..., [0.9,1.0] over the WINNING-class confidence
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
            all_fake_probs.extend(probs[:, 0].cpu().tolist())  # class 0 = fake
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
    """Standard equal-width-bin ECE: bins the WINNING-class confidence
    max(p_fake, 1-p_fake) into n_bins equal-width bins over [0,1], and for
    each non-empty bin computes |accuracy - mean_confidence|, weighted by the
    bin's share of all samples. This is the textbook ECE definition (Guo et
    al. 2017), stated explicitly here since the task requires a clearly
    stated binning method."""
    y_true_fake = np.asarray(y_true_fake)
    p_fake = np.asarray(p_fake)
    confidence = np.maximum(p_fake, 1 - p_fake)
    pred_fake = (p_fake >= 0.5).astype(int)
    correct = (pred_fake == y_true_fake).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p_fake)
    ece = 0.0
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = mask.sum()
        if count == 0:
            bin_rows.append((lo, hi, 0, None, None))
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)
        bin_rows.append((lo, hi, int(count), float(bin_acc), float(bin_conf)))
    return float(ece), bin_rows


def main():
    print("=== Verifying inputs ===")
    for p in (PROD_CKPT, NEW_CKPT):
        if not p.exists():
            print(f"ERROR: missing checkpoint {p}")
            raise SystemExit(1)
        print(f"  OK checkpoint: {p}")
    if not TEST_DIR.exists():
        print(f"ERROR: missing test dir {TEST_DIR}")
        raise SystemExit(1)
    test_csv_path = SPLITS_DIR / "test.csv"
    if not test_csv_path.exists():
        print(f"ERROR: missing {test_csv_path}")
        raise SystemExit(1)
    print(f"  OK test dir: {TEST_DIR}")
    print(f"  OK split csv: {test_csv_path}")

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)
    class_to_idx = test_dataset.class_to_idx
    assert class_to_idx == {"fake": 0, "real": 1}, "Class mapping drifted from the production convention"
    fake_idx, real_idx = class_to_idx["fake"], class_to_idx["real"]
    paths = [p for p, _ in test_dataset.samples]
    true_labels_idx = [c for _, c in test_dataset.samples]
    y_true_fake = [1 if c == fake_idx else 0 for c in true_labels_idx]
    print(f"  OK {len(test_dataset):,} test crops loaded, class_to_idx={class_to_idx}")

    with open(test_csv_path, newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    stem_to_row = {video_stem_of(r): r for r in test_rows}

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print("\n=== Running production model ===")
    prod_model = load_model(PROD_CKPT)
    prod_pfake = run_inference(prod_model, loader, len(test_dataset), "production")
    del prod_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print("\n=== Running new (yunet_clean) model ===")
    new_model = load_model(NEW_CKPT)
    new_pfake = run_inference(new_model, loader, len(test_dataset), "new_yunet_clean")
    del new_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prod_pred = ["fake" if p >= 0.5 else "real" for p in prod_pfake]
    new_pred = ["fake" if p >= 0.5 else "real" for p in new_pfake]
    true_label_str = ["fake" if c == fake_idx else "real" for c in true_labels_idx]
    abs_diff = [abs(n - p) for n, p in zip(new_pfake, prod_pfake)]
    disagree = [pp != np_ for pp, np_ in zip(prod_pred, new_pred)]

    # --- per-sample CSV ---
    print(f"\nWriting per-sample CSV -> {CSV_PATH}")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "stem", "method", "subject_id", "true_label",
                    "production_pfake", "new_pfake", "production_pred", "new_pred",
                    "abs_diff", "disagree"])
        for path, true_l, pp, np_, ppr, npr, ad, dis in zip(
                paths, true_label_str, prod_pfake, new_pfake, prod_pred, new_pred, abs_diff, disagree):
            m = CROP_RE.match(Path(path).name)
            stem = m.group("stem") if m else ""
            row = stem_to_row.get(stem)
            method = Path(row["filepath"]).parent.name if row else ""
            subject_id = row["subject_id"] if row else ""
            w.writerow([Path(path).name, stem, method, subject_id, true_l,
                        f"{pp:.6f}", f"{np_:.6f}", ppr, npr, f"{ad:.6f}", dis])

    # ================= 1. reliability / calibration =================
    print("\n=== Computing calibration metrics ===")
    brier_prod = brier_score_loss(y_true_fake, prod_pfake)
    brier_new = brier_score_loss(y_true_fake, new_pfake)
    ece_prod, bins_prod = expected_calibration_error(y_true_fake, prod_pfake)
    ece_new, bins_new = expected_calibration_error(y_true_fake, new_pfake)

    real_mask = [c == real_idx for c in true_labels_idx]
    fake_mask = [c == fake_idx for c in true_labels_idx]
    prod_real = sorted(p for p, m in zip(prod_pfake, real_mask) if m)
    new_real = sorted(p for p, m in zip(new_pfake, real_mask) if m)
    prod_fake = sorted(p for p, m in zip(prod_pfake, fake_mask) if m)
    new_fake = sorted(p for p, m in zip(new_pfake, fake_mask) if m)

    def stats_block(sorted_vals, high_percentiles=True):
        d = {"mean": statistics.mean(sorted_vals), "median": statistics.median(sorted_vals)}
        if high_percentiles:
            d["p90"] = percentile(sorted_vals, 0.90)
            d["p95"] = percentile(sorted_vals, 0.95)
        else:
            d["p10"] = percentile(sorted_vals, 0.10)
            d["p05"] = percentile(sorted_vals, 0.05)
        return d

    real_stats_prod = stats_block(prod_real, True)
    real_stats_new = stats_block(new_real, True)
    fake_stats_prod = stats_block(prod_fake, False)
    fake_stats_new = stats_block(new_fake, False)

    # ================= 2. threshold-independent =================
    roc_auc_prod = roc_auc_score(y_true_fake, prod_pfake)
    roc_auc_new = roc_auc_score(y_true_fake, new_pfake)
    # PR-AUC reported for BOTH choices of positive class, explicitly labeled (avoids hiding either)
    pr_auc_fake_pos_prod = average_precision_score(y_true_fake, prod_pfake)
    pr_auc_fake_pos_new = average_precision_score(y_true_fake, new_pfake)
    y_true_real = [1 - v for v in y_true_fake]
    prod_preal = [1 - p for p in prod_pfake]
    new_preal = [1 - p for p in new_pfake]
    pr_auc_real_pos_prod = average_precision_score(y_true_real, prod_preal)
    pr_auc_real_pos_new = average_precision_score(y_true_real, new_preal)

    # ================= 3. decision-boundary @0.50 =================
    prod_pred_idx = [fake_idx if p >= 0.5 else real_idx for p in prod_pfake]
    new_pred_idx = [fake_idx if p >= 0.5 else real_idx for p in new_pfake]
    cm_prod = confusion_matrix(true_labels_idx, prod_pred_idx, labels=[fake_idx, real_idx])
    cm_new = confusion_matrix(true_labels_idx, new_pred_idx, labels=[fake_idx, real_idx])
    prec_p, rec_p, f1_p, _ = precision_recall_fscore_support(true_labels_idx, prod_pred_idx, labels=[fake_idx, real_idx])
    prec_n, rec_n, f1_n, _ = precision_recall_fscore_support(true_labels_idx, new_pred_idx, labels=[fake_idx, real_idx])
    macro_f1_prod = f1_p.mean()
    macro_f1_new = f1_n.mean()
    real_total = sum(real_mask)
    real_fp_prod = sum(1 for p, m in zip(prod_pfake, real_mask) if m and p >= 0.5)
    real_fp_new = sum(1 for p, m in zip(new_pfake, real_mask) if m and p >= 0.5)
    n_disagree = sum(disagree)

    # ================= 4. probability-shift =================
    deltas_real = [n - p for n, p, m in zip(new_pfake, prod_pfake, real_mask) if m]
    deltas_fake = [n - p for n, p, m in zip(new_pfake, prod_pfake, fake_mask) if m]
    pct_real_higher = sum(1 for d in deltas_real if d > 0) / len(deltas_real) * 100
    pct_fake_higher = sum(1 for d in deltas_fake if d > 0) / len(deltas_fake) * 100

    def quantile_table(sorted_vals):
        qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        return {f"q{int(q*100)}": percentile(sorted_vals, q) for q in qs}

    real_quantiles_prod = quantile_table(prod_real)
    real_quantiles_new = quantile_table(new_real)
    fake_quantiles_prod = quantile_table(prod_fake)
    fake_quantiles_new = quantile_table(new_fake)

    # ================= 5. identity-level (real identities only) =================
    per_video = defaultdict(lambda: {"prod": [], "new": [], "true": None, "subject_id": None})
    unmatched = 0
    for path, true_l, pp, np_ in zip(paths, true_label_str, prod_pfake, new_pfake):
        m = CROP_RE.match(Path(path).name)
        row = stem_to_row.get(m.group("stem")) if m else None
        if not m or row is None:
            unmatched += 1
            continue
        v = per_video[m.group("stem")]
        v["prod"].append(pp)
        v["new"].append(np_)
        v["true"] = true_l
        v["subject_id"] = row["subject_id"]

    real_identities = {stem: v for stem, v in per_video.items() if v["true"] == "real"}
    identity_rows = []
    for stem, v in real_identities.items():
        identity_rows.append({
            "stem": stem, "subject_id": v["subject_id"],
            "prod_mean": statistics.mean(v["prod"]), "prod_median": statistics.median(v["prod"]),
            "new_mean": statistics.mean(v["new"]), "new_median": statistics.median(v["new"]),
        })
    identity_rows.sort(key=lambda r: -(r["new_mean"] - r["prod_mean"]))
    n_identities = len(identity_rows)
    prod_id_fake_pct = sum(1 for r in identity_rows if r["prod_mean"] >= 0.5) / n_identities * 100
    new_id_fake_pct = sum(1 for r in identity_rows if r["new_mean"] >= 0.5) / n_identities * 100
    newly_broken = [r for r in identity_rows if r["prod_mean"] < 0.5 <= r["new_mean"]]
    fixed = [r for r in identity_rows if r["prod_mean"] >= 0.5 > r["new_mean"]]

    # ================= write report =================
    print(f"\nWriting report -> {REPORT_PATH}")
    L = []
    L.append("# Model calibration / decision-boundary diagnostic: production vs. yunet_clean")
    L.append("")
    L.append("**Read-only diagnostic.** Did not modify `phase4_fusion_alerting/web_app.py`, did not modify or "
             "overwrite `phase2_visual/visual_model.pth`, did not integrate the new checkpoint, changed no "
             "threshold/fusion weight/hysteresis, and trained nothing. Both checkpoints were only read; this "
             "script only writes its own report and CSV.")
    L.append("")
    L.append(f"Test set: `{TEST_DIR}` -- {len(test_dataset):,} crops "
             f"({sum(fake_mask):,} fake / {sum(real_mask):,} real), same sealed set used in the earlier "
             f"same-test-set comparison and root-cause audit. Crops matched back to a CSV video row: "
             f"{len(paths) - unmatched}/{len(paths)}.")
    L.append("")

    L.append("## 1. Reliability / calibration analysis")
    L.append("")
    L.append(f"**Brier score** (mean squared error between P(fake) and the true fake/real indicator, lower is "
             f"better-calibrated *and* better-separated -- it conflates calibration and discrimination, which "
             f"is exactly why it's paired with ECE and ROC-AUC below):")
    L.append(f"- Production: {brier_prod:.4f}")
    L.append(f"- New (yunet_clean): {brier_new:.4f}")
    L.append("")
    L.append(f"**Expected Calibration Error (ECE)** -- {N_ECE_BINS} equal-width bins of width 0.1 over the "
             f"winning-class confidence `max(P(fake), 1-P(fake))`; each bin's contribution is "
             f"`(bin_count/N) * |bin_accuracy - bin_mean_confidence|`, summed over all non-empty bins "
             f"(standard definition, Guo et al. 2017):")
    L.append(f"- Production: {ece_prod:.4f}")
    L.append(f"- New (yunet_clean): {ece_new:.4f}")
    L.append("")
    L.append("<details><summary>ECE bin detail (production)</summary>\n")
    L.append("| bin range | n | accuracy | mean confidence |")
    L.append("|---|---|---|---|")
    for lo, hi, cnt, acc, conf in bins_prod:
        L.append(f"| [{lo:.1f}, {hi:.1f}] | {cnt} | {'-' if acc is None else f'{acc:.3f}'} | "
                 f"{'-' if conf is None else f'{conf:.3f}'} |")
    L.append("\n</details>\n")
    L.append("<details><summary>ECE bin detail (new yunet_clean)</summary>\n")
    L.append("| bin range | n | accuracy | mean confidence |")
    L.append("|---|---|---|---|")
    for lo, hi, cnt, acc, conf in bins_new:
        L.append(f"| [{lo:.1f}, {hi:.1f}] | {cnt} | {'-' if acc is None else f'{acc:.3f}'} | "
                 f"{'-' if conf is None else f'{conf:.3f}'} |")
    L.append("\n</details>\n")

    L.append("**P(fake) on true REAL samples** (lower is better -- this is the real-false-positive-relevant view):")
    L.append("")
    L.append("| | production | new |")
    L.append("|---|---|---|")
    L.append(f"| mean | {real_stats_prod['mean']:.4f} | {real_stats_new['mean']:.4f} |")
    L.append(f"| median | {real_stats_prod['median']:.4f} | {real_stats_new['median']:.4f} |")
    L.append(f"| p90 | {real_stats_prod['p90']:.4f} | {real_stats_new['p90']:.4f} |")
    L.append(f"| p95 | {real_stats_prod['p95']:.4f} | {real_stats_new['p95']:.4f} |")
    L.append("")
    L.append("**P(fake) on true FAKE samples** (higher is better):")
    L.append("")
    L.append("| | production | new |")
    L.append("|---|---|---|")
    L.append(f"| mean | {fake_stats_prod['mean']:.4f} | {fake_stats_new['mean']:.4f} |")
    L.append(f"| median | {fake_stats_prod['median']:.4f} | {fake_stats_new['median']:.4f} |")
    L.append(f"| p10 | {fake_stats_prod['p10']:.4f} | {fake_stats_new['p10']:.4f} |")
    L.append(f"| p05 | {fake_stats_prod['p05']:.4f} | {fake_stats_new['p05']:.4f} |")
    L.append("")

    L.append("## 2. Threshold-independent comparison (no winner selected here by design)")
    L.append("")
    L.append("| metric | production | new |")
    L.append("|---|---|---|")
    L.append(f"| ROC-AUC (fake vs. real ranking) | {roc_auc_prod:.4f} | {roc_auc_new:.4f} |")
    L.append(f"| PR-AUC, FAKE as positive class | {pr_auc_fake_pos_prod:.4f} | {pr_auc_fake_pos_new:.4f} |")
    L.append(f"| PR-AUC, REAL as positive class (minority class, {sum(real_mask)}/{len(test_dataset)} = "
             f"{sum(real_mask)/len(test_dataset)*100:.1f}% base rate) | {pr_auc_real_pos_prod:.4f} | "
             f"{pr_auc_real_pos_new:.4f} |")
    L.append("")

    L.append("## 3. Decision-boundary comparison @ threshold 0.50")
    L.append("")
    L.append(f"Confusion matrices [rows=actual, cols=predicted, order=(fake, real)]:")
    L.append("")
    L.append(f"- Production: `{cm_prod.tolist()}`")
    L.append(f"- New: `{cm_new.tolist()}`")
    L.append("")
    L.append("| metric | production | new |")
    L.append("|---|---|---|")
    L.append(f"| Real false-positive rate | {real_fp_prod/real_total*100:.2f}% ({real_fp_prod}/{real_total}) | "
             f"{real_fp_new/real_total*100:.2f}% ({real_fp_new}/{real_total}) |")
    L.append(f"| Fake recall | {rec_p[0]*100:.2f}% | {rec_n[0]*100:.2f}% |")
    L.append(f"| Macro F1 | {macro_f1_prod*100:.2f}% | {macro_f1_new*100:.2f}% |")
    L.append("")
    L.append(f"**Prediction disagreement at threshold 0.50: {n_disagree:,} / {len(test_dataset):,} "
             f"({n_disagree/len(test_dataset)*100:.2f}%)**")
    L.append("")

    L.append("## 4. Probability-shift analysis")
    L.append("")
    L.append("Quantiles of P(fake), true REAL samples only:")
    L.append("")
    L.append("| quantile | production | new |")
    L.append("|---|---|---|")
    for k in ["q5", "q10", "q25", "q50", "q75", "q90", "q95"]:
        L.append(f"| {k} | {real_quantiles_prod[k]:.4f} | {real_quantiles_new[k]:.4f} |")
    L.append("")
    L.append("Quantiles of P(fake), true FAKE samples only:")
    L.append("")
    L.append("| quantile | production | new |")
    L.append("|---|---|---|")
    for k in ["q5", "q10", "q25", "q50", "q75", "q90", "q95"]:
        L.append(f"| {k} | {fake_quantiles_prod[k]:.4f} | {fake_quantiles_new[k]:.4f} |")
    L.append("")
    L.append("Paired delta (new − production), computed per sample on the identical crop:")
    L.append("")
    L.append(f"- REAL samples: mean delta = {statistics.mean(deltas_real):+.4f}, "
             f"median delta = {statistics.median(deltas_real):+.4f}, "
             f"stdev = {statistics.pstdev(deltas_real):.4f}")
    L.append(f"- FAKE samples: mean delta = {statistics.mean(deltas_fake):+.4f}, "
             f"median delta = {statistics.median(deltas_fake):+.4f}, "
             f"stdev = {statistics.pstdev(deltas_fake):.4f}")
    L.append(f"- % of REAL samples where new P(fake) > production P(fake): {pct_real_higher:.1f}%")
    L.append(f"- % of FAKE samples where new P(fake) > production P(fake): {pct_fake_higher:.1f}%")
    L.append("")

    L.append("## 5. Identity-level analysis (real identities only, subject-disjoint sealed test)")
    L.append("")
    L.append(f"- Number of real identities: {n_identities}")
    L.append(f"- % of real identities with mean P(fake) >= 0.5 -- production: {prod_id_fake_pct:.1f}% "
             f"({sum(1 for r in identity_rows if r['prod_mean']>=0.5)}/{n_identities})")
    L.append(f"- % of real identities with mean P(fake) >= 0.5 -- new: {new_id_fake_pct:.1f}% "
             f"({sum(1 for r in identity_rows if r['new_mean']>=0.5)}/{n_identities})")
    L.append("")
    L.append(f"**Identities that flipped mostly-REAL (production) -> mostly-FAKE (new): {len(newly_broken)}**")
    if newly_broken:
        L.append("")
        L.append("| stem | subject_id | prod mean | prod median | new mean | new median |")
        L.append("|---|---|---|---|---|---|")
        for r in sorted(newly_broken, key=lambda r: -(r["new_mean"] - r["prod_mean"])):
            L.append(f"| {r['stem']} | {r['subject_id']} | {r['prod_mean']:.3f} | {r['prod_median']:.3f} | "
                     f"{r['new_mean']:.3f} | {r['new_median']:.3f} |")
    L.append("")
    L.append(f"**Identities that flipped mostly-FAKE (production) -> mostly-REAL (new): {len(fixed)}**")
    if fixed:
        L.append("")
        L.append("| stem | subject_id | prod mean | prod median | new mean | new median |")
        L.append("|---|---|---|---|---|---|")
        for r in sorted(fixed, key=lambda r: (r["new_mean"] - r["prod_mean"])):
            L.append(f"| {r['stem']} | {r['subject_id']} | {r['prod_mean']:.3f} | {r['prod_median']:.3f} | "
                     f"{r['new_mean']:.3f} | {r['new_median']:.3f} |")
    L.append("")

    L.append("## 6. Full identity table (all real identities, sorted by delta)")
    L.append("")
    L.append("<details><summary>expand full table</summary>\n")
    L.append("| stem | subject_id | prod mean | new mean | delta |")
    L.append("|---|---|---|---|---|")
    for r in identity_rows:
        L.append(f"| {r['stem']} | {r['subject_id']} | {r['prod_mean']:.3f} | {r['new_mean']:.3f} | "
                 f"{r['new_mean']-r['prod_mean']:+.3f} |")
    L.append("\n</details>\n")

    L.append("## 7. Interpretation")
    L.append("")
    L.append("Distinguishing four possible explanations, per the measurements above:")
    L.append("")
    L.append("- **Pure calibration problem** would show: similar or better ROC-AUC/PR-AUC (ranking preserved), "
             "but real/fake P(fake) distributions shifted in a way a single rescaling (e.g. temperature "
             "scaling) could fix -- i.e. the *ordering* of samples by P(fake) stays close to production's "
             "ordering, just the numeric scale differs.")
    L.append("- **Boundary/generalization problem** would show: ROC-AUC/PR-AUC change (not just probability "
             "scale) and/or the real and fake P(fake) distributions overlap differently -- some previously "
             "well-separated samples now sit on the wrong side, which a single monotonic rescaling cannot fix.")
    L.append("- **Identity-specific generalization problem** would show: the identity-level flip lists in "
             "section 5 are large in both directions (churn) rather than the new model being a strict, "
             "monotonic improvement over production for every identity.")
    L.append("- **Dataset/domain shift** cannot be directly measured from this sealed-test-only diagnostic -- "
             "it was separately addressed via the raw live-camera A/B test "
             "(`live_camera_ab_test_report.md`), which already showed the same shift reproduces on raw, "
             "non-FF++ input.")
    L.append("")
    L.append("*(Filled in narratively by the analysis accompanying this report -- see the chat response for "
             "the specific verdict drawn from the numbers above.)*")
    L.append("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")

    # console summary for the analysis step
    print("\n" + "=" * 90)
    print("SUMMARY FOR INTERPRETATION")
    print("=" * 90)
    print(f"Brier: prod={brier_prod:.4f} new={brier_new:.4f}")
    print(f"ECE:   prod={ece_prod:.4f} new={ece_new:.4f}")
    print(f"ROC-AUC: prod={roc_auc_prod:.4f} new={roc_auc_new:.4f}")
    print(f"PR-AUC(fake+): prod={pr_auc_fake_pos_prod:.4f} new={pr_auc_fake_pos_new:.4f}")
    print(f"PR-AUC(real+): prod={pr_auc_real_pos_prod:.4f} new={pr_auc_real_pos_new:.4f}")
    print(f"Real P(fake) mean: prod={real_stats_prod['mean']:.4f} new={real_stats_new['mean']:.4f}")
    print(f"Real P(fake) p90:  prod={real_stats_prod['p90']:.4f} new={real_stats_new['p90']:.4f}")
    print(f"Real P(fake) p95:  prod={real_stats_prod['p95']:.4f} new={real_stats_new['p95']:.4f}")
    print(f"Fake P(fake) mean: prod={fake_stats_prod['mean']:.4f} new={fake_stats_new['mean']:.4f}")
    print(f"Fake P(fake) p10:  prod={fake_stats_prod['p10']:.4f} new={fake_stats_new['p10']:.4f}")
    print(f"Fake P(fake) p05:  prod={fake_stats_prod['p05']:.4f} new={fake_stats_new['p05']:.4f}")
    print(f"Real FP rate @0.5: prod={real_fp_prod/real_total*100:.2f}% new={real_fp_new/real_total*100:.2f}%")
    print(f"Disagreement @0.5: {n_disagree}/{len(test_dataset)} ({n_disagree/len(test_dataset)*100:.2f}%)")
    print(f"Identity flips: newly_broken={len(newly_broken)} fixed={len(fixed)} total_real_identities={n_identities}")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()

"""Standalone, diagnostic-only live-camera A/B test: production visual_model.pth
vs. the new yunet_clean checkpoint, on IDENTICAL face crops from a raw webcam
feed (cv2.VideoCapture(0) directly -- never a screen recording, never the
dashboard).

This script is completely separate from phase4_fusion_alerting/web_app.py.
It does not import it, does not modify it, does not touch visual_model.pth,
and has no effect on production thresholds, hysteresis, fusion weights, or
the live dashboard. It only READS both checkpoints and the webcam, and WRITES
its own CSV + Markdown report under phase2_visual/experiments/.

Uses the exact same detector class, detector settings, crop convention, RGB
conversion, eval_transform and MobileNetV3-Small architecture as production
(phase2_visual/face_detector.py, dataset_pipeline/face_transforms.py),
imported unchanged -- not reimplemented -- so both models see precisely the
input production inference would compute, with class 0 = FAKE, class 1 = REAL
(same convention as phase4_fusion_alerting/web_app.py and main_fusion.py).

For every detected face, BOTH models run on the exact SAME cropped+transformed
tensor (computed once, reused for both forward passes) -- not two separately
computed crops -- so "identical input" is a structural guarantee, not just an
empirical check.

No temporal smoothing is applied to the logged/aggregated measurements. An
EMA is computed and shown on-screen ONLY as a secondary display convenience;
it is clearly labeled and never substitutes for the raw P(fake) values used
in the CSV, the console statistics, or the final report.

Run as:
    python phase2_visual/experiments/live_camera_ab_test.py
    python phase2_visual/experiments/live_camera_ab_test.py --duration 60   (auto-exit after N seconds, for a quick capability check)
    python phase2_visual/experiments/live_camera_ab_test.py --no-display   (skip the cv2 window, e.g. no GUI backend available)

Controls (interactive window):
    Q = quit and write the final CSV + Markdown report
    R = reset the ON-SCREEN/PERIODIC running statistics only -- the CSV log
        and the final report always cover the FULL session regardless of how
        many times R was pressed; reset events are recorded and listed in the
        report so segments can still be told apart from the CSV timestamps.

IMPORTANT: this experiment measures REAL-person false-positive behavior and
model-to-model agreement/disagreement on genuinely real live subjects. It is
NOT a ground-truth accuracy measurement -- there are no labeled fake live
samples in this session, so no true accuracy, precision, or recall can be
computed from it. See the printed protocol and the generated report for the
full disclaimer.
"""
import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2_visual.face_detector import FaceDetector  # noqa: E402  (same detector class/settings as production)
from dataset_pipeline.face_transforms import eval_transform  # noqa: E402  (same preprocessing as production)

EXPERIMENT_DIR = PROJECT_ROOT / "phase2_visual" / "experiments"
PROD_CKPT = PROJECT_ROOT / "phase2_visual" / "visual_model.pth"
NEW_CKPT = PROJECT_ROOT / "phase2_visual" / "checkpoints" / "visual_model_yunet_clean_best.pth"
CSV_PATH = EXPERIMENT_DIR / "live_camera_ab_test.csv"
REPORT_PATH = EXPERIMENT_DIR / "live_camera_ab_test_report.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATS_INTERVAL_SECONDS = 5.0
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_MISSED_FRAMES = 15  # ~0.5s at 30fps
EMA_ALPHA = 0.3  # display-only, never used for logging/stats

PROTOCOL_TEXT = """
============================================================================
LIVE CAMERA A/B TEST -- EXPERIMENT PROTOCOL (read before starting)
============================================================================
This is a DIAGNOSTIC experiment only. It does not touch web_app.py, does not
change visual_model.pth, and has no effect on the production dashboard.

Please perform three short segments once the window opens:

  TEST A -- Neutral/normal face:
      Look naturally at the camera for 30-60 seconds.

  TEST B -- Pose variation:
      Slowly turn left, right, up and down for 30-60 seconds, remaining a
      real, unaltered person throughout.

  TEST C -- Multiple-person (optional, only if a second real person is
      available):
      Have both people appear in frame simultaneously for 30-60 seconds,
      moving naturally.

You may press R between segments to reset the on-screen running statistics
so each segment's printed summary is easier to read -- this does NOT delete
any logged data; the CSV and final report always cover the entire session.

Press Q at any time to stop and write the final CSV + Markdown report.

IMPORTANT: this experiment has no labeled fake live samples, so it CANNOT
measure ground-truth accuracy. It measures real-person false-positive
behavior and model-to-model agreement/disagreement only.
============================================================================
"""


def safe_softmax_pfake(model, x):
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    return probs[0].item()  # class 0 = FAKE (same convention as web_app.py / main_fusion.py)


def load_model(ckpt_path):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    return model.to(DEVICE).eval()


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class MultiFaceTracker:
    """Lightweight greedy-IoU multi-object tracker so simultaneous faces each
    keep a stable face_id across frames (for the per-face-track summary).
    Not a rigorous MOT algorithm -- sufficient for a short diagnostic session."""

    def __init__(self, iou_threshold=TRACK_IOU_THRESHOLD, max_missed=TRACK_MAX_MISSED_FRAMES):
        self.next_id = 0
        self.tracks = {}  # id -> {"bbox":(x,y,w,h), "last_frame": int}
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, frame_number, bboxes):
        assigned = [None] * len(bboxes)
        used = set()
        # sort by best-IoU descending across all (detection, track) pairs -- simple greedy assignment
        pairs = []
        for i, bbox in enumerate(bboxes):
            for tid, t in self.tracks.items():
                pairs.append((iou(t["bbox"], bbox), i, tid))
        pairs.sort(key=lambda p: -p[0])
        matched_det = set()
        for score, i, tid in pairs:
            if score < self.iou_threshold:
                break
            if i in matched_det or tid in used:
                continue
            assigned[i] = tid
            matched_det.add(i)
            used.add(tid)
            self.tracks[tid]["bbox"] = bboxes[i]
            self.tracks[tid]["last_frame"] = frame_number
        for i, bbox in enumerate(bboxes):
            if assigned[i] is None:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {"bbox": bbox, "last_frame": frame_number}
                assigned[i] = new_id
        stale = [tid for tid, t in self.tracks.items() if frame_number - t["last_frame"] > self.max_missed]
        for tid in stale:
            del self.tracks[tid]
        return assigned


class StatsAccumulator:
    """Holds raw (unsmoothed) observations. Two instances are kept: one for
    the full session (never reset, used for the final report) and one for the
    on-screen periodic printout (clearable with R)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prod_pfake = []
        self.new_pfake = []
        self.n_frames_with_face = 0

    def add(self, prod_p, new_p):
        self.prod_pfake.append(prod_p)
        self.new_pfake.append(new_p)

    def summary(self):
        if not self.prod_pfake:
            return None
        deltas = [n - p for p, n in zip(self.prod_pfake, self.new_pfake)]
        return {
            "n": len(self.prod_pfake),
            "prod_mean": statistics.mean(self.prod_pfake),
            "prod_median": statistics.median(self.prod_pfake),
            "new_mean": statistics.mean(self.new_pfake),
            "new_median": statistics.median(self.new_pfake),
            "prod_pct_real": sum(1 for p in self.prod_pfake if p < 0.5) / len(self.prod_pfake) * 100,
            "new_pct_real": sum(1 for p in self.new_pfake if p < 0.5) / len(self.new_pfake) * 100,
            "mean_delta": statistics.mean(deltas),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None,
                         help="Auto-exit after N seconds (for a quick capability check). Default: run until Q is pressed.")
    parser.add_argument("--no-display", action="store_true",
                         help="Skip the cv2 window (logging/CSV still run) -- for environments without a GUI backend.")
    args = parser.parse_args()

    print(PROTOCOL_TEXT)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading PRODUCTION checkpoint: {PROD_CKPT}")
    if not PROD_CKPT.exists():
        print(f"ERROR: production checkpoint not found at {PROD_CKPT}")
        raise SystemExit(1)
    prod_model = load_model(PROD_CKPT)
    print("  OK -- production checkpoint loaded.")

    print(f"Loading NEW checkpoint: {NEW_CKPT}")
    if not NEW_CKPT.exists():
        print(f"ERROR: new checkpoint not found at {NEW_CKPT}")
        raise SystemExit(1)
    new_model = load_model(NEW_CKPT)
    print("  OK -- new checkpoint loaded.")

    detector = FaceDetector()  # same class, same settings as production (phase2_visual/face_detector.py, unmodified)
    print("Face detector ready (YuNet, same settings as production).")

    print(f"Opening webcam (index {args.camera_index}) via cv2.VideoCapture -- NOT a screen recording...")
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam at index {args.camera_index}. "
              f"Check that a camera is connected and not in use by another application (e.g. the dashboard).")
        raise SystemExit(1)
    print("  OK -- webcam opened.")

    show_window = not args.no_display
    if show_window:
        try:
            cv2.namedWindow("Live A/B Test (Q=quit, R=reset stats)", cv2.WINDOW_NORMAL)
        except cv2.error as e:
            print(f"WARNING: could not create a display window ({e}); continuing with --no-display behavior.")
            show_window = False

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "frame_number", "face_id", "x", "y", "width", "height", "face_area",
        "production_pfake", "new_pfake", "production_label", "new_label",
    ])

    tracker = MultiFaceTracker()
    ema_by_track = {}  # track_id -> {"prod": float, "new": float}  (display only)
    full_stats = StatsAccumulator()
    window_stats = StatsAccumulator()
    per_track = defaultdict(lambda: {"prod": [], "new": [], "first_t": None, "last_t": None})
    reset_events = []

    frame_number = 0
    frames_processed = 0
    t_start = time.time()
    last_stats_print = t_start

    print("\nRunning. Window title bar / console will show progress. Press Q in the video window to stop.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: failed to read a frame from the webcam; stopping.")
                break
            frame_number += 1
            frames_processed += 1
            now_wall = datetime.now().isoformat(timespec="milliseconds")

            faces = detector.extract_faces(frame)  # ALL detected faces, not just the largest -- multi-face support
            bboxes = [f["bbox"] for f in faces]
            track_ids = tracker.update(frame_number, bboxes) if bboxes else []

            for idx, face in enumerate(faces):
                x, y, w, h = face["bbox"]
                face_id = track_ids[idx]
                crop = face["crop"]
                if crop is None or crop.size == 0:
                    continue

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # same RGB conversion as production
                pil_img = Image.fromarray(rgb)
                x_in = eval_transform(pil_img).unsqueeze(0).to(DEVICE)  # same eval_transform as production

                # BOTH models receive this exact same tensor -- identical input by construction
                prod_p = safe_softmax_pfake(prod_model, x_in)
                new_p = safe_softmax_pfake(new_model, x_in)
                prod_label = "FAKE" if prod_p >= 0.5 else "REAL"
                new_label = "FAKE" if new_p >= 0.5 else "REAL"

                csv_writer.writerow([
                    now_wall, frame_number, face_id, x, y, w, h, w * h,
                    f"{prod_p:.6f}", f"{new_p:.6f}", prod_label, new_label,
                ])

                full_stats.add(prod_p, new_p)
                window_stats.add(prod_p, new_p)
                pt = per_track[face_id]
                pt["prod"].append(prod_p)
                pt["new"].append(new_p)
                pt["last_t"] = time.time() - t_start
                if pt["first_t"] is None:
                    pt["first_t"] = pt["last_t"]

                # display-only EMA, never used for logging/stats
                prev = ema_by_track.get(face_id, {"prod": prod_p, "new": new_p})
                ema_prod = EMA_ALPHA * prod_p + (1 - EMA_ALPHA) * prev["prod"]
                ema_new = EMA_ALPHA * new_p + (1 - EMA_ALPHA) * prev["new"]
                ema_by_track[face_id] = {"prod": ema_prod, "new": ema_new}

                if show_window:
                    color = (0, 200, 0) if (prod_label == "REAL" and new_label == "REAL") else \
                            (0, 0, 220) if (prod_label == "FAKE" and new_label == "FAKE") else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    lines = [
                        f"Face {face_id}",
                        f"PROD: {prod_p:.2f} {prod_label}  (ema {ema_prod:.2f})",
                        f"NEW : {new_p:.2f} {new_label}  (ema {ema_new:.2f})",
                    ]
                    ty = max(15, y - 10 - 15 * (len(lines) - 1))
                    for li, line in enumerate(lines):
                        cv2.putText(frame, line, (x, ty + li * 16), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, color, 1, cv2.LINE_AA)

            if show_window:
                hud = f"frames={frames_processed}  faces_logged={full_stats.n_frames_with_face if False else len(full_stats.prod_pfake)}  Q=quit R=reset"
                cv2.putText(frame, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Live A/B Test (Q=quit, R=reset stats)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("Q pressed -- stopping.")
                    break
                if key in (ord("r"), ord("R")):
                    window_stats.reset()
                    reset_events.append(time.time() - t_start)
                    print(f"[{time.time()-t_start:6.1f}s] R pressed -- on-screen stats reset (CSV/report unaffected).")

            now = time.time()
            if now - last_stats_print >= STATS_INTERVAL_SECONDS:
                s = window_stats.summary()
                elapsed = now - t_start
                if s is None:
                    print(f"[{elapsed:6.1f}s] frames={frames_processed} faces_analyzed=0 (no face detected yet)")
                else:
                    print(f"[{elapsed:6.1f}s] frames={frames_processed} faces_analyzed={s['n']} "
                          f"prod(mean={s['prod_mean']:.3f} median={s['prod_median']:.3f} %REAL={s['prod_pct_real']:.1f}%) "
                          f"new(mean={s['new_mean']:.3f} median={s['new_median']:.3f} %REAL={s['new_pct_real']:.1f}%) "
                          f"mean_delta(new-prod)={s['mean_delta']:+.3f}")
                last_stats_print = now

            if args.duration is not None and (now - t_start) >= args.duration:
                print(f"--duration {args.duration}s reached -- stopping automatically.")
                break

            if show_window:
                try:
                    if cv2.getWindowProperty("Live A/B Test (Q=quit, R=reset stats)", cv2.WND_PROP_VISIBLE) < 1:
                        print("Window closed -- stopping.")
                        break
                except cv2.error:
                    pass

    except KeyboardInterrupt:
        print("\nInterrupted -- stopping and writing final outputs.")
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        csv_file.close()

    duration_s = time.time() - t_start
    write_report(duration_s, frames_processed, full_stats, per_track, reset_events, args)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY (full session, raw/unsmoothed measurements)")
    print("=" * 78)
    s = full_stats.summary()
    if s is None:
        print("No faces were detected during this session -- no statistics to report.")
    else:
        print(f"Duration            : {duration_s:.1f}s")
        print(f"Frames processed    : {frames_processed}")
        print(f"Face observations   : {s['n']}")
        print(f"Production  mean={s['prod_mean']:.3f} median={s['prod_median']:.3f} %REAL={s['prod_pct_real']:.1f}%")
        print(f"New model   mean={s['new_mean']:.3f} median={s['new_median']:.3f} %REAL={s['new_pct_real']:.1f}%")
        print(f"Mean delta (new - production) = {s['mean_delta']:+.3f}")
    print(f"\nCSV written to   : {CSV_PATH}")
    print(f"Report written to: {REPORT_PATH}")
    print("\nREMINDER: this experiment has NO labeled fake live samples. It measures")
    print("real-person false-positive behavior and model-to-model agreement only --")
    print("it is NOT a ground-truth accuracy, precision, or recall measurement.")


def write_report(duration_s, frames_processed, full_stats, per_track, reset_events, args):
    s = full_stats.summary()
    lines = []
    lines.append("# Live camera A/B test report: production vs. yunet_clean checkpoint")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("**This is a diagnostic experiment only.** It did not modify "
                  "`phase4_fusion_alerting/web_app.py`, did not modify or replace "
                  "`phase2_visual/visual_model.pth`, and did not change any threshold, "
                  "hysteresis, or fusion weight. Frames were read directly from "
                  "`cv2.VideoCapture(0)` -- no screen recording was used.")
    lines.append("")
    lines.append("> **NOT a ground-truth accuracy measurement.** All subjects in this "
                  "session are known-real people; there are no labeled fake live samples. "
                  "This report measures real-person false-positive behavior and "
                  "model-to-model agreement/disagreement only -- it cannot and does not "
                  "measure overall accuracy, precision, or recall for either checkpoint.")
    lines.append("")
    lines.append("## Session")
    lines.append("")
    lines.append(f"- Duration: {duration_s:.1f}s")
    lines.append(f"- Frames processed: {frames_processed}")
    lines.append(f"- Camera index: {args.camera_index}")
    lines.append(f"- On-screen-stats reset events (R key) at: "
                  f"{', '.join(f'{t:.1f}s' for t in reset_events) if reset_events else 'none'} "
                  f"-- CSV and this report cover the full session regardless.")
    lines.append("")

    if s is None:
        lines.append("## Result")
        lines.append("")
        lines.append("**No faces were detected during this session.** No statistics could be computed. "
                      "Re-run with a face clearly visible to the camera.")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return

    prod = full_stats.prod_pfake
    new = full_stats.new_pfake
    deltas = [n - p for p, n in zip(prod, new)]
    more_fake = sum(1 for d in deltas if d > 0)
    more_real = sum(1 for d in deltas if d < 0)
    same = sum(1 for d in deltas if d == 0)
    n = len(prod)

    prod_labels = ["FAKE" if p >= 0.5 else "REAL" for p in prod]
    new_labels = ["FAKE" if p >= 0.5 else "REAL" for p in new]
    disagreements = sum(1 for pl, nl in zip(prod_labels, new_labels) if pl != nl)
    prod_fake_n = prod_labels.count("FAKE")
    new_fake_n = new_labels.count("FAKE")

    lines.append("## Face observations")
    lines.append("")
    lines.append(f"- Total face observations (one row per detected face per frame): {n}")
    lines.append("")
    lines.append("## Production (`visual_model.pth`) statistics")
    lines.append("")
    lines.append(f"- Mean P(fake): {s['prod_mean']:.4f}")
    lines.append(f"- Median P(fake): {s['prod_median']:.4f}")
    lines.append(f"- % observations classified REAL: {s['prod_pct_real']:.1f}%")
    lines.append(f"- % observations classified FAKE: {100 - s['prod_pct_real']:.1f}%")
    lines.append("")
    lines.append("## New model (`visual_model_yunet_clean_best.pth`) statistics")
    lines.append("")
    lines.append(f"- Mean P(fake): {s['new_mean']:.4f}")
    lines.append(f"- Median P(fake): {s['new_median']:.4f}")
    lines.append(f"- % observations classified REAL: {s['new_pct_real']:.1f}%")
    lines.append(f"- % observations classified FAKE: {100 - s['new_pct_real']:.1f}%")
    lines.append("")
    lines.append("## P(fake) delta (new - production)")
    lines.append("")
    lines.append(f"- Mean delta: {s['mean_delta']:+.4f}")
    lines.append(f"- Median delta: {statistics.median(deltas):+.4f}")
    lines.append(f"- Observations where new model is MORE fake than production: "
                 f"{more_fake} ({more_fake/n*100:.1f}%)")
    lines.append(f"- Observations where new model is MORE real than production: "
                 f"{more_real} ({more_real/n*100:.1f}%)")
    lines.append(f"- Observations with no difference: {same} ({same/n*100:.1f}%)")
    lines.append("")
    lines.append("## Label disagreements")
    lines.append("")
    lines.append(f"- Total disagreements (production label != new model label): "
                 f"{disagreements} ({disagreements/n*100:.1f}%)")
    lines.append("")
    lines.append("## Label distributions")
    lines.append("")
    lines.append(f"- Production: REAL={prod_labels.count('REAL')} ({prod_labels.count('REAL')/n*100:.1f}%), "
                 f"FAKE={prod_fake_n} ({prod_fake_n/n*100:.1f}%)")
    lines.append(f"- New model:  REAL={new_labels.count('REAL')} ({new_labels.count('REAL')/n*100:.1f}%), "
                 f"FAKE={new_fake_n} ({new_fake_n/n*100:.1f}%)")
    lines.append("")

    if per_track:
        lines.append("## Per-face-track summary")
        lines.append("")
        lines.append("Tracks are assigned by a lightweight greedy bounding-box IoU tracker "
                      "(not face recognition) -- a track_id generally corresponds to one "
                      "physically continuous face appearance, and a person who leaves and "
                      "re-enters frame may receive a new track_id.")
        lines.append("")
        lines.append("| track_id | n obs | duration (s) | prod mean P(fake) | prod %REAL | new mean P(fake) | new %REAL | mean delta |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tid in sorted(per_track.keys()):
            t = per_track[tid]
            tp, tn = t["prod"], t["new"]
            if not tp:
                continue
            tdur = (t["last_t"] - t["first_t"]) if t["first_t"] is not None else 0.0
            t_prod_real_pct = sum(1 for p in tp if p < 0.5) / len(tp) * 100
            t_new_real_pct = sum(1 for p in tn if p < 0.5) / len(tn) * 100
            t_mean_delta = statistics.mean(n_ - p_ for p_, n_ in zip(tp, tn))
            lines.append(f"| {tid} | {len(tp)} | {tdur:.1f} | {statistics.mean(tp):.3f} | "
                         f"{t_prod_real_pct:.1f}% | {statistics.mean(tn):.3f} | {t_new_real_pct:.1f}% | "
                         f"{t_mean_delta:+.3f} |")
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- All subjects are known-real people; no ground-truth accuracy, precision, or "
                  "recall can be computed from this session.")
    lines.append("- A high production or new-model FAKE percentage here is real-person "
                  "false-positive behavior, not a measurement of forgery-detection accuracy.")
    lines.append("- Compare this report's per-model and delta statistics against the sealed "
                  "FF++ test-set and screen-recording findings in "
                  "`phase2_visual/experiments/yunet_clean_root_cause_audit.md` before drawing "
                  "any conclusion about production readiness.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

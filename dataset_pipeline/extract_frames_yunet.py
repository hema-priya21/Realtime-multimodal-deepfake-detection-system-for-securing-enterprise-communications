"""YuNet-based face-crop extraction for training data, matching EXACTLY the
same face-detection/selection convention live inference uses (see
phase2_visual/face_detector.py: FaceDetector + select_largest_face), instead
of the original extract_frames.py's Haar Cascade + 15% margin.

This exists to remove a real train/inference distribution mismatch: live
inference (web_app.py) detects faces with YuNet and crops the exact
bounding box of the single largest face, no margin -- but the model was
trained on crops from a different detector (Haar Cascade) with a 15%
margin added around the box. That mismatch is a plausible contributor to
live accuracy issues, independent of raw model uncertainty.

Reads the SAME existing subject-disjoint split CSVs from
dataset_pipeline/splits/ (never modified by this script) and writes into a
SEPARATE output directory (data/visual_processed_yunet/ by default), so the
original data/visual_processed/ dataset and visual_model.pth remain fully
intact, untouched, and immediately usable as a fallback.

Only the face detector/crop convention changes -- the frame-position
sampling strategy (FRAMES_PER_VIDEO positions spaced evenly through the
video, identical formula to extract_frames.py) is kept the same, so the
crop-convention change is the only variable in any before/after comparison.
Unlike the old script (which could save multiple face crops per frame via
Haar's detectMultiScale), this one saves at most ONE crop per frame
position -- the largest detected face -- matching exactly what live
inference acts on.

Run as:
    python -m dataset_pipeline.extract_frames_yunet --pilot 5
    python -m dataset_pipeline.extract_frames_yunet
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.config import SPLITS_DIR, FRAMES_PER_VIDEO, FACE_SIZE
from phase2_visual.face_detector import FaceDetector, select_largest_face

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "visual_processed_yunet"


def read_split_csv(split_name):
    path = SPLITS_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run build_splits.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_token(text):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def already_processed(output_dir, video_stem):
    return any(output_dir.glob(f"{video_stem}_frame*_face00.jpg"))


def select_pilot_rows(rows, limit):
    """Balanced pilot subset: up to `limit` real + `limit` fake rows, kept
    in the CSV's own order so the same rows are picked deterministically on
    repeat runs (no randomness introduced)."""
    real_rows = [r for r in rows if r["label"] == "real"][:limit]
    fake_rows = [r for r in rows if r["label"] == "fake"][:limit]
    return real_rows + fake_rows


def extract_faces_from_video(video_path, output_dir, detector, video_stem):
    """Same evenly-spaced frame-position sampling as extract_frames.py, but
    detection/cropping now uses YuNet + select_largest_face -- the exact
    live-inference convention -- keeping at most ONE face crop per frame
    position, with no margin added around the bounding box.

    Returns (saved_count, frames_with_no_detection, status), status one of
    "ok" / "no_frames" / "open_failed".
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, "open_failed"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0, 0, "no_frames"

    positions = [int(i * total_frames / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]
    saved = 0
    no_detection = 0

    for frame_idx, position in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = cap.read()
        if not success:
            continue

        faces = detector.extract_faces(frame)
        face = select_largest_face(faces)
        if face is None:
            no_detection += 1
            continue

        face_crop = face["crop"]
        if face_crop is None or face_crop.size == 0:
            no_detection += 1
            continue

        resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
        out_path = output_dir / f"{video_stem}_frame{frame_idx:02d}_face00.jpg"
        if not out_path.exists() and cv2.imwrite(str(out_path), resized):
            saved += 1

    cap.release()
    return saved, no_detection, "ok"


def format_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def process_split(split_name, detector, output_root, pilot_limit=None):
    rows = read_split_csv(split_name)
    if pilot_limit is not None:
        rows = select_pilot_rows(rows, pilot_limit)

    print(f"\n{'=' * 75}\n{split_name.upper()} -- {len(rows):,} videos"
          f"{' (PILOT SUBSET)' if pilot_limit is not None else ''}\n{'=' * 75}")

    start = time.time()
    total_crops = 0
    total_no_detection_frames = 0
    processed, skipped, failed, zero_crop_videos = 0, 0, 0, 0
    real_crops, fake_crops = 0, 0

    for i, row in enumerate(rows, start=1):
        video_path = Path(row["filepath"])
        label = row["label"]
        video_stem = f"{row['dataset']}_{safe_token(video_path.parent.name)}_{safe_token(video_path.stem)}"

        output_dir = output_root / split_name / label
        output_dir.mkdir(parents=True, exist_ok=True)

        if already_processed(output_dir, video_stem):
            skipped += 1
            continue

        if not video_path.exists():
            failed += 1
            print(f"  [{i}/{len(rows)}] MISSING: {video_path}")
            continue

        saved, no_det, status = extract_faces_from_video(video_path, output_dir, detector, video_stem)
        if status != "ok":
            failed += 1
            print(f"  [{i}/{len(rows)}] {status.upper()}: {video_path}")
            continue

        processed += 1
        total_crops += saved
        total_no_detection_frames += no_det
        if saved == 0:
            zero_crop_videos += 1
        if label == "real":
            real_crops += saved
        else:
            fake_crops += saved

        if i % 50 == 0 or i == len(rows):
            elapsed = time.time() - start
            eta = (elapsed / i) * (len(rows) - i)
            print(f"  [{i}/{len(rows)}] processed={processed} skipped={skipped} failed={failed} "
                  f"crops={total_crops:,} (real {real_crops:,}/fake {fake_crops:,}) "
                  f"no-detection-frames={total_no_detection_frames:,} "
                  f"elapsed={format_time(elapsed)} eta={format_time(eta)}")

    print(f"\n{split_name.upper()} DONE -- processed {processed:,}, skipped {skipped:,}, "
          f"failed {failed:,}, videos-with-zero-crops {zero_crop_videos:,}, "
          f"crops {total_crops:,} (real {real_crops:,} / fake {fake_crops:,}), "
          f"time {format_time(time.time() - start)}")

    return {
        "processed": processed, "skipped": skipped, "failed": failed,
        "zero_crop_videos": zero_crop_videos, "total_crops": total_crops,
        "real_crops": real_crops, "fake_crops": fake_crops,
        "no_detection_frames": total_no_detection_frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=int, default=None,
                         help="If set, only process this many real + this many fake videos "
                              "per split (balanced pilot subset), instead of the full dataset.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                         help="Output root directory (default: data/visual_processed_yunet/)")
    args = parser.parse_args()

    output_root = Path(args.output)
    detector = FaceDetector()

    print("=" * 75)
    print("YuNet-based training-data extraction (matches live inference exactly)")
    print("Output root:", output_root)
    print("Pilot mode :", f"{args.pilot} real + {args.pilot} fake per split" if args.pilot else "FULL DATASET")
    print("=" * 75)

    summary = {}
    for split_name in ("train", "val", "test"):
        summary[split_name] = process_split(split_name, detector, output_root, pilot_limit=args.pilot)

    print("\n" + "=" * 75)
    print("OVERALL SUMMARY")
    print("=" * 75)
    for split_name, s in summary.items():
        print(f"{split_name.upper():5s} | processed={s['processed']:,} skipped={s['skipped']:,} "
              f"failed={s['failed']:,} zero-crop-videos={s['zero_crop_videos']:,} "
              f"crops={s['total_crops']:,} (real {s['real_crops']:,} / fake {s['fake_crops']:,}) "
              f"no-detection-frames={s['no_detection_frames']:,}")
    print("=" * 75)


if __name__ == "__main__":
    main()

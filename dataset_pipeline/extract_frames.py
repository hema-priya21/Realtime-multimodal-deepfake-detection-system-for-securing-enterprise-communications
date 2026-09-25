"""Extracts face crops from the videos listed in dataset_pipeline/splits/*.csv
into data/visual_processed/{train,val,test}/{real,fake}/, keeping every
frame from a video inside the same split its video was assigned to.

This replaces the old workflow (prepare_dataset.py dumping every frame into
a single data/visual/train/ folder regardless of subject), which is why
Phase 2's previous accuracy figure can't be trusted.

Run as: python -m dataset_pipeline.extract_frames
"""

import csv
import time
from pathlib import Path

import cv2

from dataset_pipeline.config import (
    SPLITS_DIR,
    PROCESSED_ROOT,
    FRAMES_PER_VIDEO,
    FACE_SIZE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASCADE_PATH = PROJECT_ROOT / "phase2_visual" / "haarcascades" / "haarcascade_frontalface_default.xml"


def load_detector():
    xml_path = str(CASCADE_PATH) if CASCADE_PATH.exists() else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(xml_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar Cascade from: {xml_path}")
    return detector


def read_split_csv(split_name):
    path = SPLITS_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run build_splits.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_token(text):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def already_processed(output_dir, video_stem):
    return any(output_dir.glob(f"{video_stem}_frame*_face*.jpg"))


def extract_faces(video_path, output_dir, detector, video_stem):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return -1

    positions = [int(i * total_frames / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]
    saved = 0

    for frame_idx, position in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for face_idx, (x, y, w, h) in enumerate(faces):
            margin = int(0.15 * max(w, h))
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
            out_path = output_dir / f"{video_stem}_frame{frame_idx:02d}_face{face_idx:02d}.jpg"
            if not out_path.exists() and cv2.imwrite(str(out_path), face):
                saved += 1

    cap.release()
    return saved


def format_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def process_split(split_name, detector):
    rows = read_split_csv(split_name)
    print(f"\n{'=' * 75}\n{split_name.upper()} -- {len(rows):,} videos\n{'=' * 75}")

    start = time.time()
    total_faces = 0
    processed, skipped, failed = 0, 0, 0

    for i, row in enumerate(rows, start=1):
        video_path = Path(row["filepath"])
        label = row["label"]
        # Include the immediate parent folder (e.g. FF++'s method name) in the
        # stem: FF++'s Deepfakes/Face2Face/FaceShifter/FaceSwap/NeuralTextures
        # folders reuse identical filenames for the same identity pair (e.g.
        # every method has its own 000_003.mp4), so dataset+stem alone collides
        # across methods and silently skips distinct videos as "already done".
        video_stem = f"{row['dataset']}_{safe_token(video_path.parent.name)}_{safe_token(video_path.stem)}"

        output_dir = PROCESSED_ROOT / split_name / label
        output_dir.mkdir(parents=True, exist_ok=True)

        if already_processed(output_dir, video_stem):
            skipped += 1
            continue

        if not video_path.exists():
            failed += 1
            print(f"  [{i}/{len(rows)}] MISSING: {video_path}")
            continue

        faces = extract_faces(video_path, output_dir, detector, video_stem)
        if faces == -1:
            failed += 1
            continue

        processed += 1
        total_faces += faces

        if i % 100 == 0 or i == len(rows):
            elapsed = time.time() - start
            eta = (elapsed / i) * (len(rows) - i)
            print(f"  [{i}/{len(rows)}] processed={processed} skipped={skipped} failed={failed} "
                  f"faces={total_faces:,} elapsed={format_time(elapsed)} eta={format_time(eta)}")

    print(f"\n{split_name.upper()} DONE -- processed {processed:,}, skipped {skipped:,}, "
          f"failed {failed:,}, faces extracted {total_faces:,}, time {format_time(time.time() - start)}")


def main():
    detector = load_detector()
    for split_name in ("train", "val", "test"):
        process_split(split_name, detector)


if __name__ == "__main__":
    main()

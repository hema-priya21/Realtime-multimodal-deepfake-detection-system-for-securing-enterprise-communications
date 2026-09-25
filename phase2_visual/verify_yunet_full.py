"""READ-ONLY final verification of data/visual_processed_yunet_full/ before
training on it. Run once dataset_pipeline.extract_frames_yunet has finished.

Checks (per the task spec):
  - exactly one face crop per saved frame (structural: extract_frames_yunet.py
    only ever writes "_face00.jpg", verified empirically here too)
  - no real images inside the fake/ folder, no fake images inside real/
  - no filename collisions between classes
  - no train/val/test identity overlap (independent re-check against the
    existing subject-disjoint split CSVs -- CSVs are read-only here)
  - real/fake crop counts per split
  - no-face / failed frame counts (parsed from the extraction log)

Never writes to data/visual_processed/, dataset_pipeline/splits/, or any
checkpoint.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dataset_pipeline.config import SPLITS_DIR

DATA_ROOT = PROJECT_ROOT / "data" / "visual_processed_yunet_full"
EXTRACT_LOG = PROJECT_ROOT / "phase2_visual" / "experiments_extract_log_tmp.txt"
CROP_RE = re.compile(r"^(?P<stem>.+)_frame(?P<frame>\d+)_face(?P<face>\d+)\.jpg$")


def safe_token(text):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def video_stem_of(row):
    p = Path(row["filepath"])
    return f"{row['dataset']}_{safe_token(p.parent.name)}_{safe_token(p.stem)}"


def read_csv(name):
    with open(SPLITS_DIR / f"{name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ok = True
    print("=" * 78)
    print("YUNET_FULL DATASET VERIFICATION")
    print("=" * 78)

    csvs = {s: read_csv(s) for s in ("train", "val", "test")}
    stem_to_row = {s: {video_stem_of(r): r for r in rows} for s, rows in csvs.items()}

    # --- identity leakage re-check ---
    split_tokens = {}
    for s, rows in csvs.items():
        toks = set()
        for r in rows:
            toks.update(r["subject_id"].split("+"))
        split_tokens[s] = toks
    print("\n-- identity leakage --")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_tokens[a] & split_tokens[b]
        status = "PASS" if not overlap else f"FAIL: {sorted(overlap)[:10]}"
        print(f"  {a} <-> {b} overlap: {len(overlap)}  {status}")
        ok = ok and not overlap

    # --- per-split crop counts, non-face-crop naming, cross-class checks ---
    print("\n-- per-split counts and contamination --")
    total_real = total_fake = 0
    for split in ("train", "val", "test"):
        real_dir = DATA_ROOT / split / "real"
        fake_dir = DATA_ROOT / split / "fake"
        real_files = list(real_dir.glob("*.jpg")) if real_dir.exists() else []
        fake_files = list(fake_dir.glob("*.jpg")) if fake_dir.exists() else []
        total_real += len(real_files)
        total_fake += len(fake_files)

        multi_face = 0
        untraceable = 0
        wrong_folder = 0
        filenames = defaultdict(set)
        for files, cls in ((real_files, "real"), (fake_files, "fake")):
            for f in files:
                m = CROP_RE.match(f.name)
                filenames[f.name].add(cls)
                if not m:
                    untraceable += 1
                    continue
                if m.group("face") != "00":
                    multi_face += 1
                row = stem_to_row[split].get(m.group("stem"))
                if row is None:
                    untraceable += 1
                    continue
                if row["label"] != cls:
                    wrong_folder += 1
        collisions = sum(1 for fn, classes in filenames.items() if len(classes) > 1)

        print(f"  {split:5s}: real={len(real_files):6,} fake={len(fake_files):6,} | "
              f"multi-face-index files (face!=00): {multi_face} | "
              f"untraceable: {untraceable} | wrong-folder (contamination): {wrong_folder} | "
              f"cross-class filename collisions: {collisions}")
        ok = ok and multi_face == 0 and untraceable == 0 and wrong_folder == 0 and collisions == 0

    print(f"\nTOTAL REAL IMAGES: {total_real:,}")
    print(f"TOTAL FAKE IMAGES: {total_fake:,}")

    # --- extraction log: no-face / failed frame stats ---
    print("\n-- extraction log summary (no-face / failed frames) --")
    if EXTRACT_LOG.exists():
        text = EXTRACT_LOG.read_text(errors="replace")
        for line in text.splitlines():
            if "DONE --" in line or "no-detection-frames" in line or line.strip().startswith(("TRAIN", "VAL", "TEST")):
                print(" ", line.strip())
    else:
        print("  extraction log not found at", EXTRACT_LOG)

    print("\n" + "=" * 78)
    print("OVERALL:", "PASS" if ok else "FAIL -- see above")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)

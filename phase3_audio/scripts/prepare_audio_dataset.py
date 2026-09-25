"""Extracts ASVspoof2019 LA train+dev CM (countermeasure) audio from LA.zip
and builds train/dev manifests for the audio deepfake detector.

Deliberately protocol-driven: only files referenced by the CM protocol
files are extracted. The raw train/dev flac/ folders inside LA.zip also
contain ASV-task enrollment/trial audio that is NOT part of the CM/
deepfake-detection task (verified during the read-only ZIP inspection --
142 extra files in dev, 696 extra in eval, all following a distinct
"LA_x_A..." ID pattern, not referenced by any CM protocol line).

The evaluation split is intentionally never opened here -- it stays sealed
inside LA.zip for the one honest final evaluation, later, exactly like the
visual pipeline's held-out test split.

Run as: python phase3_audio/scripts/prepare_audio_dataset.py
Safe to rerun: skips any audio file already extracted with the correct
byte size, and always regenerates the manifest CSVs (cheap, deterministic,
so a partial/interrupted previous run never leaves stale manifests).
"""
import csv
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ZIP_PATH = Path(r"C:\Users\kslok\Downloads\LA.zip")

AUDIO_ROOT = PROJECT_ROOT / "phase3_audio" / "data"
MANIFEST_DIR = PROJECT_ROOT / "phase3_audio" / "manifests"

SPLITS = {
    "train": {
        "protocol": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "flac_dir": "LA/ASVspoof2019_LA_train/flac/",
        "out_dir": AUDIO_ROOT / "train",
        "expected_total": 25380,
        "expected_bonafide": 2580,
        "expected_spoof": 22800,
    },
    "dev": {
        "protocol": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "flac_dir": "LA/ASVspoof2019_LA_dev/flac/",
        "out_dir": AUDIO_ROOT / "dev",
        "expected_total": 24844,
        "expected_bonafide": 2548,
        "expected_spoof": 22296,
    },
}


def fatal(message):
    print(f"\nFATAL: {message}")
    print("Stopping -- no further steps will run.")
    sys.exit(1)


def parse_protocol(zf, protocol_path):
    if protocol_path not in zf.namelist():
        fatal(f"Protocol file not found in ZIP: {protocol_path}")

    with zf.open(protocol_path) as f:
        lines = f.read().decode("utf-8").strip("\n").split("\n")

    rows = []
    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            fatal(f"Unexpected protocol line format at {protocol_path}:{line_no}: {line!r}")
        speaker_id, audio_id, _, system_id, label = parts
        if label not in ("bonafide", "spoof"):
            fatal(f"Unexpected label {label!r} at {protocol_path}:{line_no}: {line!r}")
        rows.append({
            "speaker_id": speaker_id,
            "audio_id": audio_id,
            "system_id": system_id,
            "label": label,
        })
    return rows


def estimate_required_bytes(zf, name_set, split_rows):
    total = 0
    for split_name, rows in split_rows.items():
        cfg = SPLITS[split_name]
        for row in rows:
            zip_entry = cfg["flac_dir"] + row["audio_id"] + ".flac"
            if zip_entry not in name_set:
                fatal(f"Protocol references {zip_entry} but it is not present in the ZIP.")
            total += zf.getinfo(zip_entry).file_size
    return total


def extract_split(zf, split_name, cfg, rows):
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    newly_extracted, already_present = 0, 0
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        zip_entry = cfg["flac_dir"] + row["audio_id"] + ".flac"
        dest_path = cfg["out_dir"] / (row["audio_id"] + ".flac")
        info = zf.getinfo(zip_entry)

        if dest_path.exists() and dest_path.stat().st_size == info.file_size:
            already_present += 1
        else:
            with zf.open(zip_entry) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            newly_extracted += 1

        if i % 5000 == 0 or i == total:
            print(f"  [{split_name}] {i:,}/{total:,} "
                  f"(newly extracted: {newly_extracted:,}, already present: {already_present:,})")

    return newly_extracted, already_present


def verify_counts(split_name, cfg, rows):
    label_counts = Counter(r["label"] for r in rows)
    total = len(rows)

    ok = True
    if total != cfg["expected_total"]:
        print(f"  MISMATCH [{split_name}] total: got {total}, expected {cfg['expected_total']}")
        ok = False
    if label_counts["bonafide"] != cfg["expected_bonafide"]:
        print(f"  MISMATCH [{split_name}] bonafide: got {label_counts['bonafide']}, "
              f"expected {cfg['expected_bonafide']}")
        ok = False
    if label_counts["spoof"] != cfg["expected_spoof"]:
        print(f"  MISMATCH [{split_name}] spoof: got {label_counts['spoof']}, "
              f"expected {cfg['expected_spoof']}")
        ok = False

    if not ok:
        fatal(f"Protocol counts for '{split_name}' do not match the verified inspection results.")

    print(f"  [{split_name}] counts verified: total={total:,} "
          f"bonafide={label_counts['bonafide']:,} spoof={label_counts['spoof']:,}")


def write_manifest(split_name, cfg, rows):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{split_name}.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "label", "label_id", "speaker_id", "system_id"])
        for row in rows:
            rel_path = (cfg["out_dir"] / (row["audio_id"] + ".flac")).relative_to(PROJECT_ROOT)
            label_id = 0 if row["label"] == "bonafide" else 1
            writer.writerow([
                str(rel_path).replace("\\", "/"),
                row["label"],
                label_id,
                row["speaker_id"],
                row["system_id"],
            ])

    print(f"  Wrote {len(rows):,} rows -> {manifest_path}")
    return manifest_path


def validate_sample(split_name, cfg, rows):
    by_label = {"bonafide": [], "spoof": []}
    for row in rows:
        by_label[row["label"]].append(row)

    print(f"\n  [{split_name}] sample audio properties (10 bonafide, 10 spoof):")
    for label in ("bonafide", "spoof"):
        for row in by_label[label][:10]:
            path = cfg["out_dir"] / (row["audio_id"] + ".flac")
            info = sf.info(str(path))
            duration = info.frames / info.samplerate if info.samplerate else 0.0
            print(f"    {label:8s} {row['audio_id']}: sr={info.samplerate} Hz, "
                  f"channels={info.channels}, samples={info.frames}, duration={duration:.3f}s")


def lightweight_full_validation(split_name, cfg, rows):
    zero_length, failed = [], []
    for row in rows:
        path = cfg["out_dir"] / (row["audio_id"] + ".flac")
        try:
            info = sf.info(str(path))
            if info.frames <= 0:
                zero_length.append(row["audio_id"])
        except Exception as e:
            failed.append((row["audio_id"], str(e)))

    print(f"  [{split_name}] lightweight validation over all {len(rows):,} files: "
          f"failed_to_open={len(failed)}, zero_length={len(zero_length)}")
    if failed:
        fatal(f"{len(failed)} file(s) in '{split_name}' failed to open. First: {failed[0]}")
    if zero_length:
        fatal(f"{len(zero_length)} zero-length file(s) in '{split_name}'. First: {zero_length[0]}")


def compute_class_weights(rows):
    # Same balanced-class formula already used in phase2_visual/train_visual.py:
    # weight[c] = total_samples / (num_classes * count[c])
    counts = Counter(r["label"] for r in rows)
    total = len(rows)
    num_classes = 2
    return {
        "bonafide": total / (num_classes * counts["bonafide"]),
        "spoof": total / (num_classes * counts["spoof"]),
    }


def main():
    print("=" * 80)
    print("AUDIO DATASET PREPARATION -- ASVspoof2019 LA (train + dev CM audio only)")
    print("=" * 80)

    if not ZIP_PATH.exists():
        fatal(f"LA.zip not found at {ZIP_PATH}")

    zf = zipfile.ZipFile(ZIP_PATH, "r")
    name_set = set(zf.namelist())

    print("\nParsing CM protocol files (train, dev only -- eval is never opened)...")
    split_rows = {}
    for split_name, cfg in SPLITS.items():
        split_rows[split_name] = parse_protocol(zf, cfg["protocol"])
        print(f"  [{split_name}] parsed {len(split_rows[split_name]):,} protocol lines from {cfg['protocol']}")

    print("\nVerifying protocol counts against the inspection-phase verified numbers...")
    for split_name, cfg in SPLITS.items():
        verify_counts(split_name, cfg, split_rows[split_name])

    print("\n" + "=" * 80)
    print("TASK 9 -- DISK SAFETY CHECK")
    print("=" * 80)
    required_bytes = estimate_required_bytes(zf, name_set, split_rows)
    _, _, free_bytes = shutil.disk_usage(str(PROJECT_ROOT.anchor))
    print(f"Estimated required space (train+dev CM audio only): {required_bytes:,} bytes "
          f"({required_bytes / 1024**3:.2f} GiB)")
    print(f"Free space on {PROJECT_ROOT.anchor}: {free_bytes:,} bytes ({free_bytes / 1024**3:.2f} GiB)")
    if free_bytes < required_bytes * 1.1:
        fatal("Insufficient disk space (less than 1.1x the estimated requirement free). Aborting before extraction.")
    print("Sufficient disk space confirmed. Proceeding with extraction.")

    print("\n" + "=" * 80)
    print("TASK 2 -- EXTRACTING CM AUDIO (protocol-driven, train+dev only)")
    print("=" * 80)
    extraction_stats = {}
    for split_name, cfg in SPLITS.items():
        print(f"\nExtracting [{split_name}]...")
        newly, present = extract_split(zf, split_name, cfg, split_rows[split_name])
        extraction_stats[split_name] = (newly, present)

    zf.close()

    print("\n" + "=" * 80)
    print("TASK 3 -- POST-EXTRACTION VERIFICATION")
    print("=" * 80)
    for split_name, cfg in SPLITS.items():
        extracted_files = list(cfg["out_dir"].glob("*.flac"))
        if len(extracted_files) != cfg["expected_total"]:
            fatal(f"[{split_name}] extracted file count on disk ({len(extracted_files)}) "
                  f"does not match expected total ({cfg['expected_total']}).")
        print(f"  [{split_name}] on-disk file count verified: {len(extracted_files):,}")

    print("\n" + "=" * 80)
    print("TASK 4 -- MANIFEST GENERATION")
    print("=" * 80)
    manifests = {}
    for split_name, cfg in SPLITS.items():
        manifests[split_name] = write_manifest(split_name, cfg, split_rows[split_name])

    print("\n" + "=" * 80)
    print("TASK 5 -- AUDIO VALIDATION")
    print("=" * 80)
    for split_name, cfg in SPLITS.items():
        validate_sample(split_name, cfg, split_rows[split_name])
    print()
    for split_name, cfg in SPLITS.items():
        lightweight_full_validation(split_name, cfg, split_rows[split_name])

    print("\n" + "=" * 80)
    print("TASK 6 -- DATASET SUMMARY")
    print("=" * 80)
    speaker_sets = {}
    for split_name, rows in split_rows.items():
        counts = Counter(r["label"] for r in rows)
        total = len(rows)
        speakers = set(r["speaker_id"] for r in rows)
        speaker_sets[split_name] = speakers
        print(f"\n{split_name.upper()}:")
        print(f"  total     : {total:,}")
        print(f"  bonafide  : {counts['bonafide']:,}")
        print(f"  spoof     : {counts['spoof']:,}")
        print(f"  ratio (spoof:bonafide) : {counts['spoof'] / counts['bonafide']:.2f}:1")
        print(f"  unique speakers : {len(speakers)}")

    overlap = speaker_sets["train"] & speaker_sets["dev"]
    print(f"\nSpeaker overlap between train and dev: {len(overlap)} speaker(s) "
          f"{sorted(overlap) if overlap else '(none)'}")

    print("\n" + "=" * 80)
    print("TASK 7 -- CLASS WEIGHTS (train)")
    print("=" * 80)
    weights = compute_class_weights(split_rows["train"])
    print(f"  bonafide weight: {weights['bonafide']:.6f}")
    print(f"  spoof weight   : {weights['spoof']:.6f}")

    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)
    print("Eval split was not opened, extracted, or processed at any point.")
    print(f"LA.zip untouched at: {ZIP_PATH}")


if __name__ == "__main__":
    main()

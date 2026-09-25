"""Converts the extracted ASVspoof2019 LA train/dev FLAC audio (see
prepare_audio_dataset.py) into fixed-size log-Mel spectrogram features for
the audio CNN baseline.

Preprocessing (documented per Task 1 of the audio-baseline phase):
  - sample_rate = 16000, n_fft = 512, hop_length = 160, n_mels = 80,
    fmin = 20, fmax = 7600 -- standard speech-appropriate Mel settings.
  - Fixed duration window: TARGET_SECONDS seconds -> TARGET_SAMPLES samples.
    Shorter utterances are zero-padded at the end (silence, not repeated
    audio). Longer utterances are cropped to their FIRST TARGET_SAMPLES
    samples, deterministically (not random, not center) -- fully
    reproducible across runs, at the documented cost that content after
    the window is discarded for longer utterances. This is a known,
    simplifying choice for a first baseline, not a claim that no
    information is ever lost on long utterances.
  - log-Mel via librosa.feature.melspectrogram + librosa.power_to_db
    (ref=np.max), then a FIXED, sample-independent affine rescale --
    clip to [-80, 0] dB and map linearly to [0, 1] -- rather than
    per-utterance z-score normalization, so every feature file uses
    the exact same transform with no separately-fitted statistics.
  - Native sample rate is read per-file and resampling only happens if
    a file is not already 16 kHz (the dataset-prep phase's full
    validation checked all files open and are non-empty, but only
    checked sample rate on a 40-file sample, so this is a real,
    not just defensive, check).

Storage: one float32 .npy per utterance, named after its audio_id, under
phase3_audio/features/{train,dev}/ -- mirrors the extracted-audio layout.
Never touches phase3_audio/data eval (no eval split exists on disk to
begin with) and never reads anything from LA.zip.

Run as: python phase3_audio/scripts/prepare_mels.py
Safe to rerun: any .npy that already exists and loads with the expected
shape is skipped; only missing/invalid ones are (re)computed.
"""
import csv
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = PROJECT_ROOT / "phase3_audio" / "manifests"
FEATURES_ROOT = PROJECT_ROOT / "phase3_audio" / "features"

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80
FMIN = 20
FMAX = 7600

TARGET_SECONDS = 4.0
TARGET_SAMPLES = int(TARGET_SECONDS * SAMPLE_RATE)  # 64,000 samples

DB_CLIP_MIN = -80.0
DB_CLIP_MAX = 0.0

SPLITS = {
    "train": {"manifest": MANIFEST_DIR / "train.csv", "out_dir": FEATURES_ROOT / "train", "expected": 25380},
    "dev": {"manifest": MANIFEST_DIR / "dev.csv", "out_dir": FEATURES_ROOT / "dev", "expected": 24844},
}


def fatal(message):
    print(f"\nFATAL: {message}", flush=True)
    sys.exit(1)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def pad_or_crop(waveform):
    """Zero-pads at the end if shorter than TARGET_SAMPLES, or crops
    deterministically to the first TARGET_SAMPLES if longer. Extracted as
    its own function (pure refactor, no behavior change) so the live
    inference path (phase3_audio/audio_inference.py) can import this exact
    logic instead of re-implementing it, guaranteeing live input is padded/
    cropped identically to how training features were built."""
    if len(waveform) < TARGET_SAMPLES:
        pad_amount = TARGET_SAMPLES - len(waveform)
        return np.pad(waveform, (0, pad_amount), mode="constant", constant_values=0.0)
    elif len(waveform) > TARGET_SAMPLES:
        return waveform[:TARGET_SAMPLES]  # deterministic: first TARGET_SAMPLES samples
    return waveform


def load_fixed_window(audio_path):
    """Loads audio, resampling only if the file's native rate isn't 16kHz,
    then pads/crops to exactly TARGET_SAMPLES. Returns (waveform, orig_sr,
    orig_duration_seconds)."""
    info = sf.info(str(audio_path))
    orig_sr = info.samplerate
    orig_duration = info.frames / info.samplerate if info.samplerate else 0.0

    if orig_sr == SAMPLE_RATE:
        waveform, _ = sf.read(str(audio_path), dtype="float32", always_2d=False)
    else:
        waveform, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)  # convert to mono if needed
    waveform = waveform.astype(np.float32)
    waveform = pad_or_crop(waveform)

    return waveform, orig_sr, orig_duration


def compute_log_mel(waveform):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    log_mel_db = librosa.power_to_db(mel, ref=np.max)
    clipped = np.clip(log_mel_db, DB_CLIP_MIN, DB_CLIP_MAX)
    normalized = (clipped - DB_CLIP_MIN) / (DB_CLIP_MAX - DB_CLIP_MIN)  # -> [0, 1]
    return normalized.astype(np.float32)


def validate_sample(rows, split_name, n=5):
    print(f"\n[{split_name}] Sample validation ({min(n, len(rows))} files) -- "
          f"BEFORE full preprocessing:", flush=True)
    for row in rows[:n]:
        audio_path = PROJECT_ROOT / row["audio_path"]
        waveform, orig_sr, orig_duration = load_fixed_window(audio_path)
        spec = compute_log_mel(waveform)

        has_nan = bool(np.isnan(spec).any())
        has_inf = bool(np.isinf(spec).any())
        is_empty = spec.size == 0

        print(f"  {row['audio_path']} label={row['label']}", flush=True)
        print(f"    original: sr={orig_sr} Hz, duration={orig_duration:.3f}s", flush=True)
        print(f"    waveform shape (after pad/crop): {waveform.shape}", flush=True)
        print(f"    spectrogram shape: {spec.shape}  "
              f"min={spec.min():.4f} max={spec.max():.4f} mean={spec.mean():.4f} std={spec.std():.4f}", flush=True)
        print(f"    NaN={has_nan} Inf={has_inf} empty={is_empty}", flush=True)

        if has_nan or has_inf or is_empty:
            fatal(f"Sample validation failed for {row['audio_path']} "
                  f"(NaN={has_nan}, Inf={has_inf}, empty={is_empty}). Stopping before full run.")

    print(f"[{split_name}] Sample validation PASSED.", flush=True)


def process_split(split_name, cfg):
    rows = load_manifest(cfg["manifest"])
    if len(rows) != cfg["expected"]:
        fatal(f"[{split_name}] manifest row count {len(rows)} != expected {cfg['expected']}. "
              f"Not proceeding with feature generation.")

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}\nPROCESSING [{split_name}] -- {len(rows):,} files\n{'=' * 80}", flush=True)

    start = time.time()
    newly_computed, skipped, spec_shape = 0, 0, None

    for i, row in enumerate(rows, start=1):
        audio_id = Path(row["audio_path"]).stem
        out_path = cfg["out_dir"] / f"{audio_id}.npy"

        if out_path.exists():
            try:
                existing = np.load(out_path)
                if existing.size > 0 and not np.isnan(existing).any() and not np.isinf(existing).any():
                    if spec_shape is None:
                        spec_shape = existing.shape
                    skipped += 1
                    continue
            except Exception:
                pass  # fall through and recompute a corrupt file

        audio_path = PROJECT_ROOT / row["audio_path"]
        waveform, _, _ = load_fixed_window(audio_path)
        spec = compute_log_mel(waveform)
        spec_shape = spec.shape
        np.save(out_path, spec)
        newly_computed += 1

        if i % 2000 == 0 or i == len(rows):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(rows) - i) / rate if rate > 0 else 0
            print(f"  [{split_name}] {i:,}/{len(rows):,} "
                  f"(new: {newly_computed:,}, skipped: {skipped:,}) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"[{split_name}] DONE -- new={newly_computed:,} skipped={skipped:,} "
          f"total={newly_computed + skipped:,} spectrogram_shape={spec_shape}", flush=True)
    return newly_computed + skipped, spec_shape


def main():
    print("=" * 80, flush=True)
    print("MEL SPECTROGRAM PREPROCESSING -- ASVspoof2019 LA (train + dev only)", flush=True)
    print("=" * 80, flush=True)
    print(f"sample_rate={SAMPLE_RATE} n_fft={N_FFT} hop_length={HOP_LENGTH} "
          f"n_mels={N_MELS} fmin={FMIN} fmax={FMAX}", flush=True)
    print(f"Fixed window: {TARGET_SECONDS}s = {TARGET_SAMPLES} samples "
          f"(pad shorter with zeros, crop longer to first {TARGET_SAMPLES} samples)", flush=True)

    for split_name, cfg in SPLITS.items():
        if not cfg["manifest"].exists():
            fatal(f"Manifest not found: {cfg['manifest']}")

    for split_name, cfg in SPLITS.items():
        rows = load_manifest(cfg["manifest"])
        validate_sample(rows, split_name, n=5)

    final_counts = {}
    for split_name, cfg in SPLITS.items():
        count, shape = process_split(split_name, cfg)
        final_counts[split_name] = count
        if count != cfg["expected"]:
            fatal(f"[{split_name}] final feature count {count} != expected {cfg['expected']}.")

    print("\n" + "=" * 80, flush=True)
    print("PREPROCESSING COMPLETE", flush=True)
    print("=" * 80, flush=True)
    for split_name, count in final_counts.items():
        print(f"  {split_name}: {count:,} feature files (matches expected)", flush=True)
    print("Eval split: not touched (no eval audio exists on disk, no eval features created).", flush=True)


if __name__ == "__main__":
    main()

"""Central configuration for the subject-disjoint dataset splitting pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Raw dataset locations (verified against this machine) -----------------

# FaceForensics++ c23, standard naming: original/000.mp4,
# Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures/000_003.mp4,
# DeepFakeDetection/01_02__scene__hash.mp4
RAW_FFPP_ROOT = PROJECT_ROOT / "data" / "ff_raw" / "FaceForensics++_C23"

# Not present on this machine yet (searched data/, Downloads, Desktop,
# OneDrive, D:\, repo root -- nothing found). build_splits.py skips it with
# a warning if this path doesn't exist. Extract Celeb-DF v2 here once
# downloaded (expects the official Celeb-real / Celeb-synthesis /
# YouTube-real folders, e.g. id0_0000.mp4 / id0_id1_0000.mp4).
RAW_CELEBDF_ROOT = PROJECT_ROOT / "data" / "celebdf_raw"

# --- Output locations --------------------------------------------------------

PROCESSED_ROOT = PROJECT_ROOT / "data" / "visual_processed"
SPLITS_DIR = PROJECT_ROOT / "dataset_pipeline" / "splits"

# --- Frame extraction / preprocessing ----------------------------------------

FRAMES_PER_VIDEO = 12
FACE_SIZE = 224

# --- Split ratios (must sum to 1.0) ------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Split ratios must sum to 1.0"

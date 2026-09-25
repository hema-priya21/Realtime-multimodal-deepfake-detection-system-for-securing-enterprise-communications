"""Canonical identity extraction for FF++ and Celeb-DF v2 video filenames.

Naming conventions this was written against (verified on-disk, not guessed):

FF++ (data/ff_raw/FaceForensics++_C23/):
    original/003.mp4                                  -> real, identity 003
    Deepfakes|Face2Face|FaceShifter|FaceSwap|
    NeuralTextures/000_003.mp4                         -> fake, identities 000 + 003
    DeepFakeDetection/01_02__meeting_serious__HASH.mp4 -> fake, actors 01 + 02
        (DeepFakeDetection uses Google/Jigsaw's own 2-digit actor pool,
        entirely separate people from the 3-digit YouTube pool used by
        everything else, so its tokens are namespaced "ffpp_dfd_*" and can
        never merge with "ffpp_yt_*" tokens.)

Celeb-DF v2 (once downloaded into RAW_CELEBDF_ROOT):
    Celeb-real/id0_0000.mp4          -> real, identity id0
    Celeb-synthesis/id0_id5_0003.mp4 -> fake, identities id0 + id5
    YouTube-real/00000.mp4           -> real, unique distractor identity
"""

import re
from pathlib import Path

_FFPP_ORIGINAL_RE = re.compile(r"^(\d{3})$")
_FFPP_PAIR_RE = re.compile(r"^(\d{3})_(\d{3})$")
_FFPP_DFD_RE = re.compile(r"^(\d{2})_(\d{2})__")

_CELEBDF_SYNTH_RE = re.compile(r"^id(\d+)_id(\d+)_\d+$")
_CELEBDF_REAL_RE = re.compile(r"^id(\d+)_\d+$")
_CELEBDF_YT_RE = re.compile(r"^(\d+)$")


def extract_subject_id(filepath, dataset):
    """Returns a canonical, order-independent identity key for a video file.

    For fakes involving two identities, the key incorporates BOTH
    identities, sorted and joined with "+" (e.g. "ffpp_yt_000+ffpp_yt_003").
    Splitting on "+" recovers the individual identity tokens.

    A per-file key alone is NOT sufficient to guarantee subject-disjoint
    splits: a real video's key ("ffpp_yt_003") and a fake pair-video's key
    ("ffpp_yt_000+ffpp_yt_003") both touch identity 003 but are different
    strings, so naively grouping-by-key would let identity 003 end up in
    two different splits. build_splits.py handles this correctly by
    union-finding the individual tokens across all records before splitting.
    """
    filepath = Path(filepath)
    stem = filepath.stem
    parts_lower = {p.lower() for p in filepath.parts}

    if dataset == "ffpp":
        if "deepfakedetection" in parts_lower:
            match = _FFPP_DFD_RE.match(stem)
            if not match:
                raise ValueError(f"Unrecognized DeepFakeDetection filename: {filepath}")
            tokens = sorted({f"ffpp_dfd_{match.group(1)}", f"ffpp_dfd_{match.group(2)}"})
            return "+".join(tokens)

        match = _FFPP_PAIR_RE.match(stem)
        if match:
            tokens = sorted({f"ffpp_yt_{match.group(1)}", f"ffpp_yt_{match.group(2)}"})
            return "+".join(tokens)

        match = _FFPP_ORIGINAL_RE.match(stem)
        if match:
            return f"ffpp_yt_{match.group(1)}"

        raise ValueError(f"Unrecognized FF++ filename convention: {filepath}")

    if dataset == "celebdf":
        match = _CELEBDF_SYNTH_RE.match(stem)
        if match:
            tokens = sorted({f"celebdf_id_{match.group(1)}", f"celebdf_id_{match.group(2)}"})
            return "+".join(tokens)

        match = _CELEBDF_REAL_RE.match(stem)
        if match:
            return f"celebdf_id_{match.group(1)}"

        match = _CELEBDF_YT_RE.match(stem)
        if match:
            return f"celebdf_yt_{match.group(1)}"

        raise ValueError(f"Unrecognized Celeb-DF filename convention: {filepath}")

    raise ValueError(f"Unknown dataset tag: {dataset!r}")


def get_label(filepath, dataset):
    """Returns 'real' or 'fake' based on the containing folder name."""
    parts_lower = {p.lower() for p in Path(filepath).parts}

    if dataset == "ffpp":
        return "real" if "original" in parts_lower else "fake"

    if dataset == "celebdf":
        if "celeb-real" in parts_lower or "youtube-real" in parts_lower:
            return "real"
        if "celeb-synthesis" in parts_lower:
            return "fake"
        raise ValueError(f"Cannot determine label for Celeb-DF file: {filepath}")

    raise ValueError(f"Unknown dataset tag: {dataset!r}")

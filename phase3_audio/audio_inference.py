"""Live inference wrapper around the trained audio CNN baseline
(phase3_audio/checkpoints/audio_baseline_best.pth).

Reuses the EXACT training/feature-extraction preprocessing by importing it
from prepare_mels.py (pad_or_crop, compute_log_mel, and the frozen
constants) rather than re-implementing it -- this guarantees live input is
processed identically to how the model's training features were built, so
there is no drift between offline and live preprocessing.

Reuses the exact AudioCNN architecture from train_audio.py (imported, not
redefined) so the loaded state_dict always matches the model definition
that produced it.

Dev-set performance (accuracy 98.06%, ROC-AUC 0.9951) is an OFFLINE
ASVspoof2019 LA dev-split result only. It says nothing about live
microphone accuracy, which depends on microphone quality, room acoustics,
and background noise never present in the training data.
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_audio.scripts.prepare_mels import SAMPLE_RATE, TARGET_SAMPLES, pad_or_crop, compute_log_mel
from phase3_audio.train_audio import AudioCNN

CHECKPOINT_PATH = PROJECT_ROOT / "phase3_audio" / "checkpoints" / "audio_baseline_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = {0: "bonafide", 1: "spoof"}  # must not change: matches training class mapping

_model = None


def load_audio_model():
    """Loads the checkpoint once and caches it; safe to call repeatedly."""
    global _model
    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Audio checkpoint not found at {CHECKPOINT_PATH}")

    model = AudioCNN(num_classes=2)
    model.load_state_dict(torch.load(str(CHECKPOINT_PATH), map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    _model = model
    print(f"Loaded audio weights from: {CHECKPOINT_PATH}")
    return _model


def predict_audio(waveform, sample_rate):
    """Runs one inference pass on a 1-D waveform.

    waveform: 1-D numpy array (mono). sample_rate: the waveform's actual
    sample rate -- only resampled if it doesn't already match SAMPLE_RATE
    (16000), exactly like the offline preprocessing.

    Returns {"score": <float, P(spoof)>, "label": "spoof" | "bonafide"}.
    score is the softmax probability of class 1 (spoof) -- class mapping
    is fixed at 0=bonafide, 1=spoof and must never change independently of
    the trained checkpoint.
    """
    model = load_audio_model()

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)  # convert to mono, matching prepare_mels.py

    if sample_rate != SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    waveform = pad_or_crop(waveform)
    spec = compute_log_mel(waveform)  # (80, 401), already normalized to [0, 1]

    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 80, 401)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        spoof_score = probs[1].item()  # class 1 = spoof

    return {"score": spoof_score, "label": LABEL_NAMES[1 if spoof_score >= 0.5 else 0]}

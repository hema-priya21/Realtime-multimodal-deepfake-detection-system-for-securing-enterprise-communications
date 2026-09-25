# Real-Time Multimodal Deepfake Detection System for Enterprise Communications

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.14%20(developed%20on)-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-success.svg)]()

A real-time, multimodal (video + audio) deepfake detection pipeline for live enterprise communications. It analyses a live camera and microphone stream continuously and produces an early security indication while the call is happening, not after it.

The system combines a face-level visual classifier and an audio spoof detector, fuses their scores, stabilises the result over time, and shows a live "Zero-Trust" security decision on a web dashboard.

---

## Contents

1. [What it does](#what-it-does)
2. [System architecture](#system-architecture)
3. [Decision algorithm](#decision-algorithm)
4. [Dataset and splits](#dataset-and-splits)
5. [Results (honest evaluation)](#results-honest-evaluation)
6. [Known limitations](#known-limitations)
7. [Repository structure](#repository-structure)
8. [Setup and running](#setup-and-running)
9. [Reproducing the experiments](#reproducing-the-experiments)

---

## What it does

- Detects faces in a live camera stream and scores each face for visual manipulation risk.
- Scores live microphone audio for synthetic-speech (spoof) risk.
- Fuses both into one risk value and turns it into a stable status: **NORMAL**, **SUSPICIOUS**, **CRITICAL**, or **NO SIGNAL**.
- Serves everything on a real-time web dashboard over a WebSocket.

---

## System architecture

```text
 LIVE CAMERA ──► YuNet face detector ──► face crop (BGR→RGB, 224×224) ──► MobileNetV3-Small ──► Visual risk P(fake)
                                                                                                        │
                                                                                                        ▼
                                                                         MULTIMODAL FUSION  (0.6·Visual + 0.4·Audio)
                                                                                                        ▲
 MICROPHONE ──► audio buffer ──► 80-band log-mel spectrogram ──► AudioCNN (ASVspoof 2019 LA) ──► Audio risk P(spoof)
                                                                                                        │
                                                                                                        ▼
                                              TEMPORAL STABILISATION (moving average + hysteresis + confirmation)
                                                                                                        │
                                                                                                        ▼
                                                     SECURITY DECISION ──► FastAPI + WebSocket ──► Dashboard
```

### Visual pipeline
- **Face detector:** OpenCV YuNet (`face_detection_yunet_2023mar.onnx`), score threshold 0.9, NMS 0.3. The largest detected face per frame is analysed. The model file is downloaded automatically into `models/` on first run if missing.
- **Classifier:** MobileNetV3-Small (torchvision, ImageNet-pretrained) with a 2-class head. Class mapping is `0 = FAKE`, `1 = REAL`, so `P(fake) = softmax(logits)[0]`.
- **Production checkpoint:** `phase2_visual/visual_model.pth` (about 6 MB).

### Audio pipeline
- **Features:** 80-band log-mel spectrogram, 16 kHz, `n_fft=512`, `hop_length=160`, fixed shape 80×401.
- **Classifier:** a small custom 3-block CNN (`AudioCNN` in `phase3_audio/train_audio.py`), trained on **ASVspoof 2019 LA** (train + dev). Class mapping is `0 = bonafide`, `1 = spoof`.
- **Live inference:** re-run about once per second on the latest 4-second window. A coarse RMS-energy silence gate skips inference on near-silent audio (see [limitations](#known-limitations)).

### Backend and dashboard
- **FastAPI + Uvicorn**, with one WebSocket endpoint `/ws/stream` pushing a JSON state message every 0.1 s.
- `VideoWorker` and `AudioWorker` are background threads that write into one lock-protected shared state. The WebSocket loop only reads snapshots from it and does no inference itself.
- The dashboard (`phase4_fusion_alerting/index.html`) is a single page of plain HTML/CSS/JavaScript. It shows the live video with a face box, visual and audio risk, a fusion gauge, the security decision, an evidence panel, an event timeline, and FPS.
- The face-box label and colour come from the same authoritative `status` field as the Security Decision badge, so the two can never disagree.

---

## Decision algorithm

Every tick (about 10 times per second) the raw scores are turned into one stable status. All constants live in `phase4_fusion_alerting/web_app.py`.

1. **Freshness-gated fusion.** `fusion = 0.6·visual + 0.4·audio`, using only fresh inputs. A stale or missing modality is excluded rather than reused. If only one modality is fresh it drives the score alone. If neither is, the status is `NO SIGNAL`.
2. **Smoothing.** The visual score is EMA-smoothed (α = 0.20), and the fused score is a 10-reading moving average (about 1 s).
3. **Hysteresis mapping.** `≥ 0.72` enters CRITICAL, `≤ 0.68` leaves it, and `0.50–0.68` is SUSPICIOUS. Below `0.50` is NORMAL. A score inside the 0.68–0.72 band keeps the current status, which prevents flicker around one fixed line.
4. **Confirmation.** A new status must hold for 5 consecutive ticks (about 0.5 s) before it is shown. A decisively-past-threshold score (margin 0.15) needs only 1 tick.

The fusion weights and thresholds are demo alert settings chosen from observed live recordings. They are not calibrated security probabilities.

---

## Dataset and splits

- **Source:** FaceForensics++ (c23): original, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, plus DeepFakeDetection (used in training only).
- **Split:** 7,000 videos, **subject-disjoint** (no identity appears in more than one split), verified with 0 identity overlap.

| Split | Videos | Real crops | Fake crops |
|---|---|---|---|
| Train | 4,900 | 7,424 | 49,019 |
| Validation | 1,056 | 2,042 | 10,221 |
| Test (sealed) | 1,044 | 2,002 | 10,007 |

- **80,715 face crops in total**, extracted with the same YuNet detector and crop convention as live inference (one face per frame, no margin), so training and inference see the same kind of crop.
- 122 of the 4,900 training videos produced no detectable face at any sampled frame and contribute no crops.
- Audit fixes made along the way: filename collisions between manipulation folders (which silently dropped fake samples) and multi-face label noise (an untouched second face in a fake video inheriting the FAKE label).

Raw datasets and generated crops are not stored in this repository (`data/` is git-ignored).

---

## Results (honest evaluation)

> An earlier version of this README reported a "99.98% validation accuracy". That number came from an earlier validation split that was not subject-disjoint and does **not** measure generalisation. It has been removed. The numbers below are all measured on the sealed, subject-disjoint test set.

### Sealed test set (12,009 crops), production vs. YuNet-clean research model

| Metric | Production | YuNet-clean |
|---|---|---|
| Accuracy | 82.07% | 91.21% |
| ROC-AUC | 0.8927 | 0.9567 |
| Macro F1 | 73.82% | 85.08% |
| Real false-positive rate (crop level) | 22.23% | 18.63% |
| Fake recall | 82.93% | 93.17% |
| Real identities misclassified as FAKE | 24 / 169 | 24 / 169 |

### Raw-camera A/B test (live webcam, real people only)

Both checkpoints were run on the exact same face crops from a live `cv2.VideoCapture` stream (313 s, 2,316 face observations). All subjects were genuine, so this measures real-person false-positive behaviour, not accuracy.

| | Production | YuNet-clean |
|---|---|---|
| Mean P(fake) | 0.388 | 0.840 |
| Classified REAL | 71.9% | 13.6% |
| Classified FAKE | 28.1% | 86.4% |

### Engineering decision

The YuNet-clean model scored better on the benchmark but read genuine live people as fake far more often. **It was not adopted, and the validated production checkpoint was retained.** Higher benchmark accuracy did not translate into reliable live behaviour.

### Recipe ablation (why the problem is not the obvious suspects)

| Experiment | Change vs. previous | Accuracy | Real FPR | Real identity FP rate |
|---|---|---|---|---|
| A | Reproduce YuNet-clean (bit-identical weights) | 91.21% | 18.63% | 14.2% |
| B | Remove RandomResizedCrop, blur, JPEG, resolution simulation | 93.15% | 17.13% | 10.65% |
| C | Also remove the balanced sampler | 93.09% | 22.13% | 17.16% |

Removing the heavier augmentation left the worst real-person cases essentially unchanged. Removing the balanced sampler made them worse, so the sampler is protective rather than the cause. The remaining hard-tail behaviour is still an open research question.

Full reports are in `phase2_visual/experiments/`:
`yunet_clean_root_cause_audit.md`, `model_calibration_boundary_diagnostic.md`, `live_camera_ab_test_report.md`, and `recipe_ablation/`.

---

## Known limitations

- **Visual false positives on real people.** Some genuine people are read as FAKE. On the sealed test set 24 of 169 real identities have a mean P(fake) ≥ 0.5, and the raw-camera test showed the same behaviour on live subjects.
- **Audio is not speech-gated.** The live microphone path only applies a coarse RMS-energy silence gate. It is not a voice-activity detector, so non-speech sound above the energy threshold can still produce an elevated spoof score. Proper speech gating is planned.
- **Calibration.** Model probabilities are not calibrated confidences. The research model in particular is overconfident.
- **Benchmark is not live accuracy.** No labelled fake live samples exist, so live testing measures false-positive behaviour and model agreement only.
- The production visual checkpoint was originally trained on Haar-cascade crops with a margin, while live inference uses YuNet crops without one. This mismatch is a documented contributing factor.

---

## Repository structure

```text
├── dataset_pipeline/          # Subject-disjoint splits + YuNet frame extraction (serial and parallel)
├── models/                    # YuNet face-detector ONNX (auto-downloaded if missing)
├── phase1_ingestion/          # Stream ingestion and buffer management
├── phase2_visual/             # Visual model: training, evaluation, face detector
│   ├── face_detector.py       #   YuNet detector + largest-face selection
│   ├── train_visual*.py       #   production and YuNet training scripts
│   ├── evaluate_visual*.py    #   sealed-test evaluation
│   ├── verify_yunet_full.py   #   dataset integrity / leakage check
│   └── experiments/           #   audits, A/B test, calibration diagnostic, recipe ablation
├── phase3_audio/              # Audio spoof detection (mel features, AudioCNN, inference)
├── phase4_fusion_alerting/    # Fusion, decisioning, FastAPI web app and dashboard
│   ├── web_app.py             #   the live backend (this is the deployed system)
│   ├── index.html             #   the dashboard
│   └── main_fusion.py         #   legacy standalone visual-only OpenCV demo
├── LICENSE
└── README.md
```

Model checkpoints (`*.pth`), datasets (`data/`), split CSVs and generated audio features are git-ignored and not included in the repository.

---

## Setup and running

There is no pinned `requirements.txt` yet. The code imports: `torch`, `torchvision`, `opencv-python`, `numpy`, `scikit-learn`, `Pillow`, `fastapi`, `uvicorn`, `sounddevice`, `soundfile` and `librosa`. It was developed on Python 3.14 with a CUDA-capable GPU (an RTX 2050 was used), and it also runs on CPU.

You need trained checkpoints at `phase2_visual/visual_model.pth` and `phase3_audio/checkpoints/audio_baseline_best.pth`. They are not in the repository, so train them (below) or copy them in.

Run the live dashboard from the repository root:

```bash
uvicorn phase4_fusion_alerting.web_app:app
```

Then open <http://localhost:8000/> and allow camera and microphone access. Only one process can hold the camera at a time.

---

## Reproducing the experiments

Place FaceForensics++ (c23) under `data/ff_raw/FaceForensics++_C23/` (see `dataset_pipeline/config.py`).

```bash
# 1. Build the subject-disjoint splits, then extract YuNet face crops
python -m dataset_pipeline.build_splits
python -m dataset_pipeline.extract_frames_yunet_parallel --workers 4

# 2. Verify dataset integrity (zero identity leakage / contamination)
python phase2_visual/verify_yunet_full.py

# 3. Train and evaluate the YuNet-clean research model (saves to phase2_visual/checkpoints/)
python phase2_visual/train_visual_yunet_clean.py
python phase2_visual/evaluate_visual_yunet_clean.py

# 4. Recipe ablation (each variant saves its own checkpoint)
python phase2_visual/experiments/recipe_ablation/train_recipe_variant.py --variant B
python phase2_visual/experiments/recipe_ablation/evaluate_recipe_variant.py --variant B

# 5. Raw-camera A/B test of production vs. the research model (press Q to stop)
python phase2_visual/experiments/live_camera_ab_test.py

# 6. Audio: prepare ASVspoof 2019 LA features, then train
python phase3_audio/scripts/prepare_audio_dataset.py
python phase3_audio/scripts/prepare_mels.py
python phase3_audio/train_audio.py
```

`prepare_audio_dataset.py` expects the ASVspoof 2019 `LA.zip` at the path set inside that script.

---

## License

MIT, see [LICENSE](LICENSE).

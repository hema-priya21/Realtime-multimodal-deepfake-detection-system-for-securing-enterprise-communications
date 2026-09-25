import asyncio
import base64
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.face_transforms import eval_transform
from phase2_visual.face_detector import FaceDetector, select_largest_face
from phase3_audio.audio_inference import SAMPLE_RATE as AUDIO_SAMPLE_RATE
from phase3_audio.audio_inference import TARGET_SAMPLES as AUDIO_TARGET_SAMPLES
from phase3_audio.audio_inference import load_audio_model, predict_audio
from phase4_fusion_alerting.main_fusion import DEVICE, load_visual_model

app = FastAPI()

# Dynamically locate index.html in the same folder as web_app.py
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.html"

@app.get("/")
async def get():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


# Loaded once when the server starts (module import time), reused by every
# WebSocket connection -- reuses main_fusion.py's proven loading pattern for
# the visual model and audio_inference.py's loader for the audio model,
# instead of duplicating either architecture here.
VISUAL_MODEL = load_visual_model()
FACE_DETECTOR = FaceDetector()
load_audio_model()  # loads + caches phase3_audio's AudioCNN once at startup

# Warm-up: a model's first forward pass on a given device is measurably
# slower than subsequent ones (CUDA kernel compilation/caching, cuDNN
# algorithm selection). Run one throwaway inference on silence now, at
# server startup, so that cost is paid once here instead of adding latency
# to the first real WebSocket connection's time-to-audio-available. The
# result is discarded; this never affects the model weights or any
# reported score.
predict_audio(np.zeros(AUDIO_TARGET_SAMPLES, dtype=np.float32), sample_rate=AUDIO_SAMPLE_RATE)

# --- Fusion / alerting configuration -----------------------------------
# Engineering integration weight, NOT a statistically optimized weight --
# no ablation study or evidence-based tuning has been done for this pair
# of modalities yet (that is future roadmap work). Chosen only to combine
# both scores into one number for the live demo.
FUSION_VISUAL_WEIGHT = 0.6
FUSION_AUDIO_WEIGHT = 0.4

# Demo alert thresholds -- NOT calibrated clinical/security probabilities.
STATUS_SUSPICIOUS_THRESHOLD = 0.50
STATUS_CRITICAL_THRESHOLD = 0.70  # documented nominal/reference boundary; the
# actual CRITICAL decision below uses CRITICAL_ENTER_THRESHOLD/
# CRITICAL_EXIT_THRESHOLD instead (see the hysteresis-band note further
# down) -- this constant is kept only as the "true" line those two
# straddle, and is still used for the STRONG_TRANSITION_MARGIN check below.

# CRITICAL hysteresis band: real-recording evidence (Recording
# 2026-09-20 235819.mp4, 52.3s) showed the smoothed fusion score
# genuinely hovering within +/-0.02 of 0.70 for the majority of the
# session (52.8% of once-per-second samples, 72% during the densest
# flicker cluster), because this session's raw visual score sat close to
# 0.50 (the visual model's own point of maximum uncertainty on this
# face/lighting) while audio was pinned at a steady 1.0 -- with the
# locked 0.6/0.4 fusion weights, that combination arithmetically centers
# the fused score almost exactly on the 0.70 line. That is a multi-second
# amplitude problem (the signal can sit arbitrarily close to the line for
# an arbitrarily long time), not a short noise-duration problem, so no
# amount of additional STATUS_CONFIRM_TICKS could fix it without adding
# unacceptable latency (verified: confirm_ticks would need to reach
# roughly 10-13+, pushing worst-case latency past 2s) -- and even then
# would not guarantee suppression. A two-sided entry/exit band targets
# the actual (amplitude) cause instead: entering CRITICAL needs a firmer
# reading (0.72) than leaving it (0.68), so a score parked anywhere
# inside [0.68, 0.72] simply holds whatever status is already displayed
# instead of being re-evaluated every tick. Validated against the real
# 0.2s-resolution measurements from that same recording's densest
# cluster (t=17.0-22.0, interpolated to the true 0.1s tick rate): the
# unchanged single-0.70-threshold logic reproduces the 4 real status
# changes actually observed there; the entry/exit band produces 0 over
# that same real trajectory, while still correctly entering CRITICAL
# (confirmed via a separate replay) once the real data sustains a climb
# through 0.72. This changes ONLY which discrete status label is shown --
# visual_score, audio_score, fusion_score, the 0.6/0.4 weights,
# FUSION_SMOOTHING_WINDOW, and STATUS_CONFIRM_TICKS are all unchanged.
CRITICAL_ENTER_THRESHOLD = 0.72
CRITICAL_EXIT_THRESHOLD = 0.68

# Audio is only re-inferred periodically (mel-spectrogram + CNN forward
# pass every video frame would be wasteful and the model was trained on a
# fixed 4-second window anyway, not a per-frame signal).
AUDIO_INFER_INTERVAL_SECONDS = 1.0

# --- Coarse speech/silence gate (Step 5 fix) -----------------------------
# The audio model (phase3_audio) was trained on ASVspoof2019 LA, which is
# entirely spoken utterances -- it has never seen silence or room noise as
# a class, so running it on a near-silent buffer produces an output with no
# calibrated meaning, not a genuine "low risk" reading. Previously the
# model ran unconditionally on whatever was in the buffer; this was already
# disclosed to the reviewer via index.html's info-hint, but a live
# recording (2026-09-23 165243.mp4) showed the resulting Audio Spoof Risk
# swinging 16%-97% within under two minutes while no sustained speech was
# occurring -- consistent with that disclosed limitation actually firing,
# not a model or fusion bug.
#
# This is a coarse RMS-energy gate, NOT a real speech/non-speech
# classifier -- it distinguishes "something audible is happening" from
# "near silence," which is the specific case the task asks for ("no
# speech / insufficient speech -> unavailable"). It will NOT filter out
# loud non-speech noise (typing, music, environmental sound); a genuine
# VAD model would be needed for that and is out of scope for this fix.
# Threshold is a conservative, documented heuristic (typical room-noise
# floor for a laptop mic is roughly 0.001-0.005 RMS on a [-1, 1] float
# stream; normal speech is usually well above 0.02), not empirically
# calibrated against this specific machine's microphone -- flagged here
# so it can be tuned later against real recordings if it proves too
# strict or too lax, rather than silently treated as exact.
AUDIO_SILENCE_RMS_THRESHOLD = 0.01

# How often the async WebSocket handler polls the shared state and sends a
# message. Decoupled from how fast either worker thread actually produces
# new results -- this is a "latest known state" push rate, not a claim
# that inference itself runs this fast.
CONSUMER_TICK_SECONDS = 0.1

# --- Visual display temporal filter (display-only, separate from fusion) -
# The raw MobileNetV3 score genuinely swings frame-to-frame (documented
# repeatedly against real recordings this project). The existing 5-frame
# moving average in VideoWorker exists to feed the fusion/security-decision
# pipeline and is left completely unchanged -- fusion, the 0.72/0.68
# hysteresis band, and confirm-tick logic all still consume that exact
# value, with no change to their behavior. This is a SEPARATE, independent
# EMA computed from the same raw per-frame score, used ONLY to smooth the
# displayed Visual Risk percentage:
#     raw MobileNetV3 score -> EMA(VISUAL_DISPLAY_EMA_ALPHA) -> display
# EMA formula: new = alpha*raw + (1-alpha)*previous. Lower alpha = smoother
# but slower to react; 0.20 is the middle of the requested 0.15-0.25
# starting range -- conservative enough that a single-frame spike (weight
# 0.20 in the blend) has limited effect, while a sustained change still
# visibly moves the displayed value within a handful of frames. Only
# updated on a genuine detection (never fed a fake zero when the face is
# absent); held steady (not reset) during an absence so a returning face
# continues from the last valid state instead of jumping.
VISUAL_DISPLAY_EMA_ALPHA = 0.20

# --- Live display stabilization (fusion layer only) ---------------------
# Fixes dashboard flicker: visual_score/audio_score and the 0.6/0.4 fusion
# FORMULA are unchanged, but the raw fusion result recomputed every 100ms
# is naturally noisy near a threshold. Two lightweight, independent layers
# on top of that unchanged formula:
#  1. A simple moving average over the last FUSION_SMOOTHING_WINDOW raw
#     fusion readings -- at CONSUMER_TICK_SECONDS=0.1s, 10 readings = ~1.0s.
#  2. Status hysteresis: the displayed status only changes once the target
#     status (from the smoothed score) has been the same for
#     STATUS_CONFIRM_TICKS consecutive ticks (5 ticks = ~0.5s), so a single
#     boundary-crossing reading can't flip the badge on its own.
# Combined worst-case latency for a genuine, sustained change to fully
# register: ~1.0s (average) + ~0.5s (confirm) = ~1.5s, at the upper end of
# the requested 0.5-1.5s target. A change that reverts before being
# confirmed never commits -- this suppresses flicker, not sustained trends.
#
# STATUS_CONFIRM_TICKS was raised from 3->5 after live-recording evidence
# showed the *raw* per-frame visual score genuinely swinging FAKE<->REAL
# within 0.3-1s even with a perfectly stationary, stably-detected face --
# real visual-model output noise on the live-camera domain, not a
# detector or fusion-math bug (confirmed by comparing frames with an
# identical, stable YuNet bounding box). Widening FUSION_SMOOTHING_WINDOW
# instead was tried first and measured (via a standalone simulation of
# this exact algorithm against synthetic noise shaped like the recording,
# 5 random seeds) to make status changes on boundary-straddling noise
# *worse*, not better (mean 20.8 changes/14s vs the original 12.2) -- a
# wider window lags further behind a fast oscillation and ends up
# crossing the threshold on its way in and out just as often. Raising
# STATUS_CONFIRM_TICKS alone, leaving the window untouched, was the
# config that was actually measured to cut changes substantially
# (mean 1.8 changes/14s, ~85% reduction) while keeping the same ~1.5s
# latency ceiling already in use.
FUSION_SMOOTHING_WINDOW = 10
STATUS_CONFIRM_TICKS = 5

# Second confirmation path: a smoothed score that lands decisively past a
# threshold (not just barely across it) is much less likely to be noise,
# so it needs fewer consecutive ticks to confirm than a borderline
# crossing does. "Decisively" here means at least STRONG_TRANSITION_MARGIN
# beyond the relevant threshold (e.g. CRITICAL normally needs the smoothed
# score >= CRITICAL_ENTER_THRESHOLD (0.72); the fast path applies once
# it's >= 0.87). Only applies to CRITICAL and NORMAL, the two outer tiers
# -- SUSPICIOUS is a transitional
# middle zone by definition, so it always uses the normal confirm path.
STRONG_TRANSITION_MARGIN = 0.15
STRONG_CONFIRM_TICKS = 1

# --- Decision-layer freshness gating -------------------------------------
# Separates the CONTINUOUS model signal (visual_score/audio_score/
# fusion_score -- always live, never frozen, never hidden) from the
# SECURITY DECISION (a deliberately debounced state built on top of that
# signal). Real-recording evidence (Recording 2026-09-21 005457.mp4)
# showed 3 of 10 status transitions in one session committing while
# `visual_score` was a frozen, several-ticks-stale value carried over from
# before the face genuinely left frame or was covered by hands -- YuNet
# correctly reported "no face" the whole time, but the decision layer had
# no way to know the visual number it was fusing was old, not fresh.
# FACE_STALE_GRACE_TICKS: how many consecutive ticks without a face
# detection are tolerated before visual data is treated as stale for
# DECISION purposes (long enough that a single missed frame -- already
# tolerated by VideoWorker's own 5-frame average -- doesn't count).
# FACE_REBUILD_TICKS: once a stale face becomes fresh again, how many
# consecutive fresh ticks are required before the decision layer trusts
# it enough to commit a NEW transition (Requirement 7: wait for fresh
# evidence before a new decision). Both ~1.0s, matching the existing
# FUSION_SMOOTHING_WINDOW timescale so the whole system reasons in a
# consistent ~1s unit.
FACE_STALE_GRACE_TICKS = 10
FACE_REBUILD_TICKS = 10

# How many recent frame timestamps VideoWorker keeps to compute a live
# processing FPS figure for the dashboard's live-analysis panel.
FPS_WINDOW = 30

# Bounded rolling event log sent to the dashboard's event timeline panel.
EVENT_TIMELINE_MAXLEN = 20

# --- Face-box position tracking (NOT classification) -----------------------
# The box drawn on the video used to run its OWN independent REAL/FAKE/
# ANALYZING classification (OverlayStabilizer: a hysteresis state machine
# over visual_display_score) and burn that verdict into the JPEG server-side.
# That could -- and, on real recordings, did -- disagree with the dashboard's
# Security Decision, because Security Decision is a PER-CONNECTION value
# (its own fusion history + hysteresis live in ws_stream, per the "browser
# refresh gets clean history" design) while the video frame is drawn ONCE by
# the single SHARED VideoWorker and sent identically to every connection.
# A shared, server-burned box literally cannot correctly represent a
# per-connection decision -- it can only ever show one thing to everyone.
#
# The fix: VideoWorker no longer classifies or draws anything. It only
# tracks WHERE the detected face is (position continuity: hold the last
# bbox through a brief loss, same OVERLAY_HOLD_SECONDS timing as before, so
# the box doesn't blink off on a single missed frame). The frame sent to the
# browser is the plain camera image. The LABEL and COLOUR are decided
# per-connection in ws_stream from the exact same `displayed_status` already
# computed for the Security Decision badge, and drawn by the browser as an
# HTML overlay positioned from the bbox coordinates sent in the JSON -- so
# there is exactly one status computation (ws_stream's), consumed by both
# the badge and the video overlay, per connection.
OVERLAY_HOLD_SECONDS = FACE_STALE_GRACE_TICKS * CONSUMER_TICK_SECONDS


class FaceBoxTracker:
    """Holds the last detected face bbox through a brief detection loss
    (up to OVERLAY_HOLD_SECONDS), so the box doesn't blink off on a single
    missed YuNet frame. Position-only -- makes no REAL/FAKE/security
    judgement; that comes from ws_stream's Security Decision, per connection.

    update(now, bbox) is called once per processed frame with bbox=None
    when no face was detected this frame. Returns (bbox_to_send, holding):
      bbox_to_send  the detected box; the last box while holding; None if
                    genuinely gone (past the hold window)
      holding       True when this is the last-known box, not a live
                    detection this exact frame
    """

    def __init__(self):
        self.last_face_t = None
        self.last_bbox = None

    def update(self, now, bbox):
        if bbox is not None:
            self.last_face_t = now
            self.last_bbox = bbox
            return bbox, False
        if self.last_face_t is not None and now - self.last_face_t <= OVERLAY_HOLD_SECONDS:
            return self.last_bbox, True
        self.last_face_t = None
        self.last_bbox = None
        return None, False


def run_visual_inference(frame):
    """Detects the largest face in `frame` and returns
    (s_v_raw, face_found, bbox). Detection, face selection, cropping and the
    model forward pass are unchanged.

    s_v_raw is the model's raw, instantaneous probability that the detected
    face is fake (class 0 = fake, class 1 = real, matching dataset_pipeline's
    ImageFolder mapping) -- exactly what the model output, unsmoothed.
    face_found tells the caller whether a real detection happened this
    frame, so the caller's temporal smoothing only ever averages genuine
    detections and never treats a missed detection as evidence of "real".
    bbox is (x, y, w, h) of the detected face, None when there is none.

    This function deliberately draws nothing. Nothing server-side draws a
    label/box into the frame any more -- VideoWorker only tracks the bbox's
    position (FaceBoxTracker); the frontend draws the overlay from the bbox
    coordinates plus ws_stream's per-connection `status`.
    """
    faces = FACE_DETECTOR.extract_faces(frame)

    if not faces:
        return 0.0, False, None

    # Largest detected face -- avoids locking onto a small face in the
    # background instead of the person in front of the camera. Shared with
    # dataset_pipeline's YuNet-based training extraction (see
    # face_detector.select_largest_face) so live inference and training-data
    # cropping can never silently diverge on this rule again.
    face = select_largest_face(faces)
    x, y, w, h = face["bbox"]
    face_crop = face["crop"]

    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_face)
    input_tensor = eval_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = VISUAL_MODEL(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        s_v_raw = probs[0].item()  # class 0 = fake, instantaneous raw model output

    return s_v_raw, True, (x, y, w, h)


class MicrophoneBuffer:
    """Server-side rolling microphone buffer, captured directly from the OS
    audio device -- the same pattern already used for video
    (cv2.VideoCapture(0) captures the OS camera device server-side). True
    browser-microphone capture would require the browser to record audio
    and send it back over the WebSocket, but the current WebSocket is
    strictly one-directional (server -> browser only; index.html never
    calls ws.send()) and adding a client -> server audio channel would mean
    restructuring the connection handler to read and write concurrently --
    a real architecture change, not a small addition. Server-side capture
    gives a genuinely real (not fabricated) live audio signal today without
    that rewrite; true browser-mic streaming is a documented follow-up.

    Always holds exactly AUDIO_TARGET_SAMPLES (64,000 = 4s at 16kHz) of the
    most recently captured audio, matching the model's fixed training
    window exactly. `filled` tracks how many real samples have ever been
    written, so callers can tell "still warming up" from "device broken".

    `last_callback_t` fixes a real staleness gap: `filled` is clamped at
    AUDIO_TARGET_SAMPLES and never decreases, so if the OS audio callback
    stopped firing entirely (device unplugged, driver issue) after having
    been ready once, `ready` would keep reporting True forever and
    AudioWorker would keep re-running the spoof model on a frozen buffer,
    silently presenting a stale reading as a fresh one every second. Now
    `ready` also requires a callback within the last couple of seconds.
    """

    _STALE_AFTER_SECONDS = 2.0

    def __init__(self):
        self.buffer = np.zeros(AUDIO_TARGET_SAMPLES, dtype=np.float32)
        self.filled = 0
        self.lock = threading.Lock()
        self.stream = None
        self.error = None
        self.last_callback_t = None

        try:
            self.stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
        except Exception as e:
            self.error = str(e)
            self.stream = None

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata
        n = len(mono)
        with self.lock:
            if n >= AUDIO_TARGET_SAMPLES:
                self.buffer[:] = mono[-AUDIO_TARGET_SAMPLES:]
            else:
                self.buffer[:-n] = self.buffer[n:]
                self.buffer[-n:] = mono
            self.filled = min(AUDIO_TARGET_SAMPLES, self.filled + n)
            self.last_callback_t = time.monotonic()

    @property
    def ready(self):
        if self.stream is None or self.filled < AUDIO_TARGET_SAMPLES:
            return False
        if self.last_callback_t is None:
            return False
        return (time.monotonic() - self.last_callback_t) <= self._STALE_AFTER_SECONDS

    def snapshot(self):
        with self.lock:
            return self.buffer.copy()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


class SharedState:
    """Thread-safe latest-known-state, written by VideoWorker/AudioWorker
    (background threads) and read by the async WebSocket handler. This is
    the "producer/consumer" hand-off point: no queue, because a live
    dashboard only ever wants the newest value, never a backlog of stale
    ones (Step 10: avoid unbounded queues, prefer latest-state)."""

    def __init__(self):
        self.lock = threading.Lock()
        # visual_score (5-frame moving average) is kept only as a secondary/
        # diagnostic statistic now -- see ws_stream, which reads
        # visual_display_score (the EMA) as THE authoritative visual signal
        # for fusion, the security decision, and the evidence text, because
        # that is also what the face overlay and the Visual Manipulation
        # Risk gauge use. Previously visual_score (a differently-windowed
        # statistic of the same raw score) fed fusion while a SEPARATE
        # visual_display_score fed the overlay/gauge -- the two could
        # genuinely disagree (different memory/lag), which is exactly what
        # produced a real observed case of the face overlay reading FAKE
        # while Security Decision read NORMAL from the same face at the
        # same moment. One authoritative visual number now feeds all of
        # them; a real difference between the visual reading and the
        # security decision can now only come from the audio contribution,
        # which the evidence text says explicitly (see below).
        self.visual_score = None
        self.raw_visual_score = None
        self.visual_display_score = None
        self.face_detected = False
        self.frame_data_uri = ""
        # bbox/bbox_holding/frame_width/frame_height: the face's position in
        # the shared camera frame only -- NOT a classification. The frontend
        # uses these purely to position the overlay box; the box's colour/
        # label come from ws_stream's per-connection `status`, sent
        # separately. See FaceBoxTracker for why classification moved out of
        # VideoWorker entirely.
        self.bbox = None
        self.bbox_holding = False
        self.frame_width = None
        self.frame_height = None
        self.audio_score = None
        self.audio_raw_score = None
        self.audio_available = False
        self.frames_analyzed = 0
        self.fps = 0.0

    def update_visual(self, visual_score, face_detected, frame_data_uri, frames_analyzed, fps,
                       raw_visual_score, visual_display_score,
                       bbox=None, bbox_holding=False, frame_width=None, frame_height=None):
        with self.lock:
            self.visual_score = visual_score
            self.face_detected = face_detected
            self.frame_data_uri = frame_data_uri
            self.frames_analyzed = frames_analyzed
            self.fps = fps
            self.raw_visual_score = raw_visual_score
            self.visual_display_score = visual_display_score
            self.bbox = bbox
            self.bbox_holding = bbox_holding
            self.frame_width = frame_width
            self.frame_height = frame_height

    def update_audio(self, audio_score, audio_available, audio_raw_score=None):
        with self.lock:
            self.audio_score = audio_score
            self.audio_available = audio_available
            if audio_raw_score is not None:
                self.audio_raw_score = audio_raw_score

    def snapshot(self):
        with self.lock:
            return {
                "visual_score": self.visual_score,
                "face_detected": self.face_detected,
                "frame_data_uri": self.frame_data_uri,
                "bbox": self.bbox,
                "bbox_holding": self.bbox_holding,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "audio_score": self.audio_score,
                "audio_raw_score": self.audio_raw_score,
                "audio_available": self.audio_available,
                "frames_analyzed": self.frames_analyzed,
                "fps": self.fps,
                "raw_visual_score": self.raw_visual_score,
                "visual_display_score": self.visual_display_score,
            }


class VideoWorker(threading.Thread):
    """Owns the camera and runs YuNet + MobileNetV3 inference in a
    dedicated background thread, entirely outside the asyncio event loop.
    This is the actual fix for the microphone-starvation bug: previously
    this exact loop ran inline inside the async WebSocket handler, which
    kept the asyncio event loop's own thread busy almost continuously and
    starved the separate PortAudio audio-callback thread of scheduling
    time within the real uvicorn process (confirmed by instrumented
    testing -- isolated reproductions outside uvicorn did not show the
    starvation, only the real server did). Moving this work to its own
    thread means the event-loop thread now spends nearly all its time
    genuinely blocked on `await`, which is what actually frees up
    scheduling time for the microphone's callback thread.

    Keeps the existing 5-frame visual smoothing here (not in the
    consumer), since it must average every real frame this thread
    processes, not just the subset the consumer happens to poll.
    """

    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = threading.Event()
        self.cap = cv2.VideoCapture(0)
        self.opened = self.cap.isOpened()

    def run(self):
        if not self.opened:
            return

        s_v_history = deque(maxlen=5)
        last_smoothed = None
        visual_display_score = None
        frame_times = deque(maxlen=FPS_WINDOW)
        frames_analyzed = 0
        box_tracker = FaceBoxTracker()

        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                break

            s_v_raw, face_found, bbox = run_visual_inference(frame)
            # Raw score/bbox untouched. visual_display_score still advances
            # only on a genuine detection -- unchanged EMA, still feeds
            # fusion/the gauge/the box label's underlying number via
            # ws_stream. The frame itself is sent as-is: no box, no label,
            # no colour burned in server-side any more (see FaceBoxTracker
            # docstring above for why) -- only position continuity is
            # tracked here, for the bbox coordinates sent in the JSON.
            if face_found:
                s_v_history.append(s_v_raw)
                visual_display_score = (
                    s_v_raw if visual_display_score is None
                    else VISUAL_DISPLAY_EMA_ALPHA * s_v_raw + (1 - VISUAL_DISPLAY_EMA_ALPHA) * visual_display_score
                )
            box_to_send, box_holding = box_tracker.update(time.monotonic(), bbox if face_found else None)
            annotated_frame = frame
            if s_v_history:
                last_smoothed = sum(s_v_history) / len(s_v_history)
            # Unavailable (None), not auto-treated as fake, only when no
            # face has EVER been seen this connection -- same reasoning as
            # the previous single-thread implementation.
            visual_score = last_smoothed if s_v_history else None
            # visual_display_score is left untouched (held, not reset) on a
            # miss -- Requirement 8/9: preserve the last known display
            # value through an absence, and resume from it (not from
            # scratch) once the face returns, rather than a fresh/jumpy
            # start.

            frames_analyzed += 1
            frame_times.append(time.time())
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                fps = 0.0

            encoded, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_data_uri = ""
            if encoded:
                frame_data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            self.state.update_visual(
                visual_score, face_found, frame_data_uri, frames_analyzed, fps,
                raw_visual_score=(s_v_raw if face_found else None),
                visual_display_score=visual_display_score,
                bbox=box_to_send, bbox_holding=box_holding,
                frame_width=frame.shape[1], frame_height=frame.shape[0],
            )

    def stop(self):
        self.stop_event.set()

    def close(self):
        self.cap.release()


class AudioWorker(threading.Thread):
    """Owns the microphone buffer and runs the audio CNN in its own
    background thread, at most once per AUDIO_INFER_INTERVAL_SECONDS.
    Keeps the existing exponential smoothing here, for the same reason
    VideoWorker keeps the visual moving average: it must persist across
    every inference this thread performs, independent of the consumer's
    poll rate.
    """

    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = threading.Event()
        self.mic = MicrophoneBuffer()
        if self.mic.error:
            print(f"Microphone unavailable, audio will stay unavailable this session: {self.mic.error}")

    def run(self):
        smoothed_audio_score = None

        while not self.stop_event.is_set():
            if self.mic.ready:
                waveform = self.mic.snapshot()
                rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
                if rms < AUDIO_SILENCE_RMS_THRESHOLD:
                    # Insufficient signal to be evidence either way (see
                    # AUDIO_SILENCE_RMS_THRESHOLD docstring) -- explicitly
                    # marked unavailable rather than reusing/guessing a
                    # score, and the smoothed value is NOT advanced on a
                    # buffer this quiet, so a later genuine reading isn't
                    # blended with a silence sample.
                    self.state.update_audio(smoothed_audio_score, False)
                else:
                    result = predict_audio(waveform, sample_rate=AUDIO_SAMPLE_RATE)
                    raw_audio_score = result["score"]
                    # Live UI stabilization only -- does not touch the offline
                    # dev-set metrics reported for the trained audio model.
                    smoothed_audio_score = (
                        raw_audio_score if smoothed_audio_score is None
                        else 0.7 * smoothed_audio_score + 0.3 * raw_audio_score
                    )
                    self.state.update_audio(smoothed_audio_score, True, audio_raw_score=raw_audio_score)
                # Once live, throttle actual inference calls to once per
                # interval -- no need to re-run the CNN faster than that.
                self.stop_event.wait(AUDIO_INFER_INTERVAL_SECONDS)
            else:
                self.state.update_audio(None, False)
                # Not yet ready: poll quickly so "buffer just filled" is
                # noticed promptly, instead of inheriting the 1-second
                # inference throttle as extra, avoidable startup latency.
                self.stop_event.wait(0.1)

    def stop(self):
        self.stop_event.set()

    def close(self):
        self.mic.close()


class SharedPipeline:
    """One camera + microphone pipeline shared by every WebSocket connection.

    Each connection used to build its own VideoWorker, i.e. its own
    cv2.VideoCapture(0). On this machine a second capture handle on the same
    camera invalidates the first: the older handle's read() starts returning
    False after ~3 frames and never recovers (measured directly with two
    handles, and reproduced end to end with two overlapping WebSocket
    clients). VideoWorker.run() ends on a failed read, so the older
    connection's video thread died silently and its Face status, Visual
    Inference FPS and Frames Analyzed froze at their last values while the
    socket stayed open and kept re-sending them ("STREAM ACTIVE" with dead
    numbers). Any second tab, preview pane or test client did this to the
    first one.

    The workers are now started by the first connection, shared by every
    concurrent one, and stopped when the last one leaves. Everything
    per-connection (fusion window, security state machine, event timeline)
    still lives in ws_stream; the workers, models, temporal filters, face
    overlay and message schema are untouched.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._clients = 0
        self._state = None
        self._video = None
        self._audio = None

    def acquire(self):
        """Returns the shared SharedState (starting the workers for the first
        client), or None if the camera cannot be opened."""
        with self._lock:
            if self._clients == 0:
                state = SharedState()
                video = VideoWorker(state)
                audio = AudioWorker(state)
                if not video.opened:
                    video.close()
                    audio.close()
                    return None
                video.start()
                audio.start()
                self._state, self._video, self._audio = state, video, audio
            self._clients += 1
            return self._state

    def release(self):
        with self._lock:
            self._clients -= 1
            if self._clients > 0:
                return
            video, audio = self._video, self._audio
            self._state = self._video = self._audio = None
            video.stop()
            audio.stop()
            video.join(timeout=2.0)
            audio.join(timeout=2.0)
            video.close()
            audio.close()


PIPELINE = SharedPipeline()


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()

    state = PIPELINE.acquire()

    if state is None:
        await websocket.send_json({
            "type": "inference",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame": "",
            "visual_score": None, "visual_raw": None, "visual_smoothed": None,
            "audio_score": None, "audio_raw": None, "fusion_score": None, "fusion_fresh": False,
            "status": "ERROR: camera unavailable",
            "face_detected": False, "visual_fresh": False, "audio_available": False,
            "bbox": None, "bbox_holding": False, "frame_width": None, "frame_height": None,
            "fps": 0.0, "frames_analyzed": 0,
            "evidence": ["Camera device could not be opened."],
            "event_timeline": [], "visual_display_score": None,
        })
        return

    # Per-connection display-stabilization state (Step: fix status flicker).
    # Fresh for every connection, same as everything else per-connection --
    # a browser refresh/reconnect naturally gets a clean history, not stale
    # smoothing carried over from a previous session.
    fusion_history = deque(maxlen=FUSION_SMOOTHING_WINDOW)
    # security_state is the INTERNAL hysteresis-tracked value -- exactly the
    # same band + confirm-tick machinery as before, just renamed. It starts
    # at "NORMAL" (the neutral baseline) rather than "NO SIGNAL": once the
    # decision layer has a full smoothing window of real data to evaluate
    # (see `analyzing` below), the first real evaluation should be able to
    # commit immediately if that data genuinely reads NORMAL, rather than
    # forcing an arbitrary confirm delay on the safe/no-risk case.
    security_state = "NORMAL"
    pending_status = None
    pending_streak = 0
    # Freshness-gating state (new -- see FACE_STALE_GRACE_TICKS/
    # FACE_REBUILD_TICKS above for the evidence and reasoning).
    ticks_since_face_fresh = 0
    was_stale = False
    rebuild_ticks_remaining = 0
    # displayed_status is the OUTER value the dashboard actually sees --
    # NO SIGNAL / ANALYZING / security_state. Separating this from
    # security_state is what lets the decision layer hold its internal
    # hysteresis state steady during NO SIGNAL/ANALYZING without losing it.
    displayed_status = "NO SIGNAL"
    last_displayed_status = None
    event_timeline = deque(maxlen=EVENT_TIMELINE_MAXLEN)

    try:
        while True:
            snap = state.snapshot()
            # visual_score (5-frame MA) is kept only for the JSON payload's
            # backward-compatible/diagnostic field below -- it is NOT what
            # drives fusion, the security decision, or the evidence text.
            # visual_display_score (the EMA) is THE authoritative visual
            # signal: it is also what the face overlay and the Visual
            # Manipulation Risk gauge use, so there is exactly one visual
            # number in play, not two differently-windowed statistics of
            # the same raw score that could silently disagree with each
            # other. See SharedState.__init__ for the full reasoning and
            # the real recording that demonstrated the disagreement.
            visual_score = snap["visual_score"]
            visual_smoothed = snap["visual_display_score"]
            audio_score = snap["audio_score"]
            audio_available = snap["audio_available"]
            face_detected = snap["face_detected"]

            # --- Freshness bookkeeping: computed BEFORE fusion, not after,
            # so a stale visual reading can be excluded from fusion_raw
            # itself instead of only gating the security decision below it.
            # (Bug found via a real recording, 2026-09-23 165243.mp4: during
            # a face-loss gap the Visual Risk gauge correctly showed a
            # STALE-tagged held value, but Multimodal Risk kept silently
            # blending that same stale number with fresh audio and showed
            # no staleness indicator at all -- e.g. observed 69% STALE
            # visual blended with 16% fresh audio produced an unflagged 48%
            # multimodal reading.) See FACE_STALE_GRACE_TICKS above for why
            # the grace period exists at all (a single missed frame must
            # not count as staleness).
            if face_detected:
                ticks_since_face_fresh = 0
            else:
                ticks_since_face_fresh += 1
            visual_is_stale = (
                visual_smoothed is not None
                and ticks_since_face_fresh > FACE_STALE_GRACE_TICKS
            )
            # audio_available now means "the last audio reading was both
            # from live mic hardware AND above the silence-energy floor"
            # (see MicrophoneBuffer.ready and AUDIO_SILENCE_RMS_THRESHOLD) --
            # i.e. it already IS a freshness flag, not just a hardware-
            # present flag, so it is used the same way visual_is_stale is.
            visual_for_fusion = None if visual_is_stale else visual_smoothed
            audio_for_fusion = audio_score if audio_available else None

            # --- Fusion calculation: same 0.6/0.4 formula and same
            # single-modality/unavailable fallback shape as before -- the
            # only change is that a STALE input is treated exactly like a
            # MISSING input here, so the continuous number can never
            # silently be built from data that stopped being current. ---
            if visual_for_fusion is not None and audio_for_fusion is not None:
                fusion_raw = FUSION_VISUAL_WEIGHT * visual_for_fusion + FUSION_AUDIO_WEIGHT * audio_for_fusion
            elif visual_for_fusion is not None:
                fusion_raw = visual_for_fusion  # audio unavailable/stale -> visual-only fallback
            elif audio_for_fusion is not None:
                fusion_raw = audio_for_fusion   # visual unavailable/stale -> audio-only fallback
            else:
                fusion_raw = None               # neither fresh -> no fabricated score

            analyzing = False

            if fusion_raw is None:
                # No data at all: NO SIGNAL is immediate and bypasses
                # smoothing/hysteresis entirely (Requirement 8) -- and
                # resets ALL stabilization/freshness state so a later
                # reconnection of data starts clean rather than averaging
                # in stale history. This condition and its reset behavior
                # are UNCHANGED from before the redesign.
                fusion_history.clear()
                security_state = "NORMAL"
                pending_status = None
                pending_streak = 0
                ticks_since_face_fresh = 0
                was_stale = False
                rebuild_ticks_remaining = 0
                fusion_smoothed = None
                displayed_status = "NO SIGNAL"
            else:
                # Layer 1 (CONTINUOUS SIGNAL, always live): simple moving
                # average of the last FUSION_SMOOTHING_WINDOW raw fusion
                # readings (~1.0s). This is the number the dashboard's
                # Multimodal Risk gauge shows every tick, regardless of
                # what the decision layer below is doing -- the goal is to
                # make uncertainty visible, not hide it behind the badge.
                fusion_history.append(fusion_raw)
                fusion_smoothed = sum(fusion_history) / len(fusion_history)

                # ticks_since_face_fresh / visual_is_stale are already
                # computed above, before fusion_raw -- not recomputed here.
                if was_stale and not visual_is_stale:
                    # Face just came back after being genuinely stale --
                    # Requirement 7: wait for fresh evidence before a new
                    # decision, instead of trusting the average immediately.
                    rebuild_ticks_remaining = FACE_REBUILD_TICKS
                was_stale = visual_is_stale
                if rebuild_ticks_remaining > 0:
                    rebuild_ticks_remaining -= 1

                window_filling = len(fusion_history) < FUSION_SMOOTHING_WINDOW
                analyzing = window_filling or rebuild_ticks_remaining > 0

                if analyzing:
                    # Not enough fresh evidence yet to trust a decision --
                    # hold. The hysteresis pending/confirm counters are
                    # simply not touched this tick (frozen, not reset), so
                    # whatever was already accumulating resumes cleanly
                    # once analyzing ends.
                    pass
                elif visual_is_stale:
                    # Requirement 6: a stale visual score must not create a
                    # NEW decision. Hold security_state exactly where it is;
                    # a pending streak toward some other target is dropped
                    # rather than allowed to keep accumulating on stale data.
                    pending_status = None
                    pending_streak = 0
                else:
                    # --- Layer 2: hysteresis band + confirm-tick logic ---
                    # UNCHANGED from the previously-validated implementation
                    # -- only the variable name (security_state, was
                    # displayed_status) and its gating above are new.
                    if security_state == "CRITICAL":
                        if fusion_smoothed <= CRITICAL_EXIT_THRESHOLD:
                            target_status = "SUSPICIOUS" if fusion_smoothed >= STATUS_SUSPICIOUS_THRESHOLD else "NORMAL"
                        else:
                            target_status = "CRITICAL"
                    else:
                        if fusion_smoothed >= CRITICAL_ENTER_THRESHOLD:
                            target_status = "CRITICAL"
                        elif fusion_smoothed >= STATUS_SUSPICIOUS_THRESHOLD:
                            target_status = "SUSPICIOUS"
                        else:
                            target_status = "NORMAL"

                    if target_status == security_state:
                        pending_status = None
                        pending_streak = 0
                    else:
                        if target_status == pending_status:
                            pending_streak += 1
                        else:
                            pending_status = target_status
                            pending_streak = 1

                        if (target_status == "CRITICAL"
                                and fusion_smoothed >= CRITICAL_ENTER_THRESHOLD + STRONG_TRANSITION_MARGIN):
                            required_ticks = STRONG_CONFIRM_TICKS
                        elif (target_status == "NORMAL"
                                and fusion_smoothed <= STATUS_SUSPICIOUS_THRESHOLD - STRONG_TRANSITION_MARGIN):
                            required_ticks = STRONG_CONFIRM_TICKS
                        else:
                            required_ticks = STATUS_CONFIRM_TICKS

                        if pending_streak >= required_ticks:
                            security_state = target_status
                            pending_status = None
                            pending_streak = 0

                displayed_status = "ANALYZING" if analyzing else security_state

            # --- Human-readable evidence for the current displayed_status,
            # generated server-side so there's one source of truth instead
            # of the frontend re-deriving "why" from raw numbers. ---
            evidence = []
            if displayed_status == "NO SIGNAL":
                evidence.append("No face detected and no audio signal available.")
            elif displayed_status == "ANALYZING":
                if fusion_raw is not None and len(fusion_history) < FUSION_SMOOTHING_WINDOW:
                    evidence.append("Accumulating signal history before committing a security decision.")
                if rebuild_ticks_remaining > 0:
                    evidence.append("Face reacquired -- waiting for fresh visual evidence before a new decision.")
            else:
                if visual_is_stale:
                    evidence.append("Face not currently visible -- holding last confirmed decision.")
                if visual_smoothed is not None and visual_smoothed >= STATUS_SUSPICIOUS_THRESHOLD:
                    evidence.append(f"Elevated visual manipulation signal ({visual_smoothed * 100:.0f}%).")
                if audio_score is not None and audio_score >= STATUS_SUSPICIOUS_THRESHOLD:
                    evidence.append(f"Elevated audio spoof signal ({audio_score * 100:.0f}%).")
                if fusion_smoothed is not None and fusion_smoothed >= CRITICAL_ENTER_THRESHOLD:
                    evidence.append(
                        f"Multimodal risk at or above the CRITICAL decision threshold "
                        f"({fusion_smoothed * 100:.0f}% >= {CRITICAL_ENTER_THRESHOLD * 100:.0f}%)."
                    )
                elif fusion_smoothed is not None and fusion_smoothed >= STATUS_SUSPICIOUS_THRESHOLD:
                    evidence.append(
                        f"Multimodal risk above the SUSPICIOUS decision threshold "
                        f"({fusion_smoothed * 100:.0f}% >= {STATUS_SUSPICIOUS_THRESHOLD * 100:.0f}%)."
                    )
                # Explicit reconciliation: the face box no longer runs its
                # own classification (see FaceBoxTracker) -- it shows this
                # exact displayed_status, so it can never disagree with the
                # Security Decision badge. The Visual Manipulation Risk
                # gauge, however, is still deliberately visual-only (see
                # VISUAL_DISPLAY_EMA_ALPHA), so IT can still read elevated
                # while Security Decision reads NORMAL/SUSPICIOUS (not
                # CRITICAL) for a genuine, documented multimodal reason --
                # audio pulling the 0.6/0.4 blend down (or visual currently
                # stale, already covered above). Say so explicitly instead
                # of leaving the reviewer to reconcile a seemingly
                # contradictory gauge and security badge themselves.
                if (not visual_is_stale and visual_smoothed is not None
                        and visual_smoothed >= STATUS_SUSPICIOUS_THRESHOLD
                        and displayed_status in ("NORMAL", "SUSPICIOUS")):
                    audio_txt = f"{audio_score * 100:.0f}%" if audio_score is not None else "unavailable"
                    evidence.append(
                        f"Visual model alone reads elevated ({visual_smoothed * 100:.0f}%), but the multimodal "
                        f"decision also weighs audio (currently {audio_txt}) at 0.6/0.4 -- Security Decision "
                        f"(and the face box, which mirrors it) reflects the combined signal, not visual alone."
                    )
                if not evidence:
                    evidence.append("No elevated signals observed.")

            # --- Event timeline: edge-triggered, only logs on a genuine
            # change of displayed_status, not every 100ms tick. ---
            now_iso = datetime.now(timezone.utc).isoformat()
            if displayed_status != last_displayed_status:
                if displayed_status == "NO SIGNAL":
                    message = "Signal lost -- no face and no audio available."
                elif displayed_status == "ANALYZING":
                    message = "Analyzing -- building fresh evidence before a decision."
                elif fusion_smoothed is not None:
                    message = (
                        f"Security status: {last_displayed_status or 'START'} -> {displayed_status} "
                        f"(multimodal risk {fusion_smoothed * 100:.0f}%)"
                    )
                else:
                    message = f"Security status -> {displayed_status}"
                event_timeline.append({"timestamp": now_iso, "message": message})
                last_displayed_status = displayed_status

            audio_raw_score = snap["audio_raw_score"]

            await websocket.send_json({
                "type": "inference",
                "timestamp": now_iso,
                "visual_score": round(visual_score, 4) if visual_score is not None else None,
                # Authoritative state fields (see SharedState.__init__ /
                # ws_stream top-of-loop comments): visual_raw is the
                # instantaneous per-frame model output; visual_smoothed is
                # visual_display_score, the SAME number feeding the Visual
                # Risk gauge and fusion below. The face-box overlay does NOT
                # use this number for its label any more -- it uses `status`
                # (below), the exact same authoritative value the Security
                # Decision badge shows, so the two can never disagree.
                "visual_raw": round(snap["raw_visual_score"], 4) if snap["raw_visual_score"] is not None else None,
                "visual_smoothed": round(visual_smoothed, 4) if visual_smoothed is not None else None,
                "audio_score": round(audio_score, 4) if audio_score is not None else None,
                "audio_raw": round(audio_raw_score, 4) if audio_raw_score is not None else None,
                "fusion_score": round(fusion_smoothed, 4) if fusion_smoothed is not None else None,
                "status": displayed_status,
                "face_detected": face_detected,
                "visual_fresh": not visual_is_stale,
                "audio_available": snap["audio_available"],
                # Face position only (see FaceBoxTracker) -- the frontend
                # positions the overlay box from these and colours/labels it
                # from `status` above. bbox is [x, y, w, h] in the ORIGINAL
                # camera frame's pixel coordinates (frame_width/frame_height
                # below), None when no face is currently shown (genuinely
                # gone, past the hold window). bbox_holding is True when this
                # is the last-known position during a brief detection loss,
                # not a live detection this exact frame.
                "bbox": list(snap["bbox"]) if snap["bbox"] is not None else None,
                "bbox_holding": snap["bbox_holding"],
                "frame_width": snap["frame_width"],
                "frame_height": snap["frame_height"],
                # True whenever fusion_raw was built from at least one
                # genuinely fresh input this tick -- by construction this
                # matches (fusion_score is not None), since a stale/missing
                # visual or audio reading is excluded from fusion_raw above
                # rather than silently included. Sent explicitly (not just
                # inferred from fusion_score) so the frontend/any future
                # consumer never has to re-derive freshness from a number.
                "fusion_fresh": fusion_raw is not None,
                "frame": snap["frame_data_uri"],
                "fps": round(snap["fps"], 1),
                "frames_analyzed": snap["frames_analyzed"],
                "evidence": evidence,
                "event_timeline": list(event_timeline),
                # Kept as a separate field name for the existing frontend
                # (index.html reads data.visual_display_score for the Visual
                # Risk gauge and the face overlay's underlying state) -- same
                # value as visual_smoothed above, not a second computation.
                "visual_display_score": round(visual_smoothed, 4) if visual_smoothed is not None else None,
            })

            await asyncio.sleep(CONSUMER_TICK_SECONDS)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket stream error: {e}")
    finally:
        PIPELINE.release()

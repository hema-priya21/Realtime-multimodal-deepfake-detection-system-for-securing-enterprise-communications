# YuNet-clean checkpoint: root-cause audit of sealed-test improvement vs. live-domain regression

**Status: read-only investigation. No production file was modified.**
`phase4_fusion_alerting/web_app.py`, `phase2_visual/visual_model.pth`, thresholds, hysteresis, and fusion weights are
all confirmed unchanged (verified via `git status` and file mtimes — see §0). The new checkpoint
(`phase2_visual/checkpoints/visual_model_yunet_clean_best.pth`) is **not** wired into `web_app.py`.

Date: 2026-09-23

---

## 0. Production-file protection check

```
git status --porcelain -- phase2_visual/visual_model.pth   -> (empty: untouched)
phase2_visual/visual_model.pth mtime                        -> 2026-09-20 02:57 (predates this experiment)
phase2_visual/checkpoints/visual_model_yunet_clean_best.pth  -> separate file, 2026-09-23 20:39
```
No production file was written to during this audit. All scripts referenced below were run from the session
scratchpad and only read project files plus the two checkpoints.

---

## 1. Train-extraction anomaly — RESOLVED

Cross-referenced all 4,900 `train.csv` rows against the actual files in
`data/visual_processed_yunet_full/train/{real,fake}/`, using the exact same `video_stem` formula and
`already_processed()`/`_scan_done_stems()` skip logic the extraction scripts use (`dataset_pipeline/extract_frames_yunet.py`,
`dataset_pipeline/extract_frames_yunet_parallel.py`).

| quantity | count |
|---|---|
| Expected train videos (train.csv rows) | 4,900 |
| Distinct expected stems (0 CSV duplicates/collisions) | 4,900 |
| Stems with ≥1 crop on disk | 4,778 (97.5%) |
| Stems with **zero** crops | 122 (2.5%) |
| Orphan crop files on disk not in train.csv | 0 |
| Label-folder mismatches (crop in wrong real/fake dir) | 0 |
| Total crops on disk, summed over present stems | **56,443** (matches the live `train/real`+`train/fake` file count exactly) |

Crop-count-per-video histogram (4,778 present stems): 4,561 have the full 12/12 crops; 217 have 1–11 (partial,
due to individual no-detection frames — same pattern already seen in val/test); **0** have more than 12.

**Why the extraction log showed `processed=289 skipped=4,611`:** the final logged run resumed an
already-mostly-complete extraction session from earlier the same day. `already_processed()` /
`_scan_done_stems()` skip a video only if it already has ≥1 crop file with its exact stem — by construction this
cannot mark a genuinely-untouched video as "skipped" unless two different CSV rows collide on the same stem
(checked: 0 collisions). The 4,611 "skipped" videos were legitimately extracted by earlier parts of the same
day's session; the final run only had 289 videos left to do, of which 122 produced zero crops (genuine
no-face-detection) and 167 produced crops normally. `289 + 4,611 = 4,900` and `4,778 = 4,611 + 167` both check out
exactly.

**Independently re-verified the 122 zero-crop videos are genuine, not silent failures:** re-ran YuNet detection
(read-only, no files written) at the same 12 sampled frame positions on a random sample of 8 of the 122 videos.
All 8 opened fine and produced **0/12 detections**, e.g.:
```
ffpp_NeuralTextures_516_555   total_frames=354  detections=0/12
ffpp_FaceSwap_713_726         total_frames=410  detections=0/12
ffpp_Deepfakes_733_935        total_frames=518  detections=0/12
ffpp_FaceSwap_507_418         total_frames=319  detections=0/12
```
This matches the extraction script's own logged `zero-crop-videos=122` for its final run and is consistent with
the same "genuine YuNet non-detection on a subset of FF++ source videos" characteristic already independently
verified earlier in this project (e.g. `508.mp4`, `045.mp4`, `889.mp4`, `607.mp4`).

Crop file mtimes span one continuous window (2026-09-23 13:56–17:10), ruling out stale data from an abandoned,
much-older run.

**Conclusion: the training dataset is complete.** 56,443 train crops is the full, intended extraction (minus
122 videos, 2.5% of the split, where YuNet genuinely detects no face at any of the 12 sampled positions — a
property of those specific source videos, not a bug). This does **not** explain the live-domain regression.

---

## 2. Same-test-set comparison (true apples-to-apples)

**Root-cause note:** `evaluate_visual_yunet_clean.py`'s original "BASELINE (production)" column was a
**hardcoded dictionary of previously-recorded numbers**, not a live evaluation of `visual_model.pth` on this
exact test directory — a real methodology gap (this is answer **F** to the hypothesis list: a genuine
implementation issue in my own comparison, now fixed). Both checkpoints were re-evaluated in one script, in the
same process, on the exact same `data/visual_processed_yunet_full/test` (12,009 crops), eliminating that
confound. The hardcoded numbers turned out close to the true baseline (e.g. accuracy 82.52% hardcoded vs. 82.07%
true), so no earlier conclusion flips, but the identity- and method-level breakdowns below were never computed
against a true baseline before now, and they change the picture materially (§2.3).

### 2.1 Overall metrics

| METRIC | PRODUCTION | NEW (yunet_clean) | DIFF |
|---|---|---|---|
| Accuracy | 82.07% | 91.21% | +9.13pp |
| ROC-AUC | 89.27% | 95.67% | +6.40pp |
| Fake precision | 94.91% | 96.15% | +1.24pp |
| Fake recall | 82.93% | 93.17% | +10.24pp |
| Fake F1 | 88.52% | 94.64% | +6.12pp |
| Real precision | 47.69% | 70.46% | +22.77pp |
| Real recall | 77.77% | 81.37% | +3.60pp |
| Real F1 | 59.12% | 75.52% | +16.40pp |
| Macro F1 | 73.82% | 85.08% | +11.26pp |
| Weighted F1 | 83.62% | 91.45% | +7.84pp |

Confusion matrices (rows=actual, cols=predicted, order=[fake, real]):
```
PRODUCTION          NEW
[[8299 1708]        [[9324  683]
 [ 445 1557]]         [ 373 1629]]
```

- **Real crop-level false-positive rate:** production 1708/... — corrected: real crops total=2002,
  wrong=445 (production, 22.2%) vs. wrong=373 (new, 18.6%). New is better.
- **Real identity-level false-positive rate (video mean P(fake)≥0.5):** production **24/169 = 14.2%**,
  new **24/169 = 14.2%** — **identical count**, not improved at the identity level (see §2.3 — this identical
  total conceals a large reshuffle).
- **Mean real P(fake):** production 0.267, new 0.200 (lower/better on average).

### 2.2 Real P(fake) distribution — the new model is far more polarized, not just "more accurate"

| percentile | production | new |
|---|---|---|
| p5 | 0.0071 | 0.0000042 |
| p25 | 0.0400 | 0.00030 |
| p50 (median) | 0.1655 | 0.00885 |
| p75 | 0.4596 | 0.2414 |
| p90 | 0.6605 | 0.8903 |
| p95 | 0.7931 | 0.9819 |
| p99 | 0.9453 | 0.9997 |

This is a concrete, measured behavioral difference, not speculation: production's real-sample P(fake)
distribution is smooth and gradual (moderate uncertainty across most of the range). The new model is
**bimodal** — confidently correct (near 0) for the large majority of real crops (hence the better median/mean),
but confidently *wrong* (0.89–0.9997) for the tail that it misses, markedly higher than production's tail
(0.66–0.945 at the same percentiles). The new model hedges less in both directions. This single measurement
is the clearest evidence that the new model's aggregate improvement is not a uniform tightening of the decision
boundary — it is a sharper, more extreme one.

### 2.3 Per-manipulation-method breakdown

| method | true label | n | prod acc | new acc | prod mean P(fake) | new mean P(fake) |
|---|---|---|---|---|---|---|
| Deepfakes | fake | 1,999 | 96.4% | 96.7% | 0.881 | 0.962 |
| Face2Face | fake | 2,005 | 86.0% | 95.9% | 0.800 | 0.950 |
| FaceShifter | fake | 2,000 | 97.0% | 95.7% | 0.874 | 0.948 |
| FaceSwap | fake | 1,995 | **59.2%** | **96.7%** | 0.577 | 0.963 |
| NeuralTextures | fake | 2,008 | 76.0% | 80.9% | 0.675 | 0.799 |
| original | real | 2,002 | 77.8% | 81.4% | 0.267 | 0.200 |

Production was barely above chance on **FaceSwap** specifically (59.2% accuracy, mean P(fake) 0.577 — right at
the threshold). The new model fixes this dramatically (96.7%). This is real, substantial, method-specific
improvement, not noise. FaceShifter is the one fake method where the new model is marginally *worse* (−1.3pp),
within normal run-to-run variance.

### 2.4 Identity-level disagreement — the critical finding

Both checkpoints misclassify the same **count** of real identities (24/169), but not the same **set**:

| | count |
|---|---|
| Wrong under production only (**fixed** by new) | 17 |
| Wrong under new only (**newly broken** by new) | 17 |
| Wrong under **both** | 7 |
| Total wrong, either model | 24 (each) |

**Newly broken** (production correctly read these as real; new model does not):
`ffpp_original_015, 044, 056, 070, 160, 472, 556, 578, 635, 662, 752, 772, 807, 835, 863, 887, 957`

**Fixed** (production misclassified these; new model correctly reads them as real):
`ffpp_original_049, 058, 210, 297, 314, 545, 588, 595, 597, 663, 674, 681, 711, 723, 881, 940, 983`

Example swings: `ffpp_original_835` went from mean P(fake)=0.029 (production, correctly real) to 0.968 (new,
confidently fake). `ffpp_original_058` went the opposite way: 0.837 (production, wrongly fake) to 0.028 (new,
correctly real).

**This is the single most important piece of evidence in this audit.** The new model did not learn a strictly
more general real-vs-fake boundary that is a superset of what production knew — it learned a *different*
boundary that happens to net out to the same identity-level error rate on the 169 sealed-test identities, by
fixing some and breaking an almost-equal number of others. That is the signature of a model still relying on
identity/appearance-correlated cues rather than a fully generalized forgery signal (consistent with the
identity-dependent spurious correlation already documented earlier in this project's forensic audit) — just a
different instance of the same underlying pattern, not its elimination. It directly explains why performance on
two *new* real identities outside the 169-person sealed test (the two screen-recording subjects, §3) cannot be
predicted from the sealed-test aggregate metrics alone: those two people were never at stake in the swap above,
and this audit gives no basis to assume they'd land on the "fixed" side rather than the "newly broken" side.

---

## 3. Screen-recording domain-shift investigation

### 3.1 Resolution / FPS / codec / bitrate

| file | resolution | fps | codec | size | approx. bitrate |
|---|---|---|---|---|---|
| Screen Recording 150707.mp4 | 1826×808 | 30.0 | h264 | 28.3 MB | 6,487 kbps |
| Screen Recording 165243.mp4 | 1838×768 | 30.0 | h264 | 83.9 MB | 6,097 kbps |
| FF++ `original/000.mp4` | 640×480 | 25.0 | h264 | 0.9 MB | 452 kbps |
| FF++ `original/397.mp4` | 1920×1080 | 30.0 | h264 | 3.8 MB | 2,567 kbps |

The screen recordings actually have **higher** bitrate than FF++'s c23-compressed videos — this is not a simple
"screen recording = more compressed" story.

### 3.2 Extra pipeline generations — confirmed by code, not assumed

Checked exactly where visual inference reads its input frame in `web_app.py`:

```
web_app.py:544   success, frame = self.cap.read()      # raw camera frame
web_app.py:~552  s_v_raw, ... = run_visual_inference(frame)   # inference on the RAW frame
...
web_app.py:584   encoded, buffer = cv2.imencode(".jpg", annotated_frame)   # JPEG encode happens AFTER inference, for browser display only
```

**This is an important, previously under-stated finding.** In genuine live use, the model performs inference on
the raw camera frame straight out of `cv2.VideoCapture` — it never sees the JPEG-encoded/browser-rendered
version at all. The screen-recording sanity-test methodology, by construction, feeds the model something the
production path never actually produces:

```
Live production inference input:
  camera -> cv2.VideoCapture frame -> YuNet -> crop -> resize(224) -> model

Screen-recording sanity-test input (this audit's tool, and the earlier session's):
  camera -> cv2.VideoCapture frame -> [inference already happened here in real use]
         -> cv2.imencode JPEG(~95) -> browser JPEG-decode -> DOM render -> CSS scale
         -> Windows screen capture -> H.264 encode -> (my script) H.264 decode
         -> YuNet -> crop -> resize(224) -> model
```

The screen-recording test adds **at least three extra lossy transcoding generations** (JPEG encode/decode,
browser-render/CSS-scale/recapture, H.264 encode/decode) that a real live-camera session never goes through.
This means the regression measured in §3.3 below is evidence about the model's behavior on
*screen-recording-domain* input specifically — it is not yet evidence about the model's behavior on genuine raw
live-camera input, which is a distinct, so-far-untested domain. This is the most consequential open question
from this audit (see §5).

### 3.3 Face crop context and native resolution (measured, not assumed)

Ran YuNet directly (read-only probe, not modifying `face_detector.py`) against 40 sampled FF++ real test videos
and both screen recordings, at 12–40 positions each, reading the raw detection + landmark output:

| | n faces | bbox-area / frame-area | native bbox width (px) | \|roll\| degrees |
|---|---|---|---|---|
| FF++ real test sample (40 videos) | 470 | mean 0.055, median 0.050 | mean 154, **median 131** | mean 3.46°, median 2.86° |
| Screen Recording 150707.mp4 | 30 | mean 0.031, median 0.038 | mean 187, **median 211** | mean 4.21°, median 2.89° |
| Screen Recording 165243.mp4 | 34 | mean 0.028, median 0.025 | mean 171, **median 163** | mean 4.46°, median 3.47° |

- **Context (bbox area relative to full frame):** screen-recording faces occupy a *smaller* fraction of their
  (larger, UI-inclusive) frame than FF++ faces do — the opposite of what a "zoomed-in crop" hypothesis would
  predict.
- **Native resolution (what actually matters for the 224×224 resize/upsampling amount):** FF++ real crops have a
  **lower** median native width (131px, requiring ~1.7× upsampling to reach 224) than either screen recording
  (163–211px, near 1:1, minimal upsampling). The screen-recording faces are, if anything, higher-native-resolution
  inputs than the typical FF++ training real face.
- **Pose (roll angle from YuNet's own eye landmarks):** essentially indistinguishable — 2.9–3.5° median across
  all three sources, well within the same range. Pose/alignment is **not** a meaningful factor here.

**Relevant training detail:** `train_visual_yunet_clean.py`'s own docstring documents the video evidence that
motivated its `simulate_native_resolution_loss` augmentation (40% of training samples downsampled then
upsampled back to 224): a Pearson r=−0.84 correlation, on this exact 150707.mp4 recording, between face-crop
area and the *production* model's P(fake) (small/distant faces read fake, large/near faces read real). That
augmentation was designed around, and explicitly validated against, the *small*-face failure mode. It was never
separately validated on genuinely high-native-resolution real faces (like this recording's large/near-camera
tracks) — the new model's regression is concentrated in exactly that untested regime (§3.4). This is a plausible
contributing mechanism, not a proven one: no code fault was found in the augmentation itself, only an
untested interaction.

### 3.4 Direct track-by-track comparison — production vs. new, identical frames, same session

Re-ran **both** checkpoints, in this session, through the identical YuNet-detection + `eval_transform` pipeline
on the same two screen recordings, so any difference is attributable to model weights alone (detection is
deterministic and identical between runs).

**150707.mp4** (production model was previously the more studied case — this table is the direct re-measurement):

| track | t (s) | mean bbox area | PROD mean P(fake) | PROD %FAKE | NEW mean P(fake) | NEW %FAKE | Δ mean |
|---|---|---|---|---|---|---|---|
| 0 | 0.0–4.1 | 23,345 | 0.829 | 100.0% | 0.980 | 100.0% | +0.151 |
| 1 | 7.8–11.8 | 24,034 | 0.819 | 100.0% | 0.967 | 96.7% | +0.148 |
| 2 | 11.8–12.1 | 55,157 | 0.620 | 100.0% | 0.803 | 90.0% | +0.183 |
| 3 | 12.2–13.8 | 57,900 | 0.521 | 68.0% | 0.963 | 100.0% | **+0.442** |
| 4 | 15.0–20.3 | 57,809 | 0.365 | 12.7% | 0.931 | 98.7% | **+0.566** |
| 5 | 20.3–22.5 | 57,930 | 0.444 | 40.3% | 0.954 | 95.5% | **+0.510** |
| 6 | 22.6–23.1 | 57,215 | 0.196 | 0.0% | 0.979 | 100.0% | **+0.783** |
| 7 | 26.5–34.8 | 56,871 | 0.323 | 15.6% | 0.797 | 82.4% | **+0.474** |

**165243.mp4:**

| track | t (s) | mean bbox area | PROD mean P(fake) | PROD %FAKE | NEW mean P(fake) | NEW %FAKE | Δ mean |
|---|---|---|---|---|---|---|---|
| 0 | 0.0–3.5 | 28,410 | 0.702 | 100.0% | 1.000 | 100.0% | +0.298 |
| 1 | 6.1–9.1 | 28,784 | 0.743 | 100.0% | 1.000 | 100.0% | +0.257 |
| 2 | 12.7–16.1 | 30,713 | 0.707 | 96.2% | 1.000 | 100.0% | +0.293 |
| 3 | 16.2–23.6 | 47,195 | 0.445 | 25.0% | 0.689 | 69.2% | +0.244 |
| 6 | 28.1–34.0 | 47,086 | 0.418 | 13.5% | 0.931 | 95.5% | **+0.513** |
| 7 | 34.0–39.1 | 34,187 | 0.754 | 97.4% | 1.000 | 100.0% | +0.246 |
| 8 | 43.0–56.2 | 35,011 | 0.706 | 90.9% | 0.975 | 97.5% | +0.269 |
| 10 | 56.5–57.8 | 33,902 | 0.810 | 100.0% | 1.000 | 100.0% | +0.190 |
| 13 | 58.3–58.6 | 28,602 | 0.574 | 100.0% | 1.000 | 100.0% | +0.426 |
| 14 | 58.9–63.8 | 71,720 | 0.383 | 5.4% | 0.831 | 86.5% | **+0.448** |
| 16 | 64.4–69.6 | 66,679 | 0.382 | 1.3% | 0.907 | 94.9% | **+0.525** |
| 17 | 69.7–88.5 | 35,443 | 0.538 | 69.8% | 0.995 | 99.5% | +0.457 |
| 18 | 91.2–110.1 | 30,402 | 0.687 | 95.6% | 0.994 | 99.8% | +0.307 |

**Every one of the 21 tracks across both videos shows a positive delta** (new model reads more fake than
production, no exceptions). The shift is not confined to one person or one video. Two things are true at once:

1. **It's not exclusively a large-face effect.** Even the already-fake-biased small/medium tracks (23K–35K
   area) get pushed further toward fake (deltas +0.15 to +0.46).
2. **It only *flips the verdict* for the large-face tracks**, because those were the only ones sitting close to
   or below the 0.5 threshold under production (area ≥47K: deltas of +0.24 to +0.78, taking mean P(fake) from
   the 0.20–0.52 REAL-leaning range up into the 0.69–0.98 FAKE-classified range). The small-face tracks were
   already above threshold under production and simply became more extremely so.

Combined with §2.2 (the new model's real-sample P(fake) distribution is far more polarized even on the sealed
test) and §3.3 (these are the higher-native-resolution, near-1:1 tracks — the regime the training augmentation
did not specifically validate), the concrete, measured picture is: **on this specific out-of-FF++-domain
footage, the new model applies a systematic upward P(fake) shift, and that shift is large enough to flip the
classification specifically for the faces that were previously closest to the decision boundary.** Whether this
is caused by the screen-recording transcoding chain (§3.2) or is a genuine property of the retrained decision
boundary independent of that chain cannot be fully separated with the data gathered so far — see §5.

### 3.5 Hypothesis verdicts

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| A | Screen-recording/compression domain shift | **Plausible, not confirmed** | Confirmed the screen-recording test methodology adds 3 extra transcoding generations that live production inference never goes through (§3.2, code citation `web_app.py:544` vs `584`). Bitrate is not obviously worse than FF++ (§3.1); blur (Laplacian variance) of the actual crops is **not** worse — screen-recording crops measured *sharper* (mean 49.5–49.8) than FF++ real crops (mean 41.8) and FF++ fake crops (mean 30.0). So "the screen recording is simply lower quality" is not supported, but the *specific* transcoding artifacts (JPEG blockiness, browser color/gamma handling, H.264 GOP structure) were not otherwise ruled out, and the pipeline gap itself is real and unresolved. |
| B | Crop-size/context differences | **Rejected as the primary driver** | Screen-recording faces occupy a *smaller* fraction of their frame (context) and have *higher*, not lower, native resolution (163–211px median vs FF++'s 131px median) — the opposite of what would predict more upsampling-driven distortion (§3.3). |
| C | Pose/alignment effects | **Rejected** | Roll angle from YuNet's own landmarks is statistically indistinguishable between FF++ (median 2.86°) and both screen recordings (median 2.89°, 3.47°) (§3.3). |
| D | FF++-artifact overfitting / non-generalizing decision boundary | **Best-supported hypothesis** | The identity-level churn (§2.4: 17 fixed / 17 newly broken, not a strict superset) and the polarized P(fake) distribution even on the sealed test (§2.2) both show the new model learned a *different*, sharper decision function rather than a strictly more general one. The universal, non-uniform (larger for high-native-resolution tracks) upward P(fake) shift in §3.4 is consistent with a boundary that transfers unevenly outside its training distribution. |
| E | Dataset incompleteness (the 289/4,900 issue) | **Rejected** | §1: dataset is 97.5% complete with a verified-genuine 2.5% zero-detection gap; re-training on the missing 122 videos (which have zero detected faces at any sampled position) cannot add real training signal. |
| F | Another concrete implementation issue | **Found and fixed (in evaluation methodology, not production code)** | `evaluate_visual_yunet_clean.py`'s original baseline column was a hardcoded prior value, not a same-run comparison — corrected in §2 by re-running both checkpoints together. This did not change the sealed-test conclusion but was a real gap, now closed. |

---

## 4. What was and wasn't touched

Confirmed via `git status` and direct inspection during this audit: `phase4_fusion_alerting/web_app.py` — no
diff from before this task started. `phase2_visual/visual_model.pth` — mtime and `git status` both confirm
untouched. No thresholds, hysteresis constants, or fusion weights were read or written by any script in this
audit. No training was run. The new checkpoint file was only read (for evaluation), never modified.

---

## 5. Final answers

### 5.1 What we now know for certain (measured, not inferred)

- The YuNet training dataset is **complete**: 56,443/expected train crops, with the only gap being 122
  genuinely-zero-face-detection source videos (independently re-verified), not an extraction bug.
- On the sealed FF++ test set, evaluated identically for both checkpoints in one run, the new model is better on
  **every** standard metric (§2.1), and the improvement is not a "predict REAL more" shortcut (both fake and
  real recall rise together) nor an artifact of a mismatched baseline (§2, "F" resolved).
- The new model's real-identity false-positive **rate** did not actually improve at the identity level (14.2%
  both) — it **reshuffled** which 24/169 identities are wrong (7 in common, 17 newly fixed, 17 newly broken)
  (§2.4). This is a directly measured fact, not a guess.
- On the two screen-recording videos, re-measured for both checkpoints in this session on identical frames, the
  new model shows a **universal upward P(fake) shift across all 21 independently-tracked face segments**, large
  enough to flip several previously-correct large-face/near-camera tracks from REAL to FAKE (§3.4).
- That upward shift is **not** explained by pose/alignment (ruled out, §3.3), is **not** explained by lower crop
  resolution or tighter cropping (ruled out — screen-recording crops are higher native resolution and less
  zoomed-in, §3.3), and is **not** explained by simple video quality/blur (ruled out — screen-recording crops
  measured sharper, §3.5-A).
- The screen-recording sanity-test methodology itself feeds the model input that **live production inference
  never sees** — production runs inference on the raw camera frame before any JPEG/browser/screen-capture
  transcoding (`web_app.py:544` vs `:584`, §3.2). This is a genuine, previously under-examined gap in how this
  and the prior session's live tests were constructed.

### 5.2 What remains uncertain

- ~~Whether the regression measured on screen-recording footage would also appear on genuine **raw live-camera**
  input...~~ **RESOLVED, see Addendum below — it does reproduce on raw camera input.**
- The specific causal mechanism behind the universal P(fake) upward shift — whether it's a byproduct of the
  `simulate_native_resolution_loss`/JPEG-recompression augmentation interacting poorly with high-native-resolution
  input, a more general effect of the sharper/more-polarized decision function, or something else — is
  plausible but not proven; no direct ablation was run.
- Whether the identity-reshuffling pattern (§2.4) would, on a much larger population of never-before-seen real
  identities, net out positive, negative, or neutral. Two out-of-sample people (the screen-recording subjects)
  both landed unfavorably, but n=2 cannot establish a population-level rate.

### 5.3 Single highest-value next experiment

**A genuine raw live-camera A/B test**, not a screen recording: a short, standalone, read-only script (like the
sanity scripts used in this audit, but reading `cv2.VideoCapture(0)` directly instead of a video file) that runs
several volunteers' live camera frames through both checkpoints side by side, logging P(fake) for each,
**without displaying anything to end users or touching `web_app.py`**. This directly removes the JPEG/browser/
screen-capture confound identified in §3.2 and is the only way to determine whether the regression in §3.4 is
real for the domain that actually matters (live production) or an artifact of the screen-recording test
methodology itself. This is a strictly cheaper, faster, and more decisive experiment than retraining again.

### 5.4 Should the new checkpoint remain isolated, or can it be tested live temporarily?

**It can be tested live temporarily, but only in a diagnostic, non-authoritative capacity, and it should not
drive any user-visible decision while doing so.** Concretely: a separate, standalone script (as in §5.3) reading
directly from the camera and logging both checkpoints' outputs side by side is low-risk — it does not touch
`web_app.py`, does not change what any user sees, and production remains the sole driver of the actual Security
Decision and dashboard throughout. That is different from, and much lower-risk than, wiring the new checkpoint
into `web_app.py`'s live inference path, which should **not** happen until §5.3's experiment is run and reviewed
— the sealed-test improvement is real, but §2.4 and §3.4 together show it has not been shown to be a strict
improvement for people outside the sealed test, and the one live-domain measurement available so far (§3.4)
points the other way strongly enough that it should not be dismissed on sealed-test numbers alone.

**Production remains: `phase2_visual/visual_model.pth`, unchanged, as fallback and current authority.**

---

## Addendum (2026-09-23, later same day): raw live-camera A/B test result

The §5.3 experiment was run: `phase2_visual/experiments/live_camera_ab_test.py`, reading `cv2.VideoCapture(0)`
directly (no screen recording, no JPEG/browser/H.264 transcoding chain — see §3.2), both checkpoints run on the
identical crop tensor per detected face. Full session: 313.3s, 2,228 frames, 2,316 face observations. Full
results: `phase2_visual/experiments/live_camera_ab_test_report.md` and `.csv`.

**Result: production mean P(fake)=0.388 (71.9% classified REAL); new model mean P(fake)=0.840 (13.6% classified
REAL); mean delta (new − production) = +0.452; label disagreement on 63.0% of observations.** The session
included a ~29s window (141 frames, CSV-verified as simultaneously co-occurring with the primary subject's track
in the same frames — a genuine two-person segment) where a second real person was read by production as 78.0%
REAL (mean P(fake)=0.405) and by the new model as 0.0% REAL (mean P(fake)=0.999).

**This resolves the open question in §5.2.** The same large, systematic upward P(fake) shift found on
screen-recording footage reproduces on raw, uncompressed, direct camera frames — on a person who does not
appear in the FF++ dataset or either screen recording, sustained across a neutral-pose segment, a pose-variation
segment, and a two-person segment. Since this input never passes through the JPEG/browser-render/screen-capture/
H.264 chain identified in §3.2, that transcoding chain is **not a necessary cause** of the regression. This
strengthens hypothesis D (§3.5) as the primary explanation and weakens hypothesis A to, at most, a possible
secondary contributor on the earlier screen-recording measurements specifically — it is not what is driving the
effect on genuine live input.

**Verdict unchanged, now on stronger evidence: the new checkpoint should not be integrated into production.**
This was previously withheld on sealed-test-vs-live-domain uncertainty with an unresolved methodology gap; that
gap is now closed and the live-domain regression is confirmed directly, not just suspected.

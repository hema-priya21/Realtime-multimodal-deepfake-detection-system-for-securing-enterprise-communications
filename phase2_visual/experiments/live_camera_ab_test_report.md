# Live camera A/B test report: production vs. yunet_clean checkpoint

Generated: 2026-09-23T22:52:01

**This is a diagnostic experiment only.** It did not modify `phase4_fusion_alerting/web_app.py`, did not modify or replace `phase2_visual/visual_model.pth`, and did not change any threshold, hysteresis, or fusion weight. Frames were read directly from `cv2.VideoCapture(0)` -- no screen recording was used.

> **NOT a ground-truth accuracy measurement.** All subjects in this session are known-real people; there are no labeled fake live samples. This report measures real-person false-positive behavior and model-to-model agreement/disagreement only -- it cannot and does not measure overall accuracy, precision, or recall for either checkpoint.

## Session

- Duration: 313.3s
- Frames processed: 2228
- Camera index: 0
- On-screen-stats reset events (R key) at: 57.7s, 58.6s, 58.9s, 60.6s, 300.4s, 303.6s, 304.1s, 304.9s, 305.0s, 305.2s, 305.5s, 305.7s, 306.0s, 306.2s, 306.3s, 306.6s, 307.2s, 307.3s, 307.4s, 310.7s, 311.0s, 311.2s, 311.4s, 311.7s -- CSV and this report cover the full session regardless.

## Face observations

- Total face observations (one row per detected face per frame): 2316

## Production (`visual_model.pth`) statistics

- Mean P(fake): 0.3879
- Median P(fake): 0.3699
- % observations classified REAL: 71.9%
- % observations classified FAKE: 28.1%

## New model (`visual_model_yunet_clean_best.pth`) statistics

- Mean P(fake): 0.8398
- Median P(fake): 0.9781
- % observations classified REAL: 13.6%
- % observations classified FAKE: 86.4%

## P(fake) delta (new - production)

- Mean delta: +0.4519
- Median delta: +0.5091
- Observations where new model is MORE fake than production: 2129 (91.9%)
- Observations where new model is MORE real than production: 187 (8.1%)
- Observations with no difference: 0 (0.0%)

## Label disagreements

- Total disagreements (production label != new model label): 1460 (63.0%)

## Label distributions

- Production: REAL=1666 (71.9%), FAKE=650 (28.1%)
- New model:  REAL=314 (13.6%), FAKE=2002 (86.4%)

## Per-face-track summary

Tracks are assigned by a lightweight greedy bounding-box IoU tracker (not face recognition) -- a track_id generally corresponds to one physically continuous face appearance, and a person who leaves and re-enters frame may receive a new track_id.

| track_id | n obs | duration (s) | prod mean P(fake) | prod %REAL | new mean P(fake) | new %REAL | mean delta |
|---|---|---|---|---|---|---|---|
| 0 | 2163 | 309.6 | 0.387 | 71.6% | 0.829 | 14.5% | +0.442 |
| 1 | 2 | 0.2 | 0.552 | 0.0% | 0.953 | 0.0% | +0.401 |
| 2 | 6 | 0.7 | 0.388 | 83.3% | 0.971 | 0.0% | +0.583 |
| 3 | 141 | 29.1 | 0.405 | 78.0% | 0.999 | 0.0% | +0.594 |
| 4 | 4 | 0.7 | 0.454 | 50.0% | 0.999 | 0.0% | +0.545 |

## Interpretation notes

- All subjects are known-real people; no ground-truth accuracy, precision, or recall can be computed from this session.
- A high production or new-model FAKE percentage here is real-person false-positive behavior, not a measurement of forgery-detection accuracy.
- Compare this report's per-model and delta statistics against the sealed FF++ test-set and screen-recording findings in `phase2_visual/experiments/yunet_clean_root_cause_audit.md` before drawing any conclusion about production readiness.

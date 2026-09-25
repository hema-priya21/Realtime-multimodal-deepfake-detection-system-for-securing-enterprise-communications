# Model calibration / decision-boundary diagnostic: production vs. yunet_clean

**Read-only diagnostic.** Did not modify `phase4_fusion_alerting/web_app.py`, did not modify or overwrite `phase2_visual/visual_model.pth`, did not integrate the new checkpoint, changed no threshold/fusion weight/hysteresis, and trained nothing. Both checkpoints were only read; this script only writes its own report and CSV.

Test set: `C:\Users\kslok\OneDrive\Desktop\Realtime-multimodal-deepfake-detection-system-for-securing-enterprise-communications-main\data\visual_processed_yunet_full\test` -- 12,009 crops (10,007 fake / 2,002 real), same sealed set used in the earlier same-test-set comparison and root-cause audit. Crops matched back to a CSV video row: 12009/12009.

## 1. Reliability / calibration analysis

**Brier score** (mean squared error between P(fake) and the true fake/real indicator, lower is better-calibrated *and* better-separated -- it conflates calibration and discrimination, which is exactly why it's paired with ECE and ROC-AUC below):
- Production: 0.1299
- New (yunet_clean): 0.0718

**Expected Calibration Error (ECE)** -- 10 equal-width bins of width 0.1 over the winning-class confidence `max(P(fake), 1-P(fake))`; each bin's contribution is `(bin_count/N) * |bin_accuracy - bin_mean_confidence|`, summed over all non-empty bins (standard definition, Guo et al. 2017):
- Production: 0.0275
- New (yunet_clean): 0.0544

<details><summary>ECE bin detail (production)</summary>

| bin range | n | accuracy | mean confidence |
|---|---|---|---|
| [0.0, 0.1] | 0 | - | - |
| [0.1, 0.2] | 0 | - | - |
| [0.2, 0.3] | 0 | - | - |
| [0.3, 0.4] | 0 | - | - |
| [0.4, 0.5] | 0 | - | - |
| [0.5, 0.6] | 1366 | 0.553 | 0.551 |
| [0.6, 0.7] | 1417 | 0.687 | 0.649 |
| [0.7, 0.8] | 1503 | 0.760 | 0.752 |
| [0.8, 0.9] | 2042 | 0.814 | 0.854 |
| [0.9, 1.0] | 5681 | 0.936 | 0.968 |

</details>

<details><summary>ECE bin detail (new yunet_clean)</summary>

| bin range | n | accuracy | mean confidence |
|---|---|---|---|
| [0.0, 0.1] | 0 | - | - |
| [0.1, 0.2] | 0 | - | - |
| [0.2, 0.3] | 0 | - | - |
| [0.3, 0.4] | 0 | - | - |
| [0.4, 0.5] | 0 | - | - |
| [0.5, 0.6] | 233 | 0.558 | 0.551 |
| [0.6, 0.7] | 240 | 0.588 | 0.652 |
| [0.7, 0.8] | 346 | 0.610 | 0.752 |
| [0.8, 0.9] | 451 | 0.630 | 0.857 |
| [0.9, 1.0] | 10739 | 0.949 | 0.994 |

</details>

**P(fake) on true REAL samples** (lower is better -- this is the real-false-positive-relevant view):

| | production | new |
|---|---|---|
| mean | 0.2670 | 0.1999 |
| median | 0.1655 | 0.0089 |
| p90 | 0.6605 | 0.8903 |
| p95 | 0.7931 | 0.9819 |

**P(fake) on true FAKE samples** (higher is better):

| | production | new |
|---|---|---|
| mean | 0.7614 | 0.9244 |
| median | 0.8702 | 1.0000 |
| p10 | 0.3201 | 0.7960 |
| p05 | 0.1544 | 0.2439 |

## 2. Threshold-independent comparison (no winner selected here by design)

| metric | production | new |
|---|---|---|
| ROC-AUC (fake vs. real ranking) | 0.8927 | 0.9567 |
| PR-AUC, FAKE as positive class | 0.9755 | 0.9910 |
| PR-AUC, REAL as positive class (minority class, 2002/12009 = 16.7% base rate) | 0.6429 | 0.8212 |

## 3. Decision-boundary comparison @ threshold 0.50

Confusion matrices [rows=actual, cols=predicted, order=(fake, real)]:

- Production: `[[8299, 1708], [445, 1557]]`
- New: `[[9324, 683], [373, 1629]]`

| metric | production | new |
|---|---|---|
| Real false-positive rate | 22.23% (445/2002) | 18.63% (373/2002) |
| Fake recall | 82.93% | 93.17% |
| Macro F1 | 73.82% | 85.08% |

**Prediction disagreement at threshold 0.50: 2,231 / 12,009 (18.58%)**

## 4. Probability-shift analysis

Quantiles of P(fake), true REAL samples only:

| quantile | production | new |
|---|---|---|
| q5 | 0.0071 | 0.0000 |
| q10 | 0.0126 | 0.0000 |
| q25 | 0.0400 | 0.0003 |
| q50 | 0.1655 | 0.0089 |
| q75 | 0.4596 | 0.2414 |
| q90 | 0.6605 | 0.8903 |
| q95 | 0.7931 | 0.9819 |

Quantiles of P(fake), true FAKE samples only:

| quantile | production | new |
|---|---|---|
| q5 | 0.1544 | 0.2439 |
| q10 | 0.3201 | 0.7960 |
| q25 | 0.6229 | 0.9964 |
| q50 | 0.8702 | 1.0000 |
| q75 | 0.9754 | 1.0000 |
| q90 | 0.9962 | 1.0000 |
| q95 | 0.9989 | 1.0000 |

Paired delta (new − production), computed per sample on the identical crop:

- REAL samples: mean delta = -0.0671, median delta = -0.0529, stdev = 0.3641
- FAKE samples: mean delta = +0.1629, median delta = +0.0848, stdev = 0.2899
- % of REAL samples where new P(fake) > production P(fake): 25.4%
- % of FAKE samples where new P(fake) > production P(fake): 90.4%

## 5. Identity-level analysis (real identities only, subject-disjoint sealed test)

- Number of real identities: 169
- % of real identities with mean P(fake) >= 0.5 -- production: 14.2% (24/169)
- % of real identities with mean P(fake) >= 0.5 -- new: 14.2% (24/169)

**Identities that flipped mostly-REAL (production) -> mostly-FAKE (new): 17**

| stem | subject_id | prod mean | prod median | new mean | new median |
|---|---|---|---|---|---|
| ffpp_original_835 | ffpp_yt_835 | 0.029 | 0.013 | 0.968 | 0.992 |
| ffpp_original_556 | ffpp_yt_556 | 0.087 | 0.027 | 0.866 | 0.986 |
| ffpp_original_160 | ffpp_yt_160 | 0.022 | 0.017 | 0.750 | 0.890 |
| ffpp_original_887 | ffpp_yt_887 | 0.180 | 0.089 | 0.812 | 0.921 |
| ffpp_original_752 | ffpp_yt_752 | 0.149 | 0.077 | 0.774 | 0.947 |
| ffpp_original_807 | ffpp_yt_807 | 0.309 | 0.181 | 0.908 | 0.999 |
| ffpp_original_772 | ffpp_yt_772 | 0.171 | 0.119 | 0.745 | 0.784 |
| ffpp_original_056 | ffpp_yt_056 | 0.222 | 0.213 | 0.796 | 0.970 |
| ffpp_original_863 | ffpp_yt_863 | 0.195 | 0.048 | 0.702 | 0.778 |
| ffpp_original_957 | ffpp_yt_957 | 0.464 | 0.535 | 0.964 | 0.996 |
| ffpp_original_044 | ffpp_yt_044 | 0.036 | 0.027 | 0.508 | 0.378 |
| ffpp_original_578 | ffpp_yt_578 | 0.143 | 0.122 | 0.596 | 0.650 |
| ffpp_original_662 | ffpp_yt_662 | 0.240 | 0.230 | 0.621 | 0.723 |
| ffpp_original_635 | ffpp_yt_635 | 0.432 | 0.390 | 0.637 | 0.845 |
| ffpp_original_015 | ffpp_yt_015 | 0.417 | 0.432 | 0.621 | 0.769 |
| ffpp_original_070 | ffpp_yt_070 | 0.478 | 0.448 | 0.628 | 0.748 |
| ffpp_original_472 | ffpp_yt_472 | 0.451 | 0.421 | 0.543 | 0.777 |

**Identities that flipped mostly-FAKE (production) -> mostly-REAL (new): 17**

| stem | subject_id | prod mean | prod median | new mean | new median |
|---|---|---|---|---|---|
| ffpp_original_058 | ffpp_yt_058 | 0.837 | 0.885 | 0.028 | 0.003 |
| ffpp_original_681 | ffpp_yt_681 | 0.890 | 0.903 | 0.095 | 0.016 |
| ffpp_original_711 | ffpp_yt_711 | 0.652 | 0.632 | 0.002 | 0.001 |
| ffpp_original_983 | ffpp_yt_983 | 0.783 | 0.783 | 0.201 | 0.116 |
| ffpp_original_663 | ffpp_yt_663 | 0.514 | 0.450 | 0.000 | 0.000 |
| ffpp_original_588 | ffpp_yt_588 | 0.515 | 0.580 | 0.002 | 0.000 |
| ffpp_original_595 | ffpp_yt_595 | 0.691 | 0.679 | 0.193 | 0.083 |
| ffpp_original_297 | ffpp_yt_297 | 0.541 | 0.537 | 0.044 | 0.004 |
| ffpp_original_940 | ffpp_yt_940 | 0.536 | 0.514 | 0.058 | 0.018 |
| ffpp_original_881 | ffpp_yt_881 | 0.656 | 0.673 | 0.208 | 0.083 |
| ffpp_original_723 | ffpp_yt_723 | 0.908 | 0.906 | 0.477 | 0.515 |
| ffpp_original_545 | ffpp_yt_545 | 0.606 | 0.616 | 0.350 | 0.298 |
| ffpp_original_049 | ffpp_yt_049 | 0.611 | 0.568 | 0.371 | 0.385 |
| ffpp_original_674 | ffpp_yt_674 | 0.632 | 0.694 | 0.450 | 0.305 |
| ffpp_original_597 | ffpp_yt_597 | 0.590 | 0.647 | 0.438 | 0.313 |
| ffpp_original_210 | ffpp_yt_210 | 0.547 | 0.546 | 0.471 | 0.418 |
| ffpp_original_314 | ffpp_yt_314 | 0.538 | 0.513 | 0.483 | 0.354 |

## 6. Full identity table (all real identities, sorted by delta)

<details><summary>expand full table</summary>

| stem | subject_id | prod mean | new mean | delta |
|---|---|---|---|---|
| ffpp_original_835 | ffpp_yt_835 | 0.029 | 0.968 | +0.939 |
| ffpp_original_556 | ffpp_yt_556 | 0.087 | 0.866 | +0.779 |
| ffpp_original_160 | ffpp_yt_160 | 0.022 | 0.750 | +0.728 |
| ffpp_original_887 | ffpp_yt_887 | 0.180 | 0.812 | +0.632 |
| ffpp_original_752 | ffpp_yt_752 | 0.149 | 0.774 | +0.625 |
| ffpp_original_807 | ffpp_yt_807 | 0.309 | 0.908 | +0.599 |
| ffpp_original_772 | ffpp_yt_772 | 0.171 | 0.745 | +0.574 |
| ffpp_original_056 | ffpp_yt_056 | 0.222 | 0.796 | +0.574 |
| ffpp_original_863 | ffpp_yt_863 | 0.195 | 0.702 | +0.507 |
| ffpp_original_957 | ffpp_yt_957 | 0.464 | 0.964 | +0.501 |
| ffpp_original_044 | ffpp_yt_044 | 0.036 | 0.508 | +0.472 |
| ffpp_original_578 | ffpp_yt_578 | 0.143 | 0.596 | +0.453 |
| ffpp_original_662 | ffpp_yt_662 | 0.240 | 0.621 | +0.381 |
| ffpp_original_093 | ffpp_yt_093 | 0.050 | 0.421 | +0.371 |
| ffpp_original_024 | ffpp_yt_024 | 0.092 | 0.419 | +0.327 |
| ffpp_original_702 | ffpp_yt_702 | 0.165 | 0.472 | +0.307 |
| ffpp_original_996 | ffpp_yt_996 | 0.209 | 0.494 | +0.285 |
| ffpp_original_908 | ffpp_yt_908 | 0.030 | 0.299 | +0.269 |
| ffpp_original_529 | ffpp_yt_529 | 0.091 | 0.347 | +0.256 |
| ffpp_original_107 | ffpp_yt_107 | 0.603 | 0.823 | +0.220 |
| ffpp_original_955 | ffpp_yt_955 | 0.217 | 0.435 | +0.218 |
| ffpp_original_635 | ffpp_yt_635 | 0.432 | 0.637 | +0.205 |
| ffpp_original_015 | ffpp_yt_015 | 0.417 | 0.621 | +0.204 |
| ffpp_original_639 | ffpp_yt_639 | 0.154 | 0.323 | +0.170 |
| ffpp_original_070 | ffpp_yt_070 | 0.478 | 0.628 | +0.150 |
| ffpp_original_879 | ffpp_yt_879 | 0.105 | 0.234 | +0.130 |
| ffpp_original_292 | ffpp_yt_292 | 0.120 | 0.232 | +0.112 |
| ffpp_original_472 | ffpp_yt_472 | 0.451 | 0.543 | +0.092 |
| ffpp_original_275 | ffpp_yt_275 | 0.115 | 0.200 | +0.085 |
| ffpp_original_012 | ffpp_yt_012 | 0.621 | 0.695 | +0.074 |
| ffpp_original_612 | ffpp_yt_612 | 0.134 | 0.200 | +0.066 |
| ffpp_original_651 | ffpp_yt_651 | 0.023 | 0.086 | +0.063 |
| ffpp_original_570 | ffpp_yt_570 | 0.672 | 0.734 | +0.061 |
| ffpp_original_270 | ffpp_yt_270 | 0.060 | 0.119 | +0.059 |
| ffpp_original_946 | ffpp_yt_946 | 0.054 | 0.110 | +0.056 |
| ffpp_original_945 | ffpp_yt_945 | 0.559 | 0.605 | +0.047 |
| ffpp_original_499 | ffpp_yt_499 | 0.065 | 0.110 | +0.045 |
| ffpp_original_186 | ffpp_yt_186 | 0.003 | 0.045 | +0.042 |
| ffpp_original_162 | ffpp_yt_162 | 0.079 | 0.118 | +0.038 |
| ffpp_original_594 | ffpp_yt_594 | 0.061 | 0.094 | +0.032 |
| ffpp_original_228 | ffpp_yt_228 | 0.025 | 0.054 | +0.030 |
| ffpp_original_207 | ffpp_yt_207 | 0.084 | 0.113 | +0.029 |
| ffpp_original_704 | ffpp_yt_704 | 0.586 | 0.614 | +0.028 |
| ffpp_original_182 | ffpp_yt_182 | 0.114 | 0.141 | +0.027 |
| ffpp_original_708 | ffpp_yt_708 | 0.033 | 0.059 | +0.025 |
| ffpp_original_353 | ffpp_yt_353 | 0.032 | 0.044 | +0.012 |
| ffpp_original_754 | ffpp_yt_754 | 0.022 | 0.029 | +0.007 |
| ffpp_original_311 | ffpp_yt_311 | 0.117 | 0.122 | +0.006 |
| ffpp_original_078 | ffpp_yt_078 | 0.008 | 0.013 | +0.006 |
| ffpp_original_841 | ffpp_yt_841 | 0.767 | 0.772 | +0.004 |
| ffpp_original_624 | ffpp_yt_624 | 0.006 | 0.006 | +0.000 |
| ffpp_original_641 | ffpp_yt_641 | 0.019 | 0.016 | -0.003 |
| ffpp_original_971 | ffpp_yt_971 | 0.173 | 0.170 | -0.004 |
| ffpp_original_837 | ffpp_yt_837 | 0.017 | 0.009 | -0.008 |
| ffpp_original_242 | ffpp_yt_242 | 0.011 | 0.002 | -0.009 |
| ffpp_original_108 | ffpp_yt_108 | 0.023 | 0.013 | -0.010 |
| ffpp_original_073 | ffpp_yt_073 | 0.011 | 0.000 | -0.011 |
| ffpp_original_539 | ffpp_yt_539 | 0.074 | 0.060 | -0.014 |
| ffpp_original_964 | ffpp_yt_964 | 0.105 | 0.090 | -0.014 |
| ffpp_original_381 | ffpp_yt_381 | 0.017 | 0.002 | -0.015 |
| ffpp_original_164 | ffpp_yt_164 | 0.022 | 0.004 | -0.018 |
| ffpp_original_519 | ffpp_yt_519 | 0.106 | 0.082 | -0.024 |
| ffpp_original_065 | ffpp_yt_065 | 0.131 | 0.107 | -0.024 |
| ffpp_original_530 | ffpp_yt_530 | 0.028 | 0.000 | -0.028 |
| ffpp_original_330 | ffpp_yt_330 | 0.058 | 0.028 | -0.030 |
| ffpp_original_800 | ffpp_yt_800 | 0.030 | 0.000 | -0.030 |
| ffpp_original_294 | ffpp_yt_294 | 0.032 | 0.000 | -0.032 |
| ffpp_original_216 | ffpp_yt_216 | 0.103 | 0.070 | -0.033 |
| ffpp_original_113 | ffpp_yt_113 | 0.067 | 0.034 | -0.033 |
| ffpp_original_751 | ffpp_yt_751 | 0.034 | 0.001 | -0.033 |
| ffpp_original_152 | ffpp_yt_152 | 0.230 | 0.195 | -0.036 |
| ffpp_original_758 | ffpp_yt_758 | 0.110 | 0.073 | -0.038 |
| ffpp_original_849 | ffpp_yt_849 | 0.042 | 0.000 | -0.042 |
| ffpp_original_057 | ffpp_yt_057 | 0.052 | 0.007 | -0.044 |
| ffpp_original_853 | ffpp_yt_853 | 0.379 | 0.334 | -0.046 |
| ffpp_original_121 | ffpp_yt_121 | 0.048 | 0.002 | -0.047 |
| ffpp_original_314 | ffpp_yt_314 | 0.538 | 0.483 | -0.055 |
| ffpp_original_840 | ffpp_yt_840 | 0.203 | 0.148 | -0.055 |
| ffpp_original_170 | ffpp_yt_170 | 0.055 | 0.000 | -0.055 |
| ffpp_original_383 | ffpp_yt_383 | 0.411 | 0.353 | -0.058 |
| ffpp_original_789 | ffpp_yt_789 | 0.107 | 0.049 | -0.058 |
| ffpp_original_289 | ffpp_yt_289 | 0.180 | 0.119 | -0.061 |
| ffpp_original_856 | ffpp_yt_856 | 0.062 | 0.000 | -0.062 |
| ffpp_original_959 | ffpp_yt_959 | 0.100 | 0.036 | -0.064 |
| ffpp_original_511 | ffpp_yt_511 | 0.075 | 0.009 | -0.066 |
| ffpp_original_928 | ffpp_yt_928 | 0.074 | 0.000 | -0.073 |
| ffpp_original_016 | ffpp_yt_016 | 0.075 | 0.000 | -0.075 |
| ffpp_original_795 | ffpp_yt_795 | 0.370 | 0.294 | -0.075 |
| ffpp_original_210 | ffpp_yt_210 | 0.547 | 0.471 | -0.076 |
| ffpp_original_818 | ffpp_yt_818 | 0.110 | 0.031 | -0.079 |
| ffpp_original_376 | ffpp_yt_376 | 0.091 | 0.011 | -0.080 |
| ffpp_original_753 | ffpp_yt_753 | 0.096 | 0.009 | -0.087 |
| ffpp_original_636 | ffpp_yt_636 | 0.090 | 0.000 | -0.089 |
| ffpp_original_642 | ffpp_yt_642 | 0.095 | 0.004 | -0.091 |
| ffpp_original_000 | ffpp_yt_000 | 0.111 | 0.003 | -0.108 |
| ffpp_original_963 | ffpp_yt_963 | 0.128 | 0.015 | -0.114 |
| ffpp_original_387 | ffpp_yt_387 | 0.377 | 0.257 | -0.120 |
| ffpp_original_553 | ffpp_yt_553 | 0.128 | 0.006 | -0.122 |
| ffpp_original_890 | ffpp_yt_890 | 0.162 | 0.040 | -0.122 |
| ffpp_original_941 | ffpp_yt_941 | 0.487 | 0.365 | -0.122 |
| ffpp_original_498 | ffpp_yt_498 | 0.173 | 0.049 | -0.123 |
| ffpp_original_919 | ffpp_yt_919 | 0.646 | 0.520 | -0.127 |
| ffpp_original_241 | ffpp_yt_241 | 0.143 | 0.015 | -0.128 |
| ffpp_original_273 | ffpp_yt_273 | 0.392 | 0.261 | -0.131 |
| ffpp_original_934 | ffpp_yt_934 | 0.132 | 0.002 | -0.131 |
| ffpp_original_071 | ffpp_yt_071 | 0.498 | 0.358 | -0.141 |
| ffpp_original_918 | ffpp_yt_918 | 0.476 | 0.335 | -0.141 |
| ffpp_original_633 | ffpp_yt_633 | 0.405 | 0.261 | -0.144 |
| ffpp_original_003 | ffpp_yt_003 | 0.144 | 0.000 | -0.144 |
| ffpp_original_682 | ffpp_yt_682 | 0.470 | 0.322 | -0.148 |
| ffpp_original_141 | ffpp_yt_141 | 0.352 | 0.203 | -0.149 |
| ffpp_original_209 | ffpp_yt_209 | 0.494 | 0.344 | -0.149 |
| ffpp_original_962 | ffpp_yt_962 | 0.184 | 0.034 | -0.150 |
| ffpp_original_597 | ffpp_yt_597 | 0.590 | 0.438 | -0.152 |
| ffpp_original_744 | ffpp_yt_744 | 0.160 | 0.002 | -0.158 |
| ffpp_original_039 | ffpp_yt_039 | 0.242 | 0.083 | -0.159 |
| ffpp_original_197 | ffpp_yt_197 | 0.166 | 0.002 | -0.165 |
| ffpp_original_174 | ffpp_yt_174 | 0.189 | 0.018 | -0.170 |
| ffpp_original_706 | ffpp_yt_706 | 0.179 | 0.006 | -0.173 |
| ffpp_original_674 | ffpp_yt_674 | 0.632 | 0.450 | -0.182 |
| ffpp_original_391 | ffpp_yt_391 | 0.352 | 0.168 | -0.185 |
| ffpp_original_013 | ffpp_yt_013 | 0.188 | 0.002 | -0.186 |
| ffpp_original_026 | ffpp_yt_026 | 0.201 | 0.008 | -0.193 |
| ffpp_original_559 | ffpp_yt_559 | 0.200 | 0.000 | -0.200 |
| ffpp_original_721 | ffpp_yt_721 | 0.206 | 0.001 | -0.205 |
| ffpp_original_023 | ffpp_yt_023 | 0.348 | 0.133 | -0.214 |
| ffpp_original_161 | ffpp_yt_161 | 0.240 | 0.025 | -0.215 |
| ffpp_original_820 | ffpp_yt_820 | 0.227 | 0.001 | -0.226 |
| ffpp_original_406 | ffpp_yt_406 | 0.233 | 0.002 | -0.231 |
| ffpp_original_948 | ffpp_yt_948 | 0.329 | 0.094 | -0.234 |
| ffpp_original_089 | ffpp_yt_089 | 0.457 | 0.221 | -0.235 |
| ffpp_original_049 | ffpp_yt_049 | 0.611 | 0.371 | -0.239 |
| ffpp_original_347 | ffpp_yt_347 | 0.310 | 0.069 | -0.241 |
| ffpp_original_771 | ffpp_yt_771 | 0.385 | 0.140 | -0.245 |
| ffpp_original_545 | ffpp_yt_545 | 0.606 | 0.350 | -0.255 |
| ffpp_original_992 | ffpp_yt_992 | 0.322 | 0.063 | -0.259 |
| ffpp_original_462 | ffpp_yt_462 | 0.371 | 0.112 | -0.259 |
| ffpp_original_564 | ffpp_yt_564 | 0.303 | 0.027 | -0.276 |
| ffpp_original_543 | ffpp_yt_543 | 0.308 | 0.023 | -0.286 |
| ffpp_original_321 | ffpp_yt_321 | 0.343 | 0.056 | -0.287 |
| ffpp_original_222 | ffpp_yt_222 | 0.290 | 0.003 | -0.287 |
| ffpp_original_923 | ffpp_yt_923 | 0.296 | 0.003 | -0.293 |
| ffpp_original_168 | ffpp_yt_168 | 0.430 | 0.129 | -0.301 |
| ffpp_original_883 | ffpp_yt_883 | 0.451 | 0.114 | -0.337 |
| ffpp_original_467 | ffpp_yt_467 | 0.436 | 0.095 | -0.341 |
| ffpp_original_149 | ffpp_yt_149 | 0.461 | 0.106 | -0.355 |
| ffpp_original_054 | ffpp_yt_054 | 0.361 | 0.000 | -0.361 |
| ffpp_original_980 | ffpp_yt_980 | 0.411 | 0.045 | -0.366 |
| ffpp_original_907 | ffpp_yt_907 | 0.434 | 0.050 | -0.385 |
| ffpp_original_929 | ffpp_yt_929 | 0.433 | 0.043 | -0.390 |
| ffpp_original_669 | ffpp_yt_669 | 0.414 | 0.006 | -0.408 |
| ffpp_original_515 | ffpp_yt_515 | 0.420 | 0.001 | -0.419 |
| ffpp_original_224 | ffpp_yt_224 | 0.434 | 0.014 | -0.420 |
| ffpp_original_723 | ffpp_yt_723 | 0.908 | 0.477 | -0.431 |
| ffpp_original_881 | ffpp_yt_881 | 0.656 | 0.208 | -0.448 |
| ffpp_original_109 | ffpp_yt_109 | 0.460 | 0.000 | -0.460 |
| ffpp_original_231 | ffpp_yt_231 | 0.482 | 0.010 | -0.472 |
| ffpp_original_715 | ffpp_yt_715 | 0.488 | 0.012 | -0.476 |
| ffpp_original_940 | ffpp_yt_940 | 0.536 | 0.058 | -0.478 |
| ffpp_original_433 | ffpp_yt_433 | 0.482 | 0.003 | -0.478 |
| ffpp_original_288 | ffpp_yt_288 | 0.494 | 0.015 | -0.479 |
| ffpp_original_297 | ffpp_yt_297 | 0.541 | 0.044 | -0.497 |
| ffpp_original_595 | ffpp_yt_595 | 0.691 | 0.193 | -0.498 |
| ffpp_original_588 | ffpp_yt_588 | 0.515 | 0.002 | -0.513 |
| ffpp_original_663 | ffpp_yt_663 | 0.514 | 0.000 | -0.514 |
| ffpp_original_983 | ffpp_yt_983 | 0.783 | 0.201 | -0.582 |
| ffpp_original_711 | ffpp_yt_711 | 0.652 | 0.002 | -0.650 |
| ffpp_original_681 | ffpp_yt_681 | 0.890 | 0.095 | -0.795 |
| ffpp_original_058 | ffpp_yt_058 | 0.837 | 0.028 | -0.809 |

</details>

## 7. Interpretation

Distinguishing four possible explanations, per the measurements above:

- **Pure calibration problem** would show: similar or better ROC-AUC/PR-AUC (ranking preserved), but real/fake P(fake) distributions shifted in a way a single rescaling (e.g. temperature scaling) could fix -- i.e. the *ordering* of samples by P(fake) stays close to production's ordering, just the numeric scale differs.
- **Boundary/generalization problem** would show: ROC-AUC/PR-AUC change (not just probability scale) and/or the real and fake P(fake) distributions overlap differently -- some previously well-separated samples now sit on the wrong side, which a single monotonic rescaling cannot fix.
- **Identity-specific generalization problem** would show: the identity-level flip lists in section 5 are large in both directions (churn) rather than the new model being a strict, monotonic improvement over production for every identity.
- **Dataset/domain shift** cannot be directly measured from this sealed-test-only diagnostic -- it was separately addressed via the raw live-camera A/B test (`live_camera_ab_test_report.md`), which already showed the same shift reproduces on raw, non-FF++ input.

### Verdict on A/B/C/D, drawn from the measurements in this report

**D -- a combination, but not an equal mix.** The evidence points to a decision-boundary/generalization shift
(B) and a training-distribution/identity-specific bias (C) as the dominant components, with a real but
secondary calibration problem (A) that amplifies how extreme the resulting errors look.

**Why this is not primarily A (pure calibration):** a pure calibration issue is, by definition, a monotonic
rescaling of the same underlying ranking -- it cannot change ROC-AUC or PR-AUC, which are rank-invariant. Both
changed substantially here (ROC-AUC 0.8927->0.9567, PR-AUC real-as-positive 0.6429->0.8212), and 18.58% of all
12,009 predictions flip across the 0.5 threshold between the two models -- far more reordering than a single
rescaling could produce. The underlying ranking of samples genuinely changed, not just the numeric scale.

**Why B (boundary shift) is clearly present:** beyond the AUC changes, the effect on REAL and FAKE samples is
asymmetric in a way a *re-drawn* boundary would produce: for FAKE samples, the new model improves across the
**entire** distribution including the hard tail (P(fake) p05: 0.154->0.244, p10: 0.320->0.796 -- even
previously-hard fakes move solidly into fake territory). For REAL samples, the bulk improves (mean
0.267->0.200) but the hard tail gets **worse**, not just less-improved (p90: 0.661->0.890, p95: 0.793->0.982).
A single global shift would move both tails in the same relative direction; this asymmetry is consistent with a
boundary redrawn using different discriminative features, not a rescaled one.

**Why C (training-distribution / identity-specific bias) is clearly present:** the identity-level flip lists in
section 5 are large in *both* directions -- 17 real identities newly misclassified, 17 different ones fixed, out
of 169 -- not a monotonic improvement. A model that generalized strictly better would shrink the wrong set, not
swap roughly a fifth of it for a different fifth. This is the same pattern independently confirmed on genuinely
out-of-sealed-test subjects in the raw live-camera A/B test (`live_camera_ab_test_report.md`): two real people
never in FF++ both landed on the "newly broken" side, consistent with a boundary that depends on
training-distribution-specific cues that don't transfer evenly to new identities.

**Why A (calibration) is present but secondary:** ECE is genuinely worse for the new model (0.0275->0.0544),
and the per-bin detail is precise about where: the new model concentrates 89.4% of all predictions in the
[0.9,1.0] confidence bin (vs. production's 47.3%), and within the smaller [0.7,0.9) confidence range it is
**badly overconfident** -- e.g. the [0.8,0.9) bin claims 85.7% mean confidence but is only 63.0% accurate (a
22.7pp gap), against production's [0.8,0.9) bin at 85.4% confidence / 81.4% accuracy (a 4.0pp gap). This
overconfidence doesn't create the misclassifications by itself, but it explains why, when the redrawn boundary
(B) and identity-specific bias (C) do put a real sample on the wrong side, the new model reports it at an
extreme value (0.9+) rather than a borderline one (0.5-0.6) -- exactly the pattern seen in the live-camera
test's two-person segment (0% classified REAL, not just below-50%-but-close).

## 8. Summary

**Strongest evidence:** the combination of (a) ROC-AUC/PR-AUC changing substantially, which rules out pure
calibration as the primary mechanism, and (b) the identity-level churn being roughly symmetric (17 fixed / 17
newly broken) rather than monotonic, which rules out "the new model is a strict superset improvement." Together
these are the clearest, most direct evidence that this is a genuine boundary/generalization difference, not a
probability-scale artifact -- independently corroborated by the raw live-camera A/B test reproducing the same
asymmetry on subjects entirely outside the sealed test.

**What remains uncertain:** the *precise* mechanism inside the training recipe that produced the redrawn
boundary (e.g. whether the balanced sampler, the heavier augmentation, or the resolution-loss simulation is
most responsible) has not been isolated; this diagnostic identifies *that* the boundary shifted and *that* it's
compounded by overconfidence, not *which recipe change* caused it. Also unmeasured: whether the same
ECE/overconfidence pattern holds on non-FF++ real subjects at the same magnitude (the live-camera test measured
the raw P(fake) shift there, but could not compute a live-domain ECE, since there are no labeled fake live
samples to compute it against).

**Single most informative next experiment:** an ablation that isolates the training-recipe changes one at a
time against this same sealed test set and identity-level churn analysis (e.g. balanced sampling alone, vs. +
heavier augmentation, vs. + resolution-loss simulation) would show which specific recipe choice drives the
boundary shift and the overconfidence, rather than attributing it to the recipe as a whole. This is a
training-required but narrowly-scoped experiment (three or four short, targeted re-training runs on the
already-extracted dataset, each evaluated the same way as here) -- more informative per run than another
full end-to-end retrain-and-hope cycle, and still does not require touching production.

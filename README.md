# Real-Time Multimodal Deepfake Detection System for Enterprise Communications

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-success.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Visual%20Val%20Acc-99.98%25-brightgreen.svg)]()

An edge-optimized, real-time multimodal deepfake detection pipeline purpose-built to secure enterprise video conferencing (WebRTC) streams against executive impersonation and Business Email Compromise 2.0 (BEC 2.0) attacks.

---

## Executive Summary

As generative AI models enable real-time identity spoofing, enterprise communication platforms face severe security vulnerabilities—highlighted by high-profile fraud events such as the $25.6M Hong Kong deepfake CFO scam. 

This repository implements a lightweight, low-latency framework that operates on live streaming feeds. By coupling a MobileNetV3 visual extraction backbone with audio spectral analysis and score-level late fusion, the system delivers continuous trust scoring and automated Zero-Trust SIEM alerting without introducing perceptible call latency.

---

## Empirical Benchmarks & Performance Highlights

* **Visual Model Accuracy:** Achieved **99.98% validation accuracy** across a hybrid dataset of **126,538 images** (combining benchmark corpora and live webcam captures).
* **Inference Throughput:** **38.53 frames per second** (~25.9 ms/frame visual pipeline execution time).
* **Real-Time Streaming Feasibility:** Total frame processing budget remains comfortably under the 33.3 ms requirement for standard 30 FPS WebRTC video streams.
* **Lightweight Footprint:** ~11.2 MB trained weight file size (`visual_model.pth`), enabling edge deployment without heavy GPU infrastructure requirements.

---

## System Architecture & Tech Stack

```text
   ┌──────────────────────┐      ┌─────────────────────────┐
   │ Live Video Stream    ├─────►│ Haar Cascade Face Crop  ├─────┐
   └──────────────────────┘      └─────────────────────────┘     │
                                                                 ▼
                                                 ┌─────────────────────────┐
                                                 │ MobileNetV3 Visual (Sv) │
                                                 └───────────┬─────────────┘
                                                             │
                                                             ▼
                                                ┌──────────────────────────┐      ┌──────────────────────┐
                                                │ Score-Level Fusion Engine├─────►│ Enterprise SIEM /    │
                                                │ S_fusion = 0.6Sv + 0.4Sa │      │ Zero-Trust Alerting  │
                                                └───────────▲──────────────┘      └──────────────────────┘
                                                            │
   ┌──────────────────────┐      ┌──────────────────────────┤
   │ Live Audio Stream    ├─────►│ Mel-Spectrogram / Librosa├─────┘
   └──────────────────────┘      └──────────────────────────┘

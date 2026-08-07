# Real-Time Multimodal Deepfake Detection System for Enterprise Communications

## Overview
This repository contains the ongoing development of an edge-optimized, real-time multimodal deepfake detection system designed to secure enterprise video conferencing (WebRTC) streams against executive impersonation and Business Email Compromise (BEC 2.0).

## Core Team Members
* **Maharsi** - Title Finalization & Literature Survey
* **Lokesh** - Problem Identification & Threat Vector Analysis
* **Rushitha** - Generalized Objectives & Timeline
* **M Hema Priya** - Innovation, Architecture & Tech Stack

## System Architecture Highlights
* **Video Branch**: OpenCV & MediaPipe (3D Facial Landmark Tracking) + MobileNetV3
* **Audio Branch**: Librosa / Torchaudio (Mel-Spectrogram Extraction) + Siamese CNN with Self-Attention
* **Sync Engine**: Phoneme-Viseme Cross-Modal Synchronization (Temporal Contrastive Loss)
* **Inference Pipeline**: ONNX Runtime with INT8 Quantization (Sub-100ms latency)
* **Backend & Frontend**: FastAPI, WebRTC Ingestion, React/Electron Trust Score Overlay

## Project Roadmap
- [x] Phase 1: Problem Formulation & Literature Survey (Review 1)
- [ ] Phase 2: Feature Extraction Pipeline & Data Preprocessing (Review 2)
- [ ] Phase 3: Model Training & 50% Functional Prototype (Review 3)
- [ ] Phase 4: WebRTC Integration & UI Dashboard (Review 4)
- [ ] Phase 5: Final Benchmarking & viva (Final Viva)

## License
Distributed under the MIT License.

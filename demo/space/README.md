---
title: YOLO26 OBB Aerial Detection
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Rotated bounding boxes on aerial images (YOLO26n-OBB, CPU)
---

# YOLO26n-OBB — Oriented Object Detection on Aerial Images (CPU demo)

> **Reference implementation only.** This Gradio/CPU version is not the deployed Space. The live
> demo is the static browser implementation in `demo/space-static/`.

Upload an aerial or satellite image and get **oriented (rotated) bounding boxes**
for the 15 DOTA aerial classes (planes, ships, vehicles, harbors, …).

<!-- claim:browser-scope -->
- Model: official `yolo26n-obb` exported to **ONNX**, inference via **ONNX Runtime CPU**
  (the nano model keeps CPU-only local or self-hosted inference responsive)
- This reference demo does not represent the fine-tuned `yolo26m-obb` checkpoint's accuracy or
  its recorded T4 latency.
<!-- /claim:browser-scope -->
- Adjustable confidence threshold and per-class filtering; the table reports each
  detection's rotation angle
- Part of a full training-to-deployment portfolio project (fine-tuning
  YOLO26m-OBB on DOTAv1, ONNX/TensorRT benchmarks, HBB-vs-OBB analysis)

## Licensing

- Code: **AGPL-3.0** (Ultralytics)
- DOTA dataset: **academic use only — commercial use prohibited**

---
title: YOLO26 OBB Aerial Detection
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: agpl-3.0
short_description: Rotated boxes on aerial images, 100% in-browser (ONNX RT)
---

# YOLO26n-OBB — Oriented Object Detection on Aerial Images (runs in your browser)

Upload an aerial or satellite image and get **oriented (rotated) bounding boxes** for the 15
DOTA aerial classes (planes, ships, vehicles, harbors, …) — inference happens **entirely in
your browser** via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM), no
server, no GPU, nothing uploaded anywhere.

<!-- claim:browser-scope -->
- Model: official `yolo26n-obb` exported to ONNX, running client-side (~10MB model, downloaded once and
  cached by the browser)
- This live demo does not represent the fine-tuned `yolo26m-obb` checkpoint's accuracy or its
  recorded T4 latency; that model is published separately for evaluation and deployment experiments
<!-- /claim:browser-scope -->
- Adjustable confidence threshold and per-class filtering; the table reports each detection's
  rotation angle
- Built as a **static Space** that requires no hosted compute tier — the whole
  demo is HTML/CSS/JS plus the ONNX file, no Python runtime involved
- Part of a full training-to-deployment portfolio project (fine-tuning YOLO26m-OBB on DOTAv1,
  ONNX/TensorRT benchmarks, HBB-vs-OBB analysis) — see
  [steven0226/yolo26m-obb-dota](https://huggingface.co/steven0226/yolo26m-obb-dota)

## Licensing

- Code: **AGPL-3.0** (Ultralytics)
- DOTA dataset: **academic use only — commercial use prohibited**

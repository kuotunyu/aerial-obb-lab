---
title: YOLO26 OBB Browser Demo
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: agpl-3.0
short_description: Bring-your-own-model rotated detection in the browser
---

# YOLO26 OBB — Bring Your Own Model Browser Demo

Select a compatible YOLO26 OBB ONNX file and an image to run **oriented (rotated) bounding-box**
inference entirely in your browser through
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM). Neither selected file is
uploaded, and this repository does not distribute a model binary.

<!-- claim:browser-scope -->
- Model: a user-supplied local ONNX file matching the documented `images` input and end-to-end
  `output0 [1,N,7]` contract
- This browser demo does not represent the fine-tuned `yolo26m-obb` checkpoint's accuracy or its
  recorded T4 latency
<!-- /claim:browser-scope -->
- Adjustable confidence threshold and per-class filtering; the table reports each detection's
  rotation angle
- Built as a static HTML/CSS/JS demo with no Python runtime or bundled ONNX file
- Part of a full training-to-deployment portfolio project (fine-tuning YOLO26m-OBB on DOTAv1,
  ONNX/TensorRT benchmarks, and HBB-vs-OBB analysis). This code-only release intentionally does
  not distribute the checkpoint, exported model, or DOTA-derived imagery.

## Licensing

- Code: **AGPL-3.0**. A user-supplied model keeps its own upstream, dataset, and weight terms;
  Ultralytics' Enterprise route is separate.
- DOTA images/annotations are academic-use-only and are not bundled. Users are responsible for
  the rights to any model or image they select.

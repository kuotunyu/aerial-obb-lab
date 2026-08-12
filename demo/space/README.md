---
title: YOLO26 OBB Aerial Detection
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Bring-your-own-model rotated detection on CPU
---

# YOLO26 OBB — Bring Your Own Model CPU Demo

> **Reference implementation only.** For a serverless browser path, use the BYOM implementation
> in `demo/space-static/`.

Set `MODEL_PATH` to an existing local `.pt` or `.onnx` file, then upload an image to get
**oriented (rotated) bounding boxes**. `MODEL_DEVICE` defaults to `cpu`.

<!-- claim:browser-scope -->
- Model: a user-supplied local model file; this app has no download, export, or named-model fallback
- This reference demo does not represent the fine-tuned `yolo26m-obb` checkpoint's accuracy or
  its recorded T4 latency.
<!-- /claim:browser-scope -->
- Adjustable confidence threshold and per-class filtering; the table reports each
  detection's rotation angle
- Part of a full training-to-deployment portfolio project (fine-tuning
  YOLO26m-OBB on DOTAv1, ONNX/TensorRT benchmarks, HBB-vs-OBB analysis)

```powershell
$env:MODEL_PATH = "C:/models/your-model.onnx"
$env:MODEL_DEVICE = "cpu"
python app.py
```

## Licensing

- Code: **AGPL-3.0**. A user-supplied model keeps its own upstream, dataset, and weight terms;
  Ultralytics' Enterprise route is separate.
- DOTA images/annotations are academic-use-only and are not bundled. Users are responsible for
  the rights to any model or image they select.

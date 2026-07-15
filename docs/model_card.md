---
license: agpl-3.0
language:
- en
tags:
- object-detection
- oriented-bounding-box
- obb
- yolo
- yolo26
- aerial-imagery
- remote-sensing
- pytorch
- onnx
- ultralytics
datasets:
- dota-v1
metrics:
- mAP
pipeline_tag: object-detection
---

# YOLO26m-OBB fine-tuned on DOTAv1

Fine-tuned [`yolo26m-obb`](https://docs.ultralytics.com/) (oriented bounding box detection) on
a re-split of the [DOTAv1](https://captain-whu.github.io/DOTA/) aerial imagery dataset. Part of
a training-to-deployment portfolio project — full writeup, HBB-vs-OBB analysis, and
ONNX/TensorRT benchmarks in the project repo (GitHub link to be added once published).

## Model description

- **Task**: oriented (rotated) bounding box detection, 15 DOTA classes (plane, ship, storage
  tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge,
  large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool)
- **Base checkpoint**: Ultralytics' official `yolo26m-obb.pt` (already DOTAv1-trained; see
  Training data below for what "fine-tuned" means here)
- **Input**: 1024×1024 RGB
- **Params**: 21.2M / **GFLOPs**: 183 (at 1024)

## Training data

DOTAv1, re-tiled with `ultralytics.data.split_dota.split_trainval` at scales `[0.8, 1.2]`
(gap 500) — 62,030 train / 21,271 val tiles at 1024px. Trained on a Colab A100 for 28/30
epochs, early-stopped (`patience=15`), best checkpoint at epoch 13.

## Evaluation

Baseline = official `yolo26m-obb.pt`, evaluated on this repo's own val split for an apples-to-apples
comparison (not the officially published test-split numbers — see caveat below).

| model | split | mAP50 | mAP50-95 |
|---|---|---:|---:|
| yolo26m-obb.pt (official, published) | DOTAv1 test | 81.0 | 55.3 |
| yolo26m-obb.pt (official, this repo's val) | DOTAv1 val | 78.2 | 63.3 |
| **this fine-tuned checkpoint** | DOTAv1 val | 78.2 | 63.1 |

Fine-tuning moved the needle by essentially nothing under matched conditions (Δ mAP50 -0.05pt,
Δ mAP50-95 -0.13pt) — expected, since the base checkpoint was already DOTAv1-converged; this
run quantifies that ceiling rather than claiming a large improvement. Full per-class breakdown
and training-curve analysis in the project repo's `docs/training_results.md`.

**Caveat**: the top row uses DOTA's own test-split evaluation pipeline (full-image stitching);
the bottom two rows use this repo's own val-split tiling and `ultralytics` evaluation — not
directly comparable to the top row, only to each other.

## Deployment

Also exported to ONNX and a TensorRT FP16 engine (Tesla T4), see `results.png` /
`confusion_matrix_normalized.png` / `results.csv` in this repo's file list and the project
README's benchmark table for latency/FPS numbers.

**Live demo** (nano variant, 100% browser-side via ONNX Runtime Web, no server):
https://huggingface.co/spaces/steven0226/yolo26-obb-aerial-detection

## Intended use & limitations

- Academic / research / portfolio use. **The DOTA dataset license prohibits commercial use** —
  this restriction carries over to any weights derived from it.
- Code is Ultralytics (AGPL-3.0); these fine-tuned weights are a derivative work and carry the
  same license.
- Trained and evaluated on DOTAv1's specific tiling scheme; performance on differently-tiled or
  differently-sourced aerial imagery is untested.

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")  # this repo's fine-tuned checkpoint
results = model.predict("aerial_image.jpg", imgsz=1024)
results[0].show()  # draws rotated boxes
```

## License

- Code: AGPL-3.0 ([Ultralytics](https://github.com/ultralytics/ultralytics))
- Weights: derivative of AGPL-3.0 Ultralytics code, same license
- Training data (DOTA): academic use only, commercial use prohibited

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

# Aerial OBB Lab — YOLO26m-OBB DOTAv1 Experiment Model Card

Fine-tuned [`yolo26m-obb`](https://docs.ultralytics.com/) (oriented bounding box detection) on
a re-split of the [DOTAv1](https://captain-whu.github.io/DOTA/) aerial imagery dataset. Part of
a companion training-to-deployment source project containing the notebooks, HBB-vs-OBB
analysis, deployment code, and ONNX/TensorRT benchmarks. This code-only release records the
historical model evidence but does not distribute the checkpoint, exports, or DOTA-derived images.

## Model description

- **Task**: oriented (rotated) bounding box detection, 15 DOTA classes (plane, ship, storage
  tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge,
  large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool)
- **Base checkpoint**: Ultralytics' official `yolo26m-obb.pt` (already DOTAv1-trained; see
  Training data below for what "fine-tuned" means here)
- **Input**: 1024×1024 RGB
- **Params**: 21.2M / **GFLOPs**: 183.3 (at 1024; [official Ultralytics OBB table](https://docs.ultralytics.com/tasks/obb))

## Training data

DOTAv1, re-tiled with `ultralytics.data.split_dota.split_trainval` at scales `[0.8, 1.2]`
(gap 500) — 62,030 train / 21,271 val tiles at 1024px. Trained on a Colab A100 for 28/30
epochs, early-stopped (`patience=15`), best checkpoint at epoch 13.

## Evaluation

<!-- claim:matched-evaluation -->
Baseline = official `yolo26m-obb.pt`, evaluated on this repo's own val split for an apples-to-apples
comparison (not the officially published test-split numbers — see caveat below).

| model | split | mAP50 | mAP50-95 |
|---|---|---:|---:|
| yolo26m-obb.pt (official, published) | DOTAv1 test | 81.0 | 55.3 |
| yolo26m-obb.pt (official, this repo's val) | DOTAv1 val | 78.2 | 63.3 |
| **this fine-tuned checkpoint** | DOTAv1 val | 78.2 | 63.1 |

Fine-tuning is a **near-tie/slight regression** under matched conditions (Δ mAP50 -0.05pt,
Δ mAP50-95 -0.13pt), not an improvement. The base checkpoint was already DOTAv1-converged; this
run quantifies that ceiling. The baseline raw console log is not committed, while the fine-tuned
aggregate is checksum-gated in the committed evidence registry. A checksum-, manifest-, and
historical-consistency-gated validation-only run completed on 2026-07-15 and restored the three
rows not preserved by the original session: `plane` 0.952147 / 0.862352, `ship` 0.909448 /
0.762681, and `storage tank` 0.850699 / 0.716696 (mAP50 / mAP50-95). The reviewed run is accepted
as **PASS** and does not need to be rerun. The complete machine-readable breakdown is in
[`docs/per_class_metrics.csv`](per_class_metrics.csv) and
[`docs/per_class_metrics.json`](per_class_metrics.json); methodology and training-curve analysis
are in [`docs/training_results.md`](training_results.md).

**Caveat**: the top row uses DOTA's own test-split evaluation pipeline (full-image stitching);
the bottom two rows use this repo's own val-split tiling and `ultralytics` evaluation — not
directly comparable to the top row, only to each other.
<!-- /claim:matched-evaluation -->

## Deployment

The accepted historical run also produced ONNX and TensorRT FP16 exports on a Tesla T4. Those
model binaries and training renders are not distributed here; the source README and committed
machine-readable registry retain the bounded latency and parity evidence.

<!-- claim:browser-scope -->
The browser demo accepts a compatible, user-supplied ONNX file and runs locally through ONNX
Runtime Web. It does not represent the fine-tuned `yolo26m-obb` checkpoint's accuracy or T4
latency, and no model file is bundled.
<!-- /claim:browser-scope -->

## Intended use & limitations

- Academic / research / portfolio use. **The DOTA dataset terms prohibit commercial use** and
  underlying image-source terms may also apply. This project therefore does not clear the weights
  for commercial use; obtain rights-holder confirmation for any broader use.
- Ultralytics provides AGPL-3.0 and Enterprise licensing routes. This project uses the AGPL route
  and makes no Enterprise-license grant; see `THIRD_PARTY_NOTICES.md` in the source release.
- Trained and evaluated on DOTAv1's specific tiling scheme; performance on differently-tiled or
  differently-sourced aerial imagery is untested.

## Usage

Supply a local, trusted `.pt` or compatible `.onnx` file. Verify its provenance and checksum
before loading it; PyTorch checkpoints may execute code during deserialization.

```powershell
uv sync --frozen --no-install-project --group demo
$env:MODEL_PATH = "C:/models/your-model.onnx"
$env:MODEL_DEVICE = "cpu"
.venv/Scripts/python.exe demo/app.py
```

## License

- Code: AGPL-3.0 ([Ultralytics](https://github.com/ultralytics/ultralytics))
- User-supplied weights: their own upstream software, dataset, and weight terms apply
- Training data (DOTA): academic use only; commercial use prohibited

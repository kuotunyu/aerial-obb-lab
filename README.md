# YOLO26 OBB on DOTA: Aerial Oriented Object Detection, Training to Deployment

> 🚧 Work in progress — see [docs/PLAN.md](docs/PLAN.md) for the full project plan.

Fine-tuning **YOLO26-OBB** (oriented bounding boxes) on the **DOTAv1** aerial dataset, with a full lifecycle:
train on Colab A100 → evaluate against the official baseline → quantify *why OBB beats horizontal boxes* on aerial imagery → export ONNX / TensorRT FP16 with a 3-framework latency benchmark → Gradio demo + Hugging Face Space.

## Status

| Phase | What | Where | Status |
|---|---|---|---|
| 0 | Scaffold & environment | local | ✅ |
| 1 | DOTA8 smoke test (train→val→predict→resume→HF push→ONNX) | local (RTX 2070) | ✅ |
| 2 | Full fine-tune on DOTAv1 | Colab A100 | ✅ |
| 3 | Evaluation vs. official baseline | Colab + local | ✅ |
| 4 | "Why OBB" quantitative analysis | local | ✅ |
| 5 | ONNX / TensorRT export + benchmark | Colab GPU | ✅ |
| 6 | Gradio demo + HF Space (CPU) | local + HF | ✅ |
| 7 | Docs, model card, wrap-up | — | ⬜ |

## Evaluation: Fine-tuned vs. Official Baseline

Fine-tuned `yolo26m-obb.pt` on a re-split DOTAv1 (Colab A100, `split_dota` rates `[0.8, 1.2]`,
28/30 epochs, early-stopped on `patience=15`). Full per-class breakdown, training-curve
analysis, and confusion-matrix findings in
[docs/training_results.md](docs/training_results.md).

| model | split | mAP50 | mAP50-95 |
|---|---|---:|---:|
| yolo26m-obb.pt (official, published) | DOTAv1 test | 81.0 | 55.3 |
| yolo26m-obb.pt (official, our reproduction) | DOTAv1 val (ours) | 78.2 | 63.3 |
| fine-tuned `best.pt` | DOTAv1 val (ours) | 78.2 | 63.1 |

Only the bottom two rows are directly comparable — same val split, same tiling, same eval
conditions. The top row uses DOTA's own test-split evaluation pipeline and is shown for
context, not comparison (see [docs/training_results.md](docs/training_results.md) for why the
test/val gap runs in an unexpected direction on mAP50 vs mAP50-95).

Under matched conditions, fine-tuning moved the needle by essentially nothing (Δ mAP50
-0.05pt, Δ mAP50-95 -0.13pt). That's expected, not a failure: `yolo26m-obb.pt` is already the
official DOTAv1-trained checkpoint, so this quantifies the ceiling of continuing to fine-tune
an already-converged model on a re-tiled version of the same dataset, rather than demonstrating
a big lift.

## Why Oriented Bounding Boxes? (quantified, not hand-waved)

Measured on **28,853 ground-truth objects across 456 DOTAv1 val images** (see
[docs/analysis_results.md](docs/analysis_results.md) for full tables,
[scripts/obb_analysis.py](scripts/obb_analysis.py) to reproduce):

**1. Horizontal boxes swallow background.** Area of the axis-aligned box ÷ area of
the true oriented box:

| class | mean | p90 | | class (control) | mean |
|---|---:|---:|---|---|---:|
| bridge | **2.43×** | 3.71× | | roundabout | 1.02× |
| harbor | **2.15×** | 3.66× | | storage tank | 1.00× |
| large vehicle | **2.14×** | 3.21× | | | |
| ship | **1.95×** | 2.69× | | | |

Elongated, arbitrarily-rotated objects force an HBB to be ~2× too large — while the
circular control classes (storage tank, roundabout) sit at 1.0×, confirming the
measurement isolates orientation, not labeling noise.

**2. In dense scenes, HBB overlap is mostly *phantom* overlap.** Among same-class
neighbor pairs whose *horizontal* boxes overlap heavily (IoU ≥ 0.3), the fraction whose
*oriented* boxes barely overlap at all (IoU < 0.1):

| class | HBB IoU≥0.3 pairs | phantom rate |
|---|---:|---:|
| large vehicle | 810 | **100%** |
| ship | 736 | **99%** |
| harbor | 43 | **98%** |
| small vehicle | 42 | **93%** |

A horizontal-box detector's NMS sees these as duplicate detections and suppresses true
positives — parked trucks and docked ships literally disappear. Oriented boxes make the
overlap (and the NMS decision) reflect reality.

**3. Seeing is believing** — same marina, same labels (all five comparisons in `assets/`):

![HBB vs OBB, 535 ships in a marina](assets/hbb_vs_obb_1_P0706_ship.jpg)

## Deployment Benchmark: PyTorch vs. ONNX Runtime vs. TensorRT FP16

Exported the fine-tuned `best.pt` to ONNX and a TensorRT FP16 engine on a Colab **Tesla T4**
(`notebooks/02_benchmark_colab.ipynb`, reproducible standalone — original plan was to build the
TensorRT engine on a local RTX 4090; actual local hardware is an RTX 2070 8GB, so this moved to
Colab, which doubles as the workaround for TensorRT's rough Windows install story). Batch=1,
imgsz=1024, 20 warmup + 100 timed runs per backend, engine build tied to this exact GPU model
and TensorRT version.

**Export accuracy check first** (dota8 val, same tool/conditions across backends): PyTorch
mAP50=0.9950, ONNX 0.9950, TensorRT 0.9950 — no accuracy loss from either export, clears the
<1pt tolerance this project set going in given YOLO26's known end-to-end export issues
(ultralytics#23397 et al.).

| backend | size (MB) | mean latency | p50 | p95 | FPS |
|---|---:|---:|---:|---:|---:|
| PyTorch (FP32) | 48.7 | 71.54 ms | 71.49 ms | 74.08 ms | 14.0 |
| ONNX Runtime GPU | 85.3 | 79.09 ms | 79.67 ms | 81.43 ms | 12.6 |
| TensorRT FP16 | 45.0 | 20.22 ms | 20.14 ms | 21.09 ms | 49.4 |

**TensorRT FP16 is ~3.5× faster than eager PyTorch** and comes in at the smallest file size.
**ONNX Runtime GPU is actually slightly slower than native PyTorch here** — without a
graph-compilation backend like TensorRT behind it, ONNX Runtime's GPU execution provider doesn't
automatically beat PyTorch's own cuDNN kernels; it's included because it's the backend the free-tier
HF Space demo runs on CPU (Phase 6), not because it's the fastest GPU option.

Getting these numbers took several rounds of environment debugging on Colab's side, none of it
this project's own code: a broken `torch._dynamo` build, HF's file CDN intermittently returning
signed URLs with an invalid key, and ONNX Runtime's CUDA execution provider crashing the whole
Colab runtime outright (worked around by running that inference in an isolated subprocess). Full
writeup in [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md).

## Licensing

- Code: **AGPL-3.0** (required by [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0; fine-tuned weights are derivative and carry the same license)
- Dataset: **DOTA** is released for **academic use only — commercial use is prohibited**

*(Demo GIF and reproduction steps land in later phases. 中文版見 README.zh-TW.md)*

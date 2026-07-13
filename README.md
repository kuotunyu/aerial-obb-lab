# YOLO26 OBB on DOTA: Aerial Oriented Object Detection, Training to Deployment

> 🚧 Work in progress — see [docs/PLAN.md](docs/PLAN.md) for the full project plan.

Fine-tuning **YOLO26-OBB** (oriented bounding boxes) on the **DOTAv1** aerial dataset, with a full lifecycle:
train on Colab A100 → evaluate against the official baseline → quantify *why OBB beats horizontal boxes* on aerial imagery → export ONNX / TensorRT FP16 with a 3-framework latency benchmark → Gradio demo + Hugging Face Space.

## Status

| Phase | What | Where | Status |
|---|---|---|---|
| 0 | Scaffold & environment | local | ✅ |
| 1 | DOTA8 smoke test (train→val→predict→resume→HF push→ONNX) | local (RTX 2070) | ⬜ |
| 2 | Full fine-tune on DOTAv1 | Colab A100 | ⬜ |
| 3 | Evaluation vs. official baseline | Colab + local | ⬜ |
| 4 | "Why OBB" quantitative analysis | local | ⬜ |
| 5 | ONNX / TensorRT export + benchmark | Colab GPU | ⬜ |
| 6 | Gradio demo + HF Space (CPU) | local + HF | ⬜ |
| 7 | Docs, model card, wrap-up | — | ⬜ |

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

## Licensing

- Code: **AGPL-3.0** (required by [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0; fine-tuned weights are derivative and carry the same license)
- Dataset: **DOTA** is released for **academic use only — commercial use is prohibited**

*(Full README with results tables, benchmark numbers, HBB-vs-OBB analysis, and reproduction steps lands in Phase 7. 中文版見 README.zh-TW.md)*

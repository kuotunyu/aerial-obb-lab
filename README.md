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

## Licensing

- Code: **AGPL-3.0** (required by [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0; fine-tuned weights are derivative and carry the same license)
- Dataset: **DOTA** is released for **academic use only — commercial use is prohibited**

*(Full README with results tables, benchmark numbers, HBB-vs-OBB analysis, and reproduction steps lands in Phase 7. 中文版見 README.zh-TW.md)*

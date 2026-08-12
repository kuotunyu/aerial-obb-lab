# Project plan — historical execution record

> Phases 0–7 are complete. This file preserves the engineering sequence; use the root README for
> the current code-only release, demo, installation, and evidence boundaries.

## Objective

Build an end-to-end OBB portfolio project around YOLO26 and DOTAv1:

1. validate the OBB workflow at smoke-test scale;
2. fine-tune on a reproducible DOTAv1 split;
3. compare against the official baseline under matched conditions;
4. quantify OBB-vs-HBB ground-truth geometry;
5. export and benchmark PyTorch, ONNX Runtime, and TensorRT;
6. provide one local-file browser inference surface; and
7. publish honest results, limitations, licenses, and reproducibility gates.

## Final architecture

- **Local CPU release path:** Python 3.11, analysis/tests, package build, privacy and claim gates,
  clean-export rebuild, and a synthetic Playwright browser smoke.
- **Browser-native demo:** static HTML/CSS/JavaScript with ONNX Runtime Web 1.20.1 WASM. The user
  supplies a compatible local ONNX model and image; neither file is uploaded.
- **Historical Colab A100 workflow:** DOTAv1 tiling, training, validation-only metric recovery, and
  checkpoint persistence.
- **Historical Colab T4 workflow:** ONNX/TensorRT export, DOTA8 smoke parity, and the bounded
  batch-1 1024px latency benchmark.
- **Historical Hugging Face artifacts:** retained privately by the owner and excluded from the
  public code-only release.

Local environments, datasets, caches, weights, model exports, and training runs are intentionally
not versioned. The current release does not retrain, run full validation, build TensorRT, or execute
GPU inference.

## Repository layout

```text
<project-root>/
├── src/obbkit/          # geometry analysis, browser reference, visualization helpers
├── scripts/             # CPU release, privacy, package, browser, and repository gates
├── tests/               # deterministic CPU and synthetic browser fixtures
├── notebooks/           # self-contained historical Colab workflows
├── demo/web/            # single Browser-native ONNX Runtime Web BYOM workbench
├── release/             # machine-readable evidence and artifact exclusion manifest
└── docs/                # results, model card, decisions, and release evidence
```

## Completed phases

| Phase | Scope | Execution | Accepted evidence |
|---|---|---|---|
| 0 | Scaffold, license, and environment | Local CPU | package and repository checks |
| 1 | DOTA8 train/val/predict/resume/export smoke | Historical workstation | workflow completed; retained only as smoke evidence |
| 2 | DOTAv1 fine-tune | Historical Colab A100 | 28/30 epochs, early stop, best epoch 13 |
| 3 | Matched baseline evaluation | Historical Colab + local audit | 15-class metrics and checksum-gated recovery |
| 4 | HBB-vs-OBB geometry | Local CPU | 28,853 objects across 456 val images |
| 5 | Export and latency benchmark | Historical Colab T4 | DOTA8 parity plus bounded T4 measurements |
| 6 | Interactive deployment | Local browser | BYOM WASM workbench and synthetic Playwright smoke |
| 7 | Release hardening | Windows/Ubuntu CPU contract | claims, privacy, license, package, and clean export |

## Decisions and trade-offs

- `yolo26m-obb` balanced model size and the official reported DOTAv1 accuracy for the experiment.
- DOTAv1 tiling used `[0.8, 1.2]` after measuring the data-volume and throughput trade-off.
- The matched fine-tuning result is reported as a near-tie/slight regression: approximately
  -0.05pt mAP50 and -0.13pt mAP50-95. It is not reframed as an improvement.
- DOTA8 0.9950 across three backends is an export smoke, not full DOTAv1 certification.
- TensorRT FP16 20.22 ms / 49.4 FPS is historical evidence for one Tesla T4, batch=1, 1024px,
  and the recorded measurement procedure—not a universal SLA.
- The maintained demo is one Browser-native BYOM path. Removing a server UI reduced dependency,
  privacy, hosting, and documentation drift while demonstrating client-side deployment.
- The release is code-only because DOTA is academic-use-only and weights inherit separate
  Ultralytics, dataset, and source-image obligations.

## Reproducibility boundaries

- DOTAv1 data is not redistributed. Independent reproduction must follow its academic-use terms.
- TensorRT engines are hardware/runtime-bound and must be rebuilt from an authorized checkpoint.
- Official test-split scores and this project's tile-level val scores are not directly comparable.
- Three fine-tuned per-class rows were recovered by a checksum-, manifest-, and
  historical-consistency-gated validation-only workflow. The accepted values and provenance are in
  [per_class_metrics.csv](per_class_metrics.csv), [per_class_metrics.json](per_class_metrics.json),
  and [training_results.md](training_results.md); no rerun is required for this release.
- The browser screenshot uses a synthetic SVG and stubbed output. It verifies integration and UI,
  not model quality or the historical T4 latency.

## Optional future work after v1.0

- Adapt to a genuinely new aerial domain instead of continuing a converged DOTAv1 model.
- Train or quantize a separately licensed lightweight model specifically for browser deployment.
- Publish only after the local committed-export gates and hosted Ubuntu/Windows CPU checks pass.

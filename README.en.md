[正體中文](README.md) | English

# Aerial OBB Lab

[![CI](https://github.com/kuotunyu/aerial-obb-lab/actions/workflows/release-gates.yml/badge.svg)](https://github.com/kuotunyu/aerial-obb-lab/actions/workflows/release-gates.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![Historical Evidence](https://img.shields.io/badge/Evidence-Historical%20A100%20%7C%20T4-F9AB00?logo=googlecolab&logoColor=white)
![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Web%20WASM-005CED?logo=onnx&logoColor=white)
![BYOM Demo](https://img.shields.io/badge/Demo-Browser%20BYOM-green)

This project takes **YOLO26-OBB on DOTAv1** from A100 fine-tuning and matched baseline evaluation
through OBB/HBB geometry analysis, ONNX/TensorRT benchmarking, and a browser-native BYOM demo.
The result is reported as-is: matched fine-tuning was a near-tie/slight regression.

**Demo:** [`demo/web/`](demo/web/) · **Model card:** [`docs/model_card.md`](docs/model_card.md) ·
**Release scope:** code, evidence, three public-domain NAIP aerial derivatives, and one privacy-sanitized AGPL demo model.

<!-- claim:browser-scope -->
> The browser demo first displays the official aerial original. Only after **Start Detect** does it
> load the same-origin privacy-sanitized AGPL derivative and pinned runtime, then perform genuine
> inference in the current browser session. BYOM still accepts a user-supplied compatible ONNX model
> and image; no model/image bytes are uploaded. This integration demo **does not represent** the
> fine-tuned `yolo26m-obb` checkpoint's accuracy, evaluation, or recorded T4 latency; its screenshot
> is subject to the same boundary.
<!-- /claim:browser-scope -->

## Highlights

| Area | Result | Evidence boundary |
|---|---|---|
| Matched evaluation | Fine-tuned vs. official baseline: Δ mAP50 **-0.05pt**, Δ mAP50-95 **-0.13pt** | Near-tie/slight regression, not an accuracy gain |
| OBB geometry | 456 val images, 28,853 objects; overall weighted mean **1.76×**, bridge mean **2.43×** | Ground-truth geometry, not a detector benchmark |
| Deployment | TensorRT FP16 **20.22 ms / 49.4 FPS** | Historical Tesla T4, batch=1, 1024px environment |
| Browser-native real-image demo | Choose one of three public-domain NAIP originals → user presses Detect → genuine local inference; advanced BYOM remains available | Detect lazy-loads pinned jsDelivr runtime and the same-origin privacy-sanitized derivative; non-zero network, no DOTA pixels |

![OBB workbench after genuine local-browser inference on the default airfield sample; not model-quality evidence](docs/assets/browser-workbench.png)

*After explicit Detect, the selected public-domain NAIP original is processed by genuine local-browser inference and shown with rotated polygons, numeric runtime, and the result table. These curated clarity examples are integration evidence, not accuracy, evaluation, or historical T4 latency evidence; their guardrails are drift checks only.*

---

## Project Flow

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph DataStage ["Phase 1: Data Preparation & Tiling"]
        direction LR
        Raw[("DOTAv1 Dataset<br/>(15 aerial categories)")] --> Split["Multi-scale Tiling<br/>(split_dota [0.8, 1.2])"] --> ValSet[("Frozen Val Split<br/>(456 images · 28,853 objects)")]
    end

    subgraph TrainStage ["Phase 2: Cloud Training & Matched Baseline Eval"]
        direction LR
        ValSet --> Train["YOLO26-OBB Fine-tune<br/>(Colab A100 · 28 Epochs)"] --> Best[("Best Checkpoint<br/>(best.pt / Resumable)")] --> Eval["Matched Baseline Evaluation<br/>(15-class recovery audit)"]
    end

    subgraph DeployStage ["Phase 3: Geometric Analysis & Multi-Backend Deployment"]
        direction LR
        Eval --> Geom["OBB Geometry Quantification<br/>(overall mean 1.76× · ghost overlap)"] --> Export[("ONNX / TensorRT FP16 Export<br/>(Tesla T4 20.22ms benchmark)")] --> Demo(["Browser BYOM Demo<br/>(ONNX Runtime Web / WASM)"])
    end

    DataStage --> TrainStage --> DeployStage

    classDef srcStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef procStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef pubStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class Raw,ValSet,Best,Export srcStyle
    class Split,Train,Eval,Geom procStyle
    class Demo pubStyle

    style DataStage fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,color:#1971c2,stroke-dasharray: 4 4
    style TrainStage fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2,stroke-dasharray: 4 4
    style DeployStage fill:#f4fbf7,stroke:#0ca678,stroke-width:2px,color:#0ca678,stroke-dasharray: 4 4
```

Detailed training, evaluation, analysis, and deployment branches are documented below.

---

## Status

| Phase | Milestone | Execution Target | Status |
|---|---|---|---|
| Phase 0 | Scaffold, dependencies, and environment pinning | Local CPU | Complete |
| Phase 1 | DOTA8 smoke test (train→val→predict→resume→ONNX) | Workstation | Complete |
| Phase 2 | Full fine-tune on DOTAv1 (28 Epochs Early-stop) | Colab A100 | Complete |
| Phase 3 | Evaluation vs. official baseline & 15-class recovery | Colab + Local | Complete |
| Phase 4 | "Why OBB" quantitative geometry & ghost overlap analysis | Local | Complete |
| Phase 5 | ONNX / TensorRT export + 3-backend benchmark | Colab Tesla T4 | Complete |
| Phase 6 | Static browser-native BYOM demo | Local / Web | Complete |
| Phase 7 | Comprehensive technical docs, model card, release gates | Local CI | Complete |

---

## Evaluation: Fine-tuned vs. Official Baseline

<!-- claim:matched-evaluation -->
- **Training:** Colab A100; re-split DOTAv1 (`split_dota` `[0.8, 1.2]`); `yolo26m-obb.pt`
  ran 28/30 epochs and early-stopped at `patience=15`.
- **Recovery:** On 2026-07-15, a checksum/provenance-gated validation-only run restored `plane`
  0.952147/0.862352, `ship` 0.909448/0.762681, and `storage tank` 0.850699/0.716696
  (mAP50/mAP50-95).
- **Evidence:** [CSV](docs/per_class_metrics.csv) · [JSON](docs/per_class_metrics.json) ·
  [recovery notebook](notebooks/03_recover_per_class_metrics_colab.ipynb) ·
  [full analysis](docs/training_results.md).

| Model Variant | Evaluation Split | mAP50 | mAP50-95 |
|---|---|---:|---:|
| yolo26m-obb.pt (official, published) | DOTAv1 test | 81.0 | 55.3 |
| yolo26m-obb.pt (official, our reproduction) | DOTAv1 val (ours) | 78.2 | 63.3 |
| fine-tuned `best.pt` | DOTAv1 val (ours) | 78.2 | 63.1 |

**Comparison rule:** Only the bottom two rows are directly comparable: same val split, tiling,
and evaluation. The official test row is context only.

**Conclusion:** Fine-tuning is a **near-tie/slight regression** (Δ mAP50 **-0.05pt**,
Δ mAP50-95 **-0.13pt**), not an accuracy gain. The official checkpoint was already trained on
DOTAv1; this measures the marginal value of continuing an already-converged model.

**Limitation:** The baseline raw console log is not committed; four-decimal values and rounded
deltas are preserved historical results. Evidence strength and limits are machine-readable in
[`release/evidence.json`](release/evidence.json).
<!-- /claim:matched-evaluation -->

---

## Why Oriented Bounding Boxes? (quantified, not hand-waved)

<!-- claim:analysis -->
Measured on **28,853 ground-truth objects across 456 DOTAv1 val images**. The overall weighted mean
HBB/OBB area ratio is **1.76×**; the `bridge` mean is **2.43×**.
[Full tables](docs/analysis_results.md) · [Reproduction script](scripts/obb_analysis.py)

**1. Horizontal boxes swallow background.** Area of the axis-aligned box ÷ area of
the true oriented box:

| Object Class | Mean Area Inflation | p90 Inflation | | Control Class (Circular) | Mean Area Inflation |
|---|---:|---:|---|---|---:|
| bridge | 2.43× | 3.71× | | roundabout | 1.02× |
| harbor | 2.15× | 3.66× | | storage tank | 1.00× |
| large vehicle | 2.14× | 3.21× | | | |
| ship | 1.95× | 2.69× | | | |

**Reading:** Elongated, rotated objects inflate by about 2×; circular controls stay near 1.0×.

**2. In dense scenes, horizontal box overlap is mostly "ghost overlap".** Among adjacent
same-class objects with horizontal box IoU ≥ 0.3, the fraction where true oriented box IoU < 0.1:

| Object Class | Pairs with HBB IoU≥0.3 | Ghost Overlap Rate |
|---|---:|---:|
| large vehicle | 810 | 100% |
| ship | 736 | 99% |
| harbor | 43 | 98% |
| small vehicle | 42 | 93% |

**Limit:** This is a ground-truth geometry proxy, not a detector/NMS experiment. It identifies
potential HBB overlap and suppression risk, not actual prediction loss for a specific detector.

**3. Visual source:** Five DOTA-derived comparison crops are excluded from this public release;
only reproducible aggregate data remains.
<!-- /claim:analysis -->

---

## Deployment Benchmark: PyTorch vs. ONNX Runtime vs. TensorRT FP16

<!-- claim:t4-benchmark -->
- **Hardware:** Colab Tesla T4.
- **Input:** batch=1, imgsz=1024.
- **Measurement:** 20 warmups + 100 timed runs per backend.
- **Artifact:** fine-tuned `best.pt` → ONNX/TensorRT FP16; engine bound to its build GPU/TensorRT.
- **Notebook:** [02_benchmark_colab.ipynb](notebooks/02_benchmark_colab.ipynb).

<!-- claim:export-smoke -->
**Export smoke:** On DOTA8 val, PyTorch, ONNX, and TensorRT each reached mAP50 **0.9950** and met
the <1pt parity gate. This checks export consistency, **not full DOTAv1 production certification**;
raw logs are not committed.
<!-- /claim:export-smoke -->

| Inference Backend | File Size (MB) | Mean Latency | Median (p50) | 95th Percentile (p95) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| PyTorch (FP32) | 48.7 | 71.54 ms | 71.49 ms | 74.08 ms | 14.0 |
| ONNX Runtime GPU | 85.3 | 79.09 ms | 79.67 ms | 81.43 ms | 12.6 |
| TensorRT FP16 | 45.0 | 20.22 ms | 20.14 ms | 21.09 ms | 49.4 |

**Result:** Within this comparison, TensorRT FP16 is ~3.5× faster with the smallest file; ONNX Runtime
GPU is slightly slower than PyTorch. ONNX remains the interchange format for `demo/web/` and ONNX
Runtime **Web**.

**Evidence boundary:** These are historical results for the specified T4, batch=1, 1024px, and that
Colab/TensorRT environment. Exact runtime version strings were not preserved; the numbers do not
promise browser, other-hardware, throughput, or production-SLA performance.

**Environment notes:** `torch._dynamo`, HF CDN, and ONNX Runtime CUDA-provider issues are documented
in [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md).
<!-- /claim:t4-benchmark -->

---

## Reproduction Steps

### 1. Local Development (CPU-only, no training or GPU required)

Requirements: Python 3.11 and CPU. Do not copy `.venv` across machines.

```powershell
uv python install 3.11
uv venv --python 3.11
uv sync --frozen --no-install-project
.venv/Scripts/python.exe scripts/repo_check.py
.venv/Scripts/python.exe scripts/release_check.py
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m http.server 8765 --directory demo/web
```

- Demo: open `http://localhost:8765`, inspect the official aerial original, then press **Start Detect**.
  The image and model are not uploaded; first Detect reads the same-origin privacy-sanitized derivative
  and loads the pinned runtime through CDN + SRI.
- Default env: no Torch, CUDA, Ultralytics, or Python ONNX Runtime.
- Broken `.venv`: run `uv venv --clear --python 3.11`; use `.venv/bin/python` on Linux/macOS.
- Windows non-ASCII path: keep `--no-install-project` and call Python directly; see
  [DESIGN_NOTES T6](docs/DESIGN_NOTES.md).
- `tool.uv.link-mode = "copy"`: avoids environment/cache contamination.

The clean-export gate rebuilds a clean HEAD, checks tests/links/privacy/artifacts/browser, then
builds and installs the wheel. Browser smoke runs genuine sample inference and BYOM safety paths in Chromium:

```powershell
.venv/Scripts/playwright.exe install chromium
.venv/Scripts/python.exe scripts/clean_export_check.py
```

### 2. Historical Colab Workflows (not part of this release gate)

These notebooks preserve provenance; **publishing, reviewing, or presenting this RC does not require
running them**. The first two use GPUs, DOTA, and authenticated private Hugging Face writes. Run them
only for an intentional reproduction after permission checks and a cell-by-cell review. Results are
frozen and must not be rerun merely to improve the numbers.

- [01_train_dotav1_a100.ipynb](notebooks/01_train_dotav1_a100.ipynb): original A100 fine-tuning,
  private checkpoint backup, and resume workflow.
- [02_benchmark_colab.ipynb](notebooks/02_benchmark_colab.ipynb): original T4 ONNX/TensorRT FP16
  export, parity, and three-backend benchmark workflow.
- [03_recover_per_class_metrics_colab.ipynb](notebooks/03_recover_per_class_metrics_colab.ipynb):
  validation-only recovery of three class rows from an owner-supplied, checksummed checkpoint; it
  does not download a model or retrain.

**Accepted evidence:** [evidence.json](release/evidence.json) ·
[per_class_metrics.json](docs/per_class_metrics.json) · [training results](docs/training_results.md)

### 3. Browser Real-image Demo + BYOM

- The shared viewport has three compact choices—airfield (default), sports complex, and harbor—of
  public-domain NAIP originals. **Start Detect** then
  lazy-loads pinned ONNX Runtime Web and the same-origin privacy-sanitized YOLO26n-OBB derivative,
  performs genuine inference in the current browser session, and replaces the original with rotated
  polygons, numeric runtime, provenance, and the result table in that same viewport.
- Original/result view switching and confidence/class filters redraw the same cached output without
  another inference. BYOM remains a collapsed advanced path; user-supplied model/image bytes are also
  processed only inside the browser and are never uploaded.
- First Detect or a cache miss fetches pinned JavaScript and WASM assets from jsDelivr, so the demo is
  not zero-network. SHA-384 SRI covers `ort.min.js` only; later WASM requests are outside that script SRI.
- Asset, runtime, integrity, model-contract, inference, output, render, and image-decode failures clear
  stale results, retain a safe original/retry path, and expose fixed recovery copy without local filenames.
  An invalid replacement BYOM model does not displace the last validated session.

---

## Citation and License

- Repository code: **AGPL-3.0-or-later** as declared in `pyproject.toml`; Ultralytics components
  remain subject to their respective AGPL / Enterprise licensing.
- This candidate includes one AGPL YOLO26n-OBB derivative whose private build metadata was removed
  without changing graph or weights, plus three public-domain NAIP aerial derivatives. It contains no DOTA pixels,
  annotations, or derived renders. DOTAv1 training provenance is disclosed; the project implies no
  Ultralytics endorsement and makes no commercial-use clearance claim. User-supplied models/images
  remain subject to their dataset, software, weight, and image-rights terms.
- Artifact hashes, third-party notices, and release gates: see
  [artifact manifest](release/artifact-manifest.json), [third-party notices](THIRD_PARTY_NOTICES.md),
  and [release checklist](RELEASE_CHECKLIST.md).

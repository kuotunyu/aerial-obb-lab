# Project plan — historical execution record

> This document preserves the plan and completion criteria that guided the project. Phases 0–7
> are complete. Use the root README for current installation and reproduction instructions.

## Objective

Build an end-to-end oriented-bounding-box portfolio project around YOLO26 and DOTAv1:

1. validate the complete OBB workflow at smoke-test scale;
2. fine-tune on a reproducible DOTAv1 split in Colab;
3. compare against the official baseline under matched conditions;
4. quantify why OBBs matter in dense aerial scenes;
5. export and benchmark PyTorch, ONNX Runtime, and TensorRT;
6. provide local and public interactive demos; and
7. document results, limitations, and engineering decisions honestly.

## Execution architecture

- **Local development**: Python 3.11, lightweight analysis/test dependencies, documentation,
  static demo serving, and optional Gradio inference.
- **Colab A100**: DOTAv1 tiling, training, validation, and checkpoint persistence.
- **Colab T4**: ONNX/TensorRT export, parity validation, and latency benchmark.
- **Hugging Face Hub**: model artifacts and resumable checkpoints. DOTA-derived tiles are not
  published because the dataset is restricted to academic use; recovery does not depend on a
  mutable private cache.
- **Hugging Face static Space**: public browser-side ONNX Runtime Web demo with no hosted Python
  compute.

Local virtual environments, datasets, caches, weights, and training runs are intentionally not
versioned. Common placeholders used by the scripts and notes are `<project-root>`,
`<dataset-dir>`, and `<hf-cache-dir>`.

## Repository layout

```text
<project-root>/
├── src/obbkit/            # analysis, visualization, and HF checkpoint helpers
├── scripts/               # smoke test, analysis, and repository preflight
├── tests/                 # CPU-only unit tests
├── notebooks/             # self-contained Colab training, eval recovery, and benchmark notebooks
├── demo/
│   ├── app.py             # optional local Gradio demo using fine-tuned weights when available
│   ├── space/             # server-side CPU reference implementation; not deployed
│   └── space-static/      # deployed browser-only ONNX Runtime Web implementation
├── assets/                # HBB-vs-OBB comparison images
└── docs/                  # results, model card, plan, and design notes
```

The public static demo deliberately uses the official lightweight `yolo26n-obb` checkpoint to
keep the browser download near 10 MB. The fine-tuned `yolo26m-obb` checkpoint is evaluated,
exported, benchmarked, and published separately on the model Hub.

## Completed phases

| Phase | Scope | Execution | Completion evidence |
|---|---|---|---|
| 0 | Scaffold, license, environment, HF helpers | local | project imports and OBB prediction established |
| 1 | DOTA8 train → val → predict → resume → HF push → ONNX | original workstation | smoke workflow completed end to end |
| 2 | DOTAv1 fine-tune | Colab A100 | 28/30 epochs, early stop, best epoch 13, checkpoints published |
| 3 | Fine-tuned vs official baseline | Colab | matched-split metrics and per-class analysis recorded |
| 4 | Quantitative HBB-vs-OBB analysis | local | 28,853 objects, 456 images, five rendered comparisons |
| 5 | Export and benchmark | Colab T4 | parity passed; three-backend latency table recorded |
| 6 | Interactive demos | local + HF | local Gradio reference and deployed static browser Space |
| 7 | Documentation and model card | local + HF | bilingual README, results notes, model card, live links |

## Key decisions reached during implementation

- Use `yolo26m-obb` for the experiment: materially lighter than l/x while retaining most of
  their reported DOTAv1 accuracy.
- Tile DOTAv1 with `[0.8, 1.2]` rather than the initial three-scale plan after measuring the
  data-volume and throughput trade-off.
- Persist checkpoints to the Hub every two epochs so a temporary Colab runtime can resume.
- Treat exported-model parity as a gate: ONNX and TensorRT must stay within one mAP point of
  PyTorch before latency numbers are accepted.
- Report the near-tie against the already-DOTAv1-trained baseline rather than reframing it as an
  improvement.
- Replace the planned server-side public Space with browser-side inference after the hosting
  option available during implementation required a paid backend.
- Keep local development portable: Python 3.11 is pinned, GPU/ML dependencies are optional, and
  the default environment does not depend on a specific CUDA driver or workstation.

The detailed reasoning and incident history are retained in [DESIGN_NOTES.md](DESIGN_NOTES.md).

## Completion criteria and current evidence

- **Training**: resumable notebook, best/last checkpoints, results CSV, and model artifacts exist.
- **Evaluation**: baseline and fine-tuned checkpoints were measured on the same val split.
- **Analysis**: generated Markdown/JSON tables and five visual comparisons are versioned.
- **Deployment**: exported-model parity passed and TensorRT FP16 reached the recorded T4 result.
- **Demo**: the local Gradio reference was tested; the static Space is public and browser-side.
- **Reproducibility**: notebooks are self-contained; a fresh local clone can install the
  lightweight environment and run `scripts/repo_check.py` plus CPU-only tests without a GPU.

## Reproducibility boundaries

- DOTAv1 data is not redistributed by this repository. Reproduction must follow the dataset's
  academic-use license and the notebook's download/tiling steps.
- TensorRT engines are tied to the GPU architecture and TensorRT build environment; reproduce the
  engine from `best.pt` rather than treating an engine binary as portable.
- The official published test-split scores and this project's tile-level val scores use different
  evaluation pipelines and are not directly comparable.
- Three fine-tuned per-class values (`plane`, `ship`, and `storage tank`) were not captured before
  the original Colab session ended. The checksum-, manifest-, and historical-consistency-gated
  validation-only workflow completed and was reviewed on 2026-07-15 with Ultralytics 8.4.93. It
  recovered 0.952147 / 0.862352, 0.909448 / 0.762681, and 0.850699 / 0.716696 respectively
  (mAP50 / mAP50-95), with aggregate 0.781614 / 0.631422. The accepted complete result and
  provenance are stored in [CSV](per_class_metrics.csv), [JSON](per_class_metrics.json), and
  [training_results.md](training_results.md); no rerun is required.
- The recovery run initially printed `FAIL` only because the first gate required raw label-line
  counts to equal loader-validated instance counts. Ultralytics removes duplicate label rows and
  validates labels before metric counting, so the saved bundle was reviewed with `0 < instances
  <= raw label lines`. The reproducibility notebook now additionally compares metric instance and
  image counts exactly against the loader-validated `val.cache`; raw counts are only an upper-bound
  diagnostic. All checkpoint, dataset, split, class-order, aggregate, and historical-class checks
  passed; the initial status was a false negative rather than invalid evidence.

## Optional future work

- Adapt to a genuinely new aerial domain instead of continuing to train a converged DOTAv1 model.
- Quantize or train a lightweight fine-tuned model specifically for the browser demo.
- Add hosted CI after the source repository is published.

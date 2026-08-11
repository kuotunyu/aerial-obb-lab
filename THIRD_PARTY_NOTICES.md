# Third-Party Notices and Use Boundaries

This repository combines original project code with upstream software, an Ultralytics-derived
model export, and visuals derived from DOTA imagery. Project code is declared
`AGPL-3.0-or-later` in `pyproject.toml`; that declaration does not replace the independent terms
below. This document is a release inventory, not legal advice.

## Bundled artifacts

### Ultralytics YOLO26 OBB model export

`demo/space-static/yolo26n-obb.onnx` is an export of the official `yolo26n-obb` model. Release
hardening reserialized its protobuf to remove a stale absolute exporter path. ONNX checker passed,
and the graph SHA-256 is identical before and after sanitation; no inference was run. Its upstream
and local size, SHA-256, source revision, and restrictions are recorded in
[`release/artifact-manifest.json`](release/artifact-manifest.json). Ultralytics publishes its
software and models under an [AGPL-3.0 or Enterprise licensing route](https://www.ultralytics.com/license).
This release uses the AGPL route and makes no Enterprise-license grant.

The model was trained on DOTAv1. This project therefore treats the bundled export and the external
fine-tuned weights as academic/non-commercial artifacts unless the relevant rights holders confirm
otherwise. AGPL compliance and training-data/image rights are separate requirements; satisfying
one does not satisfy the other.

### DOTA-derived comparison images

The five JPEG files under `assets/` were rendered from DOTAv1 validation imagery and annotations.
The [DOTA dataset page](https://captain-whu.github.io/DOTA/dataset) states that its images and
annotations are for academic use only and prohibits commercial use. It also says Google Earth
images remain subject to Google Earth terms. The repository does not assert that these underlying
image rights have been cleared for commercial redistribution.

## Runtime and development dependencies

These dependencies are referenced or installed but are not vendored as source code here. Their
own licenses and notices continue to apply.

| Component | Role | Upstream license/source |
|---|---|---|
| Ultralytics 8.4.93 | Training, validation, export, optional demos | [AGPL-3.0 / Enterprise](https://github.com/ultralytics/ultralytics/tree/v8.4.93) |
| PyTorch / TorchVision | Optional local training and inference runtime | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| ONNX / ONNXSlim | Optional export and graph tooling | [Apache-2.0](https://github.com/onnx/onnx/blob/main/LICENSE), [MIT](https://github.com/inisis/OnnxSlim/blob/main/LICENSE) |
| ONNX Runtime Web 1.20.1 | Browser WASM inference, fetched from jsDelivr | [MIT](https://github.com/microsoft/onnxruntime/blob/v1.20.1/LICENSE) |
| ONNX Runtime | Reference server-side demo | [MIT](https://github.com/microsoft/onnxruntime/blob/v1.20.1/LICENSE) |
| Gradio | Optional local/reference UI | [Apache-2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE) |
| Hugging Face Hub | Optional artifact transfer | [Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| NumPy | Numeric arrays and reference parity | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| OpenCV | Image I/O and analysis | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| Pillow | Image support | [HPND](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| Shapely | Polygon geometry | [BSD-3-Clause](https://github.com/shapely/shapely/blob/main/LICENSE.txt) |
| Jupytext / pytest / Hatchling | Notebook sync, tests, package build | [MIT](https://github.com/mwouts/jupytext/blob/main/LICENSE), [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [MIT](https://github.com/pypa/hatch/blob/master/LICENSE.txt) |

Package-specific transitive notices are available in each installed distribution and lockfile.

## External Hugging Face artifacts

The model repository and Space are external publications, not contents automatically relicensed by
this source repository. Their anonymously verified revisions and metadata are recorded in
[`release/evidence.json`](release/evidence.json). PyTorch `.pt` files use pickle-based loading;
only load a checkpoint from a trusted, checksum-verified source.

## Owner actions before broader use

Before commercial use, closed-source integration, or redistribution outside academic/portfolio
review, the owner must:

1. obtain written DOTA/underlying-image permission covering the intended use and redistributed
   visuals/weights;
2. choose and document either full AGPL-3.0 compliance or an applicable Ultralytics Enterprise
   license;
3. review the intended deployment and artifacts with qualified legal counsel; and
4. re-check the exact external model/Space revisions and third-party terms at release time.

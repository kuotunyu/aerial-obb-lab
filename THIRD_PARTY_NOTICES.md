# Third-Party Notices and Use Boundaries

This public candidate distributes original project code, documentation, aggregate evidence, one
official sample image, and one privacy-sanitized AGPL model derivative. It distributes no DOTA image,
annotation, or DOTA-derived raster render. Project code is declared `AGPL-3.0-or-later` in
`pyproject.toml`; that declaration does not replace independent upstream or user-supplied-artifact
terms. This document is a release inventory, not legal advice or a warranty against liability.

## Bundled artifacts

The exact bundled inventory and hashes are recorded in
[`release/artifact-manifest.json`](release/artifact-manifest.json): the official Ultralytics
`boats.jpg` sample, the privacy-sanitized YOLO26n-OBB derivative, its complete unmodified
AGPL-3.0-only license, and the self-hosted OFL display font. The derivative was modified on
2026-08-31 by removing one non-inference metadata entry; the committed sanitization receipt records
that its graph and weights remain structurally unchanged. Its training provenance is DOTAv1.

The modification record is
[`demo/web/third_party/yolo26n-obb-privacy-sanitization.json`](demo/web/third_party/yolo26n-obb-privacy-sanitization.json),
and the sanitizer source is [`scripts/sanitize_demo_model.py`](scripts/sanitize_demo_model.py).
The public demo and screenshot are integration evidence, not ground truth, accuracy, evaluation, or
latency evidence. This project is not endorsed by Ultralytics and makes no commercial-use clearance
claim. Ultralytics offers separate
[AGPL-3.0 and Enterprise routes](https://www.ultralytics.com/license); this repository grants no
Enterprise license. DOTA images and annotations are academic-use-only and may also carry underlying
image-source restrictions. AGPL compliance and data/image rights are separate obligations.

Demo and BYOM inference are not zero-network modes: first Detect or selecting a BYOM model lazy-loads pinned ONNX Runtime Web
JavaScript and WASM assets from jsDelivr on a cache miss. SHA-384 SRI covers `ort.min.js` only, not
the WASM assets subsequently fetched by that runtime. The official derivative is fetched only from
the same origin after Detect; model and image processing remains local to the browser. User-supplied
models and images remain subject to their own provenance, software, dataset, weight, and image-rights terms.

## Runtime and development dependencies

These dependencies are referenced or installed but are not vendored as source code here. Their
own licenses and notices continue to apply.

| Component | Role | Upstream license/source |
|---|---|---|
| Ultralytics 8.4.93 | Historical training, validation, and export workflows | [AGPL-3.0 / Enterprise](https://github.com/ultralytics/ultralytics/tree/v8.4.93) |
| PyTorch / TorchVision | Historical training and export runtime | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| ONNX / ONNXSlim | Optional export and graph tooling | [Apache-2.0](https://github.com/onnx/onnx/blob/main/LICENSE), [MIT](https://github.com/inisis/OnnxSlim/blob/main/LICENSE) |
| ONNX Runtime Web 1.20.1 | Browser WASM inference, fetched from jsDelivr | [MIT](https://github.com/microsoft/onnxruntime/blob/v1.20.1/LICENSE) |
| Hugging Face Hub | Historical notebook artifact transfer; not used by release gates or demos | [Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| NumPy | Numeric arrays and reference parity | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| OpenCV | Image I/O and analysis | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| Pillow | Image support | [HPND](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| Shapely | Polygon geometry | [BSD-3-Clause](https://github.com/shapely/shapely/blob/main/LICENSE.txt) |
| Jupytext / pytest / Hatchling | Notebook sync, tests, package build | [MIT](https://github.com/mwouts/jupytext/blob/main/LICENSE), [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [MIT](https://github.com/pypa/hatch/blob/master/LICENSE.txt) |
| Playwright | Headless real-demo and BYOM browser smoke | [Apache-2.0](https://github.com/microsoft/playwright-python/blob/main/LICENSE) |
| IBM Plex Sans Condensed 2.0.0 | Self-hosted display type for the browser workbench | [SIL Open Font License 1.1](demo/web/fonts/IBM-Plex-OFL.txt) |

Package-specific transitive notices are available in each installed distribution and lockfile.

## Historical external artifacts

Historical model/Space revision and checksum facts are retained without advertising an owner-hosted
download location. They are evidence records, not redistributed contents or license grants.
PyTorch `.pt` files use pickle-based loading; only load a trusted, checksum-verified checkpoint.

## Before broader use

For any future commercial use, closed-source integration, or redistribution of DOTA-derived
weights/visuals, the owner must separately obtain the relevant permissions, satisfy either the
applicable AGPL obligations or an Enterprise agreement, and seek qualified legal review when needed.

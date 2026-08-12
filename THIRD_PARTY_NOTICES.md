# Third-Party Notices and Use Boundaries

This public candidate distributes original project code, documentation, aggregate evidence, and
synthetic fixtures. It deliberately distributes no model binary, trained weight, DOTA image or
annotation, or DOTA-derived raster render. Project code is declared `AGPL-3.0-or-later` in
`pyproject.toml`; that declaration does not replace independent upstream or user-supplied-artifact
terms. This document is a release inventory, not legal advice or a warranty against liability.

## Bundled artifacts

There are no bundled third-party model or dataset artifacts. The empty distributable inventory and
the checksums of six excluded historical artifacts are recorded in
[`release/artifact-manifest.json`](release/artifact-manifest.json). Those records are audit metadata,
not permission or distribution of the files.

The browser and Python demos are bring-your-own-model tools. A model selected by a user is not part
of this repository and remains subject to its own provenance, software license, dataset terms, and
weight terms. Ultralytics offers separate
[AGPL-3.0 and Enterprise routes](https://www.ultralytics.com/license); this repository grants no
Enterprise license. DOTA images and annotations are academic-use-only and may also carry underlying
image-source restrictions. AGPL compliance and data/image rights are separate obligations.

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
| Hugging Face Hub | Historical notebook artifact transfer; not used by release gates or demos | [Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| NumPy | Numeric arrays and reference parity | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| OpenCV | Image I/O and analysis | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| Pillow | Image support | [HPND](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| Shapely | Polygon geometry | [BSD-3-Clause](https://github.com/shapely/shapely/blob/main/LICENSE.txt) |
| Jupytext / pytest / Hatchling | Notebook sync, tests, package build | [MIT](https://github.com/mwouts/jupytext/blob/main/LICENSE), [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [MIT](https://github.com/pypa/hatch/blob/master/LICENSE.txt) |
| Playwright | Headless synthetic browser smoke | [Apache-2.0](https://github.com/microsoft/playwright-python/blob/main/LICENSE) |

Package-specific transitive notices are available in each installed distribution and lockfile.

## Historical external artifacts

Historical model/Space revision and checksum facts are retained without advertising an owner-hosted
download location. They are evidence records, not redistributed contents or license grants.
PyTorch `.pt` files use pickle-based loading; only load a trusted, checksum-verified checkpoint.

## Owner actions before broader use

Authenticated publication tasks are listed in [`docs/OWNER_ACTIONS.md`](docs/OWNER_ACTIONS.md).
For any future commercial use, closed-source integration, or redistribution of DOTA-derived
weights/visuals, the owner must separately obtain the relevant permissions, satisfy either the
applicable AGPL obligations or an Enterprise agreement, and seek qualified legal review when needed.

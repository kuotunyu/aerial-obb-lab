# Third-Party Notices and Use Boundaries

This public candidate distributes original project code, documentation, aggregate evidence, and
synthetic fixtures. It deliberately distributes no model binary, trained weight, DOTA image or
annotation, or DOTA-derived raster render. Project code is declared `AGPL-3.0-or-later` in
`pyproject.toml`; that declaration does not replace independent upstream or user-supplied-artifact
terms. This document is a release inventory, not legal advice or a warranty against liability.

## Bundled artifacts

There are no bundled third-party model or dataset artifacts. The distributable inventory contains
only one self-hosted OFL display font; its hash and the checksums of six excluded historical
artifacts are recorded in [`release/artifact-manifest.json`](release/artifact-manifest.json). Those
records are audit metadata, not permission or distribution of the files.

The committed Synthetic Showcase SVG and its fixed result data are authored, first-party test and
presentation fixtures, not third-party artifacts. They contain no DOTA pixels and do not change the
one-entry bundled third-party inventory. The code-only exclusions still cover model binaries,
trained weights, DOTA images and annotations, DOTA-derived renders, and owner-private artifacts.

The browser demo is a bring-your-own-model tool. A model selected by a user is not part
of this repository and remains subject to its own provenance, software license, dataset terms, and
weight terms. Ultralytics offers separate
[AGPL-3.0 and Enterprise routes](https://www.ultralytics.com/license); this repository grants no
Enterprise license. DOTA images and annotations are academic-use-only and may also carry underlying
image-source restrictions. AGPL compliance and data/image rights are separate obligations.

BYOM inference is not a zero-network mode: selecting a model lazy-loads pinned ONNX Runtime Web
JavaScript and WASM assets from jsDelivr on a cache miss. SHA-384 SRI covers `ort.min.js` only, not
the WASM assets subsequently fetched by that runtime. Model and image bytes remain local to the
browser. The Synthetic Showcase neither loads this external runtime nor performs model inference.

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
| Playwright | Headless synthetic browser smoke | [Apache-2.0](https://github.com/microsoft/playwright-python/blob/main/LICENSE) |
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

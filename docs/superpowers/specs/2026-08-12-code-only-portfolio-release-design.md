# Code-Only Portfolio Release Design

**Date:** 2026-08-12  
**State:** Approved design for the next local release-hardening pass  
**Target branch:** `portfolio/obb-v1.0-release-hardening`

## Purpose

Convert the existing local `v1.0.0rc1` candidate into a public GitHub portfolio release that
retains its engineering and evidence value without redistributing DOTA-derived images, datasets,
annotations, trained weights, or model binaries. The public repository remains an open-source
AGPL-3.0-or-later project. It is not presented as a paper, commercial product, production
certification, or grant of rights to third-party data or models.

This design minimizes practical publication risk but is not a guarantee of zero legal exposure
and is not legal advice. DOTA's official terms restrict its images and annotations to academic
purposes and prohibit commercial use. Ultralytics identifies its code and trained/fine-tuned
models as covered by AGPL-3.0 unless a separate Enterprise license applies.

## Chosen approach

Use a **code-only, bring-your-own-model (BYOM) portfolio release**.

The repository will distribute source code, documentation, aggregate evaluation results,
notebook sources without outputs, synthetic fixtures, and release tooling. It will not distribute
model files or DOTA-derived raster images. The browser demo will accept an ONNX model selected
from the visitor's local machine and will never fetch a model from this repository or Hugging
Face.

Two rejected approaches are documented for clarity:

1. **Automatic external model download:** visually convenient, but adds availability, provenance,
   privacy, and licensing ambiguity to the public demo.
2. **Continue bundling the ONNX and DOTA-derived JPEGs with notices:** easiest technically, but it
   preserves the redistribution exposure this release is intended to remove.

## Public artifact policy

### Included

- Repository-owned Python and JavaScript source.
- Stripped `.ipynb` files and their synchronized Jupytext `.py` sources.
- Aggregate JSON/CSV metrics and bounded historical benchmark transcriptions.
- Synthetic JSON and SVG browser fixtures.
- CI workflows, tests, package metadata, citation metadata, license, notices, and release gates.
- Documentation that accurately attributes DOTA and Ultralytics and states the evidence limits.

### Excluded from Git and release archives

- DOTA source images, annotations, archives, tiles, and dataset directories.
- The five existing DOTA-derived HBB-versus-OBB comparison JPEGs.
- All trained, fine-tuned, exported, or compiled model binaries, including `.pt`, `.onnx`,
  `.engine`, `.torchscript`, `.tflite`, and `.mlpackage` files.
- Training runtime directories, caches, secrets, private notes, and interview material.

The former bundled artifacts remain described in the manifest as deliberately excluded historical
artifacts, including their accepted hashes and provenance. Their files are not present in the
public tree or clean export.

## Browser demo design

The static demo keeps separate controls for:

1. A local `.onnx` model file.
2. A local input image.
3. Confidence and class filters.
4. Detection execution and results.

Selecting a model reads it through the browser `File` API, converts its `ArrayBuffer` to a
`Uint8Array`, and passes those bytes to `ort.InferenceSession.create`. The application contains no
model URL, Hugging Face endpoint, implicit download, or network fallback. Replacing a selected
model disposes the previous session when the runtime supports disposal and resets model status.

Detection stays disabled until both a model session and image are ready. Model-read, runtime-load,
schema, and inference failures are shown in the existing status region and never trigger a remote
fallback. The established output contract remains strict: named `output0`, shape `[1,N,7]`, with
`[cx, cy, width, height, confidence, class_id, angle_radians]` in letterboxed coordinates.

ONNX Runtime Web may continue loading from its version-pinned upstream CDN; that runtime dependency
is disclosed and is distinct from distributing or automatically fetching a model. The synthetic
headless test intercepts the runtime script so CI does not perform external requests.

## Local Python demo design

The optional Gradio/reference paths no longer download weights automatically and no longer fall
back to official weight names. A user must provide an explicit local path, such as through
`MODEL_PATH`. Missing, unreadable, or unsupported model paths fail with a clear message before
inference. The public repository therefore demonstrates integration code without distributing or
silently acquiring a model.

## Evidence and documentation

The matched negative result, DOTA8 export-smoke result, T4 latency, and geometry analysis remain
because they are aggregate factual records rather than redistributed dataset samples. Existing
scope limits continue to apply:

- Fine-tuning is a near-tie/slight regression of about `-0.05pt` mAP50 and `-0.13pt` mAP50-95.
- DOTA8 `0.9950` is export-smoke parity, not full DOTAv1 production certification.
- `20.22 ms / 49.4 FPS` applies only to the recorded Tesla T4, batch-1, 1024-pixel environment.
- The BYOM browser UI carries no fine-tuned-medium accuracy or T4 latency claim.
- Geometry measurements describe overlap risk, not observed detector or NMS losses.

Public-facing links to the owner's current Hugging Face model repository and Space are removed.
Documentation may link to official DOTA, Ultralytics, ONNX Runtime, and AGPL sources. Historical
workflow descriptions may name `best.pt` as an output, but must not provide an owner-hosted download
URL or imply that a model is included.

The README replaces the DOTA-derived image embed with a concise statement that comparison renders
were excluded from the public release. The quantitative tables remain the visual evidence.

## Manifest and release gates

`release/artifact-manifest.json` becomes a public-distribution manifest with:

- An empty `bundled_third_party_artifacts` list.
- An `excluded_historical_artifacts` list recording the former paths, hashes, provenance, and
  reason for exclusion.
- A policy declaring that model binaries and DOTA-derived raster images must not appear in the
  committed tree or clean export.

The release verifier will enforce:

- No committed or archived file with a forbidden model suffix.
- No committed file under dataset/training runtime paths.
- No known DOTA-derived asset path or hash.
- No owner Hugging Face model/Space URL in public-facing documents or demo source.
- No model URL or fetch fallback in the static application.
- Manifest excluded entries are absent, and any bundled artifact entry is present and hash-bound.

The repository preflight will require the BYOM controls and static source files but no ONNX file.
The clean-export required-member inventory will remove the bundled model and require the updated
manifest and browser fixtures.

## Testing strategy

Every release-blocking behavior change follows red-green TDD:

1. Add release-policy tests that fail while restricted assets and owner HF links are still tracked.
2. Add browser contract tests that require model bytes and reject embedded model URLs.
3. Add headless browser assertions for model-file selection, image selection, enabled-state
   transitions, decoding, drawing, and result rendering using synthetic in-memory model bytes.
4. Add Python demo tests for explicit `MODEL_PATH` behavior without importing an ML runtime.
5. Update manifest, repository, package, privacy, and clean-export tests.

Final verification runs in CPU-only mode and includes the complete pytest suite, repository gate,
release gate, browser smoke, package build, clean committed export rebuild, isolated wheel install,
Git identity/trailer audit, and remote audit. It does not train, validate a model, initialize CUDA,
build TensorRT, download DOTA, or perform model inference.

## Commit structure

Use small English commits without amend, squash, rebase, or co-author trailers:

1. `docs: define code-only portfolio release`
2. `fix: exclude restricted release artifacts`
3. `feat: require user-supplied browser model`
4. `fix: require explicit local demo model`
5. `docs: publish code-only portfolio guidance`
6. `test: verify code-only clean export`

Only explicit paths are staged; `git add -A` is prohibited. Author and committer remain
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`.

## Owner-only external actions

The local implementation does not modify any remote service. After local verification, the owner
must perform these actions:

1. Back up the current Hugging Face model repository and Space if desired.
2. Set the existing Hugging Face model repository to private because it hosts `best.pt`.
3. Set the existing Hugging Face Space to private because it hosts the bundled ONNX, or replace its
   contents later with the verified BYOM static folder before making it public again.
4. Create a new empty GitHub repository without generated README, license, or `.gitignore` files.
5. Add that repository as the first remote and push only after reviewing the local code-only diff.
6. Require the Ubuntu, Windows, and browser-smoke GitHub Actions jobs to pass before merge or tag.
7. Create a tag or GitHub Release only after hosted CI succeeds.

These actions require the owner's authenticated sessions and are intentionally not automated by
the local release process.

## Completion criteria

The conversion is complete when the working tree is clean, every restricted artifact is absent
from `HEAD` and the clean export, no public-facing owner HF artifact link remains, BYOM browser and
local demos fail safely without a user model, all CPU gates pass, a new clean-export hash is
recorded, author/committer/trailer checks pass, and no remote mutation has occurred.

# YOLO26 OBB v1.0 Release-Hardening Design

## Status and scope

This document records the approved feature-frozen design for a local, unpushed
`1.0.0rc1` release candidate. It preserves the existing A100 and T4 results as
historical evidence. It does not authorize retraining, a new DOTA split, full
validation, GPU inference, TensorRT builds, remote repository changes, tags, or
publishing.

The release candidate is accepted only when the named branch is clean, every new
commit has the approved identity, no collaboration trailer is present, and a clean
archive made from `HEAD` can pass the CPU-only gates without weights, DOTA, Torch,
CUDA, Hugging Face credentials, or any other secret.

## Evidence architecture

Historical measurements and reproducible release checks are deliberately separate.
Committed JSON records the observed A100/T4 values, their provenance, and their
limits. A standard-library verifier parses the English README, Traditional Chinese
README, model card, result documents, evidence records, and artifact manifest so a
number cannot silently drift between them.

The matched evaluation remains a slightly negative result: approximately
`-0.05` percentage point mAP50 and `-0.13` percentage point mAP50-95. The raw
full-precision baseline output was not preserved, so the verifier treats those two
deltas as historical observations and says so; it must not manufacture extra
precision from rounded tables. DOTA8 `0.9950` values are export smoke parity only.
T4 latency is a single batch-1, 1024 px, 20-warmup/100-run observation. The exact
TensorRT and ONNX Runtime versions were not retained and remain an explicit limit.

## Artifact, privacy, and licensing architecture

The artifact manifest inventories every redistributed binary or DOTA-derived image
with its SHA-256, byte size, origin, intended role, and governing restriction. The
fine-tuned checkpoint stays external at a pinned Hugging Face revision. DOTA and
training weights stay out of Git. The bundled browser ONNX file may remain because it
is the deployed static product path, but embedded absolute build paths are removed
without changing the graph.

Repository checks reject unexpected weights, datasets, archives, oversized files,
notebook outputs, execution counts, secret-shaped text, personal paths, and private
material. `notes.private.md` is checked only for Git ignore/tracking state; its
contents are never consumed. Third-party notices distinguish the repository's AGPL
license from DOTA's academic-only restriction and Ultralytics' own licensing policy.
External legal clearance is recorded as an owner action, not guessed by automation.

## Browser pipeline architecture

The static demo keeps the official lightweight `yolo26n-obb` ONNX model and gains a
small pure JavaScript core for letterbox geometry, RGBA-to-RGB/CHW normalization,
`[N,7]` decoding, class filtering, confidence ordering, angle conversion, and rotated
corner construction. A Python reference implements the same public contract. Both
runtimes consume a hand-checked synthetic JSON fixture, so CI can compare behavior
without downloading DOTA or running inference.

The actual browser page imports that tested core. A local HTTP and headless Chromium
smoke verifies rendering, controls, module loading, and absence of browser console
errors. It does not claim model accuracy or T4 performance.

## CI, package, and clean-export architecture

GitHub Actions uses Python 3.11 on Ubuntu and Windows, installs only locked default
and dev dependencies, and runs unit tests, repository/release checks, JavaScript
tests, and package builds. No GPU stack or credential is installed. The package
version is `1.0.0rc1`; wheel and source distributions are inspected and the wheel is
installed into an isolated environment for an import smoke.

The final gate archives only committed `HEAD` files into a fresh ASCII-only temporary
directory, creates a fresh CPU environment, and reruns package, test, link, privacy,
artifact, JavaScript, static HTTP, and browser checks there. Docker is not introduced:
the existing product is a static browser app and Python library, so a service image
would add no release value.

## Release documentation and owner boundary

`CHANGELOG.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md`, and a release checklist form
the publication handoff. Remaining owner actions include legal review of DOTA-derived
assets/weights, review of upstream Ultralytics terms for the intended use, creating
and describing the GitHub repository, enabling CI, and intentionally syncing updated
HF cards/demo files later. This hardening run performs none of those remote writes.

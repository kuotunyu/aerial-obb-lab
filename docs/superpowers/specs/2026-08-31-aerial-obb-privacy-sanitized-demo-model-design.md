# Aerial OBB Privacy-sanitized Demo Model Design

- **Date:** 2026-08-31
- **Repository:** `kuotunyu/aerial-obb-lab`
- **Design branch:** `feat/pages-live-real-image-demo`
- **Design parent:** `8116e79399af3523c9972216fdf2287e586098db`
- **Status:** Approved for implementation planning
- **Scope:** preserve the approved real-image-first browser demo by deriving one privacy-safe,
  inference-equivalent AGPL ONNX from the exact official Ultralytics release model

## Decision and supersession

The owner approved this design after the exact official `yolo26n-obb.onnx` reached the mandatory
Task 2 privacy stop condition. The source model passed source, redirect, length, digest, media, license,
and receipt verification, but a read-only ONNX parse found exactly one absolute-user-path-shaped value
in `ModelProto.metadata_props[0].value`. The value is deliberately never copied into this specification,
logs, reports, screenshots, commits, receipts, UI, or console output.

This design supersedes only these requirements in:

- `docs/superpowers/specs/2026-08-31-aerial-obb-live-real-image-demo-design.md`; and
- `docs/superpowers/plans/2026-08-31-aerial-obb-live-real-image-demo.md`:

1. the public model is no longer the unmodified upstream ONNX byte stream;
2. the public model path becomes
   `demo/web/models/yolo26n-obb-privacy-sanitized.onnx`;
3. the public provenance and third-party notices describe a modified AGPL derivative; and
4. model admission includes deterministic sanitization and source-versus-derivative equivalence gates.

The prior design's user journey, real `boats.jpg`, lazy loading, browser-only inference, BYOM path,
decode/filter/corner/render pipeline, accessibility requirements, release separation, and remote-gate
prohibitions remain binding unless this document explicitly refines them.

This document authorizes design documentation only. It does not authorize product code, asset publication,
push, pull request, merge, workflow dispatch, Pages mutation or deployment, GitHub About, Hugging Face,
release, tag, visibility, or worktree cleanup.

## Problem and verified technical fact

Publishing the exact official binary would expose a known absolute path embedded by an upstream export
environment. Relaxing the artifact scanner would contradict the repository's public privacy boundary.
Changing the desired one-click experience back to BYOM-only would fail the owner's product requirement.

The verified model facts make a surgical derivative possible:

- the immutable upstream URL is
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx`;
- the admitted upstream body is exactly 10,207,250 bytes with SHA-256
  `02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38`;
- the source is one ONNX `ModelProto`, IR version 9;
- it has no external tensor files;
- exactly one absolute-user-path match exists;
- that match is a string value in `model.metadata_props[0].value`, not tensor data, a node attribute,
  graph input/output, operator definition, or external-data location; and
- ONNX defines model metadata as additional descriptive information and separately defines the graph and
  serialized model structure.

The design therefore removes the complete offending metadata entry, not a substring within it, and proves
that every non-metadata protobuf field is unchanged.

## Considered approaches

### Selected: deterministic single-entry metadata sanitization

Acquire the exact pinned upstream model outside every repository, verify its identity, remove exactly the
one offending `metadata_props` entry through the official ONNX protobuf API, and publish only the resulting
derivative after structural, privacy, checker, reproducibility, and browser-inference equivalence gates pass.

This is the smallest change that preserves the requested YOLO26 one-click demo. It does not retrain,
quantize, optimize, simplify, rename graph values, change tensor bytes, change opsets, or rewrite the graph.

### Rejected: select another official OBB model

Another model could avoid this particular metadata value, but it would change the project's YOLO26 identity,
inference characteristics, provenance, digest, and admission evidence. It would also require a new source and
license investigation without proving that the replacement is privacy-clean.

### Rejected: publish the exact upstream model and waive the finding

The finding is a genuine structured metadata value, not a random binary coincidence. Allowlisting it would
knowingly publish an absolute path and weaken the privacy gate for the artifact most likely to contain build
environment details.

### Rejected: return to sample-image-plus-BYOM

This keeps models out of the Pages artifact, but a visitor with no local model still cannot press Detect and
experience real inference. It does not meet the approved primary journey.

## Components and responsibilities

### Exact upstream acquisition boundary

`scripts/prepare_demo_assets.py` remains the only network acquisition boundary. It accepts no alternate
model or image override. It downloads the exact official source into a newly created repository-external
review root and verifies:

- immutable source URL and allowed redirect hosts;
- expected upstream byte length and source SHA-256;
- ONNX media detection;
- the exact official image and license identities; and
- a closed, path-safe receipt with fixed diagnostics.

The source ONNX never enters a Git worktree, Pages staging tree, screenshot directory, test fixture, archive,
or commit. A failed sanitization leaves no derivative in the repository.

The existing Windows Unicode defect is corrected at this boundary: Git subprocess output is decoded as
UTF-8 explicitly, with a fixed safe error on decode failure. No command output, worktree path, or exception
is copied to user-visible diagnostics.

### Sanitizer

Add `scripts/sanitize_demo_model.py` backed by exactly locked `onnx==1.22.0` and its locked protobuf runtime.
The library is a development/release tool dependency only; it is not shipped to the browser.

The public callable operation consumes an admitted source file and writes into a contained, newly created
staging directory outside every repository. It performs this closed sequence:

1. recompute and require the exact upstream source SHA-256;
2. parse one in-memory `ModelProto` with external-data loading disabled;
3. recursively inspect protobuf string and byte fields without printing values;
4. require exactly one absolute-user-path match and require its field identity to be
   `ModelProto.metadata_props[0].value`;
5. remove the entire matched `StringStringEntryProto` from `metadata_props`;
6. preserve every other metadata entry in its original order and byte value;
7. serialize once with deterministic protobuf serialization;
8. parse the result again and run `onnx.checker.check_model`;
9. require zero forbidden path, secret, URL-query, stack, or private-marker matches in the output bytes; and
10. atomically publish the staged derivative and transformation receipt only after all gates pass.

Zero matches, more than one match, a match in any other field, parse/check failure, structural drift,
non-deterministic output, or a remaining privacy match is a stop condition. The sanitizer never performs a
generic metadata purge and never replaces the sensitive value with a placeholder.

### Structural-equivalence verifier

The verifier makes independent source and derivative clones, clears `metadata_props` on both clones, and
requires deterministic serialized protobuf bytes to be identical. It additionally freezes and compares:

- IR version and opset imports;
- producer/model version fields;
- graph name and every node/operator/domain/input/output/attribute;
- graph inputs, outputs, and value-info contracts;
- initializer count, names, data types, shapes, storage mode, and raw-data SHA-256 values;
- function and sparse-initializer content; and
- absence of external tensor data.

These checks establish that only the admitted metadata entry changed. They do not claim mathematical
equivalence after an optimizer, because no optimizer is permitted.

### Transformation receipt and public manifest

The repository-external transformation receipt and the committed public transformation record contain only:

- exact upstream URL, release identity, source size, and source SHA-256;
- derivative public path, size, and SHA-256;
- sanitizer source path and committed version;
- exact ONNX and protobuf versions;
- `removed_metadata_entries: 1`;
- `modified_field: "ModelProto.metadata_props[0].value"`;
- modification date `2026-08-31`;
- structural-equivalence, checker, privacy, deterministic-rerun, and browser-parity verdicts; and
- license/provenance identifiers.

They contain no removed key or value, local path, local filename, redirect query, raw header, exception,
stack, tensor values, browser profile, or temporary-directory identity. The browser manifest names the
derivative rather than describing it as the exact upstream binary.

`prepare_demo_assets.py publish` may copy only these public asset paths:

- `demo/web/samples/boats.jpg`;
- `demo/web/models/yolo26n-obb-privacy-sanitized.onnx`;
- `demo/web/demo-model.json`;
- `demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt`;
- `demo/web/third_party/yolo26n-obb-privacy-sanitization.json`; and
- `demo/web/THIRD_PARTY_NOTICES.md`.

The unmodified source ONNX is expressly forbidden from the Pages tree and release archive.

## Source-versus-derivative inference parity

Structural identity is necessary but the admission gate also exercises both byte streams in the same pinned
browser runtime. A local-only Playwright harness serves the admitted source and derivative from separate
opaque loopback routes backed by the external review root. Route names, logs, reports, screenshots, and
console output never contain filesystem paths or removed metadata.

Using the exact `boats.jpg`, identical preprocessing, and ONNX Runtime Web 1.20.1 WASM, it creates source and
derivative sessions sequentially and requires:

- identical input name, type, and `[1,3,1024,1024]` shape;
- identical output name, type, and `[1,N,7]` shape;
- byte-identical complete `Float32Array` output for `output0`;
- finite rows under the existing schema;
- identical decoded, sorted detections at confidence `0.25`; and
- at least one accepted `ship` detection.

The source session is released before the public-demo admission continues. Any output-byte difference is a
hard stop; the plan may not substitute a tolerance, change preprocessing, alter the threshold, edit boxes,
or accept a visually similar result.

The committed derivative is then exercised through the actual public demo path, not only the parity harness.
It must create a real browser session, complete Detect, render oriented polygons, populate the accessible
table/description, and show a rounded numeric runtime.

## User experience and claim presentation

The primary experience remains intentionally simple:

1. the page opens with the real official aerial image visible and labelled `原圖 · 尚未 Detect`;
2. the first primary control is `開始 Detect`;
3. only that click requests the browser runtime, manifest, and derivative model;
4. successful local inference displays the annotated image, summary, detection list, and original/result
   switch; and
5. cached confidence/class filters redraw the same completed result without another inference run.

Advanced BYOM remains collapsed under `使用自己的模型與圖片（進階）`. It uses the same preprocess,
decode, filter, rotated-corner, render, error, and session-lifecycle contracts.

The first-control claim notice states that the image is real, inference runs locally in the browser, files
are not uploaded, and the demo is not accuracy, evaluation, or latency-benchmark evidence. It does not
burden the first-time visitor with protobuf terminology.

The result provenance is exactly:

`Ultralytics YOLO26n-OBB · privacy-sanitized AGPL derivative`

`模型與素材來源` opens the detailed notice explaining that one non-inference metadata entry containing an
upstream local path was removed on 2026-08-31, while graph and weights were verified unchanged. The notice
links the exact upstream release, sanitizer source, transformation record, AGPL text, repository source,
and non-endorsement/commercial-clearance disclaimer.

The modified-work notice is prominent in the public third-party notice and model transformation record.
The repository does not describe the derivative as an official Ultralytics release binary or imply that
Ultralytics produced, endorsed, or commercially cleared the derivative.

## State, loading, and failure behavior

The prior single active source/session/result state machine remains authoritative. Sanitization is a release
process and never runs in a visitor's browser.

Initial state requests only same-origin HTML/CSS/JS/font/image assets. Detect transitions to loading and then
requests the pinned jsDelivr ORT runtime plus same-origin manifest and derivative. Loading, reset, error, and
no-cache states show runtime `—`; only a completed `session.run` displays numeric milliseconds.

Manifest digest mismatch, derivative fetch failure, Web Crypto mismatch, ORT load/session failure, contract
mismatch, run failure, output-schema failure, image decode failure, and render failure use fixed public error
codes with one actionable retry or BYOM recovery step. Every failure clears stale cache, canvas, table,
description, completed badge, numeric runtime, and result toggle while returning to the real original image.

The UI and console never emit raw response bodies, source/derivative metadata, local paths or filenames,
signed queries, raw exceptions, tensor values, or stacks.

## Accessibility and responsive requirements

All previously approved accessibility improvements remain required:

- skip link is the first focusable element and targets `main#mainContent`;
- the immutable claim notice precedes the first interactive control;
- file, range, and class inputs retain stable semantic names and labels;
- the result canvas uses `aria-describedby` linked to a non-live description generated from the same sorted,
  filtered detections as the visible table;
- empty, reset, loading, and error descriptions never retain stale results;
- progress/error status uses the existing deliberate `aria-live` region without duplicating the complete
  detection description;
- keyboard focus, heading order, 200% zoom, reduced motion, and contrast remain verified; and
- 1280×720 and 390×844 layouts have no horizontal overflow and keep the real image, primary action, status,
  source, and license information readable.

## TDD and verification design

Every behavior change follows RED → minimal GREEN → refactor. Batch tests record only the earliest assertion
actually reached; later checkpoints are not claimed as independent RED evidence until run independently.

### Sanitizer and acquisition tests

Focused unit/integration coverage must prove:

- exact source identity is mandatory;
- zero, multiple, or wrong-field path matches fail before output;
- exactly one matching metadata entry is removed and all other metadata is preserved;
- graph, initializer, function, opset, input, and output mutations are rejected;
- two independent sanitizations produce identical bytes and digest;
- output parses, passes ONNX checker, contains no external data, and passes the binary privacy scan;
- receipts have exact closed schemas and no sensitive values;
- staging, symlink/reparse containment, atomic replacement, and failure cleanup remain safe;
- the unmodified model can never be published or allowlisted;
- the Windows Git subprocess handles a Unicode worktree path under the default system code page without
  requiring callers to set `PYTHONUTF8`; and
- every CLI success/error line is fixed and does not echo arguments or exceptions.

### Browser behavior tests

Real Playwright coverage must prove:

- initial page shows the official original image and performs zero manifest/ORT/WASM/model requests;
- first Detect lazily requests exactly the admitted runtime and derivative assets;
- source-versus-derivative `output0` bytes are identical in the parity harness;
- the derivative's genuine run yields at least one `ship`, polygon pixels, synchronized table/description,
  the exact provenance, and numeric runtime;
- original/result switching and cached filters perform no additional run;
- repeated Detect reuses only a valid session;
- demo/BYOM transitions retain atomic candidate-session replacement and safe release;
- every documented failure clears stale state and offers a usable recovery path; and
- UI, console, network report, screenshot, and committed evidence contain no sensitive metadata or path.

### Repository, artifact, and release gates

The Pages verifier, package/release tests, manifest, clean-export checker, and CI must agree on one exact
allowlisted derivative path and digest. They reject every other model suffix/path, the upstream model digest,
unknown files, DOTA pixels/annotations, secrets, absolute paths, browser storage/telemetry, and unexpected
origins.

The complete local gate remains separated into:

1. sanitizer/acquisition unit and static-contract tests;
2. parity and real-demo browser smoke;
3. full Python/JS regression;
4. Pages/repository/release verifier;
5. binary/text privacy and license scan;
6. strict clean export with browser enabled;
7. desktop/mobile local operation and screenshot inspection; and
8. fresh task-scoped plus broad whole-branch review with no unresolved Critical/Important finding.

## Release, license, and public-boundary requirements

The public model is a modified AGPL derivative. Public notices therefore state the modification and date,
retain the full AGPL license, provide the corresponding sanitizer source and repository source, identify the
upstream release and source digest, and preserve the DOTAv1 training disclosure. They make no accuracy,
evaluation, T4 latency, endorsement, warranty, or commercial-clearance claim.

The root and Pages third-party notices, release manifests, READMEs, changelog, screenshot evidence, workflow
labels, and clean-export rules must consistently say `privacy-sanitized AGPL derivative` and
`modification_status: metadata-only`. No evidence file may label the derivative `unmodified` or use the
upstream binary's digest as the published digest.

Only local implementation, tests, assets, preview, evidence, and commits may be authorized by a later plan.
Remote Gate A (push/PR), Gate B (candidate/integration), Gate C (Pages dispatch), Gate D (live review), and
Gate E (About/Portfolio Control receipt) remain separate. Nothing in this design authorizes any remote gate,
Pages enable/disable/deploy, About edit, Hugging Face operation, release/tag, visibility change, force push,
branch/worktree deletion, or cleanup of the former PR #6 worktree.

## Stop conditions

Implementation stops without weakening a gate if any of the following occurs:

- upstream source URL, redirect identity, size, digest, image, or license drifts;
- the sensitive match count is not exactly one or its protobuf field is not `metadata_props.value`;
- deterministic sanitization yields different bytes across two clean runs;
- any non-metadata protobuf field or initializer digest changes;
- ONNX checker fails or external tensor data appears;
- any sensitive path/secret/private marker remains in the derivative or evidence;
- source-versus-derivative browser output bytes differ;
- the derivative contract differs from `images [1,3,1024,1024]` and `output0 [1,N,7]`;
- the exact official image produces no accepted finite `ship` result at confidence `0.25`;
- the real public path can be satisfied only with a mock or precomputed result;
- an unexpected external origin, license contradiction, artifact mismatch, or unresolved Critical/Important
  review finding appears; or
- the branch/base, authorization, remote state, or privacy boundary drifts.

No stop condition permits a different model, alternate mirror, tolerance-based parity, edited tensor,
changed threshold, manually adjusted detection, proxy, private model, DOTA image, hosted inference, or
weakened scanner without a new owner-approved design.

## Design self-review checklist

- **No placeholder:** all public paths, source identity, exact match count/field, runtime, UI copy, test layers,
  and remote boundaries are specified.
- **No contradiction:** the derivative exception replaces only the prior unmodified-model requirement; the
  real-image-first UI and every privacy/accessibility constraint remain intact.
- **No ambiguity:** one complete metadata entry is removed; no substring replacement or blanket metadata
  purge is permitted.
- **Focused scope:** one source model, one metadata-only derivative, one real image, one primary Detect flow,
  one BYOM fallback, and the evidence required to publish them safely.
- **Truthful claim:** the artifact is never called an unmodified official binary; modification, date, source,
  AGPL, sanitizer, and non-endorsement are disclosed.

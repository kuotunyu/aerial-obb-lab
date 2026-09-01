# Aerial OBB Single-harbor Demo Design

- **Date:** 2026-09-02
- **Repository:** `kuotunyu/aerial-obb-lab`
- **Branch:** `feat/pages-live-real-image-demo`
- **Design parent:** `c465ec858a95fd9b0d9a06716a7e1c78ae87cadc`
- **Status:** Design approved; awaiting written spec review
- **Scope:** replace the three-option real-sample gallery with one fixed low-density harbor example while preserving the approved original-before-Detect browser-inference workbench

## Authorization and relationship to earlier designs

This design supersedes the final public behavior in
`2026-09-01-aerial-obb-curated-real-sample-gallery-design.md` only where that design requires three public
images, a sample selector, `airfield` as the default, multi-sample switching, or three-sample release
evidence. The approved harbor source, derivation, digest, inference, privacy, accessibility, license, and
claim boundaries remain authoritative.

This document authorizes documentation only. It does not authorize product code, tests, asset deletion,
push, pull request, merge, Pages configuration or dispatch, deployment, GitHub About, Hugging Face,
release, tag, visibility change, branch deletion, or worktree cleanup.

## Owner outcome

The public demo contains one built-in real-world image: the currently approved low-density harbor aerial
sample. A visitor sees its unannotated original immediately, presses `開始 Detect`, and then sees genuine
local-browser OBB inference in the same viewport. The airfield and sports-complex images are removed from
the current public catalog and exported artifact.

The interface no longer presents a choice that does not exist. The `選擇範例` fieldset and all three
selector buttons are removed. The left rail identifies the fixed harbor sample, shows its state, and keeps
the existing Detect, original/result toggle, confidence/class filters, and advanced BYOM controls.

## Selected approach and alternatives

### Selected: fixed harbor original with no selector

The page loads the harbor original as the only built-in sample. A compact, non-interactive identity block
shows `低密度港區航拍範例` and `真實航拍原圖`; it is not a button, listbox, radio group, or pressed-state
control. The primary Detect action follows it directly.

This is selected because it matches the actual product choice, removes unnecessary interaction, shortens
the left rail, and gives visitors the simplest path from original image to genuine inference.

### Rejected: retain one selectable card

A one-item selector implies that another choice may exist and adds keyboard and state semantics without a
real decision.

### Rejected: hide two entries inside the three-sample catalog

Keeping inactive airfield and sports-complex records would leave unnecessary assets, branches, tests, and
release ambiguity. The current product contract must be one sample, not three samples with CSS hiding.

## Fixed sample identity

The only current built-in sample is:

| Field | Exact value |
| --- | --- |
| ID | `harbor` |
| Visible title | `低密度港區航拍範例` |
| Alternative text | `低密度港區的真實航拍原圖` |
| Public path | `samples/harbor.jpg` |
| Bytes | `241046` |
| SHA-256 | `916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0` |
| Source service | `https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer` |
| Product ID | `m_3411955_sw_11_060_20220514` |
| Agency/year/date | `USDA`, `2022`, `2022-05-14` |
| Public-domain record | `https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39` |
| WGS84 bbox | `[-119.216719, 34.14417, -119.200719, 34.15417]` |
| Derivation | `1280×800`, sRGB, JPEG quality 90, crop/resample/metadata removal |
| Guardrail classes | ship, storage tank, harbor (`[1, 2, 7]`) |
| Guardrail count | `16..26` at confidence `0.25` |

No airfield or sports-complex JPEG, active catalog entry, thumbnail, current notice record, release-manifest
entry, evidence entry, or current product test remains. Historical design documents and Git history may
describe the superseded gallery, but they are not current product instructions or public artifact members.
The reproducibility tooling and its current fixtures are narrowed to the approved harbor recipe and receipt;
airfield/sports candidate recipes and approval outputs are not retained as active source-admission choices.

## User experience and component responsibilities

### Initial state

- The notice remains before the first interactive workbench control and states that the page shows one
  curated USGS/USDA NAIP public-domain harbor example, performs inference only after Detect, keeps the image
  local, and is not accuracy, evaluation, model-quality, or latency-benchmark evidence.
- The left rail heading remains compact. The fixed identity block is informational text, not a selectable
  control.
- The right viewport shows `samples/harbor.jpg` with the exact alternative text above.
- Detection count is `0`; confidence, top confidence, and runtime are `—`; mode is `尚未 Detect`; filters are
  unavailable until a result exists; the primary action is `開始 Detect`.
- Initial loading may request the same-origin harbor JPEG and static application assets. It makes zero ORT,
  WASM, or ONNX requests and performs no inference.

### Detect and result

Detect continues through the existing verified path: validate the closed one-sample manifest, lazily load
pinned ORT/WASM and the privacy-sanitized model, create or reuse the verified demo session, preprocess the
decoded harbor image, execute a genuine `session.run`, decode/filter/compute rotated corners, and render one
shared cached result.

Success keeps the approved behavior:

- the result replaces the original inside the shared viewport;
- mode is `LOCAL BROWSER INFERENCE` and runtime is finite numeric browser execution time;
- provenance identifies the reviewed model and harbor sample without a correctness claim;
- canvas, sorted table, summary, and non-live textual alternative derive from the same filtered output;
- `查看原圖` and `查看 Detection 結果` switch views without another inference; and
- confidence/class filtering rerenders the active cached result without another `session.run`.

`再次 Detect` explicitly reruns inference. There is one current built-in image identity, one model session,
one active result, one generation token, and no selector state or per-image cache.

### Advanced BYOM

BYOM remains a collapsed advanced path. Opening it is network-silent. Selecting a model or image clears the
built-in harbor result before replacement work begins. Returning from BYOM restores the harbor original and
requires an explicit Detect. Existing lazy loading, atomic session replacement, generation-token, safe-error,
and privacy contracts remain unchanged.

## State simplification

The built-in path removes `sampleCatalog`, `selectedSampleId`, sample-button pressed state, sample-switch
listeners, cross-sample decode races, and three-sample result guardrails. It retains a single immutable
`demoSample` record for the harbor image.

```text
source: demo | byom
demoSample: immutable harbor record
phase: idle | loading-runtime | loading-model | ready | running | result | error
session: null | verified active session
sessionKind: null | demo | byom
image: decoded harbor or BYOM image
cached: null | active output + image transform + elapsedMs
view: original | result
generation: monotonically increasing token
```

The closed asset manifest owns the one harbor identity, source, bytes, digest, dimensions, derivation, alt,
and guardrails. `app.js` owns phase, session, decoded image, active result, view, and generation. The shared
render pipeline remains the only result presentation path. No hidden image entry, selector-only state,
precomputed output, or presentation cache is retained.

## Error and empty-result behavior

If the harbor JPEG cannot be fetched, authenticated, or decoded, the page shows the fixed actionable message
`範例影像目前無法顯示。請重新整理後重試，或使用進階 BYOM。` Detect is unavailable, and stale runtime,
canvas, table, summary, description, and completed state are cleared. The UI and console expose no URL,
local path, filename, response body, model metadata, raw exception, or stack.

Runtime, model, session, inference, and render failures preserve the already decoded harbor original, clear
all stale result surfaces, and use the existing safe recovery action. A successful inference with no
detections after filtering is not an error: it retains numeric runtime, shows count `0`, and clears polygons
and rows with an explicit empty textual description.

## Accessibility and responsive behavior

- Removing the selector removes its redundant `fieldset`, legend, button names, pressed states, and keyboard
  traversal; no disabled or hidden replacement controls are added.
- The skip link remains the first focusable element and the notice remains visually before Detect.
- The fixed harbor title is an `h3` heading followed by the text `真實航拍原圖`; neither element has button,
  pressed-state, selection, or live-region semantics.
- Detect, original/result toggle, filters, BYOM inputs, labels/names, focus-visible styles, heading order,
  reduced motion, forced colors, Source/AGPL links, and `aria-describedby` canvas alternative retain their
  accepted behavior.
- At desktop, mobile, and 200%-zoom-equivalent widths, the shorter left rail and shared viewport have no
  page-level horizontal overflow.

## Public artifact, evidence, and claims

The current sample tree is exactly:

```text
demo/web/samples/
  harbor.jpg
```

`airfield.jpg`, `sports-complex.jpg`, and the earlier dense `boats.jpg` are absent from the current Pages
artifact. The release receipt, demo manifest/loader, artifact manifest, Pages checker, clean-export checker,
evidence, notices, README files, checklist, changelog, and canonical screenshot are updated to the single
harbor contract. Each current source record remains exact and independently verifiable.

The claim remains: this is one curated integration example that runs genuine local-browser inference. It is
not ground truth, accuracy/evaluation evidence, a benchmark, a representative dataset, proof of general model
quality, or USDA/USGS endorsement. The separate Ultralytics model, privacy-sanitization, DOTAv1 provenance,
and AGPL/commercial-clearance notices remain intact.

No rejected candidate, source tile, DOTA image/annotation/render, annotated result image, private model,
local path, filename, raw diagnostic, token, signed query, model metadata dump, telemetry, or browser profile
enters Git, the artifact, UI, console, screenshot metadata, or evidence.

## Test and release matrix

Implementation follows strict TDD. Each behavior change first gets a real failing test, and only the earliest
actually reached failure is recorded before the minimal fix.

### Static and unit contracts

- the closed manifest accepts exactly one `harbor` sample and rejects zero, extra, duplicate, renamed,
  external, traversal, malformed, wrong-byte, wrong-digest, wrong-source, wrong-alt, wrong-derivation, and
  wrong-guardrail records;
- no current runtime/catalog/documentation/release contract references `airfield.jpg` or
  `sports-complex.jpg`;
- exactly one public sample JPEG is admitted and it is byte-identical to the reviewed harbor derivative;
- selector markup/state/listeners and cross-sample caches are absent;
- source/public-domain/derivation/no-endorsement claims remain exact; and
- the model/license/privacy exception remains one exact, separate record.

### Real browser behavior

- initial paint shows the harbor original and fixed title, exposes no selector, and requests no ORT/WASM/ONNX;
- explicit Detect performs one real model run, yields finite output and numeric runtime, satisfies the harbor
  count/representative guardrails, and renders a reconciled canvas/table/summary/description;
- original/result toggling, cached filters, repeated Detect, responsive layout, keyboard/focus, canvas text
  alternative, reduced motion, forced colors, error recovery, BYOM transitions, privacy, origin allowlist,
  console, and overflow contracts pass;
- malformed image/model/runtime/render paths clear stale state and expose only fixed recovery copy; and
- successful coverage uses the exact reviewed harbor/model bytes and a real browser ORT session, not mocks,
  fixed tensors, precomputed boxes, or a committed annotated image.

### Release gates

- focused one-sample/static/browser tests, full browser smoke, and the full pytest suite pass;
- artifact, Pages, repo, release, privacy, origin, license, and stale-reference verifiers pass;
- a strict committed-files-only clean export passes tests, browser smoke, package build/install/import, and
  byte/canonical-text equality;
- the canonical screenshot shows the unfiltered harbor result at confidence `0.25`, with no class filter,
  numeric runtime, claim notice, Source/AGPL context, closed BYOM, and no visually dominant wrong polygon;
- repository-external desktop/mobile original/result screenshots pass visual review; and
- an independent whole-branch reviewer finds no unresolved Critical or Important issue.

## Non-goals and remote boundary

- No second sample, selector, carousel, random image, upload service, webcam/video, map, geolocation, automatic
  Detect, side-by-side before/after wall, or persistent storage.
- No new model, threshold, class filter, NMS, preprocessing, hand-edited box, accuracy claim, or evaluation.
- No redesign of the approved compact workbench, inference geometry, session architecture, BYOM contract,
  license boundary, or Pages workflow.
- No push, pull request, merge, Pages configuration/dispatch/deployment, GitHub About, Hugging Face, release,
  tag, visibility change, branch deletion, worktree cleanup, or other-repository modification is authorized.

## Self-review record

- **Owner intent:** only the approved low-density harbor image remains; the interface no longer suggests a
  choice between samples.
- **Truthfulness:** every displayed result follows an explicit real browser inference; visual curation is not
  presented as evaluation.
- **Consistency:** one sample, one threshold, one reviewed model, one active result, one shared pipeline.
- **Privacy and license:** the harbor public-domain record and model AGPL/privacy boundaries remain exact and
  separate.
- **Error boundary:** failures clear stale result presentation and expose fixed actionable copy only.
- **Scope:** current sample inventory, redundant selector state/UI, documentation/evidence, tests, and release
  gates only; no unrelated polish or remote action.

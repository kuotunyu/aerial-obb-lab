# Aerial OBB Curated Real-sample Gallery Design

- **Date:** 2026-09-01
- **Repository:** `kuotunyu/aerial-obb-lab`
- **Design branch:** `feat/pages-live-real-image-demo`
- **Design parent:** `40a6eb130b6e2cf46b89469750eb10f9133d8a83`
- **Status:** Approved for implementation planning
- **Scope:** replace the single dense harbor example with three switchable, public-domain real aerial
  examples while preserving the approved workbench and genuine click-to-run browser inference

## Authorization and relationship to approved designs

This design refines the local real-image workbench currently committed on the design branch. It supersedes
only these earlier single-sample decisions:

- `boats.jpg` as the sole and default public example;
- the single hard-coded sample identity in the demo manifest, application, tests, release evidence, and
  public notices; and
- the earlier deferral of a multiple-example gallery.

The following approved boundaries remain load-bearing:

- the compact left-control/right-result workbench;
- one reviewed, privacy-sanitized, same-origin YOLO26n-OBB derivative;
- no model, ORT, or WASM request before an explicit Detect action;
- genuine local-browser inference through the shared preprocess, decode, filter, rotated-corner, render,
  table, and textual-description pipeline;
- no upload, telemetry, hosted inference, precomputed output, or committed annotated result image;
- advanced BYOM with its existing lazy-load, atomic session replacement, generation-token, privacy, and
  recovery contracts; and
- no accuracy, evaluation, latency-benchmark, or endorsement claim.

This document authorizes documentation only. It does not authorize product code, tests, asset acquisition,
push, pull request, merge, Pages configuration or dispatch, deployment, GitHub About, Hugging Face, release,
tag, visibility, or worktree cleanup.

## Problem and outcome

The current real-image-first interaction is correct: a visitor sees an original aerial image, presses
`開始 Detect`, and then sees genuine browser inference in the same viewport. The sole `boats.jpg` example is
not a good first impression, however. It contains many small, tightly packed vessels. At the default
confidence threshold, minor localization errors, duplicate-looking outlines, and detections crossing dock or
image boundaries become visually dominant.

The approved outcome retains the interaction and replaces the example experience:

1. the left rail offers exactly three compact, labelled real-aerial example choices;
2. the best reviewed example is selected initially and its unannotated original is visible immediately;
3. selecting another example replaces the original and clears every result belonging to the former image;
4. the visitor presses the existing primary Detect action to run the reviewed model on the selected image;
5. the result appears in the same viewport and can still be toggled back to the original; and
6. all three examples are curated for visual legibility but are explicitly not presented as accuracy or
   evaluation evidence.

The dense Ultralytics harbor sample leaves the public demo and artifact. It may remain only in historical Git
objects or isolated tests that have an independent, documented reason to use it; it is not an option in the
new selector and is not copied into the exported Pages tree.

## Selected approach and alternatives

### Selected: three compact selectors and one shared Detect workspace

The current official-example card becomes a labelled sample selector containing three compact thumbnail
buttons. The existing primary Detect button, filters, result summary, unified viewport, table, and BYOM
disclosure remain shared. Only one sample is active, only one result identity is retained, and switching a
sample always returns the workspace to an original-before-Detect state.

This approach is selected because it preserves the UI the owner approved, makes all choices discoverable,
and keeps the mental model explicit: choose one real image, inspect the original, press Detect, inspect the
result.

### Rejected: three independent full-size cards

Giving every image its own viewport and Detect button would make the page long, duplicate status and
accessibility regions, and weaken the compact workbench. It also creates unnecessary session and result-state
coordination.

### Rejected: one image with a random or sequential “換一張” action

This is compact but hides the available scenes, makes screenshots and browser tests less deterministic, and
does not let a visitor deliberately compare different target types.

### Rejected: pre-rendered before/after gallery

Pre-rendered boxes could be visually curated but would repeat the experience already rejected by the owner.
Every displayed result must come from a real `session.run` initiated by the visitor in the current browser
session.

## Fixed sample identities and admission source

The public catalog has exactly these stable identities and paths:

| ID | Visible title | Intended scene | Public path |
| --- | --- | --- | --- |
| `airfield` | `小型機場航拍範例` | a low-density airfield with a small number of separated aircraft | `samples/airfield.jpg` |
| `sports-complex` | `運動場館航拍範例` | separated baseball diamonds, courts, or track facilities | `samples/sports-complex.jpg` |
| `harbor` | `低密度港區航拍範例` | a small number of separated large vessels | `samples/harbor.jpg` |

`airfield` is the preferred initial example because a few separated objects give the clearest introduction.
It becomes the default only after it passes the same admission gate as the other two. If it does not pass,
implementation stops for design review rather than silently changing the default or category.

All candidates must come from National Agriculture Imagery Program imagery for the contiguous United States
distributed by the U.S. Geological Survey. The authoritative records are:

- `https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip`;
- `https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39`; and
- DOI `10.5066/F7QN651G`.

Those USGS records identify the collection as public-domain aerial orthoimagery. Commercial basemaps,
Google/Bing/Apple imagery, map screenshots, DOTA/DOTAv1 pixels or annotations, social-media images, stock
photography, and any source whose exact redistribution basis is unclear are forbidden.

### Exact asset derivation and provenance

Implementation evaluates two or three candidates for each fixed category outside the repository. Each
candidate is derived from one exact NAIP source product by a recorded, deterministic operation:

- source product identifier, acquisition year, official download URL, retrieval date, source byte length,
  and source SHA-256;
- pixel crop rectangle in the source image and any orientation normalization;
- output width `1280`, output height `800`, sRGB color, JPEG quality `90`, and metadata stripped;
- output byte length, decoded dimensions, media signature, and SHA-256; and
- a plain-language scene description that does not claim a detection result.

No boxes, labels, masks, color emphasis, blur, object insertion/removal, or model-informed retouching are
burned into an admitted image. Cropping and ordinary resampling are allowed only to establish a consistent
inspection viewport. Exact source facts and final digests are measured during controlled acquisition and
written as literal values before product code or release evidence depends on them; fake or placeholder facts
are forbidden.

The acquisition tooling may download large source products into a repository-external temporary directory.
Only the three admitted, metadata-stripped JPEGs enter Git or the Pages artifact. Rejected candidates,
source tiles, geospatial sidecars, credentials, local paths, and operator data never enter the repository,
reports, screenshots, or commits.

## Visual suitability admission gate

Asset selection is a release-engineering step, not a claim that the samples measure model quality. All
candidates are tested using the exact reviewed public demo model, the normal browser preprocessing and
postprocessing pipeline, and the single existing default confidence threshold `0.25`. Per-image thresholds,
hidden class filters, hand-deleted detections, altered non-maximum suppression, manual box movement, and
different model bytes are prohibited.

A candidate is admitted only when all of the following hold in the real browser:

- inference completes with finite output and at least one visible oriented detection corresponding to the
  scene's intended target family;
- the intended objects that dominate the scene are represented by visually aligned boxes rather than a
  result dominated by empty background;
- there is no obvious high-confidence detection centred on a clearly unrelated background feature;
- no single visibly incorrect polygon dominates the frame, crosses a large unrelated region, or makes the
  result immediately appear broken;
- the result is not a dense wall of overlapping or edge-clipped polygons comparable to the rejected harbor
  example;
- the unannotated original remains understandable at desktop and mobile sizes; and
- before/after screenshots from the exact candidate artifact pass independent visual review.

Machine-readable guardrails record, for each admitted sample, the expected target class family, a bounded
detection-count range, and broad centre/size ranges for at least one stable representative detection. These
guardrails detect accidental asset/model/preprocess drift; they do not label ground truth and are not exposed
as accuracy metrics. Browser assertions use tolerances that accommodate legitimate float/runtime variation
without allowing a blank, synthetic, mocked, or unrelated result.

If any category cannot produce a compliant candidate at the shared default threshold, implementation stops
and reports that category. It must not raise the threshold, replace the category, use a private image, or
commit an annotated substitute without a new design decision.

## User experience

### Initial presentation

The existing header, skip link, Browser/WASM/Local-files indicators, workbench layout, and notice-first order
remain. Before the first workbench control, the non-collapsible notice becomes:

> **真實航拍範例 · 實際瀏覽器推論**
>
> 選擇一張 USGS／USDA NAIP 公領域航拍範例，再按下 Detect。頁面才會載入 OBB 模型並在你的瀏覽器中執行推論；影像不會上傳。範例經挑選以便清楚展示操作，不是 accuracy、evaluation 或 latency benchmark。

The left control rail begins with `範例與設定` and the instruction
`先選擇一張真實航拍原圖，再由目前的 browser 執行 Detect。` A `fieldset` labelled `選擇範例` contains
the three sample buttons in the fixed order `airfield`, `sports-complex`, `harbor`.

Each button contains:

- an uncropped thumbnail sourced from the same admitted image used for inference;
- the fixed visible title from the catalog;
- the neutral label `真實航拍原圖`; and
- selected state conveyed visually and through `aria-pressed="true"`.

The selected sample's original fills the existing deep-navy unified viewport with `object-fit: contain`.
The label is `原圖 · 尚未 Detect`; count is `0`; confidence and runtime are `—`; mode is `尚未 Detect`; and
provenance is `USGS／USDA NAIP · 尚未執行`. Filters remain visible but disabled until a result exists. The
primary action is `開始 Detect`.

All three images may be fetched from the same site origin during normal page load so their thumbnails switch
without delay. Initial load must still make zero ORT, WASM, manifest, or ONNX requests.

### Selecting a different sample

A real sample button activation performs one atomic transition:

1. increment the generation token so late work cannot publish into the new sample;
2. set `selectedSampleId` to the selected fixed catalog ID;
3. clear active cached output, elapsed time, result view, polygons, table, textual description, summary,
   completion status, and retry/error state;
4. decode and show the selected original in the shared viewport;
5. publish the neutral selected-sample provenance and status; and
6. leave the verified demo model session available for reuse without running it.

Returning to a previously detected sample does not restore a hidden per-sample result. The visitor presses
Detect again. This preserves one active result identity and avoids a second presentation cache.

### Detect and result

Detect uses the already approved generation-protected path. The first run lazily loads pinned ORT/WASM,
validates the manifest and privacy-sanitized model, creates the candidate session, preprocesses the selected
decoded image, and executes a real `session.run`. Later runs may reuse that verified session but always run
inference again for the currently selected image.

Success preserves the existing workbench behavior:

- the result canvas replaces the original inside the same viewport;
- the mode remains `LOCAL BROWSER INFERENCE`;
- runtime is the finite rounded `session.run` time;
- provenance identifies the reviewed model and the selected NAIP sample without presenting a correctness
  claim;
- summary, canvas polygons, sorted table, and non-live canvas description derive from the same active cached
  output and filters; and
- `查看原圖` / `查看 Detection 結果` toggles the two views without another inference.

The button becomes `再次 Detect`. Confidence and class filters operate on the active cached result and never
rerun inference. No result count, confidence value, runtime, or wording such as “correct”, “accurate”, “best”,
or “validated accuracy” is shown on the selector itself.

### Advanced BYOM

The advanced BYOM disclosure remains below the sample controls. Opening it is network-silent. Selecting
either BYOM file clears the selected demo result presentation but does not expose a local path, filename,
model metadata, response data, raw exception, or stack. Returning from BYOM to any public sample restores
that sample's original and requires an explicit Detect.

## State and component responsibilities

The existing single-session/single-result model gains one catalog identity:

```text
sampleCatalog: immutable three-entry manifest catalog
selectedSampleId: airfield | sports-complex | harbor
source: demo | byom
phase: idle | loading-runtime | loading-model | ready | running | result | error
session: null | verified active session
sessionKind: null | demo | byom
image: decoded currently selected demo or BYOM image
cached: null | active output + image transform + elapsedMs
view: original | result
generation: monotonically increasing token
```

Responsibilities remain separated:

- the closed asset manifest owns the three stable IDs, titles, paths, digests, dimensions, source records,
  and admission guardrails;
- `app.js` owns selection, generation, phase, current decoded image, active result, and session lifecycle;
- the selector renders only from the validated catalog and emits one sample ID;
- the unified viewport consumes the active decoded image and result view;
- the shared inference pipeline receives the currently active image and model session;
- `renderCachedOutput()` remains the sole confidence/class rerender path; and
- release tooling owns exact inventory, digest, provenance, license, origin, and forbidden-artifact checks.

There is no per-sample result cache, hidden inference, sample-specific presentation cache, or manually stored
polygon set. Browser HTTP caching of the three immutable same-origin JPEGs and reuse of the one verified model
session are allowed.

## Error, empty, and stale-state behavior

Selecting a sample whose admitted JPEG cannot be fetched or decoded shows the fixed message
`這張範例影像目前無法顯示。請選擇其他範例，或重新整理後重試。` The active result, numeric runtime,
canvas, table, description, and completed badge are empty; Detect is unavailable for that sample; the other
two selectors remain usable. No URL, path, filename, response body, or raw exception is displayed or logged.

Runtime/model/inference/render failures preserve the currently selected original and use the existing fixed,
actionable safe messages. They clear every stale result surface and restore a retryable Detect action when
appropriate. Switching samples clears the error and never permits a late completion from the former sample
to change the new image or status.

A successful inference with no detections after filtering is not an error. It keeps the selected original or
result canvas available, shows count `0`, retains the completed numeric runtime, clears all polygons and rows,
and gives the existing explicit empty description. It never falls back to precomputed boxes.

## Accessibility and responsive behavior

- The existing skip link remains the first focusable element and the claim notice remains visually before
  the sample selector.
- The selector has one programmatic group label; each option is a real button with a stable name, fixed title,
  meaningful image alternative, visible selected state, and `aria-pressed` state.
- Selection is available through keyboard and pointer input without carousel gestures or hover dependence.
- The concise polite status announces the newly selected title once. The detailed canvas description remains
  non-live, preventing duplicate detection announcements.
- After selecting a sample, focus remains on the activating selector. After explicit Detect success, focus
  moves through the existing result-heading behavior.
- The result canvas remains `aria-describedby` by the same filtered textual alternative as the table.
- At 1280×720 the three compact options remain within the left rail and the viewport stays to the right. At
  390×844 and 200%-zoom-equivalent width they stack above the viewport without page-level horizontal
  overflow.
- Thumbnails use fixed dimensions to avoid layout shift and do not rely on color alone to show selection.
- Reduced motion, focus visibility, contrast, form names/labels, heading order, and Source/AGPL links retain
  their current accepted behavior.

## Public artifact, notices, and privacy boundary

The sample portion of the public artifact becomes:

```text
demo/web/
  samples/
    airfield.jpg
    sports-complex.jpg
    harbor.jpg
```

`boats.jpg` is absent. The closed artifact manifest admits exactly those three sample JPEGs plus the existing
reviewed application/model/license inventory. `demo-model.json` or its versioned replacement lists the three
sample records and one default ID. Unknown IDs, external runtime URLs, traversal, duplicate paths, wrong
dimensions, non-JPEG signatures, digest mismatch, non-public-domain source records, and missing admission
guardrails fail closed.

`THIRD_PARTY_NOTICES.md`, the root notices, README files, evidence, release checklist, acquisition tooling,
artifact verifier, Pages checker, and clean-export rules are updated from “one Ultralytics sample” to the
three exact USGS/USDA NAIP derivatives. Each record identifies source product, public-domain basis,
modification as crop/resample/metadata removal, final digest, and no-endorsement statement. The Ultralytics
model and AGPL notice remain separate and unchanged except where prose must stop claiming that `boats.jpg` is
the public input.

No source tile, rejected candidate, DOTA content, private model, model metadata dump, geospatial sidecar,
local path, local filename, user profile, raw exception, stack, token, signed query, telemetry, or annotated
result image enters Git, the exported artifact, UI, console, screenshot metadata, or committed evidence.

## Test and review matrix

Implementation follows strict browser-first TDD. Every behavior change first receives a real failing test and
the actually reached failure is recorded before minimal production changes.

### Unit and static contracts

- manifest validation accepts exactly the three fixed sample IDs, paths, titles, digests, dimensions, source
  records, and default ID, and rejects unknown/duplicate/external/traversal/malformed entries;
- selector/state reducers clear one active result identity and increment generation on sample change;
- no per-sample result cache or precomputed polygon data is admitted;
- acquisition receipts reproduce each final JPEG from the exact official source facts;
- `boats.jpg`, single-sample claims, and deprecated Ultralytics-image notice are absent from current public
  contracts; and
- names, labels, notice text, license links, and public-domain records are exact.

### Real browser behavior

- first paint shows `airfield` selected, its real unannotated image in the unified viewport, neutral summary,
  three discoverable selectors, and zero ORT/WASM/manifest/ONNX requests;
- each selector displays the matching admitted image, clears the former cached result/table/description/
  runtime/toggle, updates selected accessibility state, and issues no inference request;
- each of the three samples completes a genuine real-model `session.run`, produces finite output and numeric
  runtime, satisfies its bounded admission guardrails, and renders the same filtered list in canvas, table,
  summary, and description;
- model/session loading occurs only when no verified demo session is active; that session is reused across
  sample switches without suppressing later per-image inference, while a BYOM replacement continues to use
  the existing single-session lifecycle;
- original/result toggling and confidence/class filtering preserve the selected sample identity and do not
  trigger another model run;
- switching during delayed image decode, model creation, or inference rejects the stale generation and cannot
  publish the former sample's result;
- image, runtime, model, inference, and render failures clear stale state, retain safe recovery, keep other
  samples usable, and expose no forbidden details;
- BYOM transitions remain lazy, private, atomic, recoverable, and mutually exclusive with demo results; and
- keyboard, focus, group/button semantics, canvas alternative, live announcements, reduced motion, desktop,
  mobile, zoom, origin allowlist, console, and overflow contracts pass.

Mocked network/error scenarios may cover failures, but the three successful admission paths must use the exact
reviewed JPEG/model bytes and a real browser ORT session. Source grep, mock-only output, fixed tensors, or
committed result images cannot satisfy the gate.

### Visual and release acceptance

For each exact admitted sample, capture repository-external original and post-Detect screenshots from the
verified clean export at desktop and mobile sizes. Review confirms scene readability, non-dominant errors,
consistent viewport geometry, selector clarity, notice priority, and absence of stale sample/result state.
Screenshots and reports use public sample IDs only and contain no local path or browser profile metadata.

Before any remote action, run and record separately:

- focused catalog/state/static tests;
- focused three-sample browser scenarios;
- the complete browser smoke;
- the full pytest suite;
- repo, release, Pages workflow, license, origin, privacy, and artifact verifiers;
- a strict clean export built only from tracked files, with canonical text and binary bytes identical to the
  reviewed `demo/web` tree;
- `pages_artifact_check.py` against that exact export;
- a local desktop/mobile/zoom review served from the exact export; and
- an independent whole-branch scope, claim, privacy, quality, and visual review.

## Remote gates

Every remote mutation remains a later, separately authorized gate:

1. **Gate A — candidate PR:** non-force push and one review-only PR after local acceptance.
2. **Gate B — candidate artifact and merge:** download, byte-compare, serve, visually review all three
   examples, and merge only after every check passes.
3. **Gate C — Pages dispatch:** deploy only the exact reviewed merged-main SHA after its automatic CI passes.
4. **Gate D — live review:** repeat all three original/Detect/result flows, network/privacy/accessibility,
   notices, desktop/mobile, and forbidden-artifact verification on the live URL.
5. **Gate E — About and Portfolio Control receipt:** change About only after independent Gate D approval and
   then record the final receipt separately.

This spec authorizes none of those gates and does not authorize force push, branch deletion, worktree cleanup,
release, tag, visibility change, Hugging Face operation, or modification of another repository.

## Non-goals

- No automatic carousel, random sample, autoplay Detect, batch inference, side-by-side three-result wall,
  upload service, webcam/video, map, geolocation UI, or persistent browser storage.
- No ground-truth annotations, accuracy percentage, precision/recall, evaluation set, benchmark, best-model
  statement, or implication that visual curation proves general model quality.
- No per-image confidence, class filter, NMS, preprocessing, model, hand-edited box, or precomputed output.
- No new model, runtime framework, analytics, cookie, service worker, telemetry, external runtime image origin,
  commercial imagery, DOTA pixels/annotations, or private data.
- No redesign of the approved workbench, inference geometry, session architecture, BYOM contract, license
  boundary, Pages workflow, GitHub About, or portfolio-wide UI.

## Self-review record

- **Owner intent:** three visible real-world choices preserve the successful original-before-Detect flow and
  replace the visually weak dense harbor sample.
- **Truthfulness:** every result is genuine browser inference; curation is disclosed and never described as
  evaluation evidence.
- **Consistency:** one default threshold, one reviewed model, one active result, one shared pipeline, and no
  hidden per-image tuning or output.
- **Asset identity:** three fixed IDs, titles, paths, dimensions, derivation rules, and one authoritative
  public-domain source family; exact measured source/digest facts are mandatory admission outputs, not
  placeholders.
- **Failure boundary:** any category that lacks a compliant licensed image stops implementation instead of
  weakening the source, visual, or inference gate.
- **Privacy:** no private/local/model metadata, rejected source, DOTA content, raw error, stack, token, or
  result image enters public or committed surfaces.
- **Accessibility:** grouped keyboard-operable selectors, stable pressed state, no duplicate live detail,
  synchronized canvas alternative, and retained desktop/mobile/zoom behavior.
- **Scope:** sample catalog, selection/reset state, three genuine inference paths, provenance/notices,
  tests/evidence, and release inventory only; no remote action.
- **Contradictions:** the earlier single-sample and “multiple examples deferred” statements are explicitly
  superseded; the workbench, model, inference, privacy, BYOM, and remote-gate contracts remain intact.

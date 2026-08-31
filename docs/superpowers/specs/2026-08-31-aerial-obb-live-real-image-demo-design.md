# Aerial OBB Live Real-image Demo Design

- **Date:** 2026-08-31
- **Repository:** `kuotunyu/aerial-obb-lab`
- **Design branch:** `docs/pages-real-sample-before-after-design`
- **Current design parent:** `633b1d44c72e6a1eef51dbc63b29304cf2f4c000`
- **Implementation dependency:** the approved accessibility/runtime follow-up at
  `24039db9e07e327b55433086241ac574430c4531`
- **Status:** Approved for implementation planning
- **Scope:** replace the public synthetic-first experience with one real aerial image and one genuine,
  click-to-run browser OBB demo; keep BYOM as an advanced secondary path

## Supersession and authorization boundary

This design supersedes the product direction in:

- `docs/superpowers/specs/2026-08-31-aerial-obb-real-sample-before-after-design.md`; and
- `docs/superpowers/plans/2026-08-31-aerial-obb-real-sample-before-after.md`.

The superseded design attempted to publish two public-domain images with precomputed outputs. The owner
rejected that experience because it still did not let a visitor see a real image and then press Detect.
Its candidate-selection tooling, partial Task 2 work, and local reports are historical implementation
artifacts only. They are not requirements for this design and must not be mixed into a future implementation
commit without a new TDD justification.

The owner has explicitly approved publishing one official public demo model so a visitor without an image
or model can perform real local-browser inference. This approval replaces the former “no weights or ONNX in
the Pages artifact” boundary for exactly the model identified below. It does not authorize private weights,
other models, training data, DOTA images or annotations, uploads, telemetry, hosted inference, deployment, or
any remote mutation.

This turn authorizes only this design document. Product code, tests, assets, model acquisition, push, pull
request, merge, Pages configuration or dispatch, GitHub About, Hugging Face, release, tag, and visibility
changes remain unauthorized until their later written gates.

## Problem and desired outcome

The current public page asks a first-time visitor to load an abstract Synthetic Showcase. That path proves
the UI and geometry pipeline, but it does not answer the visitor's natural question: “What does detection
look like on a real aerial image?” It also presents model and image pickers before a visitor has enough
context to know why they would need them.

The approved outcome is deliberately simple:

1. the page opens with one real aerial photograph already visible and unannotated;
2. the visitor presses one primary `開始 Detect` button;
3. the browser lazily loads a reviewed public OBB model and executes genuine inference locally;
4. the same photograph displays oriented detection boxes, a concise result summary, and an accessible
   result list; and
5. the visitor can switch between the original and annotated view without running inference again.

A visitor needs neither a local image nor a local model for this primary experience. The existing BYOM
workflow remains available below it for advanced users who want to use their own compatible files.

## Selected approach and alternatives

### Selected: one same-origin real image and one same-origin lazy demo model

The page publishes the exact official Ultralytics `boats.jpg` sample and the exact official
`yolo26n-obb.onnx` release asset as reviewed same-origin Pages assets. The image is requested during normal
page load so the original is immediately understandable. The ONNX model is not requested until the visitor
presses `開始 Detect`.

The application continues to use its existing pinned ONNX Runtime Web loader and shared
preprocess/decode/filter/rotated-corner/render pipeline. It does not add the Ultralytics browser package,
another framework, or another presentation cache. The committed model is treated as a versioned third-party
binary: source, release tag, digest, size, contract, training provenance, and license are recorded and
verified before it can enter either Git or the Pages artifact.

This is the recommended approach because it produces the requested one-click real inference while avoiding
runtime model-origin drift. Ultralytics' own browser documentation states that GitHub Release assets do not
send `Access-Control-Allow-Origin`; a browser therefore cannot reliably fetch the release URL directly and
must use a same-origin or explicitly CORS-enabled copy.

### Rejected: browser fetch from the GitHub Release URL

This would keep the repository smaller, but the official Release asset lacks the CORS response header a
cross-origin browser fetch requires. A design that knowingly fails at the primary action is unacceptable.
No proxy, permissive third-party mirror, query-string workaround, or CORS bypass is allowed.

### Rejected: precomputed Before/After output

This avoids distributing a model, but it is the experience the owner explicitly rejected. It demonstrates
rendering, not the requested action of pressing Detect and observing real inference.

### Deferred: multiple examples or automatic gallery

Additional images would increase model suitability, image rights, visual QA, payload, and interaction scope.
The first release proves one clear path. More samples require evidence from the live experience and a new
design review.

## Reviewed public assets and license boundary

### Real image

The sole primary image candidate is the exact byte stream served by the official Ultralytics OBB example:

- documentation use: `https://docs.ultralytics.com/tasks/obb`;
- acquisition URL: `https://ultralytics.com/images/boats.jpg`;
- public path after admission: `demo/web/samples/boats.jpg`; and
- presentation: the unmodified original bytes, with no crop, box, label, metadata injection, or derivative
  “result image” committed.

Before admission, implementation must prove that the final redirected bytes correspond to an official
Ultralytics sample asset covered by an explicit repository/file license record. It records the full redirect
chain, retrieval date, byte length, media signature, decoded dimensions, SHA-256, source commit or immutable
record when available, and license basis. If image-specific provenance or permission cannot be proven, the
implementation stops for a new owner decision; it does not silently substitute another image.

### Public demo model

The only authorized model is:

- identity: `yolo26n-obb.onnx`;
- official release: Ultralytics assets `v8.4.0`;
- acquisition URL:
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx`;
- public path after admission: `demo/web/models/yolo26n-obb.onnx`;
- task: oriented bounding-box detection;
- expected input contract: `images [1,3,1024,1024]`;
- expected output contract: `output0 [1,N,7]`, with the exact admitted `N` recorded in the manifest; and
- license: Ultralytics AGPL-3.0, with the repository's existing AGPL-3.0-or-later code license kept separate
  from the third-party asset notice.

Implementation performs an anonymous acquisition from the exact tag, confirms that the asset is attached to
that official release, computes its digest and size, inspects the ONNX contract without printing raw metadata,
and proves one real local inference against the exact admitted `boats.jpg`. Admission fails closed if the
source is missing, redirected outside the approved GitHub asset hosts, larger than 15 MiB, malformed, not an
OBB model, incompatible with the declared contract, or unable to produce a finite browser result.

The official model is pretrained on DOTAv1. That training provenance is disclosed in the notice manifest.
No DOTAv1/DOTA image, annotation, archive, evaluation output, dataset-derived screenshot, or training record
is copied into this repository or Pages artifact. This spec makes no legal claim about dataset/model
derivative status; it relies only on the official model's stated distribution license and the repository
owner's explicit decision to distribute this one reviewed model under that license.

### Required public notices

The Pages artifact adds:

```text
demo/web/
  THIRD_PARTY_NOTICES.md
  third_party/
    ULTRALYTICS-AGPL-3.0.txt
```

`THIRD_PARTY_NOTICES.md` records the image and model separately: title, exact source, release/tag or immutable
record, SHA-256, byte length, retrieval date, copyright/licensor statement, license, modification status,
training provenance disclosure for the model, and a no-endorsement statement. It must not contain a local
path, private model identity, raw metadata dump, access token, or operator identity.

The visible footer retains `Source` and `AGPL-3.0-or-later` and adds `模型與素材來源`. The primary result
provenance links to the reviewed notice. Ultralytics, GitHub, DOTAv1, or any contributor is not represented as
endorsing the detection, repository, portfolio, or output.

## Public file and data contracts

The intended public layout is:

```text
demo/web/
  index.html
  style.css
  app.js
  obb.js
  demo-model.json
  samples/
    boats.jpg
  models/
    yolo26n-obb.onnx
  THIRD_PARTY_NOTICES.md
  third_party/
    ULTRALYTICS-AGPL-3.0.txt
```

`demo-model.json` is a strict, closed schema with these required values:

- schema version `1` and ID `ultralytics-yolo26n-obb-demo`;
- same-origin relative image path `samples/boats.jpg`, exact SHA-256, byte length, media type, width, and
  height;
- same-origin relative model path `models/yolo26n-obb.onnx`, exact SHA-256, byte length, official source URL,
  release tag `v8.4.0`, and license ID `AGPL-3.0-only`;
- input name, dimensions, data type, normalization, channel order, letterbox size and padding value;
- output name, exact admitted dimensions, row layout, angle unit/convention, and 15-class label mapping;
- default confidence `0.25`; and
- the public notice path.

The exact hashes, lengths, decoded image dimensions, and admitted output count are facts generated by the
controlled acquisition and validation step. The implementation plan must require those literal values before
any production code depends on the manifest. No illustrative or placeholder digest is allowed.

Application validation rejects unknown fields, external asset URLs, traversal, non-lowercase SHA-256,
non-finite numbers, unexpected dimensions or names, duplicate/missing class indexes, unapproved licenses,
digest mismatch, MIME/signature mismatch, and files outside the exact allowlist. The release verifier compares
both the repository tree and exported Pages tree to this closed manifest.

The authored synthetic SVG and fixed output leave the public artifact. They may remain only as test fixtures
outside `demo/web`, where deterministic geometry tests can continue to use them without exposing a
synthetic-first public path.

## User experience and exact presentation

### Initial page

The first viewport has this hierarchy:

1. product title `Aerial OBB Lab`;
2. plain-language introduction:
   `先看真實航拍原圖，再按下 Detect；模型會在你的瀏覽器中找出帶方向的目標。`;
3. the non-collapsible claim and privacy notice;
4. one large unannotated `boats.jpg` figure labelled `原圖 · 尚未 Detect`; and
5. one primary `開始 Detect` button.

The exact notice before the first visible control is:

> **真實航拍範例 · 實際瀏覽器推論**
>
> 下方是 Ultralytics 官方航拍範例。按下 Detect 後，頁面才會載入官方 OBB 模型並在你的瀏覽器中執行推論；影像不會上傳。這是操作示範，不是 accuracy、evaluation 或 latency benchmark。

The initial viewport does not show the ONNX picker, image picker, class-filter wall, tensor shape, or
Synthetic Showcase action. Those details do not compete with the one-click task.

The initial result summary is neutral: detection count `0`, confidence `—`, runtime `—`, mode `尚未 Detect`,
and provenance `官方範例 · 尚未執行`. No box, result row, or stale description is present.

### Detect interaction

Pressing `開始 Detect` performs one generation-token-protected operation:

1. clear prior result, overlay, table, canvas description, runtime, and completion state;
2. set the button busy and publish `正在載入偵測元件…`;
3. lazily load the existing pinned ONNX Runtime script with its exact integrity and anonymous-CORS contract;
4. fetch `demo-model.json` and the same-origin ONNX file only if no verified demo session is cached;
5. verify the ONNX byte digest with Web Crypto before creating a candidate session;
6. validate the session input/output contract and atomically replace any old session only after the new
   session is ready;
7. publish `正在分析航拍影像…`, preprocess the already decoded official image, and run the session;
8. measure only the `session.run` interval as inference runtime;
9. decode, filter, calculate rotated corners, draw, summarize, list, and describe from the same output; and
10. focus the result heading and publish the fixed completed state.

After success the same viewport is labelled `Detection 結果`, and the mode/provenance are exactly:

- mode badge: `DEMO MODEL · LOCAL BROWSER INFERENCE`;
- provenance: `Ultralytics YOLO26n-OBB · official AGPL model`;
- runtime: a finite rounded numeric value ending in `ms`; and
- status: `Detect 完成。可切換查看原圖與偵測結果。`.

The primary button becomes `再次 Detect`. A secondary two-state button appears only after success. It is
labelled `查看原圖` while boxes are visible and `查看 Detection 結果` while the original is visible. Switching
views changes no cached detection, filter, runtime, table, or description and never reruns inference.

The image and annotated result use the exact same decoded source and viewport transform. The result canvas
draws the image and boxes through the shared renderer; no flattened result JPEG/WebP is committed.

### Confidence and class filters

Filters appear only after a successful result under a compact `調整結果` disclosure. They operate on the
cached output and never rerun the model. The default view uses confidence `0.25` and all classes. Table,
summary, visible boxes, and non-live canvas description consume the same confidence-descending filtered list.

A filtered-empty result states `目前沒有符合篩選條件的偵測結果。` The original image remains visible, the
result count is zero, the overlay contains no boxes, and the numeric inference runtime remains the completed
run's value because filtering did not invalidate that result.

### Advanced BYOM

Below the public demo, one disclosure labelled `使用自己的模型與圖片（進階）` contains the existing BYOM
model picker, image picker, filters, and Detect action. Opening it performs no network request and does not
alter the current demo result.

Selecting a BYOM model follows the existing candidate-session-then-release lifecycle. The public demo model
name may be shown because it is a disclosed public asset; a local BYOM filename, path, metadata, and raw
exception remain private and never appear in UI or console. A successful BYOM result keeps the exact badge
`BYOM · LOCAL BROWSER INFERENCE` and a finite numeric runtime.

Demo and BYOM results are mutually exclusive. Selecting either BYOM file clears the demo result, overlay,
table, description, runtime, and toggle. Returning to the public demo restores the unannotated official image
without inference; pressing `開始 Detect` then uses the verified cached demo session when valid. Late fetch,
decode, session, or inference completion cannot republish stale state because every async boundary checks the
current generation token.

## State and data flow

The state model has one active result identity and one active session:

```text
source: demo | byom
phase: idle | loading-runtime | loading-model | ready | running | result | error
session: null | verified active session
sessionKind: null | demo | byom
cached: null | decoded current output + image transform + elapsedMs
view: original | result
generation: monotonically increasing token
```

There is no presentation cache beyond `cached`. Every completed render derives the current list, statistics,
runtime, table, canvas description, and polygons from the active source, phase, cached output, and current
filters. Runtime is numeric only for a completed real demo or BYOM `session.run`; idle, loading, reset, error,
and no-cache states use `—`.

```text
same-origin boats.jpg
    -> decode once -> initial original figure

Detect
    -> pinned ORT lazy loader
    -> same-origin model manifest + ONNX fetch
    -> SHA-256 + contract validation
    -> candidate session -> atomic active-session replacement
    -> shared letterbox preprocessing
    -> session.run -> output0 [1,N,7]
    -> shared decode -> confidence/class filter -> confidence sort
    -> shared rotated corners
    -> annotated canvas + summary + table + non-live description
```

Repeated demo Detect reuses the active verified demo session and the decoded source image. A BYOM session is
released only after its replacement is successfully created; reset and terminal errors release unusable
candidate sessions and clear the complete result presentation.

## Error handling and recovery

Every failure is atomic. The unannotated official image remains available, while stale boxes, cached output,
statistics, table rows, description, numeric runtime, completed badge, and comparison toggle are cleared.
The button returns to an enabled `重新 Detect` state.

User-facing fixed messages are:

- runtime or network load: `偵測元件無法載入。請檢查網路後重試，或開啟進階 BYOM。`;
- model manifest, digest, license, or contract: `Demo 模型目前無法使用。請稍後重試，或開啟進階 BYOM。`;
- image decode: `範例影像無法讀取。請重新整理後重試，或開啟進階 BYOM。`;
- inference: `這次 Detect 未完成。請重試，或開啟進階 BYOM。`; and
- render: `結果無法顯示。請重試，或開啟進階 BYOM。`.

UI, DOM attributes, accessibility text, console, screenshots, test reports, and commits must not contain a
local path, user filename, private model identity, private model metadata, raw exception, response body, raw
tensor, stack, access token, or browser profile path. Development diagnostics use fixed error codes and
boolean/count state only.

Integrity or contract failure never falls back to an unverified model, remote proxy, private model, synthetic
output, or precomputed boxes. It is a hard failure with the recovery actions above.

## Accessibility and responsive behavior

- The existing skip link remains the first focusable element and targets `main#mainContent`.
- The claim/privacy notice remains visually before `開始 Detect`.
- The original uses meaningful alternative text describing the aerial boat scene without claiming a
  detection result.
- The result canvas uses `aria-describedby` to reference a non-live description generated from the same
  filtered detections as the table. Each item contains class, confidence, centre x/y, width/height, and angle.
- Initial, loading, filtered-empty, reset, and error descriptions are explicit and contain no stale result.
- The concise status region is the only `aria-live` announcer; the detailed canvas description is not live.
- Busy state uses `aria-busy`, disables duplicate Detect activation, and preserves visible progress text.
- The original/result switch is a real button with `aria-pressed` state, not a hover or pointer-only gesture.
- Focus moves to the result heading only after an explicit Detect completes; filters and view switching do
  not steal focus.
- Model/image inputs and all filter controls retain stable labels and names in the advanced BYOM disclosure.
- Controls have at least a 44-pixel target. The layout has no horizontal overflow at 1280×720, 390×844, or
  200% zoom, and reduced-motion mode introduces no essential animation.

## Network, privacy, and supply-chain boundary

Initial page load may request only the exact same-origin allowlist: HTML, CSS, JavaScript, reviewed font, and
`samples/boats.jpg`. Notice/license links are ordinary user-activated links. Initial load must request neither
`demo-model.json`, ONNX Runtime, nor `models/yolo26n-obb.onnx`.

The first Detect may additionally request:

- the existing exact pinned jsDelivr ONNX Runtime script and its required pinned WASM resources; and
- the same-origin `models/yolo26n-obb.onnx`.

There are no other runtime origins. In particular, the browser does not call GitHub Releases, Ultralytics,
Hugging Face, an inference API, analytics, telemetry, logging, storage, cookies, or service workers. Image
and inference bytes stay in browser memory and are not uploaded. The design intentionally retains the
existing small ONNX Runtime integration rather than adopting a new browser package whose runtime, CDN, or
telemetry behavior would expand the review boundary.

The model response must be HTTP 200, same-origin, exact expected byte length and SHA-256, and a valid ONNX
signature before session creation. A cache hit is permitted only after the bytes were verified in the current
application version. No persistent IndexedDB/localStorage model cache is added.

## Test design and release gates

Implementation must use strict TDD: add a real failing behavioral assertion, observe the intended failure,
then make the smallest production change and rerun the focused and cumulative suites.

### Static, unit, and asset contracts

- the manifest schema accepts only the one approved image/model identity and closed field set;
- the exact official image and model sources, tag, SHA-256, lengths, signatures, dimensions, model contract,
  class mapping, training provenance, and licenses are pinned;
- altered/truncated image or model bytes, wrong hash, bad redirect host, oversize model, unknown field,
  traversal, external runtime path, wrong tensor name/dim/type, and missing notice fail closed;
- the official demo model produces a finite real result on the exact sample in a controlled local acceptance
  harness before admission;
- no committed precomputed detection output or flattened annotated result exists;
- no public synthetic CTA, SVG, fixed showcase output, private model identity, local path, DOTA image/label,
  telemetry code, source original duplicate, or unallowlisted binary enters the artifact; and
- third-party notices and license text are byte-checked and linked from the public UI.

### Real browser behavior

- initial load shows the unannotated real image, plain-language instruction, notice-first order, enabled
  `開始 Detect`, neutral summary, no boxes, and no ORT/model request;
- first Detect requests the pinned ORT resources and same-origin model exactly once, verifies model bytes
  before session creation, then performs a real session run rather than using a mocked/precomputed result;
- successful output shows at least one finite oriented polygon on the exact image, a matching table and
  canvas description, the exact demo badge/provenance/status, and a finite numeric runtime;
- original/result switching changes only the view, issues no network request, and preserves result data;
- repeated Detect reuses the verified demo session and does not redownload the model;
- confidence and class filters synchronize canvas, table, summary, and description without a new session run;
- filtered-empty, retry, stale generation, runtime/model/image/contract/inference/render failures, and recovery
  clear stale presentation and show only fixed safe messages;
- opening BYOM is network-silent; selecting a BYOM model preserves its lazy/session/release/privacy contract;
- transitions between demo and BYOM release the superseded session safely and never mix result identities;
- UI and console contain no forbidden private markers, raw errors, stacks, or filenames; and
- keyboard, labels/names, skip link, heading order, aria-live, canvas alternative, reduced motion, 200% zoom,
  1280×720 desktop, and 390×844 mobile pass with no horizontal overflow.

The real-browser inference acceptance is allowed to stub network failures, but its successful Detect path
must use the exact reviewed ONNX bytes and a real browser ORT session. Mock-only coverage cannot satisfy it.

### Full local acceptance

Before any remote gate, run separately and record:

- focused unit/static contract tests;
- the full browser smoke, including the real-model sample path;
- the repository's complete pytest suite;
- repo, release, Pages workflow, artifact, license, origin, and privacy verifiers;
- a strict clean export built from tracked files, with the model/image bytes and canonical text identical to
  the reviewed `demo/web` tree;
- local serving of that exact export with desktop and mobile screenshots; and
- a final whole-branch scope and quality review.

The model makes this acceptance slower and larger than the former code-only artifact. Tests may use a small
deterministic fixture for isolated decode/error cases, but the final real-model browser gate is mandatory and
cannot be replaced by a source grep, mock, precomputed tensor, or private model.

## Remote gates

All remote actions remain independent and require fresh written authorization:

1. **Gate A — candidate PR:** non-force push and one review/candidate-only PR after clean local review.
2. **Gate B — artifact review and merge:** download the candidate, compare exact image/model/text bytes,
   serve it outside the repository, review origins/privacy/UI, then use an authorized merge method.
3. **Gate C — Pages deployment:** dispatch only from the exact reviewed merged-main SHA after main CI passes.
4. **Gate D — live review:** verify the real initial image, genuine Detect, same-origin model, network origins,
   license links, desktop/mobile/accessibility, safe failures, and no stale Synthetic experience.
5. **Gate E — About and Portfolio Control receipt:** only after the live SHA independently passes Gate D.

This spec does not authorize any gate, branch cleanup, old worktree cleanup, model upload outside this one
future reviewed Pages artifact, Hugging Face operation, release, tag, or visibility change.

## Non-goals

- No multiple-sample gallery, autoplay, webcam/video, map, slider, upload service, server inference, benchmark,
  accuracy claim, evaluation claim, or comparison against ground truth.
- No private/owner test model, custom trained weight, alternate mirror, CORS proxy, runtime model chooser in
  the primary path, model download button, or raw output inspector.
- No manual box movement, deletion, relabeling, hand-picked detection, hidden confidence change, precomputed
  output, or committed annotated screenshot presented as the result.
- No new framework, package manager, build framework, telemetry, persistent cache, service worker, cookie,
  analytics, or additional runtime origin.
- No change to other repositories, GitHub Pages configuration, About, Hugging Face, releases, tags,
  repository visibility, or existing remote deployment in the design turn.

## Self-review record

- **Owner intent:** a visitor with no files sees a real image immediately and can press one button to run real
  detection; Synthetic is not the public experience.
- **CORS contradiction resolved:** the official release URL is acquisition provenance only. The browser loads
  the reviewed model from the same Pages origin after Detect.
- **Claim boundary:** the run is genuine local-browser inference, while the notice explicitly denies accuracy,
  evaluation, and benchmark claims.
- **License boundary:** one exact public AGPL model and one exact official sample are admitted only after
  image-specific/model-specific verification; private weights and DOTA dataset content remain prohibited.
- **Privacy boundary:** no upload, telemetry, filename, local path, private model data, raw exception, tensor,
  or stack enters public output or diagnostics.
- **State consistency:** one active result/session, generation-token checks, atomic replacement, derived
  presentation, complete stale-state clearing, and numeric runtime only for completed real inference.
- **Placeholders:** no fake asset, digest, result, or filename is specified. Literal hashes and dimensions are
  required facts produced by the future controlled acquisition gate before implementation can proceed.
- **Scope:** one real image, one reviewed demo model, one Detect flow, one result comparison, advanced BYOM,
  required notices/tests, and no remote action.

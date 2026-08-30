# Aerial OBB Pages Showcase Design

- **Date:** 2026-08-31
- **Repository:** `kuotunyu/aerial-obb-lab`
- **Status:** Design approved by the owner-delegated Technical Lead; written spec pending final review
- **Target:** GitHub Pages project site published from `demo/web`

## Context

`aerial-obb-lab` already contains a static Browser BYOM workbench in `demo/web`. It accepts a
user-supplied compatible ONNX model and image, performs inference with ONNX Runtime Web, and keeps
the selected files in the browser. The repository deliberately distributes code and evidence only:
it contains no model weights, ONNX binary, DOTA image, or DOTA-derived raster render.

The current workbench is technically suitable for static hosting, but a recruiter without the exact
model contract sees only disabled controls. The approved website therefore adds an interactive,
clearly labelled synthetic showcase while preserving BYOM as the only real inference path.

This design follows the approved portfolio strategy: use the existing privacy-preserving browser
workbench as the repository-specific Website destination, publish only after local and hosted review,
and leave the GitHub About Website field empty until the canonical URL passes its release gates.

The repository normally ignores `docs/superpowers/`. This owner-approved spec is an intentional
single-file exception; the ignore rule itself remains unchanged.

## Goals

- Give a recruiter a one-action way to inspect the real decode, filter, rotated-corner, canvas, and
  result-table pipeline without providing a model.
- Keep synthetic and BYOM modes visibly distinct and impossible to confuse.
- Preserve the code-only public boundary and local processing of user-selected files.
- Publish the existing static site directly from `demo/web` without copying it into `/docs` or a
  generated branch.
- Extend the existing deterministic, model-free test and privacy gates before any deployment.

## Non-goals

- Bundling or downloading a named model, checkpoint, weight, or ONNX binary.
- Publishing DOTA pixels, annotations, or DOTA-derived raster renders.
- Presenting the synthetic fixture as inference, accuracy, evaluation, latency, or production evidence.
- Adding a framework, backend, API, telemetry, storage, service worker, custom domain, or user account.
- Re-running training, changing accepted project metrics, or adding new model-performance claims.
- Enabling Pages, deploying, pushing, or updating GitHub About as part of the design/spec task.

## Approaches considered

### 1. Single-page, dual-mode workbench — selected

Add explicit Synthetic Showcase and BYOM states to the current page. Both states feed the same output
validation, decode, filtering, corner calculation, canvas, summary, and table pipeline. Only their data
sources differ. This reuses the mature workbench, minimizes new surface area, and provides the strongest
evidence that the showcase exercises production frontend logic rather than a visual imitation.

### 2. Two independent workbenches on one page

Keep a synthetic workbench above a separate BYOM workbench. The boundary is obvious, but controls,
result regions, responsive layout, accessibility behavior, and tests would be duplicated.

### 3. Recruiter landing page plus a separate BYOM page

Use an explanatory landing page with a synthetic explorer and link to another workbench route. This
supports more narrative content but introduces navigation, duplicated assets, and maintenance cost that
the current project does not need.

## 1. Architecture and public boundary

The site remains a framework-free static single-page application:

```text
GitHub Pages index
    -> project header
    -> mandatory claim-boundary notice
    -> mode entry
       -> Synthetic Showcase
       -> BYOM inference
    -> shared workbench
       -> controls
       -> OBB canvas renderer
       -> result summary
       -> detection table
```

The page order is normative:

1. The header identifies the project and its Browser, WASM, and local-file behavior without adding
   model-performance numbers.
2. A static, non-collapsible claim-boundary notice appears in the HTML before every showcase button,
   slider, class filter, file picker, or other interactive control. JavaScript must not be required to
   make this notice visible.
3. The notice is followed by the `載入 Synthetic Showcase` action and a clearly separated BYOM path.
4. Both modes share the formal output/decode/render pipeline. Synthetic mode never constructs an ONNX
   Runtime session and never reports inference latency.
5. Results always display their mode and provenance.
6. The footer provides Source, AGPL-3.0-or-later, and public-boundary links.

The Pages artifact may contain only the static application code, the existing OFL font and license,
the synthetic SVG and numeric fixture, and other explicitly reviewed static presentation assets. It may
not contain weights, an ONNX binary, DOTA pixels or annotations, a DOTA-derived render, a secret, a
private path, or an unreviewed binary.

GitHub Actions will upload `demo/web` itself as the Pages artifact. The design does not move or copy the
site into `/docs`, does not create a `gh-pages` content branch, and does not publish the repository root.

## 2. Component responsibilities and state model

### `index.html`

- Owns the semantic order, static claim-boundary notice, mode actions, BYOM inputs, results, provenance,
  and footer.
- Keeps the notice visible even when JavaScript or the external runtime cannot load.

### `showcase-fixture.js` and synthetic SVG

- Hold the single production/test source for the synthetic image, `output0` dimensions and data, and
  fixture provenance.
- Contain no DOM, ONNX Runtime, model, remote fetch, or inference logic.
- Replace test-only duplication: production showcase and browser/parity tests consume the same fixture.

### `obb.js`

- Remains the pure computation layer for output schema validation, letterbox geometry, RGB CHW
  conversion, decode/filter behavior, and rotated corners.
- Has no DOM or ONNX Runtime dependency and does not know which mode supplied its input.
- Receives an `output0` object with dimensions and data from both synthetic and BYOM sources.

### `app.js`

- Is the single UI controller for state transitions, lazy runtime loading, session lifecycle, file
  reading, cached output, and presentation.
- Uses one canvas, summary, table, and filter implementation for both modes.
- Re-decodes cached validated output when confidence or class filters change; it does not rerun inference.

### `style.css`

- Distinguishes Synthetic Showcase from BYOM inference without hiding the public-boundary notice.
- Preserves the existing desktop/mobile layout, visible keyboard focus, 44 px control target, and minimum
  secondary-text size.

The controller uses orthogonal state rather than unrelated Boolean flags:

```text
mode:  none | synthetic | byom
phase: idle | loading | ready | running | result | error
```

Required invariants:

- `mode=synthetic` implies `session=null` and `elapsedMs=null`.
- `mode=byom, phase=running` requires both a compatible session and a decoded image.
- `phase=result` requires validated cached output and letterbox geometry.
- Synthetic runtime text is exactly `N/A · no inference`.
- Switching modes clears the previous result cache. Switching to synthetic releases a BYOM session.
- A replacement model is first used to create and validate a new session. The old session is released
  only after the new session succeeds; a failed replacement leaves the old compatible session usable.
- Model and image operations use a monotonically increasing generation token. A stale asynchronous
  completion is discarded and cannot overwrite the active state.
- Error surfaces never include a local path, filename, model metadata, tensor contents, raw exception,
  or stack trace.

## 3. Data flow, interaction, and claim presentation

### Synthetic Showcase

```text
載入 Synthetic Showcase
    -> load same-origin synthetic SVG
    -> read committed output0 {dims, data}
    -> selectEndToEndOutput
    -> decodeDetections + confidence/class filter
    -> rotatedCorners
    -> shared canvas/summary/table renderer
    -> cache validated raw output and geometry
```

This path does not touch `ort`, create a session, or request the external runtime. Filter changes operate
on cached output. Provenance is `Committed synthetic fixture`, the mode badge is
`SYNTHETIC FIXTURE · NO INFERENCE`, and runtime remains `N/A · no inference`.

### BYOM inference

```text
Select ONNX
    -> lazy-load ONNX Runtime Web 1.20.1 JavaScript with SRI and anonymous CORS
    -> configure the ONNX Runtime Web 1.20.1 package directory as the WASM asset base
    -> File.arrayBuffer in browser memory
    -> create and validate a new session
    -> replace the previous session only on success

Select image
    -> local object URL
    -> browser decode
    -> revoke object URL

Detect
    -> offscreen 1024 x 1024 letterbox and RGB CHW conversion
    -> session.run
    -> validate output0 [1,N,7]
    -> shared decode/corners/render
    -> cache output, geometry, and measured elapsed time
```

BYOM model and image bytes stay in browser memory. The only approved third-party network boundary is
`https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js` and the same package directory for
its WASM companions. The JavaScript entry retains `crossorigin="anonymous"` and integrity
`sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp`. The SRI claim applies only
to the JavaScript resource that the browser actually validates with SRI; the site must not imply that
all WASM files receive SRI validation. It must not call the website offline or zero-network.

### Interaction rules

- Initial load produces no result. The claim notice precedes the first action.
- Loading the showcase moves focus to the results heading and announces the synthetic provenance through
  the existing live status region.
- Selecting either BYOM file switches to BYOM mode and clears synthetic results.
- Detect is enabled only when a compatible session and decoded image are ready. Repeated Detect clicks
  are locked until the active run completes.
- Generation tokens prevent an older model, image, or run from changing the current UI.
- Mobile order remains notice, mode/actions, controls, then results.

### Mandatory claim presentation

The non-collapsible notice uses this approved meaning and wording:

> **Synthetic Showcase — 沒有執行模型推論**
>
> 此展示使用本 repository 提交的 synthetic SVG 與固定 output，僅驗證 UI、decode、filter、
> rotated-corner 計算與 rendering。它不是 accuracy、evaluation 或 latency evidence。實際推論請
> 使用 BYOM，自行提供相容 ONNX model 與影像。

BYOM results carry the badge `BYOM · LOCAL BROWSER INFERENCE`. The BYOM controls separately disclose
that user files remain in the browser, the runtime may require the pinned jsDelivr resources, and no model
is bundled. The website adds no new accuracy, benchmark, or production claim; it links back to the
repository's committed evidence instead.

## 4. Error handling

The controller maps internal conditions to fixed safe error codes and fixed zh-Hant-TW recovery copy.
UI and console output never interpolate error objects or user/model details. The console records only a
stable code such as `[AERIAL_OBB:RUNTIME_LOAD]`.

| Error code | Fixed user-facing recovery copy |
| --- | --- |
| `SHOWCASE_ASSET` | `Synthetic fixture 無法載入。請重新整理頁面，或改用 BYOM。` No fallback result is fabricated. |
| `RUNTIME_LOAD` | `Browser runtime 無法載入。請檢查網路或 content blocker 後重試；Synthetic Showcase 仍可使用。` |
| `MODEL_CONTRACT` | `請選擇使用 images [1,3,1024,1024] 與 output0 [1,N,7] 的相容 ONNX。` An already-valid previous session remains active. |
| `IMAGE_DECODE` | `Browser 無法解碼影像。請改選 PNG、JPEG 或 WebP。` The loaded model session remains active. |
| `INFERENCE_RUN` | `推論未完成。請確認模型 contract、重新選擇影像後再試。` Partial output from the failed run is removed. |
| `OUTPUT_SCHEMA` | `模型輸出不符合 output0 [1,N,7]。請改用相容的 end-to-end OBB export。` |
| `RENDER_RESULT` | `結果無法呈現。請重新載入 Synthetic Showcase，或重新執行 Detect。` The result cache is cleared first. |

Stale generation completions are expected control flow and are discarded silently.

## Test matrix

| Layer | Required coverage |
| --- | --- |
| Pure JavaScript/Node | Production fixture schema, output validation, decode/filter/corners, malformed values failing closed, and synthetic operation with an unavailable or throwing `ort` global. |
| Python-to-JavaScript parity | Production fixture letterbox, float32 conversion, decoded values, angle, and corners match the Python reference. |
| Playwright synthetic | Static notice precedes the first interactive control in DOM and visual order; showcase produces no external runtime request; badge, provenance, runtime, filters, canvas, summary, and table are correct. |
| Playwright BYOM | Lazy runtime load, exact 1.20.1 URL, SRI and anonymous-CORS attributes, external-origin allowlist, atomic session replacement/release, explicit Detect, cached re-filtering, and delayed generation-token races using deterministic stubs. |
| Failure states | Every safe error code exposes an actionable recovery step; UI and console reject injected filenames, paths, metadata, tensors, exceptions, and stacks. |
| Accessibility/layout | Keyboard focus, live status, control size, minimum type size, desktop 34/66 layout, mobile single-column order, and persistent notice. |
| Release/privacy | The exact artifact contains no ONNX, weight, DOTA pixel or derived render, secret, absolute user path, symlink/hard link, unexpected binary, broken local reference, or unreviewed external origin. |

Deterministic CI remains model-free and performs no real inference. Before deployment, an owner-controlled
manual BYOM smoke must use a trusted compatible model stored outside the repository and a synthetic or
rights-cleared non-DOTA image. The model must never be copied into the checkout, artifact, log, screenshot
metadata, or commit. If this smoke cannot be completed, the site is not deployed.

## Release gates

1. Work in an isolated worktree. Run the existing pytest suite, repository preflight, release/privacy
   gate, Node parity tests, and the extended browser smoke.
2. Construct the candidate from `demo/web` only. Inventory and scan the exact Pages artifact, including
   file paths, types, sizes, hashes, symlink state, forbidden extensions, sensitive strings, and external
   origins.
3. Serve that exact artifact locally. Review desktop and mobile layouts, the synthetic and BYOM paths,
   every error recovery, keyboard behavior, provenance, and the network allowlist.
4. Require all pull-request CI checks to pass. A future Pages deploy job accepts only protected `main`
   and uses job-level minimum permissions: `contents: read`, `pages: write`, and `id-token: write`.
5. Treat initial Pages enablement as a separate remote mutation requiring fresh owner authorization. A
   merge or passing artifact does not authorize enablement or deployment.
6. After the first deployment, keep GitHub About Website blank while reviewing the canonical HTTPS URL,
   claim notice, synthetic no-runtime path, BYOM runtime loading, desktop/mobile rendering, source/license
   links, and external requests.
7. If live review fails, do not change About. Fix and redeploy after review. Disabling Pages also requires
   explicit authorization; otherwise retain the failed URL unadvertised while remediation occurs.
8. Update GitHub About Website only in a separate authorized task after the canonical URL passes live
   review. Read the API value back and verify the exact URL.

## Acceptance criteria

- A visitor can load the synthetic showcase with one action and immediately see provenance, the
  no-inference badge, rotated polygons, summary, and detection table.
- The mandatory notice is visible before the first interactive control without JavaScript.
- Synthetic mode creates no ONNX Runtime session, requests no external runtime, and reports no latency.
- BYOM remains the only inference path; user-selected bytes are not uploaded or persisted.
- Synthetic and BYOM modes share output validation, decode, filters, corners, and result rendering.
- Mode transitions, session replacement, and asynchronous races preserve all state invariants.
- User-facing errors are actionable and all user/console error surfaces remain non-sensitive.
- Existing and new automated gates pass, the owner-controlled BYOM smoke passes, and the candidate Pages
  artifact satisfies the public-boundary inventory.
- Pages remains disabled and GitHub About remains unchanged until their separate release approvals.

## References

- Existing application: `demo/web/index.html`, `demo/web/app.js`, `demo/web/obb.js`, and
  `demo/web/style.css`.
- Existing public boundary: `README.md`, `demo/web/README.md`, `THIRD_PARTY_NOTICES.md`,
  `release/artifact-manifest.json`, and `release/evidence.json`.
- Existing gates: `scripts/browser_smoke.py`, `scripts/repo_check.py`, `scripts/release_check.py`, and
  `tests/test_browser_parity.py`.
- GitHub Pages publishing sources:
  <https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site>.
- GitHub Pages custom workflows:
  <https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>.

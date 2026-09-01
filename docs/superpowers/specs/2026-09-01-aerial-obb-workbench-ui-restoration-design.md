# Aerial OBB Workbench UI Restoration Design

**Date:** 2026-09-01
**Status:** Design approved; pending written-spec review
**Base:** `b6a4cd6193a9c34bd08e437805248dbc9658e3d5` on `feat/pages-live-real-image-demo`

## Objective

Combine the strongest parts of the original Browser BYOM workbench and the approved real-image demo:

- restore the original compact left-control/right-result workbench;
- show the real official aerial image in the result viewport before inference;
- run genuine local browser inference only after the user selects **開始 Detect**;
- replace the original image with the oriented detection rendering in the same viewport;
- preserve the verified model, lazy-loading, privacy, failure recovery, accessibility, and BYOM behavior already
  committed on the branch.

This is a presentation and view-state refinement. It does not change model bytes, preprocessing, output
decoding, filtering semantics, rotated-corner geometry, inference ownership, or the public license boundary.

## Approved Direction

Use the approved visual direction **A — 原版工作台復刻**.

The page retains the claim-boundary notice above every interactive control. Below that notice, the desktop
experience returns to the original approximately 31/69 workbench:

- a compact control rail on the left; and
- a result workspace with a deep-navy viewport, summary, and table on the right.

The current full-width introduction, full-width standalone figure, and separate action strip are removed.
Their useful content is retained inside the workbench rather than repeated above it.

## Information Architecture

### Page header and claim boundary

The existing header, Browser/WASM/Local-files indicators, skip link, and non-benchmark claim remain. The claim
notice stays before the first focusable workbench control and retains its current truthful statements:

- the example is a real official aerial image;
- inference runs in the current browser session;
- image and model bytes are not uploaded; and
- the page is not accuracy, evaluation, or latency benchmark evidence.

### Left control rail

The control rail contains these sections in order:

1. **範例與設定** heading and one-line instruction.
2. A compact official-example card containing:
   - a small, uncropped thumbnail of `samples/boats.jpg`;
   - the label **官方港區航拍範例**;
   - the current demo state (`Original · ready`, loading, result ready, or retry available);
   - the primary **開始 Detect** / **再次 Detect** action; and
   - the result-only **查看原圖** / **查看結果** secondary action.
3. Confidence threshold and class filters in the same dense style as the original UI.
4. A collapsed **使用自己的模型與圖片（進階）** BYOM section at the bottom.

Confidence and class controls remain visible to preserve the original tool-workbench character. Before a
result exists they are disabled and accompanied by a short neutral message that they become available after
Detect. They are enabled for both demo and BYOM cached results. Changing them never reruns inference.

The BYOM section keeps the existing model and image file controls and **執行 BYOM Detect** action. Opening
the section does not load ORT, WASM, or a model. Selecting a BYOM file follows the existing generation-token,
result-clearing, candidate-session, privacy, and recovery contracts.

### Right result workspace

The right side restores the original hierarchy:

1. **Detection result** heading and short explanation.
2. Compact five-field summary: detections, top confidence, runtime, mode, and provenance.
3. One persistent deep-navy viewport.
4. Detection table directly below the viewport.

The summary is present in every state. It uses the already approved values for initial, loading, successful,
and failure states; no synthetic or benchmark claim is introduced.

## Unified Viewport

The original image and result canvas share one stable viewport container. The implementation uses stacked
image and canvas layers rather than moving the media between unrelated sections.

### Initial state

- The real official image is visible with `object-fit: contain`; it must not be cropped.
- The canvas is hidden.
- A small high-contrast label reads **原圖 · 尚未 Detect**.
- The table has its headers and an explicit empty state.
- Summary values are count `0`, top confidence `—`, runtime `—`, mode **尚未 Detect**, and provenance
  **官方範例 · 尚未執行**.
- Initial network activity may include same-origin page assets and `samples/boats.jpg`, but must omit the
  manifest, ORT script, WASM, and ONNX model.

### Loading state

- The original image remains visible.
- The primary action is disabled and shows a concise loading state.
- Status text identifies the recoverable stage without exposing URLs, paths, filenames, response bodies,
  model metadata, raw exceptions, or stacks.
- No stale result canvas, table row, description, numeric runtime, or result badge remains.

### Successful result state

- The result canvas replaces the image inside the same viewport bounds.
- Oriented polygons are rendered by the existing shared preprocess/decode/filter/corners/render pipeline.
- Summary, non-live canvas description, and sorted table describe the same filtered detection set.
- Runtime is numeric for genuine inference.
- **查看原圖** toggles the image layer without clearing the result.
- **查看結果** restores the cached canvas without rerunning inference.
- Confidence or class changes rerender the cached output and retain the current image/result view choice.

### Failure, reset, and source-transition states

All demo failures use the existing centralized recovery path:

- the official original image is restored;
- runtime becomes `—`;
- cache, canvas, table, description, toggle, and completed state are cleared;
- the status gives one actionable recovery step using fixed safe copy; and
- Detect becomes retryable when the relevant local prerequisites are available.

BYOM selection clears the demo result and shows the chosen local image in the same viewport. Returning to the
official demo restores the official original before any new demo inference. Existing candidate-before-current
session replacement, stale-generation rejection, and safe release behavior remain unchanged.

## Visual System

The refinement restores the original visual language instead of introducing a third design system:

- rectangular bordered panels rather than floating marketing cards;
- compact IBM Plex Sans Condensed headings and dense metadata;
- the existing cool gray page, white control surfaces, navy ink, blue accent, and deep-navy viewport;
- original summary and table rhythm;
- restrained 4/8/12/16-pixel spacing; and
- no oversized hero title or full-width image above the tool.

The official-example card may use a small thumbnail, but the primary image inspection surface is always the
right viewport. The thumbnail must not compete with or replace that surface.

Visible focus, reduced-motion behavior, color contrast, and minimum interactive target sizes remain at least
as strong as the current implementation.

## Responsive Behavior

At desktop widths of 960 CSS pixels and above, the workbench uses the 31/69 two-column layout. The control
rail remains readable without forcing the viewport below a useful size. The viewport uses the image aspect
ratio and a bounded height, with deep-navy letterboxing when necessary.

Below 960 CSS pixels, the page becomes one column in this order:

1. claim boundary;
2. example card and primary Detect action;
3. unified viewport;
4. summary;
5. filters and detection table; and
6. collapsed BYOM section.

At 390×844 and at the 200%-zoom-equivalent viewport, there is no horizontal page overflow. The result table
may use its existing labelled internal horizontal scroller. Source, code license, model license, provenance,
and sanitization links remain readable.

## State and Component Responsibilities

The existing application state remains authoritative. No second presentation cache is added.

- `app.js` continues to own source, generation, phase, session, image, cached output, elapsed time, and active
  result view.
- The unified viewport renderer derives which layer is visible from the existing phase/result-view state.
- `setResultView(view)` remains the sole image/result toggle entry point.
- `clearResultState({keepImage: true})` remains the failure/reset cleanup entry point.
- `renderCachedOutput()` remains the only confidence/class rerender path.
- `demo-assets.js`, `obb.js`, the verified model manifest, and every frozen binary/text asset remain unchanged
  unless a verifier digest must be updated for reviewed HTML/CSS/application text.

The DOM is reorganized without duplicating IDs, controls, status regions, canvas descriptions, or results.
The claim notice remains static HTML so it is present even if JavaScript fails.

## Accessibility and Privacy

- The skip link continues to target `main#mainContent`.
- Heading order remains logical: page title, control/result section headings, then table/BYOM subsections.
- Every form control retains its stable `name` and label.
- The viewport canvas remains described by the synchronized, non-`aria-live` textual description.
- The status region remains the only polite live announcement for operations and failures.
- Toggling image/result does not duplicate detection announcements.
- Keyboard focus order follows the visual workflow: demo action, optional view action, filters, BYOM, result
  table navigation, and project links.
- Page errors remain forbidden and console output remains restricted to fixed diagnostic codes.
- No local path, local filename, response body, model metadata, raw exception, stack, token, or signed query is
  permitted in UI, console, screenshot metadata, reports, or committed evidence.

## Test Strategy and Acceptance Gates

Implementation follows strict browser-first TDD.

### Required RED coverage

Before production layout changes, real browser assertions must prove the current full-width layout violates
the approved workbench contract. The first actually reached failure is recorded; unreachable later assertions
are not claimed as independently observed RED.

The layout regression must check real bounding boxes and visible state rather than source-text grep alone:

- the demo action is inside the left control rail;
- the viewport is to the right of that rail at 1280×720;
- the original image is visible inside the unified viewport before Detect;
- canvas and image occupy the same viewport bounds when their respective layers are shown;
- filters are visible but disabled before a result and enabled afterward;
- BYOM is collapsed and secondary; and
- the former standalone introduction/figure/action-strip layout is absent.

### Behavior regression

Existing browser scenarios must remain green for:

- initial zero ORT/WASM/ONNX requests;
- genuine derivative inference and numeric runtime;
- original/result toggle without new inference;
- cached confidence and class filtering;
- fixed-code failure recovery and runtime retry;
- delayed candidate creation, stale runs, BYOM transitions, and session release;
- privacy surfaces, page errors, and console allowlist;
- skip link, focus, labels/names, headings, description, live status, reduced motion, and readable links; and
- desktop, mobile, and 200%-zoom-equivalent overflow behavior.

### Repository gates

Run the complete browser smoke, current focused Python suite, Pages artifact checker, full local regression,
clean-export/artifact/privacy scans required by the active implementation plan, and exact frozen-asset digest
checks. Any changed reviewed text digest must be updated without weakening inventory, origin, model, license,
privacy, or workflow policy.

## Scope and Non-goals

In scope:

- workbench DOM reorganization;
- CSS layout and compact visual restoration;
- minimal application view-state wiring needed by the unified viewport and disabled/enabled filter presentation;
- real browser layout/behavior regression coverage; and
- reviewed text digest/evidence updates required by those exact changes.

Out of scope:

- changing, regenerating, moving, or uploading the sample image or ONNX model;
- changing preprocess, decode, rotated-corner, filtering, inference, or session architecture;
- adding more examples, a gallery, tutorial wizard, new analytics, browser storage, telemetry, or uploads;
- changing licensing, provenance, repository visibility, or GitHub About;
- push, PR, merge, Pages dispatch/deployment, release/tag, Hugging Face operations, or remote gate execution; and
- cleaning up the retained worktree or feature branch.

## Release Boundary

This design authorizes local documentation, implementation, tests, preview, evidence, and commits only. It
does not authorize any remote mutation. The public GitHub Pages site remains unchanged until the owner or the
already delegated release authority separately executes and verifies the applicable remote gates.

# Aerial OBB Pages Live-review Accessibility Follow-up Design

- **Date:** 2026-08-31
- **Repository:** `kuotunyu/aerial-obb-lab`
- **Base commit:** `00d06f012acc9b4b52417374dd4c23ef84b9797c`
- **Status:** Approved design with central runtime-refinement ruling; implementation plan separately approved
- **Scope:** Gate D follow-up for the deployed dual-mode browser workbench

## Context and boundary

The GitHub Pages workbench is already deployed from the reviewed `demo/web` artifact. A Gate D live
review found two Important issues and three Minor accessibility/metadata issues. This follow-up fixes
only those issues. It does not change detection behavior, model/session handling, Pages configuration,
GitHub About, release metadata, repository visibility, Hugging Face state, or any other repository.

The public artifact remains code, the committed synthetic fixture, the reviewed font and its license, and
other reviewed static presentation assets only. It must not gain model weights, ONNX files, DOTA pixels
or derivatives, private paths, user filenames, telemetry, or new network origins.

## Root-cause record

The failure is deterministic on the exact base commit:

1. Load Synthetic Showcase; its runtime is `N/A · no inference`.
2. Change the confidence threshold or class filter.
3. The cached output is correctly decoded, filtered, drawn, and listed, but the runtime becomes `—`.

`renderCachedOutput()` calls `renderSummary(dets, state.cached.elapsedMs)`. Synthetic cached output has
`elapsedMs: null`; `renderSummary()` renders null as `—`. `activateShowcase()` repairs the initial
presentation only by writing `N/A · no inference` after its first render. Filter-driven re-renders do not
take that path. BYOM stores a measured numeric elapsed time, so it must remain numeric.

The same cached decoded detections already contain image-pixel `cx`, `cy`, `w`, `h`, and `angle`.
`OBB.rotatedCorners()` can derive four corners and the canvas already uses those corners. The visible table
currently exposes only class, confidence, width, height, and angle. It therefore does not provide a
complete textual alternative to the position of each rendered oriented polygon.

The existing real browser smoke tests initial synthetic runtime and filtered row count, but does not check
runtime after a filter re-render or canvas alternative text. It also lacks live assertions for the skip
link, theme colour, and semantic input names.

## Selected design: render-time derived presentation

`renderCachedOutput()` remains the sole cached decode/filter/render path. It derives presentation from
the current state and the just-decoded, currently filtered detections; it does not add a second
presentation cache and does not add a filter-handler-only repair.

### Runtime

- The shared renderer displays exactly `N/A · no inference` only when the current active result identity is
  synthetic: `state.mode === "synthetic" && state.phase === "result" && state.cached !== null`.
- It displays rounded numeric milliseconds only when `state.mode === "byom"` and `elapsedMs` is a finite
  numeric value.
- Initial, loading, reset, error, and no-cache states display `—`, including a failed second Synthetic
  Showcase load after an earlier successful synthetic result.
- The BYOM measurement remains the elapsed time from its actual `session.run`; synthetic never creates a
  session and never receives a synthetic latency value.

This makes the runtime claim a property of the current active result identity rather than an accidental
consequence of a null elapsed-time sentinel or a stale mode value.

### Canvas textual alternative

`index.html` gains a persistent, visually hidden description element with a stable ID. The canvas gains
`aria-describedby` referencing that element. The description is not an `aria-live` region: the existing
live status remains the one concise announcement channel, so filter changes do not cause duplicate speech.

The renderer constructs the description from the same sorted, confidence/class-filtered detection list
used by the visible table. For every detection it includes:

- class;
- confidence;
- centre `x` and `y` in image pixels;
- width and height in image pixels; and
- rotation angle in degrees.

Centre coordinates are selected over four-corner coordinates because the table already presents dimensions
and rotation; together they describe the oriented polygon without adding a large visible coordinates table.
The existing `OBB.rotatedCorners()` remains the single canvas geometry source and is not duplicated.

The description is updated only after the current detection list has rendered successfully. It has explicit
safe values for all non-result paths:

- filtered empty result: no detections match the active filters and the canvas has no oriented polygons;
- initial/reset result: no detection result is available; and
- error result: no detection result is available and no prior detection description remains.

No local filename, local path, model metadata, raw exception, tensor content, or stack trace is permitted
in the description, table, status, or console.

### Semantic document improvements

- A visually hidden skip link is the first focusable item in the document. Its exact label is
  `跳至主要工作區`, its target is `main#mainContent`, and `main` is focusable with `tabindex="-1"`.
  The header and non-collapsible claim notice remain visually first and keep their existing order and
  prominence.
- Add `<meta name="theme-color" content="#edf1f4">`, matching the existing page background.
- Add stable semantic names without changing visible copy: `model` for the ONNX file input, `image` for
  the image file input, `confidence` for the range input, and `class-filter` for every generated class
  checkbox. Checkbox values remain their existing stable class indexes.

## Data flow and state handling

```text
cached output + current confidence/class controls
    -> decode current filtered detections
    -> draw oriented polygons from OBB.rotatedCorners
    -> sort detections for visible table and hidden canvas description
    -> render summary with active-result-derived runtime
    -> publish one non-live textual canvas alternative
```

The canvas and description describe the same filtered result. The display order of the visible table and
the hidden description is confidence-descending. Canvas stroke order is left unchanged because it is not
an accessibility ordering contract.

`resetResult()` and every result-clearing error path reset the description together with the table,
summary, mode badge, provenance, and canvas completion state. Existing safe failure copy and recovery
actions remain unchanged. A renderer exception still follows the existing `RENDER_RESULT` safe path.

## Test design

Implementation starts with the following real browser RED cases in `scripts/browser_smoke.py`; no
source-text-only detector is sufficient.

| Scenario | Required browser assertions |
| --- | --- |
| Synthetic confidence filter | After hiding and restoring the fixture row, mode/provenance remain synthetic, runtime is exactly `N/A · no inference` after each render, and ORT request count remains zero. |
| Synthetic class filter | A non-matching class produces zero visible rows and the explicit empty canvas description; restoring the filter restores the row, its description, and `N/A · no inference`, without ORT. |
| Synthetic retry after deterministic asset failure | A second Showcase selection after success clears the prior cached result and shows `—` while the fixture request deterministically fails, with zero ORT requests; a subsequent successful retry alone restores `N/A · no inference`. Task 2 additionally verifies that the new description has no stale text. |
| BYOM cached filter | After stubbed local-browser inference, filter changes retain a numeric milliseconds runtime rather than `N/A · no inference` or `—`. |
| Canvas alternative | Canvas has the exact `aria-describedby` target; populated text includes the rendered filtered class, confidence, centre coordinates, dimensions, and angle; it is ordered like the table. |
| Reset and failures | Model/image transition, result-clearing inference/schema/render failures, and reset/showcase transitions leave no stale description; empty/error wording contains no private marker, path, raw exception, or stack. |
| Keyboard/document metadata | The first Tab reaches `跳至主要工作區`; activating it focuses or scrolls to `main#mainContent`; the live DOM has theme colour `#edf1f4`; model, image, confidence, and every class checkbox have the approved names. |

Existing browser smoke coverage for no eager ORT, SRI/anonymous CORS, session replacement, safe errors,
desktop/mobile layout, and privacy remains required. The expanded tests must preserve the existing
model-free, deterministic stub and must not use a real model, upload a file, or loosen the external-origin
allowlist.

## Acceptance criteria

- Synthetic runtime never changes from `N/A · no inference` during any active cached-filter re-render, and
  returns to `—` for loading, reset, error, or no-cache state.
- BYOM runtime remains a measured numeric value through cached filter re-renders.
- The canvas has a current, non-live, filtered textual alternative with no stale result content.
- Empty, reset, and error states have explicit safe canvas description text.
- The claim-boundary notice remains before visible controls; the skip link is keyboard-first but visually
  hidden until focused.
- Theme colour and all approved input names are present in the rendered DOM.
- All existing privacy, artifact, Pages, and release checks still pass without changing their configuration.

## Non-goals and remote gates

- Do not add visible corner-coordinate columns, announce every filter change, or change existing display
  copy solely to support these fixes.
- Do not change Pages source/environment, dispatch a workflow, deploy, push, create a pull request,
  modify GitHub About, alter visibility, touch Hugging Face, tag, release, or clean up another worktree.
- This document authorizes no implementation. After central written review, the only next process step is
  `writing-plans`; implementation requires its separately approved plan and strict TDD.

## Self-review record

- **Placeholders:** none; all labels, IDs, values, base SHA, and target state are explicit.
- **Consistency:** runtime is derived from current active-result identity in every shared cached render; no
  synthetic presentation cache conflicts with the loading/reset/error/no-cache state.
- **Scope:** five Gate D findings plus their required browser regressions only; no release or remote change.
- **Ambiguity resolved:** the canvas alternative uses centre coordinates, not four corners; it is hidden and
  non-live, while the visible table remains compact and the existing status region remains the announcer.

# Aerial OBB Workbench UI Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the compact left-control/right-result Aerial OBB workbench while preserving the approved real-image-first flow: the official aerial image is visible before Detect, genuine local-browser inference runs only after the primary action, and the oriented result replaces the original in the same viewport.

**Architecture:** Reorganize the existing semantic HTML into one 31/69 desktop workbench without changing the model, manifest, inference, decode, geometry, or session contracts. Keep the existing application state and cache authoritative; add only derived presentation helpers for the shared original/result viewport and filter availability. Extend the real Playwright harness before production changes, then freeze the reviewed text digests and finish the already-approved real-demo release evidence against the final UI.

**Tech Stack:** Static HTML, CSS, and JavaScript; ONNX Runtime Web 1.20.1/WASM; Python 3.11+; Playwright through Python; pytest; standard-library Pages/release/clean-export gates; `uv`; Git.

**Plan Status:** Written from approved spec commit `af2bf9b78f02155b44d16b101b2a6e2cdbabc667`; pending written plan review. This documentation commit authorizes no product implementation or remote action.

## Global Constraints

- Product behavior starts from exact accepted commit `b6a4cd6193a9c34bd08e437805248dbc9658e3d5`; approved design commit is `af2bf9b78f02155b44d16b101b2a6e2cdbabc667` on `feat/pages-live-real-image-demo`.
- Execute in the existing isolated worktree `<repository-root>\.worktrees\aerial-obb-live-real-image-demo`; do not create, reuse, reset, delete, or clean another worktree.
- Preserve untracked `.superpowers/` visual-companion state. Commit only the exact task paths named below.
- The visible sample remains exact `demo/web/samples/boats.jpg`; the browser model remains exact `demo/web/models/yolo26n-obb-privacy-sanitized.onnx`. Do not regenerate, move, rename, replace, or upload either asset.
- Initial navigation may request same-origin static assets and `samples/boats.jpg`; it must not request `demo-model.json`, ONNX Runtime, WASM, or ONNX bytes before **開始 Detect**.
- Genuine demo and BYOM inference retain the existing shared preprocess, output selection, decode, confidence/class filter, rotated-corner, canvas render, generation-token, candidate-session, and safe-release contracts.
- `state.cached` remains the only inference-result cache. Do not add a presentation cache, precomputed boxes, flattened result image, stored result markup, browser storage, service worker, analytics, telemetry, or upload path.
- Completed inference displays rounded numeric milliseconds. Idle, loading, reset, error, and no-cache states display `—`.
- The claim-boundary notice remains static HTML and precedes every focusable workbench control.
- User-visible and console failures use existing fixed safe copy/codes. No local path, local filename, response body, model metadata, raw exception, stack, token, signed query, or private model identity may enter UI, console, screenshots, reports, commits, or evidence.
- The page remains zh-TW-first, AGPL-3.0-or-later, and explicit that the demo is not accuracy, evaluation, or latency benchmark evidence.
- Workflow action references remain exactly `actions/checkout@v7`, `actions/setup-python@v7`, `actions/setup-node@v7`, `actions/upload-artifact@v7`, `actions/configure-pages@v6`, `actions/upload-pages-artifact@v5`, and `actions/deploy-pages@v5` wherever those actions are used.
- Desktop width `>= 960px` uses approximately 31/69 columns. Width `< 960px`, 390×844, and the existing 200%-zoom-equivalent viewport use the approved single-column order without horizontal page overflow.
- Use strict browser-first TDD for every visible or behavioral change. In a batch RED, record only the earliest assertion actually reached; do not claim later blocked assertions were separately observed.
- Every implementation task gets a fresh implementer, fresh spec review, fresh quality review, and fix/re-review loop when using subagent-driven development. Keep one implementer active at a time.
- Local files, loopback preview, repo-external temporary screenshots/reports, tests, reviews, and commits are authorized only after this plan is approved.
- Push, PR creation, merge, workflow dispatch, Pages configuration/deployment, GitHub About, Hugging Face operations, release, tag, visibility changes, branch deletion, and worktree cleanup are not authorized by this plan.

---

## Execution Workspace and Ledger

Before Task 1, read `superpowers:using-git-worktrees`, `superpowers:test-driven-development`, `superpowers:subagent-driven-development`, and `frontend-design`. Verify the retained worktree rather than creating another one:

```powershell
$worktree = '<repository-root>\.worktrees\aerial-obb-live-real-image-demo'
$branch = (git -C $worktree branch --show-current).Trim()
$design = (git -C $worktree rev-parse af2bf9b78f02155b44d16b101b2a6e2cdbabc667).Trim()
$product = (git -C $worktree rev-parse b6a4cd6193a9c34bd08e437805248dbc9658e3d5).Trim()
if ($branch -ne 'feat/pages-live-real-image-demo') { throw 'Unexpected implementation branch' }
if ($design -ne 'af2bf9b78f02155b44d16b101b2a6e2cdbabc667') { throw 'Approved design commit is unavailable' }
if ($product -ne 'b6a4cd6193a9c34bd08e437805248dbc9658e3d5') { throw 'Accepted product base is unavailable' }
git -C $worktree merge-base --is-ancestor $product HEAD
git -C $worktree merge-base --is-ancestor $design HEAD
git -C $worktree diff --quiet
if ($LASTEXITCODE -ne 0) { throw 'Tracked worktree is dirty' }
git -C $worktree diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Index is dirty' }
```

Create the ignored plan ledger at:

```text
.superpowers/sdd/2026-09-01-aerial-obb-workbench-ui-restoration/progress.md
```

Record the exact starting HEAD, task dispatch, first reached RED, GREEN commands and output, exact staged paths, commit SHA, spec-review verdict, quality-review verdict, fix rounds, canonical text digests, screenshot locations, known cross-task blockers, and final review. Never stage `.superpowers/`.

## File Structure and Interfaces

| Path | Responsibility in this plan |
| --- | --- |
| `demo/web/index.html` | Static claim notice, compact control rail, official sample card, shared result viewport layers, result summary/table, filters, and collapsed BYOM semantics. |
| `demo/web/style.css` | 31/69 desktop workbench, deep-navy shared viewport, compact original visual language, filter-disabled presentation, mobile order, zoom, focus, and reduced motion. |
| `demo/web/app.js` | Existing state machine plus derived layer visibility, filter availability, explicit empty table state, and generic BYOM-original presentation. |
| `scripts/browser_smoke.py` | Real Playwright RED/GREEN coverage for initial layout, shared viewport, genuine inference, cached controls, BYOM transition, failure cleanup, accessibility, responsive order, origins, and privacy. |
| `scripts/pages_artifact_check.py` | Canonical-LF reviewed text digests for the final HTML/CSS/application bytes; all inventory/origin/privacy/model restrictions remain unchanged. |
| `tests/test_pages_artifact_check.py` | Mutation proof that the final reviewed pages tree passes and changed HTML/CSS/application bytes fail closed. |
| `scripts/repo_check.py`, `scripts/release_check.py`, `scripts/clean_export_check.py` | Finish the previously approved real-demo repository, evidence, and exact derivative-model release boundary against the final UI. |
| `tests/test_repo_check.py`, `tests/test_release_check.py`, `tests/test_clean_export.py`, `tests/test_package_release.py` | RED/GREEN contracts for final real-demo claims, workbench evidence, clean export, and real-demo CI naming. |
| `release/artifact-manifest.json`, `release/evidence.json` | Exact final asset/screenshot inventory and truthful real-demo/workbench evidence. |
| `README.md`, `README.en.md`, `demo/web/README.md`, `THIRD_PARTY_NOTICES.md`, `CHANGELOG.md` | Final user journey, source/license/privacy boundary, and historical-vs-current demo language. |
| `.github/workflows/release-gates.yml` | Real-demo browser-smoke labels only; workflow behavior and current action majors remain unchanged. |
| `docs/assets/browser-workbench.png` | Canonical desktop result screenshot generated from the exact final local artifact after genuine demo inference. |

Production presentation helpers have these exact interfaces:

```javascript
setFilterAvailability(available) // available: boolean; mutates disabled/aria-disabled/help copy only
renderEmptyTable(message)        // message: fixed safe string; creates one colspan=5 empty row
showOriginalSource(source)       // source: "demo" | "byom"; shows one original-image layer
setResultView(view)              // existing API; view: "original" | "result"; returns detections or null
renderCachedOutput()             // existing API; the sole cached filter/rerender path
clearResultState({keepImage})    // existing API; the sole reset/failure result cleanup path
```

`showOriginalSource` derives the visible original layer from `state.imageSource`; it stores no result or view cache. `setResultView` remains the only original/result toggle entry point.

---

### Task 1: Lock the Semantic Workbench and Desktop Layout

**Files:**
- Modify: `scripts/browser_smoke.py`
- Modify: `demo/web/index.html`
- Modify: `demo/web/style.css`

**Interfaces:**
- Consumes: existing static IDs, initial zero-runtime request contract, and `assert_real_demo_initial`.
- Produces: `assert_workbench_initial_layout(page: object) -> None`, `run_workbench_layout(executable_path: Path | None = None, base_url: str | None = None) -> None`, `#workbench`, `#controlRail`, `#sampleCard`, `#resultViewport`, `#demoFigure`, `#viewportByomImage`, and unchanged result/control IDs for Task 2.

- [ ] **Step 1: Add the real-browser desktop layout batch RED**

Add `assert_workbench_initial_layout(page: object) -> None` to `scripts/browser_smoke.py` and call it from a new `run_workbench_layout` scenario after `assert_real_demo_initial`. Use live DOM and bounding boxes, not source grep:

```python
def assert_workbench_initial_layout(page: object) -> None:
    if page.locator("#demoDetectBtn").evaluate(
        "node => node.closest('#controlRail')?.id || ''"
    ) != "controlRail":
        raise RuntimeError("demo action is not inside the compact control rail")
    if page.locator("#sampleCard").count() != 1:
        raise RuntimeError("official sample card is missing")
    rail = page.locator("#controlRail").bounding_box()
    viewport = page.locator("#resultViewport").bounding_box()
    if rail is None or viewport is None or viewport["x"] <= rail["x"] + rail["width"]:
        raise RuntimeError("desktop viewport is not to the right of the control rail")
    ratio = rail["width"] / (rail["width"] + viewport["width"])
    if not 0.27 <= ratio <= 0.35:
        raise RuntimeError(f"desktop workbench is not approximately 31/69: {ratio!r}")
    if page.locator(".demo-intro, .demo-action-zone").count() != 0:
        raise RuntimeError("retired full-width demo layout is still present")
```

Register `workbench-layout` in the CLI choices, dispatch, full scenario, and fixed OK output.

- [ ] **Step 2: Run the scenario and record the earliest actual RED**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario workbench-layout
```

Expected earliest RED on commit `b6a4cd6`: `demo action is not inside the compact control rail`. Do not claim the missing viewport, ratio, or retired-layout checks as separately observed until the earlier assertion is made reachable.

- [ ] **Step 3: Reorganize the static document without changing IDs or claims**

Replace the full-width `demo-intro`, standalone figure, and `demo-action-zone` with this exact box-tree mapping. Keep the existing claim notice before `<main>` and keep all existing control/result IDs exactly once:

```text
#mainContent
└─ #workbench.workbench
   ├─ #controlRail.control-rail
   │  ├─ .rail-heading (#controlsTitle + instruction)
   │  ├─ #sampleCard.sample-card
   │  ├─ #status.status
   │  ├─ #runtimeRetryBtn.runtime-retry
   │  ├─ #resultControls.result-controls (move the complete current filter subtree here)
   │  └─ #byomPanel.byom-panel (move the complete current disclosure here; inner wrapper becomes #byomControls)
   └─ #resultWorkspace.result-workspace
      ├─ .result-heading (.result-heading-copy + unchanged five-field .result-summary)
      ├─ #resultViewport.result-viewport (#demoFigure + #canvasFrame)
      └─ .detections (move the complete current table subtree here)
```

Add the new sample card and shared viewport with these exact new elements; move the unchanged filter, BYOM, summary, and table descendants according to the mapping:

```html
<main id="mainContent" tabindex="-1">
  <div id="workbench" class="workbench">
    <aside id="controlRail" class="control-rail" aria-labelledby="controlsTitle">
      <div class="rail-heading">
        <h2 id="controlsTitle">範例與設定</h2>
        <p>先查看原圖，再由目前的 browser 執行 Detect。</p>
      </div>
      <article id="sampleCard" class="sample-card" aria-labelledby="sampleTitle">
        <img id="sampleThumbnail" src="samples/boats.jpg" alt="" aria-hidden="true" />
        <div><h3 id="sampleTitle">官方港區航拍範例</h3><p id="sampleState">Original · ready</p></div>
        <button id="demoDetectBtn" type="button" aria-describedby="claimTitle">開始 Detect</button>
        <button id="viewToggleBtn" class="secondary-action" type="button" hidden>查看原圖</button>
      </article>
      <p id="status" class="status" role="status" aria-live="polite">原圖已載入 · 尚未 Detect。</p>
      <button id="runtimeRetryBtn" class="runtime-retry" type="button" hidden>重試載入 Browser runtime</button>
    </aside>
    <section id="resultWorkspace" class="result-workspace" aria-labelledby="resultTitle">
      <div id="resultViewport" class="result-viewport">
        <figure id="demoFigure" class="viewport-original">
          <figcaption id="demoFigureLabel">原圖 · 尚未 Detect</figcaption>
          <img id="demoOriginalImage" src="samples/boats.jpg" width="1920" height="1080"
               alt="海面與碼頭附近的多艘船隻航拍影像" />
          <img id="viewportByomImage" hidden alt="使用者選擇的航拍影像" />
        </figure>
        <div class="canvas-frame" id="canvasFrame" hidden>
          <canvas id="canvas" aria-label="Oriented detection result" aria-describedby="canvasDescription"></canvas>
          <p id="canvasDescription" class="visually-hidden">尚無 detection result。</p>
        </div>
      </div>
    </section>
  </div>
</main>
```

The BYOM disclosure stays inside `#controlRail` after the filter controls. Rename only its inner visual wrapper from duplicate-purpose `#controlRail` to `#byomControls`; retain `#modelInput`, `#fileInput`, `#detectBtn`, labels, names, and disclosure copy. In the Task 1 accessibility test, replace the obsolete `resultWorkspace`-before-`byomPanel` assertion with exact checks that `#byomPanel` is collapsed and its closest rail is `#controlRail`.

- [ ] **Step 4: Implement the minimum desktop visual system**

Delete rules used only by `.demo-intro`, `.demo-figure`, and `.demo-action-zone`. Establish one compact desktop workbench and one stable viewport:

```css
.workbench {
  display: grid;
  grid-template-columns: minmax(320px, 31fr) minmax(0, 69fr);
  gap: 16px;
  align-items: start;
}

.control-rail {
  min-width: 0;
  padding: 16px;
  border: 1px solid var(--line);
  background: var(--surface);
}

.sample-card {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  gap: 10px 12px;
  margin-top: 12px;
}

.sample-card button,
.sample-card .secondary-action {
  grid-column: 1 / -1;
}

#sampleThumbnail {
  width: 96px;
  aspect-ratio: 16 / 9;
  object-fit: contain;
  background: var(--canvas);
}

.result-viewport {
  position: relative;
  display: grid;
  min-height: 410px;
  overflow: hidden;
  border: 1px solid var(--canvas-line);
  background: var(--canvas);
}

.viewport-original,
.canvas-frame {
  grid-area: 1 / 1;
  width: 100%;
  min-height: 410px;
  margin: 0;
}

.viewport-original img,
.canvas-frame canvas {
  width: 100%;
  height: 100%;
  max-height: 68vh;
  object-fit: contain;
}

#byomControls {
  display: grid;
  gap: 12px;
  padding: 12px;
  border-top: 1px solid var(--line);
}

#byomControls .file-stack {
  grid-template-columns: minmax(0, 1fr);
}
```

Keep the existing palette, rectangular borders, typefaces, focus styles, table density, and deep-navy viewport. Do not introduce hero sizing, rounded marketing cards, gradients, or decorative animation.

- [ ] **Step 5: Reach all Task 1 layout assertions and run GREEN**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario workbench-layout
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility
uv run --no-sync python scripts/browser_smoke.py --scenario desktop
git diff --check
```

Expected: all three scenarios pass; initial requests still omit manifest/runtime/WASM/model; claim order, skip link, labels, source/license links, and collapsed BYOM remain valid.

- [ ] **Step 6: Review and commit Task 1**

Fresh reviewers compare the real 1280×720 page with the approved design and inspect semantic/quality scope. Resolve findings through the Task 1 implementer and rerun Step 5. Stage exactly:

```powershell
git add demo/web/index.html demo/web/style.css scripts/browser_smoke.py
git diff --cached --check
git commit -m "feat: restore compact OBB workbench layout"
```

---

### Task 2: Unify Original/Result State and Cached Controls

**Files:**
- Modify: `scripts/browser_smoke.py`
- Modify: `demo/web/index.html`
- Modify: `demo/web/style.css`
- Modify: `demo/web/app.js`

**Interfaces:**
- Consumes: Task 1 selectors and existing `state`, `setResultView`, `renderCachedOutput`, `clearResultState`, demo/BYOM inference, and failure scenarios.
- Produces: visible-but-disabled pre-result filters, one explicit empty table row, shared image/canvas bounds, generic BYOM-original layer, and unchanged inference/cache/session behavior.

- [ ] **Step 1: Add the initial/control/viewport behavior batch RED**

Extend the `workbench-layout`, `real-demo-success`, `stubbed-cache`, and `byom-transition` scenarios with real DOM assertions:

```python
controls = page.locator("#resultControls")
if not controls.is_visible():
    raise RuntimeError("result filters are not visible before Detect")
if not page.locator("#confSlider").is_disabled():
    raise RuntimeError("confidence filter is enabled without cached output")
if page.locator(".class-cb:not(:disabled)").count() != 0:
    raise RuntimeError("class filters are enabled without cached output")
if page.locator("#resultsBody tr[data-empty='true']").count() != 1:
    raise RuntimeError("initial table lacks one explicit empty state")
```

After successful genuine inference, assert the controls are enabled, the result canvas is visible, runtime remains numeric, and the canvas box matches `#resultViewport` within one CSS pixel. Assert the result-view action is exact **查看原圖**; toggle to original, assert the active original layer box matches the same viewport within one pixel, and assert the action becomes exact **查看結果**. Change confidence and class controls and assert the run counter, request count, runtime, current view, table, description, and rendered polygons remain synchronized without another inference.

In `byom-transition`, select the existing non-sensitive test image and assert `#viewportByomImage` becomes the visible original layer before BYOM Detect while neither its text nor body text contains the local test filename.

- [ ] **Step 2: Observe the earliest actual RED**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario workbench-layout
```

Expected earliest RED after Task 1: `result filters are not visible before Detect`, because current `resetResult` hides `#resultControls`. Later empty-row, enabled-state, layer-bound, and BYOM checks are recorded only when independently reached.

- [ ] **Step 3: Add derived presentation helpers and explicit empty state**

Keep `state.cached` authoritative. Add these minimum helpers to `app.js`:

```javascript
function setFilterAvailability(available) {
  confSlider.disabled = !available;
  classList.querySelectorAll(".class-cb").forEach((checkbox) => {
    checkbox.disabled = !available;
  });
  resultControls.dataset.ready = available ? "true" : "false";
  resultControls.setAttribute("aria-disabled", String(!available));
  filterAvailability.textContent = available
    ? "調整 filters 只會重繪目前的 cached result。"
    : "Detect 完成後即可調整 filters。";
}

function renderEmptyTable(message) {
  resultsBody.innerHTML = "";
  const row = document.createElement("tr");
  row.dataset.empty = "true";
  const cell = document.createElement("td");
  cell.colSpan = 5;
  cell.textContent = message;
  row.appendChild(cell);
  resultsBody.appendChild(row);
}

function showOriginalSource(source) {
  const byom = source === "byom";
  demoOriginalImage.hidden = byom;
  viewportByomImage.hidden = !byom;
  demoFigure.hidden = false;
  canvasFrame.hidden = true;
}
```

Add static `<p id="filterAvailability">Detect 完成後即可調整 filters。</p>` inside `#resultControls`; remove its `hidden` attribute. `resetResult()` calls `setFilterAvailability(false)` and `renderEmptyTable("尚未執行 Detect。")`. `fillTable([])` calls `renderEmptyTable("目前篩選條件下沒有 detections。")`; non-empty results replace the row. Successful `runActiveInference` calls `setFilterAvailability(true)` only after `state.cached` is assigned and `state.phase` is `result`.

- [ ] **Step 4: Make both original sources use the shared viewport**

`resetToDemoOriginal`, demo loading, demo failure, and `setResultView("original")` call `showOriginalSource(state.imageSource === "byom" ? "byom" : "demo")`. Result view hides `#demoFigure`, shows `#canvasFrame`, and renders the cached output.

Decode a selected BYOM image into `#viewportByomImage` using its object URL; assign that element to `state.image` only after successful decode and current-generation validation. Revoke the object URL after decode, keep the generic alt text, and never copy `File.name` into DOM or console. Demo reset restores `state.image = demoOriginalImage` and `showOriginalSource("demo")`.

Do not alter `preprocess`, `OBB.selectEndToEndOutput`, `OBB.decodeDetections`, `OBB.rotatedCorners`, manifest validation, runtime loading, candidate-session replacement, or release timing.

- [ ] **Step 5: Make disabled and active controls visually unambiguous**

Keep controls visible in every state. Use native `:disabled`, subdued opacity, and `cursor: not-allowed`; do not simulate disabled controls with pointer-events alone:

```css
.result-controls[data-ready="false"] {
  background: var(--surface-muted);
}

.result-controls :disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

#filterAvailability {
  grid-column: 1 / -1;
  margin: 0;
  color: var(--muted);
  font-size: 15px;
}
```

- [ ] **Step 6: Run focused GREEN and all affected failure paths**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario workbench-layout
uv run --no-sync python scripts/browser_smoke.py --scenario real-demo-success
uv run --no-sync python scripts/browser_smoke.py --scenario stubbed-cache
uv run --no-sync python scripts/browser_smoke.py --scenario manifest-failure
uv run --no-sync python scripts/browser_smoke.py --scenario runtime-failure
uv run --no-sync python scripts/browser_smoke.py --scenario session-failure
uv run --no-sync python scripts/browser_smoke.py --scenario run-failure
uv run --no-sync python scripts/browser_smoke.py --scenario output-failure
uv run --no-sync python scripts/browser_smoke.py --scenario render-failure
uv run --no-sync python scripts/browser_smoke.py --scenario stale-generation
uv run --no-sync python scripts/browser_smoke.py --scenario byom-transition
git diff --check
```

Expected: every scenario passes. Failure/reset states restore the correct original, runtime `—`, one empty row, disabled filters, empty synchronized canvas description, hidden toggle, no stale polygons, and actionable fixed copy. Genuine demo and BYOM success retain numeric runtime and do not rerun from toggles or filters.

- [ ] **Step 7: Review and commit Task 2**

Fresh spec and quality reviewers inspect every state transition, privacy surface, generation race, and session release. Resolve findings through the Task 2 implementer and rerun Step 6. Stage exactly:

```powershell
git add demo/web/index.html demo/web/style.css demo/web/app.js scripts/browser_smoke.py
git diff --cached --check
git commit -m "feat: unify real-image result presentation"
```

---

### Task 3: Enforce Mobile Order, Zoom, and Accessibility

**Files:**
- Modify: `scripts/browser_smoke.py`
- Modify: `demo/web/index.html`
- Modify: `demo/web/style.css`

**Interfaces:**
- Consumes: Task 2 complete workbench and existing accessibility/responsive scenarios.
- Produces: exact `<960px` single-column visual order, 390×844 and 200%-zoom safety, preserved focus/heading/label/description contracts, and repo-external review screenshots.

- [ ] **Step 1: Add the responsive visual-order batch RED**

Extend `_assert_responsive_layout` to compare live element bounding boxes. For mobile and the 640×360 200%-zoom-equivalent viewport, require this top-to-bottom order:

```python
ordered = [
    "#sampleCard",
    "#resultViewport",
    ".result-summary",
    "#resultControls",
    ".detections",
    "#byomPanel",
]
boxes = [page.locator(selector).bounding_box() for selector in ordered]
if any(box is None for box in boxes):
    raise RuntimeError(f"{label} layout hides an ordered workbench section")
tops = [box["y"] for box in boxes if box is not None]
if tops != sorted(tops):
    raise RuntimeError(f"{label} workbench visual order is wrong: {tops!r}")
```

At 1280×720, retain the right-of-rail assertion. At every width, retain overflow offender reporting and visibility of sample image, Detect, status, Source, code license, model license, and sanitization links.

- [ ] **Step 2: Observe the first actual mobile RED**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario mobile
```

Expected earliest RED after Task 2: `mobile workbench visual order is wrong`, because the desktop rail/result containers have not yet been flattened and reordered for the approved mobile sequence. Record the actual first reached assertion if overflow or a hidden element fails earlier.

- [ ] **Step 3: Implement the exact 960px responsive composition**

Replace the old 900px breakpoint with 959px. At that breakpoint, make the two semantic containers participate in one visual flow and assign explicit order without duplicating content:

```css
@media (max-width: 959px) {
  .workbench {
    display: flex;
    flex-direction: column;
  }

  #controlRail,
  #resultWorkspace,
  .result-heading {
    display: contents;
  }

  .rail-heading { order: 1; }
  .sample-card { order: 2; }
  #status { order: 3; }
  #runtimeRetryBtn { order: 4; }
  .result-heading-copy { order: 5; }
  #resultViewport { order: 6; }
  .result-summary { order: 7; }
  #resultControls { order: 8; }
  .detections { order: 9; }
  #byomPanel { order: 10; }
}
```

Give each ordered surface its own border/background/padding because the `display: contents` container boxes do not render. Retain the current internal table scroller. At 560px and below, keep two summary columns and allow only the existing table scroller to overflow internally.

- [ ] **Step 4: Extend accessibility checks for the reorganized page**

In `run_accessibility`, assert:

- skip link remains the first focus target and focuses `main#mainContent`;
- claim notice precedes `#demoDetectBtn` in DOM and visual position;
- headings are `h1`, control/result `h2`, sample/table/BYOM `h3` or equivalent logical disclosure labeling;
- all stable `name` and label counts remain exact;
- canvas keeps one non-live `aria-describedby` target synchronized with the table;
- status remains the only polite live result/failure region;
- initial keyboard order reaches Detect before filters, reaches BYOM disclosure after filters, and reaches source links after workbench controls;
- reduced-motion removes result animation and visible transitions;
- focus-visible outlines remain at least 3 CSS pixels for primary buttons, disclosure, table scroller, and source links.

- [ ] **Step 5: Run responsive/accessibility GREEN and capture local review evidence**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario desktop
uv run --no-sync python scripts/browser_smoke.py --scenario mobile
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility
$reviewRoot = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-workbench-review-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $reviewRoot | Out-Null
uv run --no-sync python scripts/browser_smoke.py --screenshot (Join-Path $reviewRoot 'desktop.png') --mobile-screenshot (Join-Path $reviewRoot 'mobile.png')
git diff --check
```

Inspect both images for uncropped original/result media, notice prominence, compact rail density, no stale labels, readable summary/table/links, and no horizontal page overflow. Keep this review root outside the repository and record only the generic location category and verdict in the ignored ledger.

- [ ] **Step 6: Review and commit Task 3**

Fresh reviewers inspect desktop, mobile, zoom, keyboard, semantic, and visual evidence. Resolve findings through the Task 3 implementer and rerun Step 5. Stage exactly:

```powershell
git add demo/web/index.html demo/web/style.css scripts/browser_smoke.py
git diff --cached --check
git commit -m "fix: preserve responsive workbench accessibility"
```

---

### Task 4: Freeze the Final Pages Artifact and Real-demo Evidence

**Files:**
- Modify: `scripts/pages_artifact_check.py`
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `scripts/repo_check.py`
- Modify: `scripts/release_check.py`
- Modify: `scripts/clean_export_check.py`
- Modify: `tests/test_repo_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `tests/test_package_release.py`
- Modify: `release/artifact-manifest.json`
- Modify: `release/evidence.json`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `demo/web/README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `CHANGELOG.md`
- Modify: `.github/workflows/release-gates.yml`
- Modify: `docs/assets/browser-workbench.png`

**Interfaces:**
- Consumes: reviewed Task 1–3 bytes and the already-approved privacy-sanitized real-demo release contract.
- Produces: exact text/screenshot digests, truthful real-demo workbench evidence, one derivative-model clean-export exception, current CI labels, and a fully verifiable local candidate.

- [ ] **Step 1: Observe the artifact digest RED before updating any digest**

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_current_pages_tree_passes -q
```

Expected RED: reviewed HTML, application, and/or stylesheet bytes differ. This proves the fail-closed artifact gate detects the reviewed UI change. Do not alter digest enforcement or add a broad exception.

- [ ] **Step 2: Add release/evidence/clean-export RED contracts**

Replace stale Synthetic/code-only assertions with these exact named tests:

- `test_browser_demo_evidence_is_genuine_local_inference_with_privacy_sanitized_derivative`
- `test_browser_demo_has_one_canonical_real_demo_source_path`
- `test_browser_ui_evidence_matches_restored_workbench`
- `test_clean_export_admits_only_the_reviewed_derivative_demo_model`
- `test_ci_names_the_real_demo_browser_smoke`

Require:

```python
assert browser["distribution_mode"] == "public-agpl-privacy-sanitized-demo-model-plus-byom"
assert browser["showcase_enabled"] is False
assert browser["demo_inference_performed"] is True
assert browser["model_bundled"] is True
assert browser["demo_image"] == "demo/web/samples/boats.jpg"
assert browser["demo_model"] == "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
assert browser["runtime_load"] == "lazy-on-demo-detect-or-byom-selection"
assert browser["layout"] == "workbench-31-69"
assert browser["responsive_breakpoint_px"] == 960
assert browser["primary_action_first_viewport"] is True
assert browser["represents_fine_tuned_medium_accuracy"] is False
assert browser["represents_t4_latency"] is False
```

The exact derivative path is the only `.onnx` allowed in the Pages tree and clean export, and only when digest, bytes, sanitization receipt, manifest, license, and artifact manifest agree. Source-model digest and every second model remain forbidden.

Run and record the earliest actual stale-evidence failure:

```powershell
uv run --no-sync python -m pytest tests/test_release_check.py::test_browser_demo_evidence_is_genuine_local_inference_with_privacy_sanitized_derivative -q
uv run --no-sync python -m pytest tests/test_clean_export.py::test_clean_export_admits_only_the_reviewed_derivative_demo_model -q
uv run --no-sync python -m pytest tests/test_package_release.py::test_ci_names_the_real_demo_browser_smoke -q
```

- [ ] **Step 3: Freeze only the reviewed changed text bytes**

Compute canonical-LF SHA-256 for `index.html`, `style.css`, and `app.js`; update their exact entries in `REVIEWED_TEXT_DIGESTS`. Do not change the digest algorithm, mutation tests, allowed origins, inventory, file-size caps, model exception, license checks, or binary privacy scans.

Confirm the frozen assets are byte-identical to the accepted product base:

```powershell
git diff --exit-code b6a4cd6193a9c34bd08e437805248dbc9658e3d5 -- demo/web/demo-assets.js demo/web/obb.js demo/web/demo-model.json demo/web/samples/boats.jpg demo/web/models/yolo26n-obb-privacy-sanitized.onnx demo/web/fonts demo/web/third_party demo/web/THIRD_PARTY_NOTICES.md
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
```

- [ ] **Step 4: Finish truthful current release evidence and documentation**

Update the release JSON and documentation to the real flow: official original is visible at first paint; first Detect lazily loads pinned SRI runtime plus exact same-origin privacy-sanitized derivative; inference stays local; result/original toggle and filters use cached output; BYOM remains advanced. Preserve DOTAv1 training provenance, AGPL modification record, no-endorsement language, `commercial_use_cleared: false`, and the exclusions for accuracy/evaluation/T4 latency.

`browser_demo.source_files` is exactly this sorted public evidence set and contains no deleted Synthetic fixture path:

```json
[
  "demo/web/THIRD_PARTY_NOTICES.md",
  "demo/web/app.js",
  "demo/web/demo-assets.js",
  "demo/web/demo-model.json",
  "demo/web/fonts/IBM-Plex-OFL.txt",
  "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
  "demo/web/index.html",
  "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
  "demo/web/obb.js",
  "demo/web/samples/boats.jpg",
  "demo/web/style.css",
  "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
  "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
  "docs/assets/browser-workbench.png"
]
```

`release/artifact-manifest.json` records exact bytes and digests for the reviewed image, derivative model, license, sanitization record, font, final reviewed text files, and canonical screenshot.

Rename only the CI job/step display text:

```yaml
browser-smoke:
  name: Live demo browser smoke / Ubuntu CPU
  # existing locked setup remains unchanged
  - name: Exercise the real-image browser demo and BYOM safety paths
    run: uv run --no-sync python scripts/browser_smoke.py
```

Keep the current official action majors, candidate artifact, CPU-only settings, and manual-only Pages workflow unchanged. Do not dispatch the workflow.

- [ ] **Step 5: Generate and inspect the canonical screenshot from exact final bytes**

```powershell
uv run --no-sync python scripts/browser_smoke.py --screenshot docs/assets/browser-workbench.png
```

Required pixels: compact 31/69 workbench, genuine annotated real image in the right viewport, numeric runtime, exact mode/provenance, visible result table, no open BYOM ready labels, and readable claim/source/license context. Inspect PNG metadata and privacy sentinels; the screenshot must contain no path, filename, private model identity, raw error, stack, token, or query. Freeze its actual digest and byte length in `release/artifact-manifest.json`.

- [ ] **Step 6: Run Task 4 GREEN**

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py tests/test_repo_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_package_release.py tests/test_readme_language.py -q
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/browser_smoke.py
git diff --check
```

Expected: focused tests and all four direct gates pass; the browser smoke exercises every real-demo, cache, failure, race, BYOM, accessibility, desktop, and mobile scenario.

- [ ] **Step 7: Review and commit Task 4**

Fresh reviewers inspect exact bytes/digests, legal/source language, evidence schema, workflow-only label diff, screenshot pixels/metadata, model exception, and privacy/origin scans. Resolve findings through the Task 4 implementer and rerun Step 6. Stage exactly the Task 4 file list and commit:

```powershell
git add scripts/pages_artifact_check.py tests/test_pages_artifact_check.py scripts/repo_check.py scripts/release_check.py scripts/clean_export_check.py tests/test_repo_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_package_release.py release/artifact-manifest.json release/evidence.json README.md README.en.md demo/web/README.md THIRD_PARTY_NOTICES.md CHANGELOG.md .github/workflows/release-gates.yml docs/assets/browser-workbench.png
git diff --cached --check
git commit -m "release: align evidence with restored real demo"
```

---

### Task 5: Complete Local Acceptance and Operable Preview

**Files:**
- Create: repo-external clean export, desktop/mobile screenshots, network report, and local review notes only.
- No tracked file is planned. A final review finding returns to the responsible Task 1–4 file list, starts with a focused RED, and receives its own exact fix commit before re-review.

**Interfaces:**
- Consumes: committed Tasks 1–4.
- Produces: full local verification evidence, strict clean export, broad review verdict, clean retained branch, and loopback preview for owner operation.

- [ ] **Step 1: Verify committed state and test inventory**

```powershell
git status --short
git diff --check
git log --oneline --decorate -12
uv run --no-sync python -m pytest --collect-only -q
```

Expected: tracked worktree and index are clean; only preserved `.superpowers/` is untracked; collected test count is at least the current 206-test baseline.

- [ ] **Step 2: Run full focused and repository regression**

```powershell
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_demo_assets.py tests/test_model_parity_smoke.py tests/test_sanitize_demo_model.py tests/test_pages_artifact_check.py tests/test_clean_export.py tests/test_release_check.py tests/test_package_release.py -q
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected: every focused test, all browser scenarios, at least 206 collected tests, and all direct gates pass with zero failures.

- [ ] **Step 3: Run strict clean export without weakening browser coverage**

```powershell
$cleanExport = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-workbench-clean-export-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $cleanExport) { throw 'Refusing to overwrite existing clean export' }
uv run --no-sync python scripts/clean_export_check.py --output $cleanExport
```

Do not pass `--skip-browser`. Expected: snapshot rebuild, package build/install/import, full tests, links, exact derivative/image/license/sanitization inventory, privacy/origin gates, and real browser smoke pass from the exported committed tree.

- [ ] **Step 4: Run privacy, forbidden-artifact, origin, and frozen-asset scans**

```powershell
rg -n -I -i "Synthetic Showcase|OBB_SHOWCASE|showcase-fixture|fixtures/showcase|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|[A-Z]:[\\/]Users[\\/]" demo/web README.md README.en.md release THIRD_PARTY_NOTICES.md .github/workflows
git diff --exit-code b6a4cd6193a9c34bd08e437805248dbc9658e3d5 -- demo/web/demo-assets.js demo/web/obb.js demo/web/demo-model.json demo/web/samples/boats.jpg demo/web/models/yolo26n-obb-privacy-sanitized.onnx demo/web/fonts demo/web/third_party
git ls-files demo/web
```

Expected: no current public Synthetic/private/token/path hit; historical changelog text is reviewed in context; frozen assets are unchanged; public inventory contains no extra model, DOTA pixels, archive, private receipt, or local-only evidence. Browser request evidence permits only loopback/same-origin assets plus the pinned jsDelivr runtime/WASM after Detect; initial navigation remains same-origin only.

- [ ] **Step 5: Serve and inspect the exact committed local artifact**

```powershell
uv run --no-sync python -m http.server 8765 --bind 127.0.0.1 --directory demo/web
```

Open `http://127.0.0.1:8765/`. At 1280×720, 390×844, and the 200%-zoom-equivalent viewport, verify: real original before Detect; compact rail/right viewport; genuine result after Detect; numeric runtime; original/result toggle; cached confidence/class filters; secondary collapsed BYOM; no overflow; claim/Source/AGPL/model/sanitization readability; keyboard focus; labels/names/headings; non-live description; reduced motion; console/page errors; and request origins. Store any additional screenshots outside the repository.

- [ ] **Step 6: Broad whole-branch review and one bounded fix wave**

Provide the final reviewer with the approved spec, this plan, the complete accepted-product-base-to-HEAD diff, task ledger/reports, test outputs, canonical screenshot, repo-external desktop/mobile images, clean-export result, and origin/privacy evidence. Critical or Important findings require one focused test-first fix wave through the responsible implementer and fresh re-review. Record Minor findings without expanding scope. A second product fix wave, asset drift, privacy leak, unexpected origin, inference/model contract failure, or unresolved Important finding stops local completion.

- [ ] **Step 7: Verification-before-completion and branch readiness**

```powershell
git status --short
git diff --check
git log --oneline --decorate -20
git diff --name-status b6a4cd6193a9c34bd08e437805248dbc9658e3d5...HEAD
git diff --quiet
git diff --cached --quiet
```

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch` only to confirm readiness and present integration choices. Keep the feature branch, worktree, `.superpowers/`, loopback preview, and repo-external evidence for feedback. Do not select or execute an integration or remote option.

---

## Remote Gates A–E — Separate and Unauthorized Here

1. **Gate A — Candidate PR:** re-check `origin/main`, branch race, auth, templates, checks, non-force push, and exactly one PR.
2. **Gate B — Candidate/integration:** download the exact CI artifact, compare every reviewed byte/canonical text digest, repeat browser/privacy/license review, then use only a repository-supported merge method.
3. **Gate C — Pages:** configure or dispatch only from the exact reviewed merged-main SHA after its automatic release gates succeed.
4. **Gate D — Live review:** validate HTTPS/assets, initial zero-model network, genuine Detect, cached controls, failures/BYOM, responsive/accessibility/privacy, and deployed SHA.
5. **Gate E — About and Portfolio Control receipt:** change homepage metadata and record portfolio completion only after independent Gate D success.

No task in this plan authorizes push, PR, merge, Pages, About, Hugging Face, release, tag, visibility changes, branch deletion, or worktree cleanup.

## Self-review Checklist

- **Spec coverage:** Tasks 1–3 cover the approved 31/69 workbench, shared original/result viewport, filters, table, BYOM, mobile order, visual system, accessibility, privacy, and real browser RED/GREEN behavior. Task 4 covers reviewed digests/evidence and the unfinished real-demo release boundary. Task 5 covers full regression, strict clean export, preview, and broad review.
- **Type and selector consistency:** `#workbench`, `#controlRail`, `#sampleCard`, `#resultViewport`, `#demoFigure`, `#viewportByomImage`, `#resultControls`, `#byomPanel`, and all pre-existing control/result IDs are defined once and used consistently. Presentation helper names and argument domains match across tasks.
- **TDD honesty:** every behavior batch names the expected first reachable RED and requires later assertions to be observed only after prior blockers are removed.
- **Asset boundary:** sample, model, manifest, loader, geometry, fonts, licenses, and sanitization record remain frozen; only reviewed HTML/CSS/application text digests and final evidence change.
- **Remote boundary:** all remote gates are explicitly separate and unauthorized.

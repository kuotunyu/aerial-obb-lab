# Aerial OBB Pages Live-review Accessibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Gate D runtime-claim and canvas-text-equivalence findings while adding the approved keyboard and document metadata improvements without weakening the public artifact boundary.

**Architecture:** Keep `renderCachedOutput()` as the sole cached decode/filter/presentation path. Derive runtime text from current active-result identity (`mode`, `phase`, and cache presence), and derive the hidden canvas alternative from the same confidence/class-filtered detections that feed the confidence-sorted table. Maintain the Pages artifact digest allowlist whenever a reviewed `demo/web` text file changes.

**Tech Stack:** Static HTML/CSS/JavaScript, Playwright through Python, pytest, `uv`, standard-library Pages artifact verifier.

## Global Constraints

- Work only in `<repository-root>\.worktrees\fix-pages-live-review-accessibility` on `fix/pages-live-review-accessibility`, based on `00d06f012acc9b4b52417374dd4c23ef84b9797c`.
- Use strict TDD: add the named real browser assertion, run it and observe its stated RED failure, then make the smallest GREEN change and rerun it.
- An active synthetic result (`state.mode === "synthetic" && state.phase === "result" && state.cached !== null`) displays exactly `N/A · no inference`; synthetic never creates an ORT session or reports a numeric latency.
- An active BYOM result displays measured rounded `… ms` only when its elapsed value is finite; it retains that runtime through confidence and class re-filtering.
- Loading, reset, error, and no-cache state display `—`; reset and every result-clearing error leave no old canvas-description text once Task 2 adds that descriptor.
- The canvas description is hidden and non-live; do not add visible corner columns or a second live announcement channel.
- Description records each filtered detection's class, confidence, centre x/y, width, height, and degree angle in confidence-descending order.
- No user filename, local path, model metadata, tensor content, raw error, or stack may enter UI, hidden description, console, screenshot, or artifact.
- Do not add weights, ONNX files, DOTA content, telemetry, storage, new network origins, dependencies, or a framework.
- Do not push, open or merge a PR, modify Pages or About, dispatch workflows, deploy, perform an HF operation, tag, release, alter visibility, or clean up a worktree.

---

## File structure and interfaces

| File | Responsibility in this change |
| --- | --- |
| `demo/web/app.js` | Active-result-derived runtime display, confidence-sorted detection view, non-live canvas-description text, and reset/error clearing. |
| `demo/web/index.html` | Skip-link/main target, theme metadata, stable control names, canvas-description node, and `aria-describedby`. |
| `demo/web/style.css` | Visually-hidden description and skip-link focus presentation without changing notice order. |
| `scripts/browser_smoke.py` | Deterministic real-browser RED/GREEN assertions using the existing ORT stub and local HTTP server. |
| `scripts/pages_artifact_check.py` | Canonical-LF SHA-256 allowlist entries for modified reviewed Pages files. |
| `tests/test_browser_parity.py` | Existing pure decode/corner parity contract, run unchanged as a separate unit/static gate. |
| `tests/test_pages_artifact_check.py` | Existing artifact-verifier contract, including the current-tree digest assertion, run unchanged as a separate unit/static gate. |

Shared implementation interfaces introduced by this plan:

```javascript
function sortedDetections(detections) // returns a new confidence-descending array
function renderSummary(detections, elapsedMs) // writes active-result-derived runtime text
function renderCanvasDescription(detections) // writes non-live current/empty text
```

`renderCanvasDescription(null)` writes the reset/error text. `renderCanvasDescription([])` writes the
filtered-empty text. `fillTable()` consumes `sortedDetections(detections)`, and the description uses the
same helper so its order matches the table. `renderCachedOutput()` invokes canvas drawing, table rendering,
summary rendering, and description rendering only after successful decode.

## Task 1: Make synthetic and BYOM cached runtime rendering active-result-derived

**Files:**
- Modify: `scripts/browser_smoke.py:423-454`, the successful-BYOM result section after line 541, and the isolated flaky-showcase scenario near line 889
- Modify: `demo/web/app.js:106-112` and `demo/web/app.js:174-199`
- Modify: `scripts/pages_artifact_check.py:REVIEWED_TEXT_DIGESTS["app.js"]`
- Test: `scripts/browser_smoke.py`
- Test: `tests/test_pages_artifact_check.py::test_current_pages_tree_passes`

**Interfaces:**
- Consumes: existing `state.mode`, `state.phase`, `state.cached`, `state.cached.elapsedMs`, and cached result filtering in `renderCachedOutput()`.
- Produces: `renderSummary(detections, elapsedMs)` whose runtime cell is exact synthetic text, numeric BYOM text, or reset/error `—`.

- [ ] **Step 1: Add the Task 1 runtime batch RED assertions**

In one edit, add the three named cached-filter checkpoints
`synthetic_confidence_refilter_preserves_no_inference_runtime`,
`synthetic_class_refilter_preserves_no_inference_runtime`, and
`byom_cached_refilter_retains_numeric_runtime`, plus the deterministic retry checkpoint
`synthetic_asset_failure_clears_runtime_before_retry`. Place the confidence assertion immediately after
the existing confidence `0.95` and restored `0.25` cached-showcase checks:

```python
if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
    raise RuntimeError("synthetic confidence refilter lost no-inference runtime")
```

In the same synthetic section, check the existing `plane` class checkbox, assert zero rows, then uncheck
it and assert the fixture row returns. The class checkpoint uses:

```python
plane = page.locator('.class-cb[value="0"]')
plane.check()
if page.locator("#resultsBody tr").count() != 0:
    raise RuntimeError("synthetic class refilter did not hide the fixture row")
if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
    raise RuntimeError("synthetic class refilter lost no-inference runtime")
plane.uncheck()
if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
    raise RuntimeError("synthetic class restore lost no-inference runtime")
```

After the existing stubbed successful BYOM result assertions, store the numeric runtime, change the
confidence slider to `0.95` and restore `0.25`; the BYOM checkpoint uses:

```python
byom_runtime = page.locator("#runtimeValue").inner_text()
if not re.fullmatch(r"\d+ ms", byom_runtime):
    raise RuntimeError(f"BYOM runtime is not numeric: {byom_runtime!r}")
page.locator("#confSlider").evaluate(
    "slider => { slider.value = '0.95'; slider.dispatchEvent(new Event('input', { bubbles: true })); }"
)
if page.locator("#runtimeValue").inner_text() != byom_runtime:
    raise RuntimeError("BYOM confidence refilter changed measured runtime")
```

Restore `0.25` before later smoke scenarios. Import `re` at the script's existing import block.

Extend the existing isolated `flaky_showcase` page so its first fixture request succeeds, its second request
deterministically aborts, and its third succeeds. Wait for each terminal status before the next click; do not
use a timeout as evidence. The new checkpoint first proves a successful synthetic result, then on the second
selection asserts status kind `error`, `#runtimeValue` exactly `—`, no synthetic table rows or canvas pixels
remain, and its page-local ORT route has recorded zero requests. It then clicks Showcase once more and asserts
only the successful retry restores exact `N/A · no inference`. Task 1 cannot assert `#canvasDescription`,
because Task 2 is the first task that introduces that node; Task 2 extends this exact deterministic scenario
to prove the description is also reset and error-clean.

- [ ] **Step 2: Run the runtime batch and record the actual RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected and recordable RED: `RuntimeError: synthetic confidence refilter lost no-inference runtime`,
because the current shared renderer writes `—` for synthetic `elapsedMs=None` after the first confidence
filter event. Do not claim that later class, BYOM, or retry checkpoints independently reached their
assertions in this RED run; they are intentionally one root-cause batch and are verified together after GREEN.

- [ ] **Step 3: Implement the minimal active-result-derived runtime change**

In `demo/web/app.js`, keep the existing `renderSummary(detections, elapsedMs)` call sites. Change only its
runtime branch:

```javascript
const activeSyntheticResult =
  state.mode === "synthetic" && state.phase === "result" && state.cached !== null;
runtimeValue.textContent = activeSyntheticResult
  ? "N/A · no inference"
  : state.mode === "byom" && Number.isFinite(elapsedMs)
    ? `${Math.round(elapsedMs)} ms`
    : "—";
```

Do not add a `runtimeText` or other presentation value to `state.cached`, and do not add a post-filter
override. `resetResult()` can run while a prior synthetic mode value remains, so the active-result guard—not
a mode-only condition—keeps loading, reset, error, and no-cache state at `—`.

- [ ] **Step 4: Run the named runtime browser GREEN checks**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected GREEN: all four named runtime checkpoints pass; synthetic makes zero ORT requests, the deterministic
second-load asset failure clears runtime/result state to `—` before retry, and BYOM runtime matches `^\d+ ms$`
before and after the cached filter.

- [ ] **Step 5: Observe the artifact verifier RED and refresh only the app digest**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_current_pages_tree_passes -q
```

Expected RED: `app.js: reviewed application bytes differ` because `app.js` changed.

Print its canonical-LF SHA-256 with:

```powershell
uv run --no-sync python -c "from pathlib import Path; import hashlib; p=Path('demo/web/app.js'); t=p.read_bytes().decode('utf-8').replace('\r\n','\n').replace('\r','\n'); print(hashlib.sha256(t.encode('utf-8')).hexdigest())"
```

Replace only `REVIEWED_TEXT_DIGESTS["app.js"]` in `scripts/pages_artifact_check.py` with that printed
digest; retain its existing reason string `reviewed application bytes differ`.

- [ ] **Step 6: Verify focused static contracts and commit**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
git diff --check
```

Expected GREEN: parity and artifact tests pass, the artifact checker prints `[OK] Pages artifact boundary`,
and `git diff --check` emits no output.

Commit only these files:

```powershell
git add demo/web/app.js scripts/browser_smoke.py scripts/pages_artifact_check.py
git commit -m "fix: preserve runtime claims across cached filters"
```

## Task 2: Add a non-live, filtered canvas textual alternative

**Files:**
- Modify: `scripts/browser_smoke.py:423-454` and its result-clearing assertion helper near lines 175-188
- Modify: `demo/web/index.html:118-141`
- Modify: `demo/web/app.js:174-199` and `demo/web/app.js:355-384`
- Modify: `demo/web/style.css` beside the page-level accessibility utility rules
- Modify: `scripts/pages_artifact_check.py:REVIEWED_TEXT_DIGESTS["index.html"]`, `["app.js"]`, and `["style.css"]`
- Test: `scripts/browser_smoke.py`
- Test: `tests/test_pages_artifact_check.py::test_current_pages_tree_passes`

**Interfaces:**
- Consumes: decoded detection objects `{cx, cy, w, h, conf, cls, angle}` and fixed `CLASS_NAMES`.
- Produces: `sortedDetections(detections)` and `renderCanvasDescription(detections)` for the canvas'
  non-live `aria-describedby` target.

- [ ] **Step 1: Add the failing populated-description browser assertion**

After the initial synthetic fixture row and polygon checks, add the named checkpoint
`canvas_description_matches_sorted_filtered_synthetic_detection`:

```python
canvas = page.locator("#canvas")
description_id = canvas.get_attribute("aria-describedby")
if description_id != "canvasDescription":
    raise RuntimeError("canvas must reference canvasDescription")
description = page.locator("#canvasDescription")
if description.get_attribute("aria-live") is not None:
    raise RuntimeError("canvas description must not be aria-live")
expected_fields = {
    "class": "ship",
    "confidence": "0.900",
    "center-x": "200.0 px",
    "center-y": "100.0 px",
    "width": "100.0 px",
    "height": "50.0 px",
    "angle": "90.0°",
}
for label, value in expected_fields.items():
    if f"{label}={value}" not in description.inner_text():
        raise RuntimeError(f"canvas description lacks structured field {label}={value}")
```

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected RED: `RuntimeError: canvas must reference canvasDescription`, because no descriptor node or canvas
reference exists on the base implementation.

- [ ] **Step 2: Add the failing empty/reset/error description browser assertions**

Extend the named checkpoint `canvas_description_clears_for_filtered_empty_reset_and_error`:

```python
if "沒有 detections" not in description.inner_text():
    raise RuntimeError("filtered-empty canvas description is not explicit")
```

Place it immediately after the Task 1 non-matching class-filter zero-row assertion. Extend the existing
`assert_fixed_failure()` helper to require exactly `尚無 detection result。` from `#canvasDescription` for
every existing safe failure code, including `MODEL_CONTRACT`, `IMAGE_DECODE`, `RUNTIME_LOAD`,
`INFERENCE_RUN`, `OUTPUT_SCHEMA`, `RENDER_RESULT`, and `SHOWCASE_ASSET`. Keep `assert_result_cleared()`
for the three inference-result cases that also verify canvas pixels. Extend Task 1's deterministic
second-showcase-load failure/retry page here—not in a separate timing-dependent scenario—to assert its
`SHOWCASE_ASSET` error has exactly this reset description before the third-request retry restores the
synthetic `ship` description. Also assert that switching from BYOM back to Synthetic replaces any prior
BYOM description with the synthetic fixture's `ship` description.

Run the same command.

Expected RED: the first descriptor lookup fails before any of the new state-specific checks; preserve that
failure ordering in the task report.

- [ ] **Step 3: Implement the minimal shared description flow**

Make the following focused changes.

1. In `index.html`, immediately after `<canvas id="canvas">`, add one visually-hidden `<p>` with
   `id="canvasDescription"` and initial exact text `尚無 detection result。`; add
   `aria-describedby="canvasDescription"` to the canvas. Do not add `aria-live`.
2. In `style.css`, add `.visually-hidden` with clipped, one-pixel, off-screen accessible text styling.
   It must remain exposed to assistive technology and must not create layout overflow.
3. In `app.js`, add `sortedDetections(detections)` returning a new confidence-descending array. Change
   `fillTable(detections)` to iterate that helper.
4. Add `renderCanvasDescription(detections)`. For a populated list, produce one confidence-sorted,
   field-labelled record per detection in this exact order:
   `class=<name>; confidence=<0.000>; center-x=<0.0 px>; center-y=<0.0 px>; width=<0.0 px>; height=<0.0 px>; angle=<0.0°>.`
   Use `toFixed(3)` for confidence and `toFixed(1)` for coordinates, dimensions, and degrees. For `[]`,
   write `目前篩選條件下沒有 detections；canvas 沒有 oriented polygons。`. For `null`, write
   `尚無 detection result。`.
5. In `renderCachedOutput()`, call `renderCanvasDescription(detections)` only after draw, table, and
   summary work completes. In `resetResult()`, call `renderCanvasDescription(null)` so reset and existing
   result-clearing safe errors cannot retain old text.

Do not calculate duplicate corners, put coordinates into the visible table, or make the description live.

- [ ] **Step 4: Run the description browser GREEN checks**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected GREEN: populated synthetic and BYOM descriptions contain the required fields, class filtering
updates the description with the table, filtered empty text is explicit, and every existing result-clearing
failure leaves only the reset text.

- [ ] **Step 5: Observe artifact-verifier RED and refresh exact changed digests**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_current_pages_tree_passes -q
```

Expected RED: digest failures for the changed reviewed files among `index.html`, `app.js`, and `style.css`.

For each of exactly those changed files, print canonical-LF SHA-256 using:

```powershell
uv run --no-sync python -c "from pathlib import Path; import hashlib; files=['demo/web/index.html','demo/web/app.js','demo/web/style.css']; [print(f'{p}='+hashlib.sha256(Path(p).read_bytes().decode('utf-8').replace('\r\n','\n').replace('\r','\n').encode('utf-8')).hexdigest()) for p in files]"
```

Update only the matching entries in `REVIEWED_TEXT_DIGESTS`, retaining all existing file paths and reason
strings.

- [ ] **Step 6: Verify static contracts and commit**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
git diff --check
```

Expected GREEN: all selected tests pass and the artifact checker reports `[OK] Pages artifact boundary`.

Commit only these files:

```powershell
git add demo/web/app.js demo/web/index.html demo/web/style.css scripts/browser_smoke.py scripts/pages_artifact_check.py
git commit -m "fix: describe filtered canvas detections accessibly"
```

## Task 3: Add keyboard skip navigation and static semantic metadata

**Files:**
- Modify: `scripts/browser_smoke.py:283-421` before the first showcase action
- Modify: `demo/web/index.html:3-10`, `demo/web/index.html:29-75`, and `demo/web/index.html:145`
- Modify: `demo/web/style.css` beside the Task 2 accessibility utility
- Modify: `scripts/pages_artifact_check.py:REVIEWED_TEXT_DIGESTS["index.html"]` and `["style.css"]`
- Test: `scripts/browser_smoke.py`
- Test: `tests/test_pages_artifact_check.py::test_current_pages_tree_passes`

**Interfaces:**
- Consumes: the existing semantic header, claim notice, `main`, file/range controls, and dynamically-created
  `.class-cb` controls.
- Produces: keyboard-first `a.skip-link[href="#mainContent"]`, `main#mainContent[tabindex="-1"]`, exact
  theme metadata, and stable input names.

- [ ] **Step 1: Add the Task 3 semantic batch RED assertions**

In one edit immediately after page load, add the three semantic assertions
`keyboard_skip_link_targets_main_content`, `document_theme_and_control_names_contract`, and the existing
notice-order assertion that must remain true. The keyboard checkpoint is:

```python
page.keyboard.press("Tab")
skip_link = page.locator("a.skip-link")
if skip_link.count() != 1 or page.evaluate("document.activeElement === document.querySelector('a.skip-link')") is not True:
    raise RuntimeError("first keyboard focus must be the main-workspace skip link")
if skip_link.inner_text() != "跳至主要工作區" or skip_link.get_attribute("href") != "#mainContent":
    raise RuntimeError("skip link text or target is wrong")
page.keyboard.press("Enter")
page.wait_for_function("document.activeElement === document.querySelector('#mainContent')")
```

The metadata/name checkpoint before showcase activation is:

```python
if page.locator('meta[name="theme-color"]').get_attribute("content") != "#edf1f4":
    raise RuntimeError("theme-color metadata is not exact")
expected_names = {"#modelInput": "model", "#fileInput": "image", "#confSlider": "confidence"}
for selector, expected_name in expected_names.items():
    if page.locator(selector).get_attribute("name") != expected_name:
        raise RuntimeError(f"stable control name is wrong for {selector}")
if any(name != "class-filter" for name in page.locator(".class-cb").evaluate_all("els => els.map(el => el.name)")):
    raise RuntimeError("class checkboxes do not share the semantic filter name")
```

- [ ] **Step 2: Run the semantic batch and record the actual RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected and recordable RED: `RuntimeError: first keyboard focus must be the main-workspace skip link`,
because the current page has no skip link. Do not claim that theme-colour or name checkpoints independently
reached their assertions in this RED run; they are intentionally verified as one semantic batch after GREEN.

- [ ] **Step 3: Implement the minimal semantic HTML and CSS**

1. Make the skip link the first focusable element inside `<body>`, before the visual page shell:

   ```html
   <a class="skip-link" href="#mainContent">跳至主要工作區</a>
   ```

2. Change the existing main element to:

   ```html
   <main id="mainContent" tabindex="-1">
   ```

3. Add exactly `<meta name="theme-color" content="#edf1f4" />` in the head next to the existing colour
   metadata.
4. Add `name="model"`, `name="image"`, and `name="confidence"` to the existing file/range inputs.
   Add `cb.name = "class-filter"` when the existing class checkbox is created. Preserve all visible labels,
   `value`s, accepts, and IDs.
5. Add `.skip-link` CSS that visually clips it off-screen by default and restores a high-contrast,
   positioned, visible focus state with `:focus-visible`. Do not use `display: none`, `visibility: hidden`,
   or change the header/notice layout.

- [ ] **Step 4: Run the semantic browser GREEN checks**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected GREEN: first Tab reaches the skip link, Enter focuses `main#mainContent`, theme colour is exact,
all approved names are present, the claim notice still precedes the first visible interactive control, and
the existing keyboard-focus/layout checks pass.

- [ ] **Step 5: Observe artifact-verifier RED and refresh exact changed digests**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_current_pages_tree_passes -q
```

Expected RED: `index.html: reviewed HTML bytes differ` and `style.css: reviewed stylesheet bytes differ`.

Print canonical-LF SHA-256 for exactly `demo/web/index.html` and `demo/web/style.css` with:

```powershell
uv run --no-sync python -c "from pathlib import Path; import hashlib; files=['demo/web/index.html','demo/web/style.css']; [print(f'{p}='+hashlib.sha256(Path(p).read_bytes().decode('utf-8').replace('\r\n','\n').replace('\r','\n').encode('utf-8')).hexdigest()) for p in files]"
```

Replace only their two `REVIEWED_TEXT_DIGESTS` values in `scripts/pages_artifact_check.py`.

- [ ] **Step 6: Verify static contracts and commit**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
git diff --check
```

Expected GREEN: selected tests pass, artifact boundary passes, and no whitespace errors are printed.

Commit only these files:

```powershell
git add demo/web/index.html demo/web/style.css scripts/browser_smoke.py scripts/pages_artifact_check.py
git commit -m "feat: improve Pages keyboard and document semantics"
```

## Task 4: Run the complete local release and live-review preflight matrix

**Files:**
- Modify: none unless a prior task left an uncommitted approved change; do not create evidence files in the repository.
- Test: `tests/test_browser_parity.py`, `tests/test_pages_artifact_check.py`, complete pytest suite, release/repository checks, browser smoke, clean export, and exact local `demo/web` preview.

**Interfaces:**
- Consumes: the three completed task commits and `demo/web` as the exact Pages artifact root.
- Produces: local evidence sufficient to request a scoped review; no remote side effect.

- [ ] **Step 1: Verify unit and static contracts separately**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: pure fixture/decode/corner parity passes independently of the browser, and the staged Pages
tree passes its strict manifest, digest, privacy, and external-origin verifier.

- [ ] **Step 2: Verify the real browser smoke separately**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected GREEN: all named synthetic, BYOM, descriptor, error-clearing, skip-link, metadata, privacy,
network-origin, desktop, and mobile checks pass using the deterministic ORT stub with no real inference.

- [ ] **Step 3: Run the complete regression and release/privacy gates**

Run:

```powershell
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/clean_export_check.py
```

Expected GREEN: the full pytest suite has zero failures; repository and release checks pass; clean export
passes both its independent artifact checks and browser smoke. Do not replace the full clean-export command
with `--skip-browser` for final evidence.

- [ ] **Step 4: Serve and inspect the exact local Pages artifact**

Run in a dedicated terminal:

```powershell
uv run --no-sync python -m http.server 8000 --directory demo/web
```

Open only `http://127.0.0.1:8000/`. At 1280×720 and 390×844, load Synthetic Showcase and confirm: no
horizontal overflow; the non-collapsible claim notice remains before visible controls; mode, provenance,
and runtime are synthetic; filters keep runtime `N/A · no inference`; canvas/table/hidden description
stay synchronized; Source and AGPL links are visible. Save any screenshots outside every repository and
exclude all private filenames, paths, models, metadata, raw errors, and stacks.

The local preview is a precondition for a later deployed live review, not a Pages dispatch or deployment.

- [ ] **Step 5: Final local branch hygiene and scoped review package**

Run:

```powershell
git status --short
git diff --check origin/main...HEAD
git log --oneline origin/main..HEAD
git diff --name-only origin/main...HEAD
```

Expected GREEN: tracked worktree is clean, no whitespace error appears, commits are limited to the approved
files, and no generated screenshots, model/data files, or new external-origin asset is present. Request a
fresh scoped code review before any remote action.

## Remote gates after local review — explicitly not authorized here

1. **Remote Gate A — publication preflight and PR publication:** fresh read-only branch/remote/PR/auth
   preflight, then separately authorized non-force push and PR creation. This plan does not perform either.
2. **Remote Gate B — PR/main integration:** candidate-artifact equality review, PR checks, merge, and exact
   `main` CI verification. This plan does not perform any of them.
3. **Remote Gate C — Pages operation:** Pages configuration and any `pages.yml` dispatch/deployment. This
   plan does not alter Pages or dispatch a workflow.
4. **Remote Gate D — deployed live review:** review the deployed HTTPS URL only after a separately
   authorized successful deployment; it must repeat the synthetic, BYOM safe-failure, privacy, accessibility,
   asset, and origin checks.
5. **Remote Gate E — About and Portfolio Control receipt:** GitHub About homepage change and Portfolio
   Control receipt occur only after a separately authorized passing Gate D. This plan leaves both untouched.

## Plan self-review

- **Spec coverage:** Task 1 implements and tests both runtime invariants; Task 2 implements and tests the
  filtered non-live equivalent canvas text plus empty/reset/error clearing; Task 3 implements and tests the
  skip link, theme colour, and names; Task 4 separates unit/static, browser, full, artifact, privacy/origin,
  and local responsive checks.
- **Path coverage:** every changed path is named in its task; `scripts/pages_artifact_check.py` digest
  updates are explicitly paired with each reviewed artifact-text change.
- **TDD ordering:** every behavior task adds named browser RED assertions, records the exact expected
  failure, then specifies a minimal GREEN change and focused verification command before its commit.
- **Boundary review:** all remote operations, Pages changes, About changes, portfolio receipt, private
  BYOM material, and artifact expansion are prohibited in this plan-writing and local implementation scope.

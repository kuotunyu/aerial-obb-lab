# Aerial OBB Pages Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing `demo/web` Browser BYOM workbench into a recruiter-ready dual-mode GitHub Pages candidate with a clearly bounded interactive synthetic showcase.

**Architecture:** Keep one framework-free static workbench. Synthetic and BYOM sources feed the same `output0` validation, decode, filtering, rotated-corner, canvas, summary, and table pipeline; only BYOM may lazy-load ONNX Runtime and create a session. Deterministic browser tests and a model-free artifact verifier protect the code-only public boundary before separately authorized remote gates.

**Tech Stack:** HTML5, CSS, browser JavaScript, Node.js 22, ONNX Runtime Web 1.20.1, Python 3.11, pytest, Playwright, GitHub Actions, GitHub Pages.

## Global Constraints

- Work in the isolated design worktree; use red/green TDD and small commits.
- Static HTML contains this non-collapsible notice before the first interactive control:

  > **Synthetic Showcase — 沒有執行模型推論**
  >
  > 此展示使用本 repository 提交的 synthetic SVG 與固定 output，僅驗證 UI、decode、filter、rotated-corner 計算與 rendering。它不是 accuracy、evaluation 或 latency evidence。實際推論請使用 BYOM，自行提供相容 ONNX model 與影像。

- Synthetic never touches `ort`, creates no session, requests no external runtime, and displays `N/A · no inference`.
- Synthetic badge is `SYNTHETIC FIXTURE · NO INFERENCE`; provenance is `Committed synthetic fixture`.
- BYOM badge is `BYOM · LOCAL BROWSER INFERENCE`; BYOM is the only inference path.
- Contract: `images [1,3,1024,1024]` float32 RGB CHW and `output0 [1,N,7]`.
- ORT entry: `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js`, `crossorigin="anonymous"`, integrity `sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp`.
- Use the same pinned package directory for WASM. Do not claim WASM SRI, offline operation, or zero-network BYOM.
- Never bundle weights, ONNX, DOTA pixels/annotations, DOTA-derived renders, secrets, telemetry, storage, or remote model fallback.
- Fixed safe errors must give recovery steps; UI/console never interpolate filename, path, metadata, tensors, raw exception, or stack.
- Preserve desktop 34/66, sub-900 px single column, 19 px body, 15 px secondary text, visible focus, and 44 px controls.
- Nothing in this plan authorizes push, PR, merge, Pages enablement/deploy, or About mutation.
- Design source: `docs/superpowers/specs/2026-08-31-aerial-obb-pages-showcase-design.md`.

---

## File Map

**Create**

- `demo/web/showcase-fixture.js` — canonical synthetic data/provenance.
- `demo/web/fixtures/showcase.svg` — canonical authored SVG.
- `scripts/pages_artifact_check.py` — exact Pages-tree verifier.
- `tests/test_pages_artifact_check.py` — verifier tests.
- `tests/test_pages_workflow.py` — remote-gate workflow tests.
- `.github/workflows/pages.yml` — manual-only protected-main deploy workflow.

**Modify**

- `demo/web/index.html:9-141`, `demo/web/app.js:7-239`, `demo/web/style.css`.
- `scripts/browser_smoke.py:24-371`, `scripts/repo_check.py:217-301`.
- `tests/js/browser_parity_runner.js`, `tests/test_browser_parity.py:11-83`, `tests/test_release_check.py`.
- `README.md:11-37,206-246`, `README.en.md:11-36,212-255`, `demo/web/README.md`.
- `THIRD_PARTY_NOTICES.md`, `release/evidence.json:158-194`, `CHANGELOG.md:1-7`.
- `.github/workflows/release-gates.yml`.

**Move**

- `tests/fixtures/browser-smoke.svg` to `demo/web/fixtures/showcase.svg`.

**Regenerate**

- `docs/assets/browser-workbench.png` from the deterministic synthetic smoke.

## Task 1: Promote the synthetic fixture into the product tree

**Files:** Create `demo/web/showcase-fixture.js`; move the SVG; modify parity runner/test and browser smoke.

**Interfaces:** Produce CommonJS/browser global `OBB_SHOWCASE` with `schemaVersion`, `provenance`, `imageUrl`, `imageWidth`, `imageHeight`, `targetSize`, and `results.output0.{dims,data}`.

- [ ] **Step 1: Add the failing production-fixture test**

Add to `tests/test_browser_parity.py`:

```python
SHOWCASE_MODULE = ROOT / "demo" / "web" / "showcase-fixture.js"

def load_showcase_fixture() -> dict:
    node = shutil.which("node")
    assert node
    script = """
const f = require(process.argv[1]);
process.stdout.write(JSON.stringify({
  schemaVersion: f.schemaVersion, provenance: f.provenance,
  imageUrl: f.imageUrl, imageWidth: f.imageWidth, imageHeight: f.imageHeight,
  targetSize: f.targetSize, dims: f.results.output0.dims,
  data: Array.from(f.results.output0.data),
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(SHOWCASE_MODULE)], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(result.stdout)

def test_production_showcase_fixture_is_canonical() -> None:
    f = load_showcase_fixture()
    assert f["schemaVersion"] == 1
    assert f["provenance"] == "Committed synthetic fixture"
    assert f["imageUrl"] == "fixtures/showcase.svg"
    assert [f["imageWidth"], f["imageHeight"], f["targetSize"]] == [400, 200, 1024]
    assert f["dims"] == [1, 2, 7]
    assert f["data"] == pytest.approx([
        512, 512, 256, 128, 0.9, 1, math.pi / 2,
        100, 100, 50, 40, 0.2, 2, 0,
    ])
```

- [ ] **Step 2: Verify red**

Run `uv run --no-sync python -m pytest tests/test_browser_parity.py::test_production_showcase_fixture_is_canonical -q`.

Expected: FAIL because the module is missing.

- [ ] **Step 3: Create the canonical module**

```javascript
(function expose(root, factory) {
  const fixture = factory();
  if (typeof module === "object" && module.exports) module.exports = fixture;
  root.OBB_SHOWCASE = fixture;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildFixture() {
  "use strict";
  return Object.freeze({
    schemaVersion: 1,
    provenance: "Committed synthetic fixture",
    imageUrl: "fixtures/showcase.svg",
    imageWidth: 400,
    imageHeight: 200,
    targetSize: 1024,
    results: Object.freeze({
      output0: Object.freeze({
        dims: Object.freeze([1, 2, 7]),
        data: Float32Array.from([
          512, 512, 256, 128, 0.9, 1, Math.PI / 2,
          100, 100, 50, 40, 0.2, 2, 0,
        ]),
      }),
    }),
  });
});
```

- [ ] **Step 4: Move the SVG and update consumers**

Run:

```powershell
New-Item -ItemType Directory -Path demo/web/fixtures -Force | Out-Null
git mv tests/fixtures/browser-smoke.svg demo/web/fixtures/showcase.svg
```

Set `FIXTURE = DEMO / "fixtures" / "showcase.svg"` in `scripts/browser_smoke.py`. Require the module from `tests/js/browser_parity_runner.js`, pass `OBB_SHOWCASE.results` through `selectEndToEndOutput`, and assert the decoded ship/corners in Python.

- [ ] **Step 5: Verify green and commit**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_browser_parity.py -q
uv run --no-sync python scripts/repo_check.py
git add demo/web/showcase-fixture.js demo/web/fixtures/showcase.svg scripts/browser_smoke.py tests/js/browser_parity_runner.js tests/test_browser_parity.py
git commit -m "test: promote browser showcase fixture"
```

## Task 2: Add the static claim boundary and mode presentation

**Files:** Modify `index.html`, `style.css`, and `browser_smoke.py`.

**Interfaces:** Produce DOM IDs `claimBoundary`, `showcaseBtn`, `modeBadge`, `provenanceValue`; preserve all existing BYOM IDs.

- [ ] **Step 1: Add failing DOM/visual-order assertions**

```python
notice = page.locator("#claimBoundary")
control = page.locator("#showcaseBtn")
assert notice.count() == 1 and control.count() == 1
assert "沒有執行模型推論" in notice.inner_text()
assert notice.evaluate(
    "(notice, control) => Boolean(notice.compareDocumentPosition(control) & Node.DOCUMENT_POSITION_FOLLOWING)",
    control.element_handle(),
)
notice_box, control_box = notice.bounding_box(), control.bounding_box()
assert notice_box and control_box and notice_box["y"] < control_box["y"]
```

Run `uv run --no-sync python scripts/browser_smoke.py`.

Expected: FAIL because the notice/button are absent.

- [ ] **Step 2: Add the exact static notice before `main`**

```html
<section id="claimBoundary" class="claim-boundary" aria-labelledby="claimTitle">
  <h2 id="claimTitle">Synthetic Showcase — 沒有執行模型推論</h2>
  <p>此展示使用本 repository 提交的 synthetic SVG 與固定 output，僅驗證 UI、decode、filter、rotated-corner 計算與 rendering。它不是 accuracy、evaluation 或 latency evidence。實際推論請使用 BYOM，自行提供相容 ONNX model 與影像。</p>
</section>
```

Make `<button id="showcaseBtn">載入 Synthetic Showcase</button>` the first control. Add `modeBadge` and `provenanceValue` to the result summary and Source/AGPL links after the workbench.

- [ ] **Step 3: Add square-edged, accessible styling**

```css
.claim-boundary { border: 2px solid var(--ink); background: #fff7d6; padding: 14px 16px; margin-bottom: 14px; }
.claim-boundary h2 { margin: 0 0 6px; font-size: 1.15rem; }
.claim-boundary p { margin: 0; max-width: 92ch; }
.mode-entry { border-bottom: 1px solid var(--line); padding-bottom: 14px; }
#showcaseBtn { min-height: 44px; width: 100%; border-radius: 0; }
```

- [ ] **Step 4: Verify green and commit**

Run the smoke at 1600x1000 and its built-in 820x1100 viewport, then:

```powershell
git add demo/web/index.html demo/web/style.css scripts/browser_smoke.py
git commit -m "feat: add synthetic showcase claim boundary"
```

## Task 3: Add the shared cached-output synthetic path

**Files:** Modify `index.html`, `app.js`, and `browser_smoke.py`.

**Interfaces:** Produce `state`, `resetResult()`, `renderCachedOutput()`, `loadImageUrl(url)`, and `activateShowcase()`. Cached shape: `{results,geometry,provenance,elapsedMs}`.

- [ ] **Step 1: Add failing interactive showcase assertions**

Click `#showcaseBtn` before uploading files. Assert exact badge/provenance/runtime, `EXPECTED_ROW`, canvas polygon, focus on `#resultTitle`, then set confidence to `0.95` and assert cached re-filtering yields zero rows.

Run `uv run --no-sync python scripts/browser_smoke.py`.

Expected: FAIL because the button has no controller.

- [ ] **Step 2: Load the fixture before `app.js`**

```html
<script src="obb.js"></script>
<script src="showcase-fixture.js"></script>
<script src="app.js"></script>
```

- [ ] **Step 3: Replace loose globals with explicit state**

```javascript
const state = {
  mode: "none", phase: "idle", generation: 0,
  session: null, image: null, cached: null, elapsedMs: null,
};
const modeBadge = document.getElementById("modeBadge");
const provenanceValue = document.getElementById("provenanceValue");
const resultTitle = document.getElementById("resultTitle");

async function releaseSession() {
  const current = state.session;
  state.session = null;
  if (current && typeof current.release === "function") await current.release();
}

function resetResult() {
  state.cached = null;
  state.elapsedMs = null;
  resultsBody.innerHTML = "";
  renderSummary([]);
  modeBadge.textContent = "NO RESULT";
  provenanceValue.textContent = "—";
}

function renderCachedOutput() {
  if (!state.cached || !state.image) return;
  const classes = new Set(
    Array.from(document.querySelectorAll(".class-cb:checked")).map((cb) => Number(cb.value))
  );
  const output = OBB.selectEndToEndOutput(state.cached.results);
  const dets = OBB.decodeDetections(
    output, state.cached.geometry, Number(confSlider.value), classes, CLASS_NAMES.length
  );
  drawDetections(dets);
  fillTable(dets);
  renderSummary(dets, state.cached.elapsedMs);
}
```

- [ ] **Step 4: Implement synthetic activation**

```javascript
function loadImageUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("SHOWCASE_ASSET"));
    image.src = url;
  });
}

async function activateShowcase() {
  const generation = ++state.generation;
  state.phase = "loading";
  await releaseSession();
  resetResult();
  const image = await loadImageUrl(OBB_SHOWCASE.imageUrl);
  if (generation !== state.generation) return;
  state.mode = "synthetic";
  state.phase = "result";
  state.image = image;
  state.cached = {
    results: OBB_SHOWCASE.results,
    geometry: OBB.letterboxGeometry(400, 200, 1024),
    provenance: OBB_SHOWCASE.provenance,
    elapsedMs: null,
  };
  modeBadge.textContent = "SYNTHETIC FIXTURE · NO INFERENCE";
  provenanceValue.textContent = OBB_SHOWCASE.provenance;
  renderCachedOutput();
  runtimeValue.textContent = "N/A · no inference";
  resultTitle.focus();
  setStatus("Synthetic fixture 已載入 · 沒有執行模型推論。", "success");
}
```

Wire confidence/class changes to `renderCachedOutput`. Selecting any BYOM file clears synthetic cache and mode before continuing.

- [ ] **Step 5: Verify green and commit**

Run browser smoke, parity tests, and full pytest. Commit:

```powershell
git add demo/web/index.html demo/web/app.js scripts/browser_smoke.py
git commit -m "feat: add shared synthetic result pipeline"
```

## Task 4: Lazy-load ORT and replace sessions atomically

**Files:** Modify `index.html`, `app.js`, and `browser_smoke.py`.

**Interfaces:** Produce `loadOrtRuntime(): Promise<object>`, `validateSessionContract(candidate)`, and `replaceModelSession(file,generation)`.

- [ ] **Step 1: Add failing lazy-load/lifecycle assertions**

Route `ORT_CDN_URL` directly in Playwright. Load/click synthetic and assert zero ORT requests. Select a model and assert exactly one request plus exact script `integrity`/`crossOrigin`. Extend the stub with `inputNames`, `outputNames`, `release()`, and create/release counters.

Run the smoke; expect FAIL because ORT is eager.

- [ ] **Step 2: Remove the eager CDN script and add the loader**

```javascript
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ORT_INTEGRITY = "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp";
let ortPromise = null;

function loadOrtRuntime() {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  if (ortPromise) return ortPromise;
  ortPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = ORT_URL;
    script.integrity = ORT_INTEGRITY;
    script.crossOrigin = "anonymous";
    script.onload = () => {
      globalThis.ort.env.wasm.wasmPaths = ORT_WASM_BASE;
      resolve(globalThis.ort);
    };
    script.onerror = () => { script.remove(); ortPromise = null; reject(new Error("RUNTIME_LOAD")); };
    document.head.appendChild(script);
  });
  return ortPromise;
}
```

- [ ] **Step 3: Implement candidate-first replacement**

Create the candidate, assert `inputNames.includes("images")` and `outputNames.includes("output0")`, discard/release stale or invalid candidates, assign the candidate, then release the previous session. Do not change the visible model label before assignment.

- [ ] **Step 4: Verify green and commit**

Run smoke/full pytest, then commit `index.html`, `app.js`, and smoke as `feat: lazy-load browser inference runtime`.

## Task 5: Add generation safety and actionable fixed errors

**Files:** Modify `app.js`, `index.html`, `style.css`, and `browser_smoke.py`.

**Interfaces:** Produce `ERROR_COPY`, `reportFailure(code)`, `nextGeneration()`, and `isCurrentGeneration(token)`.

- [ ] **Step 1: Add failing race/leak/recovery cases**

Make stub byte `0` throw a sensitive path/metadata error, byte `1` delay, byte `2` return invalid output. Assert fixed copy for runtime/model/image/inference/output/render failures; assert the injected strings never appear in UI/console. Start delayed model A, select B, resolve A, and assert B remains active.

- [ ] **Step 2: Add the exact safe map**

```javascript
const ERROR_COPY = Object.freeze({
  SHOWCASE_ASSET: "Synthetic fixture 無法載入。請重新整理頁面，或改用 BYOM。",
  RUNTIME_LOAD: "Browser runtime 無法載入。請檢查網路或 content blocker 後重試；Synthetic Showcase 仍可使用。",
  MODEL_CONTRACT: "請選擇使用 images [1,3,1024,1024] 與 output0 [1,N,7] 的相容 ONNX。",
  IMAGE_DECODE: "Browser 無法解碼影像。請改選 PNG、JPEG 或 WebP。",
  INFERENCE_RUN: "推論未完成。請確認模型 contract、重新選擇影像後再試。",
  OUTPUT_SCHEMA: "模型輸出不符合 output0 [1,N,7]。請改用相容的 end-to-end OBB export。",
  RENDER_RESULT: "結果無法呈現。請重新載入 Synthetic Showcase，或重新執行 Detect。",
});
function reportFailure(code) {
  const safe = Object.hasOwn(ERROR_COPY, code) ? code : "INFERENCE_RUN";
  console.warn("[AERIAL_OBB:" + safe + "]");
  setStatus(ERROR_COPY[safe], "error");
}
```

- [ ] **Step 3: Apply one generation source**

```javascript
function nextGeneration() { state.generation += 1; return state.generation; }
function isCurrentGeneration(token) { return token === state.generation; }
```

Use it for showcase, model, image, and Detect. Check after every `await`; release stale candidate sessions. Add a hidden `runtimeRetryBtn` shown only for `RUNTIME_LOAD`.

- [ ] **Step 4: Verify green and commit**

Run browser smoke, release tests, and release checker. Commit the four files as `fix: make browser state recovery safe`.

## Task 6: Gate the exact Pages artifact and privacy boundary

**Files:** Create `scripts/pages_artifact_check.py` and `tests/test_pages_artifact_check.py`; modify `repo_check.py`.

**Interfaces:** Produce `verify_pages_tree(root: Path) -> list[str]` and a read-only CLI.

- [ ] **Step 1: Add failing verifier tests**

```python
def test_current_pages_tree_passes() -> None:
    assert verify_pages_tree(ROOT / "demo" / "web") == []

def test_pages_tree_rejects_model_dota_secret_path_and_origin(tmp_path: Path) -> None:
    site = tmp_path / "web"
    shutil.copytree(ROOT / "demo" / "web", site)
    (site / "model.onnx").write_bytes(b"x")
    (site / "dota-derived.png").write_bytes(b"x")
    (site / "leak.js").write_text(
        'const t="ghp_' + 'x' * 24 + '";'
        'const p="C:" + "\\\\Users\\\\alice\\\\private.onnx";'
        'const u="https://unapproved.example/runtime.js";',
        encoding="utf-8",
    )
    joined = "\n".join(verify_pages_tree(site))
    assert "forbidden model/runtime artifact" in joined
    assert "forbidden DOTA-derived path" in joined
    assert "token-shaped string" in joined
    assert "absolute user path" in joined
    assert "unapproved external origin" in joined
```

Also test symlink rejection, required-file absence, unexpected binary, hard-link count, and exact ORT URL/integrity mismatch.

- [ ] **Step 2: Verify red**

Run `uv run --no-sync python -m pytest tests/test_pages_artifact_check.py -q`.

Expected: import failure.

- [ ] **Step 3: Implement the verifier**

Use stdlib only. Reject model/archive suffixes; path tokens `dota`/`hbb_vs_obb`; symlinks/hardlinks; files over 1 MiB except the allowlisted font; token patterns; Windows/macOS/Linux home paths; and unexpected binaries. Inspect runtime-capable references in HTML/JavaScript/CSS separately: executable/resource origins are limited to the pinned `cdn.jsdelivr.net` package, while reviewed navigation links may use `github.com`. URLs that appear only in `.md` or license `.txt` files are documentation, not browser requests, but those files still receive secret/path scans. Require HTML/CSS/JS, the exact ORT URL/integrity/CORS, fixture, font, and font license.

Core signature:

```python
def verify_pages_tree(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        # append stable "path: reason" messages; never mutate root
    return errors
```

- [ ] **Step 4: Integrate and verify**

Import `verify_pages_tree` in `repo_check.py`, fail with joined messages, and print `[OK] Pages artifact boundary` on success.

Run verifier tests, CLI, repo check, and release check. Commit as `test: gate Pages artifact boundary`.

## Task 7: Synchronize docs, evidence, and screenshot

**Files:** Modify both READMEs, demo README, notices, evidence, changelog, release tests; regenerate screenshot.

**Interfaces:** Add `browser_demo` fields: `showcase_enabled`, `showcase_fixture`, `showcase_image`, `showcase_inference_performed`, `showcase_runtime_label`, `showcase_external_runtime_requests`, and `runtime_load`.

- [ ] **Step 1: Add failing evidence tests**

```python
def test_browser_showcase_evidence_is_explicit_and_model_free() -> None:
    browser = load_json(ROOT / "release" / "evidence.json")["browser_demo"]
    assert browser["showcase_enabled"] is True
    assert browser["showcase_inference_performed"] is False
    assert browser["showcase_runtime_label"] == "N/A · no inference"
    assert browser["showcase_external_runtime_requests"] is False
    assert browser["runtime_load"] == "lazy-on-byom-selection"
    assert "demo/web/showcase-fixture.js" in browser["source_files"]
    assert "demo/web/fixtures/showcase.svg" in browser["source_files"]
    assert "tests/fixtures/browser-smoke.svg" not in browser["source_files"]
```

Run the test; expect missing fields.

- [ ] **Step 2: Update evidence and human docs**

Add the exact fields above. Preserve all metric sections. Explain one-action synthetic evidence, no inference/accuracy/evaluation/latency, BYOM-only inference, lazy runtime, JS-only SRI scope, non-zero-network BYOM, safe recovery, and code-only exclusions. Add `[Unreleased]` without altering historical changelog sections.

Keep `release/artifact-manifest.json` unchanged: the authored synthetic fixture is first-party, while the existing OFL font remains the only bundled third-party artifact. The existing release test that asserts this one-entry inventory must stay green.

- [ ] **Step 3: Regenerate/review the screenshot**

Run `uv run --no-sync python scripts/browser_smoke.py --screenshot docs/assets/browser-workbench.png`. Verify notice, badge, provenance, `N/A · no inference`, polygon, and no DOTA pixels.

- [ ] **Step 4: Verify and commit**

Run release/readme tests, release checker, and Pages verifier. Commit docs/evidence/test/screenshot as `docs: explain dual-mode browser evidence`.

## Task 8: Separate PR candidate and manual Pages deployment

**Files:** Create `tests/test_pages_workflow.py` and `pages.yml`; modify `release-gates.yml`.

**Interfaces:** Release workflow verifies/uploads a generic candidate without deployment permission. Pages workflow is `workflow_dispatch` only and protected-main only. Neither edits About.

- [ ] **Step 1: Add failing workflow tests**

```python
def test_release_workflow_never_deploys_pages() -> None:
    text = (ROOT / ".github/workflows/release-gates.yml").read_text()
    assert "scripts/pages_artifact_check.py" in text
    assert "actions/upload-artifact@v4" in text
    assert "actions/deploy-pages" not in text
    assert "pages: write" not in text

def test_pages_workflow_is_manual_and_about_free() -> None:
    text = (ROOT / ".github/workflows/pages.yml").read_text()
    trigger = text.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert "push:" not in trigger and "pull_request:" not in trigger
    assert "github.ref == 'refs/heads/main'" in text
    assert "actions/upload-pages-artifact@v4" in text
    assert "actions/deploy-pages@v4" in text
    assert "pages: write" in text and "id-token: write" in text
    assert "gh repo edit" not in text and "--homepage" not in text
```

- [ ] **Step 2: Verify red**

Run `uv run --no-sync python -m pytest tests/test_pages_workflow.py -q`.

Expected: missing workflow/candidate steps.

- [ ] **Step 3: Add the non-deploying candidate job**

Add `pages-candidate` to release gates. It needs `core-cpu` and `browser-smoke`, runs the Pages verifier, uploads `demo/web` via `actions/upload-artifact@v4` as `aerial-obb-pages-candidate-${{ github.sha }}`, and has `contents: read` only.

- [ ] **Step 4: Create manual-only `pages.yml`**

```yaml
name: Deploy reviewed Pages artifact
on:
  workflow_dispatch:
concurrency:
  group: pages
  cancel-in-progress: false
permissions:
  contents: read
jobs:
  verify:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with: {python-version: "3.11"}
      - uses: actions/setup-node@v7
        with: {node-version: "22"}
      - run: python -m pip install uv==0.11.18
      - run: uv sync --frozen --no-install-project
      - run: uv run --no-sync python -m pytest -q
      - run: uv run --no-sync python scripts/repo_check.py
      - run: uv run --no-sync python scripts/release_check.py
      - run: uv run --no-sync python scripts/pages_artifact_check.py
      - run: uv run --no-sync playwright install --with-deps chromium
      - run: uv run --no-sync python scripts/browser_smoke.py
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v4
        with: {path: demo/web}
  deploy:
    needs: verify
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

This file does not enable Pages and cannot run without manual dispatch.

- [ ] **Step 5: Verify and commit**

Run workflow tests, full pytest, repo/release gates. Commit as `ci: separate Pages candidate and deployment`.

## Task 9: Verify locally and stop before remote actions

**Files:** Verification only; no tracked changes.

- [ ] **Step 1: Run every deterministic gate**

```powershell
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/clean_export_check.py
```

- [ ] **Step 2: Render previews outside the repo**

```powershell
$previewDir = Join-Path ([System.IO.Path]::GetTempPath()) ("aerial-obb-pages-preview-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $previewDir | Out-Null
uv run --no-sync python scripts/browser_smoke.py --screenshot (Join-Path $previewDir "desktop.png") --mobile-screenshot (Join-Path $previewDir "mobile.png")
```

Review notice order, modes, provenance, runtime, focus/layout, and absence of DOTA.

- [ ] **Step 3: Review exact local network behavior**

Run `uv run --no-sync python -m http.server 8765 --directory demo/web`. In a clean browser profile, load `http://127.0.0.1:8765/`, click synthetic, verify no jsDelivr request, exercise cached filters, and verify Source/AGPL links.

- [ ] **Step 4: Perform owner-controlled BYOM smoke**

The owner selects a trusted compatible model stored outside the repo and a synthetic or rights-cleared non-DOTA image. Confirm ORT requests start only at BYOM model selection, Detect produces the BYOM badge and numeric runtime, and invalid model/image recovery leaks nothing. Never copy the model into checkout, artifact, log, screenshot metadata, or commit. If this cannot pass, stop as blocked.

- [ ] **Step 5: Verify branch boundaries and stop**

Run:

```powershell
git status --short
git log --oneline --decorate main..HEAD
git diff --name-status main...HEAD
git diff --check main...HEAD
```

Report tests, gates, preview paths, BYOM smoke, branch, commits, and paths. Do not push.

## Remote Gate A: Push and PR

Local completion does not authorize push. Obtain written owner authorization before pushing or creating a PR. The PR may run release gates and upload a generic candidate artifact; it must not enable/deploy Pages or change About.

## Remote Gate B: Merge

PR approval does not authorize merge. Merge only after required CI, artifact equality, desktop/mobile/privacy review, and separate owner authorization. The manual Pages workflow remains inert.

## Remote Gate C: Pages

Merge does not authorize Pages. With fresh written approval, confirm exact protected `main` SHA, enable GitHub Actions as Pages source, protect `github-pages` environment, and manually dispatch `pages.yml`. Keep About blank. No scheduled/push-triggered initial deployment.

## Remote Gate D: Live Review

Review `https://kuotunyu.github.io/aerial-obb-lab/` for HTTPS/assets, static notice order, synthetic no-runtime behavior, BYOM lazy runtime/safe failures, responsive/accessibility, Source/AGPL, exact external origins, and forbidden-artifact absence. Failure leaves About blank; disabling Pages also needs explicit approval.

## Remote Gate E: About

Live approval still does not authorize About. A separate task and written approval may set `https://kuotunyu.github.io/aerial-obb-lab/`, read the GitHub API back, verify equality, and record the portfolio ledger. No workflow may invoke `gh repo edit --homepage`.

## Definition of Done

- Tasks 1-8 finish through red/green TDD and small commits.
- Task 9 tests, artifact/privacy scan, previews, and owner-controlled BYOM smoke pass.
- Branch stays clean and unpushed until separately authorized.
- PR, merge, Pages, live review, and About remain independent approval gates.
- No weight, ONNX, DOTA visual, secret, private path/metadata, raw error, or unsupported claim enters the repo or Pages artifact.

# Aerial OBB Single-harbor Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current three-option gallery with one fixed, reviewed low-density harbor aerial image that is visible before an explicit Detect and then receives genuine local-browser OBB inference.

**Architecture:** Keep the existing static `demo/web` workbench, privacy-sanitized ONNX model, lazy ORT loader, shared decode/filter/corners/render pipeline, generation token, and advanced BYOM lifecycle. Collapse only the built-in sample boundary: one immutable harbor record is admitted by the closed receipt/manifest, the page renders a non-interactive identity block instead of a selector, and every public/release verifier admits exactly `samples/harbor.jpg`.

**Tech Stack:** Static HTML/CSS/JavaScript, ONNX Runtime Web 1.20.1, Python 3.10+, pytest, Playwright Chromium, Pillow, Git/GitHub Actions release checks.

## Global Constraints

- Work only in `D:\AI-Portfolio\.worktrees\aerial-obb-live-real-image-demo` on branch `feat/pages-live-real-image-demo`; preserve the existing branch history and the earlier gallery design/plan as historical records.
- The only current built-in sample is `harbor`, title `低密度港區航拍範例`, alt `低密度港區的真實航拍原圖`, public path `samples/harbor.jpg`, 241046 bytes, SHA-256 `916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0`.
- The exact harbor source is USGS NAIP Plus ImageServer product `m_3411955_sw_11_060_20220514`, USDA, 2022, acquired 2022-05-14, public-domain record `https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39`, WGS84 bbox `[-119.216719, 34.14417, -119.200719, 34.15417]`.
- The derivative remains 1280×800 sRGB JPEG quality 90 with crop/resample/metadata removal. Do not alter `demo/web/samples/harbor.jpg` bytes.
- The public confidence stays 0.25 with no class filter and no per-image tuning. Real-browser admission requires 16–26 detections and class IDs `[1, 2, 7]`; the representative ship remains centered near `(951.1, 443.1)` with `w=164.4`, `h=26.4`, tolerance 32.9.
- Initial load may fetch same-origin static assets and the harbor JPEG only. It must make zero ORT, WASM, manifest, or ONNX requests and perform no inference before the visitor presses `開始 Detect`.
- Detect must execute the existing genuine local-browser session path. No fixed tensors, precomputed boxes, hand-edited polygons, committed annotated image, second model, automatic Detect, server inference, or accuracy/evaluation/benchmark claim is permitted.
- Keep one session, one active result, one monotonically increasing generation token, and one shared cached-output render pipeline. Do not add a selector state, per-image cache, or presentation cache.
- Preserve advanced BYOM lazy loading, atomic session replacement, safe fixed error copy, privacy boundary, Source/AGPL links, skip link, `aria-describedby` canvas description, stable input names, focus-visible, reduced-motion, forced-colors, and responsive behavior.
- The fixed harbor identity is an informational `h3` followed by `真實航拍原圖`; neither may have button, selection, pressed-state, or live-region semantics.
- Built-in image admission failure copy is exactly `範例影像目前無法顯示。請重新整理後重試，或使用進階 BYOM。`; it clears stale runtime/canvas/table/summary/description/result state and never exposes URL, filename, path, response body, model metadata, raw exception, or stack in UI or console.
- Delete `demo/web/samples/airfield.jpg` and `demo/web/samples/sports-complex.jpg` from the current tree and remove their active runtime, test, tool, notice, evidence, and release references. Do not rewrite historical `docs/superpowers` records or Git history.
- Strict TDD applies to every behavior change: add the named test, run it against production code, record only the earliest assertion that actually fails, implement the smallest GREEN, refactor only while GREEN, and commit the task before moving on.
- Each task uses a fresh implementer plus fresh task-scoped spec and quality reviewers. Findings return to that task's implementer for one focused RED→GREEN fix and fresh re-review; the controller does not self-approve.
- Local files, tests, loopback preview, repository-external evidence, and local commits are authorized. Push, PR, merge, Pages configuration/dispatch/deployment, About, Hugging Face, release, tag, visibility, branch/worktree deletion, and changes to any other repository are not authorized.

---

## File and Responsibility Map

| File | Responsibility in the final design |
| --- | --- |
| `release/sample-gallery-sources.json` | Closed one-record public-domain harbor receipt. |
| `scripts/prepare_sample_gallery.py` | Deterministic acquisition/verification/publication for the one approved harbor recipe; no category pool or selection workflow. |
| `scripts/sample_gallery_smoke.py` | Repository-external real-browser verification of the approved harbor candidate only. |
| `scripts/prepare_demo_assets.py` | Validates the one harbor receipt alongside the existing sanitized model/license publication. |
| `tests/test_sample_gallery.py` | Source, derivation, receipt, containment, privacy, and atomic harbor publication contracts. |
| `tests/test_demo_assets.py` | One-harbor receipt/asset-publisher contract and model/license separation. |
| `demo/web/demo-model.json` | Closed runtime manifest with one `samples` member and no `defaultSampleId`. |
| `demo/web/demo-assets.js` | Deep-freezes the exact manifest and exposes `getDemoSample()` for the one immutable harbor record. |
| `demo/web/index.html` | Fixed harbor identity, initial original, Detect/toggle/filter/BYOM workbench, and claim/accessibility semantics. |
| `demo/web/app.js` | Single built-in harbor state, lazy image/session loading, genuine inference, shared cached rendering, failure recovery, and BYOM transitions. |
| `demo/web/style.css` | Compact informational sample block and removal of selector-only styles. |
| `scripts/browser_smoke.py` | Real Chromium initial/Detect/cache/toggle/error/BYOM/accessibility/responsive/network assertions. |
| `tests/test_browser_parity.py` | Existing numerical decode/corners parity; unchanged unless a genuine regression proves otherwise. |
| `scripts/pages_artifact_check.py` | Exact one-harbor Pages inventory, digest, origin, privacy, and reviewed-text enforcement. |
| `scripts/release_check.py` | Cross-binds receipt, manifest, evidence, notices, artifact records, and current public claims. |
| `scripts/clean_export_check.py` | Requires the harbor and forbids the two superseded images in committed-only archives. |
| `scripts/repo_check.py` | Existing repository/link/static-demo gate; update only its exact current UI copy/inventory assertions. |
| `tests/test_pages_artifact_check.py` | Pages one-harbor acceptance and mutation coverage. |
| `tests/test_release_check.py` | Evidence, manifest, notice, docs, and stale-gallery release contracts. |
| `tests/test_clean_export.py` | Exact archive inventory and clean-export rejection tests. |
| `tests/test_repo_check.py` | Current static UI identity/link contract. |
| `tests/test_readme_language.py` | Chinese-first README and one-harbor release-checklist wording. |
| `release/artifact-manifest.json` | Exact current bundled/reviewed artifacts and canonical/raw digests. |
| `release/evidence.json` | Truthful one-harbor genuine-inference evidence; no gallery-selection claim. |
| `README.md`, `README.en.md`, `demo/web/README.md` | Current one-harbor original→Detect→result journey and non-evaluation claim. |
| `THIRD_PARTY_NOTICES.md`, `demo/web/THIRD_PARTY_NOTICES.md` | One harbor public-domain derivative plus separate model AGPL/privacy records. |
| `RELEASE_CHECKLIST.md`, `CHANGELOG.md` | Exact current sample inventory and local/remote release boundaries. |
| `docs/assets/browser-workbench.png` | Canonical unfiltered harbor result screenshot produced from exact reviewed bytes. |

---

### Task 1: Close the Source, Receipt, and Runtime Manifest to One Harbor Sample

**Files:**
- Modify: `tests/test_sample_gallery.py`
- Modify: `tests/test_demo_assets.py`
- Modify: `scripts/prepare_sample_gallery.py`
- Modify: `scripts/sample_gallery_smoke.py`
- Modify: `scripts/prepare_demo_assets.py`
- Modify: `release/sample-gallery-sources.json`
- Modify: `demo/web/demo-model.json`
- Modify: `demo/web/demo-assets.js`
- Delete: `demo/web/samples/airfield.jpg`
- Delete: `demo/web/samples/sports-complex.jpg`
- Keep byte-identical: `demo/web/samples/harbor.jpg`

**Interfaces:**
- Consumes: the exact harbor identity/digest/source/derivation/guardrails in Global Constraints and the existing model/input/output/license/sanitization manifest fields.
- Produces: `DemoAssets.validateManifest(payload) -> Readonly<object>`, `DemoAssets.fetchVerifiedModel(manifest, options?) -> Promise<ArrayBuffer>`, and `DemoAssets.getDemoSample() -> Readonly<HarborSample>` where `HarborSample.id === "harbor"`.

- [ ] **Step 1: Replace the category-pool tests with one-harbor RED contracts**

In `tests/test_sample_gallery.py`, replace the obsolete three-category/pool/approval/publication expectations with these named contracts while retaining transport caps, official-source identity, deterministic JPEG, containment, atomic rollback, and public-safe diagnostic tests:

```python
def test_candidate_recipes_are_exact_approved_harbor_only() -> None:
    assert CANDIDATE_RECIPES == (
        CandidateRecipe(
            "harbor-port-hueneme",
            "harbor",
            (-119.216719, 34.144170, -119.200719, 34.154170),
        ),
    )


def test_approved_sample_receipt_is_exact_harbor_contract() -> None:
    receipt = json.loads((ROOT / "release/sample-gallery-sources.json").read_text("utf-8"))
    assert receipt["schemaVersion"] == 1
    assert [sample["id"] for sample in receipt["samples"]] == ["harbor"]
    sample = receipt["samples"][0]
    assert (sample["path"], sample["bytes"], sample["sha256"]) == (
        "samples/harbor.jpg",
        241046,
        "916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0",
    )
    assert sample["source"]["productId"] == "m_3411955_sw_11_060_20220514"
    assert sample["derivation"]["bboxWgs84"] == [-119.216719, 34.14417, -119.200719, 34.15417]
    assert sample["guardrails"]["classIds"] == [1, 2, 7]
    assert (sample["guardrails"]["countMin"], sample["guardrails"]["countMax"]) == (16, 26)


def test_publish_writes_only_harbor_and_removes_superseded_sample_assets_atomically(
    tmp_path: Path,
    approved_gallery_report: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_root = tmp_path
    review_name = approved_gallery_report["records"][0]["image"]["reviewName"]
    approved_harbor_bytes = (review_root / review_name).read_bytes()
    pages_root = tmp_path / "demo" / "web"
    samples = pages_root / "samples"
    samples.mkdir(parents=True)
    (samples / "airfield.jpg").write_bytes(b"superseded-airfield")
    (samples / "sports-complex.jpg").write_bytes(b"superseded-sports")
    receipt = tmp_path / "release" / "sample-gallery-sources.json"
    monkeypatch.setattr(gallery, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gallery, "_git_worktree_roots", lambda _root: {tmp_path / "repo"})
    gallery.publish_approved_gallery(approved_gallery_report, review_root, pages_root, receipt)
    assert sorted(path.name for path in (pages_root / "samples").iterdir()) == ["harbor.jpg"]
    assert (samples / "harbor.jpg").read_bytes() == approved_harbor_bytes
```

In `tests/test_demo_assets.py`, replace `test_gallery_receipt_admits_only_the_published_three_sample_bytes` and three-sample mutation fixtures with:

```python
def test_gallery_receipt_admits_only_the_published_harbor_bytes(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "release/sample-gallery-sources.json").read_text("utf-8"))
    assert [sample["id"] for sample in payload["samples"]] == ["harbor"]
    assert validate_gallery_publication(ROOT / "demo/web", ROOT / "release/sample-gallery-sources.json") == payload


def test_demo_manifest_declares_one_fixed_harbor_sample() -> None:
    manifest = json.loads((ROOT / "demo/web/demo-model.json").read_text("utf-8"))
    assert "defaultSampleId" not in manifest
    assert [sample["id"] for sample in manifest["samples"]] == ["harbor"]
    assert manifest["samples"][0]["alt"] == "低密度港區的真實航拍原圖"
```

Retain nested missing/extra/mutation parametrization, but address `samples[0]` as the harbor record and add explicit zero/extra/duplicate/renamed/path-traversal/external-path/wrong-byte/wrong-digest/wrong-source/wrong-alt/wrong-derivation/wrong-guardrail rejection cases.
Rewrite the `approved_gallery_report` fixture to contain only `CANDIDATE_RECIPES[0]` (`harbor-port-hueneme`), one matching browser observation, and one `visualReview` approval; no obsolete recipe index or category-family map remains.

- [ ] **Step 2: Run the batch and record the earliest reachable RED only**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py::test_candidate_recipes_are_exact_approved_harbor_only tests/test_sample_gallery.py::test_approved_sample_receipt_is_exact_harbor_contract tests/test_demo_assets.py::test_gallery_receipt_admits_only_the_published_harbor_bytes tests/test_demo_assets.py::test_demo_manifest_declares_one_fixed_harbor_sample -q
```

Expected against the current three-image implementation: the first executed contract fails because `CANDIDATE_RECIPES` still contains nine recipes and/or the receipt IDs are `airfield`, `sports-complex`, `harbor`. Record the actual first failure shown by pytest; do not claim later assertions were independently observed if execution stopped earlier.

- [ ] **Step 3: Narrow the source admission and external smoke to the approved harbor recipe**

In `scripts/prepare_sample_gallery.py`:

```python
@dataclass(frozen=True)
class CandidateRecipe:
    candidate_id: str
    category: Literal["harbor"]
    bbox_wgs84: tuple[float, float, float, float]


CANDIDATE_RECIPES = (
    CandidateRecipe(
        "harbor-port-hueneme",
        "harbor",
        (-119.216719, 34.144170, -119.200719, 34.154170),
    ),
)
RECIPE_BY_ID = {"harbor-port-hueneme": CANDIDATE_RECIPES[0]}
_PUBLIC_SAMPLE_IDS = ("harbor",)
```

Remove the airfield/sports recipe literals, two-or-three-per-category pool rule, multi-category approval selection, and their now-dead helpers. Keep exact official NAIP response validation, bounded transport, deterministic derivation, external-root containment, source byte re-binding, fixed diagnostics, one-record approval verification, and atomic publish/rollback. `scripts/sample_gallery_smoke.py` must accept and verify exactly `harbor-port-hueneme`, one real browser run, numeric runtime, the exact guardrails, and repository-external screenshots/report; it must reject any second candidate or category.

- [ ] **Step 4: Close the receipt, model manifest, and loader**

Reduce `release/sample-gallery-sources.json` and `demo/web/demo-model.json` to the existing harbor object only, preserving its byte-for-byte field values. Remove `defaultSampleId` from `demo-model.json`.

In `demo/web/demo-assets.js`, make the one-sample interface exact:

```javascript
const SAMPLE_IDS = Object.freeze(["harbor"]);

function getDemoSample() {
  return EXPECTED.samples[0];
}

root.DemoAssets = Object.freeze({
  validateManifest,
  fetchVerifiedModel,
  getDemoSample,
});
```

Delete `getSampleCatalog`, its export, and `defaultSampleId` validation. Keep recursive exact-key/type comparison, deep freezing, model size/digest validation, same-origin fetch, and error codes unchanged. In `scripts/prepare_demo_assets.py`, require the exact ID/path arrays `["harbor"]` and `["samples/harbor.jpg"]`, the exact alt/class IDs above, and an approved managed sample set containing only `samples/harbor.jpg`.

Delete the two superseded JPEGs with exact pathspecs; do not modify `harbor.jpg`:

```powershell
git rm -- demo/web/samples/airfield.jpg demo/web/samples/sports-complex.jpg
```

- [ ] **Step 5: Reach Task 1 GREEN and verify the retained binary**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py tests/test_demo_assets.py -q
$harbor = Get-Item -LiteralPath 'demo/web/samples/harbor.jpg'
if ($harbor.Length -ne 241046) { throw 'harbor byte size changed' }
$digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $harbor.FullName).Hash.ToLowerInvariant()
if ($digest -ne '916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0') { throw 'harbor digest changed' }
git diff --check
```

Expected: focused tests pass; harbor bytes/digest are exact; the two superseded JPEGs are deleted; current receipt/manifest/loader/source-tool tests contain no active airfield or sports-complex contract.

- [ ] **Step 6: Review and commit Task 1**

Give fresh spec and quality reviewers the Task 1 diff, RED output, GREEN output, digest evidence, and design spec. Resolve findings through the same implementer and re-review. Stage exactly:

```powershell
git add -- tests/test_sample_gallery.py tests/test_demo_assets.py scripts/prepare_sample_gallery.py scripts/sample_gallery_smoke.py scripts/prepare_demo_assets.py release/sample-gallery-sources.json demo/web/demo-model.json demo/web/demo-assets.js
git add -u -- demo/web/samples/airfield.jpg demo/web/samples/sports-complex.jpg
git diff --cached --check
git commit -m "feat: admit one harbor demo sample"
```

---

### Task 2: Replace the Selector with a Fixed Harbor Original and Genuine Detect Path

**Files:**
- Modify: `scripts/browser_smoke.py`
- Modify: `demo/web/index.html`
- Modify: `demo/web/app.js`
- Modify: `demo/web/style.css`

**Interfaces:**
- Consumes: `DemoAssets.getDemoSample()` from Task 1 and the existing `loadOrtRuntime`, `ensureDemoSession`, `runActiveInference`, `renderCachedOutput`, `setResultView`, `nextGeneration`, and `reportFailure` paths.
- Produces: a fixed `#demoSampleTitle`/`#demoSampleKind` identity, `loadDemoImage(token) -> Promise<HTMLImageElement|null>`, one built-in `DEMO_SAMPLE`, and browser scenarios `single-harbor` and `single-harbor-failures`.

- [ ] **Step 1: Write the real-browser batch RED for fixed identity, lazy network, and real inference**

Replace `assert_sample_gallery_initial`/`run_sample_gallery` with `assert_single_harbor_initial`/`run_single_harbor`. Register `--scenario single-harbor` in the CLI and remove `sample-gallery` from the current scenario choices. The scenario must use real Chromium, exact committed harbor/model bytes, and the existing instrumentation counters:

```python
def assert_single_harbor_initial(page, requests, messages) -> None:
    assert page.locator("#sampleSelector, .sample-option").count() == 0
    title = page.locator("h3#demoSampleTitle")
    assert title.inner_text() == "低密度港區航拍範例"
    assert page.locator("#demoSampleKind").inner_text() == "真實航拍原圖"
    assert title.get_attribute("role") is None
    assert page.locator("#demoSampleTitle, #demoSampleKind").evaluate_all(
        "nodes => nodes.every(node => !node.matches('button,[aria-pressed],[aria-selected]'))"
    )
    assert page.locator("#demoOriginalImage").get_attribute("src") == "samples/harbor.jpg"
    assert page.locator("#demoOriginalImage").get_attribute("alt") == "低密度港區的真實航拍原圖"
    assert page.locator("#runtimeValue").inner_text() == "—"
    assert not any(url == ORT_CDN_URL or url.startswith(ORT_WASM_BASE) for url in requests)
    assert DEMO_MODEL_PATH not in _request_paths(requests)
    assert messages == []
```

After clicking `#demoDetectBtn`, wait for the success status and assert:

```python
assert page.locator("#modeBadge").inner_text() == "LOCAL BROWSER INFERENCE"
assert re.fullmatch(r"[1-9]\d* ms", page.locator("#runtimeValue").inner_text())
count = int(page.locator("#summaryCount").inner_text())
assert 16 <= count <= 26
assert page.locator("#resultsBody tr:not([data-empty='true'])").count() == count
assert page.locator("#canvasFrame").is_visible()
assert "低密度港區航拍範例" in page.locator("#provenanceValue").inner_text()
assert page.evaluate("globalThis.__aerialObbTest.sessionRunCount") == 1
```

Also assert the decoded class-ID set is a subset of `{1, 2, 7}`, the representative ship falls within the frozen tolerance, `查看原圖`/`查看 Detection 結果` do not run inference, and confidence/class filter changes rerender the same cached output with the session-run counter still `1`.

Drive the confidence/class filters to a legitimate empty result and assert count `0`, no polygons/rows, explicit empty canvas description, numeric runtime retained, success state retained, and session-run counter still `1`; this distinguishes an empty filtered result from reset/error.

- [ ] **Step 2: Run the browser scenario and observe the actual earliest RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario single-harbor
```

Expected against the current UI: FAIL first because `#sampleSelector`/`.sample-option` still exist or the initial image is `samples/airfield.jpg`. Record only the first reached browser assertion; the numeric runtime and guardrails are not claimed RED until execution reaches them.

- [ ] **Step 3: Replace selector markup with fixed informational identity**

In `demo/web/index.html`:

- change the notice to one curated USGS/USDA NAIP public-domain harbor example and retain the non-evaluation/non-benchmark boundary before Detect;
- change the rail helper to `先查看真實航拍原圖，再由目前的 browser 執行 Detect。`;
- remove `#sampleSelector`, its legend, all `.sample-option` buttons, thumbnails, `name="demo-sample"`, `aria-pressed`, and `data-sample-id`;
- retain `#sampleCard`, and insert this non-interactive block before `#sampleState`:

```html
<div class="demo-sample-identity">
  <h3 id="demoSampleTitle">低密度港區航拍範例</h3>
  <p id="demoSampleKind">真實航拍原圖</p>
</div>
```

- set `#demoOriginalImage` to `src="samples/harbor.jpg"`, `alt="低密度港區的真實航拍原圖"`, width 1280, height 800;
- keep `#demoDetectBtn`, result controls, BYOM, summary, canvas description, table, links, and focus order intact.

In `demo/web/style.css`, delete selector/card-button/thumb/pressed-state rules that serve no other element. Style `.demo-sample-identity` as compact static content using the existing border/type/spacing tokens; do not make it hoverable or pointer-shaped and do not redesign the workbench grid.

- [ ] **Step 4: Collapse `app.js` to one immutable demo sample**

At module initialization:

```javascript
const DEMO_SAMPLE = DemoAssets.getDemoSample();

const state = {
  source: "demo",
  phase: "idle",
  generation: 0,
  generationAbort: null,
  session: null,
  sessionSource: null,
  image: null,
  imageSource: null,
  cached: null,
  elapsedMs: null,
  manifest: null,
  demoModelBytes: null,
  view: "original",
};
```

Remove `sampleOptions`, `SAMPLE_CATALOG`, `selectedSampleId`, `selectedDemoSample`, `setSampleSelection`, `selectDemoSample`, and the sample-option event listeners. Replace `loadSelectedDemoImage(sample, token)` with `loadDemoImage(token)` that always binds the exact `DEMO_SAMPLE.path/alt/width/height`, authenticates decode for the current token, and throws only `DEMO_IMAGE_DECODE` internally.

`resetToDemoOriginal()` and initial startup call `loadDemoImage(nextGeneration())` when the committed image is not decoded; otherwise they bind the existing decoded element. `runDemo()` compares only the current element/source against `DEMO_SAMPLE.path`, reloads the harbor when returning from BYOM, and then calls the unchanged `ensureDemoSession` + `runActiveInference("demo", generation)` path. Demo provenance uses `DEMO_SAMPLE.title`. There is no array search, selector update, or per-sample cache.

- [ ] **Step 5: Reach Task 2 GREEN and run covering browser/regression checks**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario single-harbor
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility
uv run --no-sync python scripts/browser_smoke.py --scenario desktop
uv run --no-sync python scripts/browser_smoke.py --scenario mobile
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_repo_check.py -q
git diff --check
```

Expected: the harbor original is initial; no selector exists; no ORT/WASM/ONNX is requested before Detect; one real Detect yields numeric runtime and accepted harbor output; cached toggle/filters do not rerun; accessibility/desktop/mobile remain GREEN.

- [ ] **Step 6: Review and commit Task 2**

Fresh spec and quality reviewers receive the exact diff, RED/GREEN output, session/request counters, and public-safe original/result screenshots outside the repository. Resolve findings and re-review. Stage exactly:

```powershell
git add -- scripts/browser_smoke.py demo/web/index.html demo/web/app.js demo/web/style.css
git diff --cached --check
git commit -m "feat: simplify demo to one harbor image"
```

---

### Task 3: Harden One-sample Failure Recovery, BYOM Transitions, and Accessibility

**Files:**
- Modify: `scripts/browser_smoke.py`
- Modify only when a named browser RED proves it necessary: `demo/web/app.js`
- Modify only when a named browser RED proves it necessary: `demo/web/index.html`
- Modify only when a named browser RED proves it necessary: `demo/web/style.css`

**Interfaces:**
- Consumes: Task 2 `loadDemoImage`, fixed identity DOM, generation token, shared result/reset functions, and existing BYOM/session lifecycle.
- Produces: deterministic `single-harbor-failures` coverage with fixed recovery copy and preserved BYOM/accessibility/network contracts.

- [ ] **Step 1: Write the deterministic single-harbor failure RED**

Replace selector-specific held-decode/invalid-selector/cross-sample/superseded-reload cases with a route-controlled one-sample case. The Playwright route must fail exactly the first `samples/harbor.jpg` response before page load, then serve the real committed bytes after `page.reload()`; it must not use sleep/timing races.

Register `--scenario single-harbor-failures` and assert on the failed load:

```python
assert page.locator("#status").inner_text() == "範例影像目前無法顯示。請重新整理後重試，或使用進階 BYOM。"
assert page.locator("#demoDetectBtn").is_disabled()
assert page.locator("#runtimeValue").inner_text() == "—"
assert page.locator("#summaryCount").inner_text() == "0"
assert page.locator("#canvasFrame").is_hidden()
assert page.locator("#resultsBody tr[data-empty='true']").count() == 1
assert page.locator("#canvasDescription").inner_text() == "尚無 detection result。"
assert not any(url == ORT_CDN_URL or url.startswith(ORT_WASM_BASE) for url in requests)
assert DEMO_MODEL_PATH not in _request_paths(requests)
assert not _contains_private_diagnostic(page.locator("body").inner_text(), messages)
```

After deterministic retry/reload, assert the harbor original is ready, explicit Detect succeeds, and stale error/result text is absent.

- [ ] **Step 2: Add BYOM-return and semantic checks to the same browser batch**

Extend existing real `run_byom_transition` and `run_accessibility` assertions:

- opening BYOM is network-silent;
- selecting either BYOM file clears harbor canvas/table/runtime/description immediately;
- model selection triggers the one pinned jsDelivr ORT script with exact integrity and anonymous CORS only when a runtime is not already cached;
- returning to the demo path restores `samples/harbor.jpg`, `低密度港區的真實航拍原圖`, runtime `—`, count `0`, mode `尚未 Detect`, and requires explicit Detect;
- the skip link is first focusable and targets `main#mainContent`;
- `h3#demoSampleTitle` is in heading order and is not interactive;
- every file/range/class input retains its stable name, canvas retains `aria-describedby="canvasDescription"`, and the description stays synchronized with the sorted filtered table without `aria-live`;
- 1280×720, 390×844, 200%-zoom-equivalent width, reduced motion, and forced colors have no page-level horizontal overflow or obscured primary action.

- [ ] **Step 3: Run the batch and record the earliest real RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario single-harbor-failures
uv run --no-sync python scripts/browser_smoke.py --scenario byom-transition
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility
```

Expected: the new failure scenario reaches a real product mismatch if Task 2 did not use the exact recovery copy or did not clear a surface. Record that earliest mismatch only. If the existing product already satisfies a later BYOM/accessibility assertion, record it as GREEN rather than fabricating a RED.

- [ ] **Step 4: Make only the minimal production correction proven by the RED**

Use the existing `reportFailure`, `clearResultState`, `resetResult`, `showOriginalSource`, `nextGeneration`, and `isCurrentGeneration` functions. Add the fixed image-load copy under the existing internal `DEMO_IMAGE_DECODE` code:

```javascript
DEMO_IMAGE_DECODE: "範例影像目前無法顯示。請重新整理後重試，或使用進階 BYOM。",
```

Do not log the caught error. Keep console output to the fixed category form `[AERIAL_OBB:DEMO_IMAGE_DECODE]`. Ensure the failure path clears cached output, elapsed time, polygons, rows, summary, description, toggle, and completed status while leaving Detect disabled until a newly decoded harbor original exists. Do not duplicate reset logic inside event handlers.

- [ ] **Step 5: Reach Task 3 GREEN**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario single-harbor-failures
uv run --no-sync python scripts/browser_smoke.py --scenario byom-transition
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility
uv run --no-sync python scripts/browser_smoke.py --scenario desktop
uv run --no-sync python scripts/browser_smoke.py --scenario mobile
uv run --no-sync python scripts/browser_smoke.py
git diff --check
```

Expected: all focused scenarios and the complete smoke pass; request origins are loopback/same-origin initially and pinned jsDelivr only after Detect/BYOM model selection; no stale sample/result state or private diagnostic appears.

- [ ] **Step 6: Review and commit Task 3**

Fresh reviewers receive route rules, RED/GREEN output, request/session counters, accessibility checks, and the exact modified path set. Resolve findings and re-review. Stage `scripts/browser_smoke.py` plus only production files whose named RED required a change:

```powershell
$task3Paths = @('scripts/browser_smoke.py')
foreach ($path in @('demo/web/app.js','demo/web/index.html','demo/web/style.css')) {
  if (git diff --name-only -- $path) { $task3Paths += $path }
}
git add -- $task3Paths
git diff --cached --check
git commit -m "test: harden single-harbor demo recovery"
```

---

### Task 4: Freeze One-harbor Public Evidence, Documentation, and Release Gates

**Files:**
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `tests/test_repo_check.py`
- Modify: `tests/test_readme_language.py`
- Modify: `scripts/pages_artifact_check.py`
- Modify: `scripts/release_check.py`
- Modify: `scripts/clean_export_check.py`
- Modify: `scripts/repo_check.py`
- Modify: `release/artifact-manifest.json`
- Modify: `release/evidence.json`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `demo/web/README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `demo/web/THIRD_PARTY_NOTICES.md`
- Modify: `RELEASE_CHECKLIST.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/assets/browser-workbench.png`

**Interfaces:**
- Consumes: reviewed Task 1–3 bytes and behavior, the exact one-record receipt, and the existing separate model/license/sanitization identity.
- Produces: exact Pages/release/archive contracts for one harbor image, canonical harbor screenshot, and truthful current documentation.

- [ ] **Step 1: Write the release-contract batch RED**

Replace the stale three-gallery test names/contracts with these exact names and assertions:

```python
def test_pages_tree_admits_exact_single_harbor_inventory(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    assert verify_pages_tree(site) == []
    assert tuple(sorted(path.name for path in (site / "samples").iterdir())) == ("harbor.jpg",)


def test_browser_demo_evidence_records_exact_single_harbor() -> None:
    browser = load_evidence()["browser_demo"]
    assert browser["demo_images"] == ["demo/web/samples/harbor.jpg"]
    assert browser["default_demo_image"] == "demo/web/samples/harbor.jpg"
    assert browser["sample_count"] == 1
    assert browser["sample_selection"] == "fixed-no-selector"
    assert browser["confidence"] == 0.25
    assert browser["per_image_tuning"] is False
    assert browser["precomputed_results"] is False
    assert browser["demo_inference_performed"] is True
    assert browser["represents_accuracy_evaluation"] is False


def test_clean_export_keeps_exact_harbor_and_omits_superseded_samples(tmp_path: Path) -> None:
    archive = _committed_candidate_archive(tmp_path)
    with zipfile.ZipFile(archive) as bundle:
        names = {item.filename for item in bundle.infolist() if not item.is_dir()}
    assert "demo/web/samples/harbor.jpg" in names
    assert "demo/web/samples/airfield.jpg" not in names
    assert "demo/web/samples/sports-complex.jpg" not in names
    assert "demo/web/samples/boats.jpg" not in names
    assert inspect_archive(archive) == []
```

Also add/rename:

- `test_real_demo_manifest_records_exact_single_harbor_artifact` cross-checks receipt ↔ `demo-model.json` ↔ bundled artifact entry for every exact harbor field;
- `test_notices_record_single_public_domain_harbor_derivation` requires the harbor title/path/product/date/agency/public-domain/derivation/digest/bytes/no-endorsement facts and the separate AGPL model record;
- `test_current_readmes_describe_fixed_harbor_detect_journey` requires one visible original, explicit genuine local inference, toggle/cached filters/advanced BYOM, and non-evaluation language in both languages;
- `test_release_checklist_matches_exact_single_harbor_inventory` requires one harbor JPEG and explicitly forbids the two superseded filenames;
- mutation tests reject zero/extra/duplicate image entries, either superseded file, wrong digest/source/alt/guardrail, changed fixed-no-selector claim, hidden threshold, precomputed result, second model, DOTA/private/token/path content, and unreviewed binary.

- [ ] **Step 2: Run the release batch and record the earliest reachable RED**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_pages_tree_admits_exact_single_harbor_inventory tests/test_release_check.py::test_browser_demo_evidence_records_exact_single_harbor tests/test_release_check.py::test_real_demo_manifest_records_exact_single_harbor_artifact tests/test_release_check.py::test_notices_record_single_public_domain_harbor_derivation tests/test_release_check.py::test_current_readmes_describe_fixed_harbor_detect_journey tests/test_clean_export.py::test_clean_export_keeps_exact_harbor_and_omits_superseded_samples tests/test_readme_language.py::test_release_checklist_matches_exact_single_harbor_inventory -q
```

Expected: FAIL first because the exact Pages/release/archive inventory still requires three sample paths or evidence still says `explicit-three-option`. Record the actual earliest failing assertion only.

- [ ] **Step 3: Make the exact verifier and evidence changes**

In `scripts/pages_artifact_check.py`, `scripts/release_check.py`, and `scripts/clean_export_check.py`, replace every current sample tuple/set with exactly:

```python
HARBOR_PATH = "demo/web/samples/harbor.jpg"  # omit demo/web/ in the Pages-root checker
HARBOR_BYTES = 241046
HARBOR_SHA256 = "916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0"
```

Use a one-item tuple/set where the checker API expects a collection. Remove airfield/sports entries from required members, public binary digests, bundled artifacts, reviewed artifacts, gallery paths, and canonical source-file lists. Continue to reject any other image/binary and preserve exact model/font/license/sanitization contracts, canonical-LF text digest behavior, link/origin/storage/telemetry/secret/path/DOTA scans, and allowlisted jsDelivr tuple.

Set `release/evidence.json` current browser fields to:

```json
{
  "demo_images": ["demo/web/samples/harbor.jpg"],
  "default_demo_image": "demo/web/samples/harbor.jpg",
  "sample_count": 1,
  "sample_selection": "fixed-no-selector",
  "confidence": 0.25,
  "per_image_tuning": false,
  "precomputed_results": false,
  "demo_inference_performed": true,
  "represents_accuracy_evaluation": false,
  "represents_t4_latency": false
}
```

Keep the existing distribution mode, lazy runtime, sanitized model, layout, responsive breakpoint, and non-representative claims. In `release/artifact-manifest.json`, retain exactly one public-domain NAIP image entry with the complete harbor receipt fields/restrictions and one reviewed `demo/web/samples/harbor.jpg`; remove the two superseded image entries. Recompute only changed canonical-LF/raw-binary bytes and SHA-256 values.

- [ ] **Step 4: Update current docs and notices without rewriting history**

Update `README.md`, `README.en.md`, `demo/web/README.md`, both third-party notices, `RELEASE_CHECKLIST.md`, and `CHANGELOG.md` to describe this exact current journey:

1. the harbor original is visible immediately;
2. no selector or automatic inference exists;
3. pressing Detect lazily loads the reviewed model/runtime and performs genuine local-browser inference;
4. original/result toggle and filters reuse the cached output;
5. BYOM remains an advanced local path; and
6. the curated harbor is integration evidence only, not ground truth, accuracy/evaluation, benchmark, representative-dataset, model-quality, or USGS/USDA endorsement evidence.

The notices include `samples/harbor.jpg`, product `m_3411955_sw_11_060_20220514`, USDA 2022/2022-05-14, public-domain record, bbox, crop/resample/metadata removal, 241046 bytes, exact SHA-256, and no-endorsement copy. Keep the Ultralytics/DOTAv1/AGPL/commercial-clearance boundary separate.

Run this current-surface scan and classify every hit. Production/runtime/receipt/evidence/notice/README positive references are failures; negative tests and reject-lists may retain the two filenames solely to prove their absence, and an explicitly historical changelog sentence may describe their removal:

```powershell
rg -n 'airfield\.jpg|sports-complex\.jpg|explicit-three-option|three sample|three-sample|三張|選擇範例|sampleSelector|sample-option|selectedSampleId' demo/web release scripts tests README.md README.en.md THIRD_PARTY_NOTICES.md RELEASE_CHECKLIST.md CHANGELOG.md
```

Do not scan or edit `docs/superpowers`, because approved historical designs/plans intentionally preserve superseded decisions.

- [ ] **Step 5: Generate and inspect the canonical exact-harbor screenshot**

Serve the exact tracked `demo/web`, execute the real single-harbor scenario at confidence 0.25 with no class filter, and write only the canonical screenshot:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario single-harbor --screenshot docs/assets/browser-workbench.png
```

Required pixels: compact fixed harbor identity in the left rail, annotated harbor result in the shared viewport, 16–26 detections, numeric runtime, `LOCAL BROWSER INFERENCE`, harbor provenance, readable table, claim notice, Source/AGPL context, collapsed BYOM, no selector, and no visually dominant wrong polygon. Inspect PNG metadata and byte strings for path, filename, private model identity, raw error, stack, token, and browser-profile leakage; update its exact artifact-manifest digest/bytes.

- [ ] **Step 6: Reach Task 4 GREEN across focused and direct gates**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_repo_check.py tests/test_readme_language.py tests/test_package_release.py -q
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/browser_smoke.py
git diff --check
```

Expected: all focused tests, direct artifact/repo/release gates, and full browser smoke pass with exactly one harbor JPEG, one reviewed model, no selector/current superseded filenames, no DOTA/private/precomputed output, and no unexpected origin.

- [ ] **Step 7: Review and commit Task 4**

Fresh reviewers inspect byte equality, artifact inventory, receipt binding, public-domain/AGPL claim separation, current documentation, stale-reference scan, canonical screenshot pixels/metadata, and all gate output. Resolve findings and re-review. Stage exactly:

```powershell
git add -- tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_repo_check.py tests/test_readme_language.py scripts/pages_artifact_check.py scripts/release_check.py scripts/clean_export_check.py scripts/repo_check.py release/artifact-manifest.json release/evidence.json README.md README.en.md demo/web/README.md THIRD_PARTY_NOTICES.md demo/web/THIRD_PARTY_NOTICES.md RELEASE_CHECKLIST.md CHANGELOG.md docs/assets/browser-workbench.png
git diff --cached --check
git commit -m "release: freeze single-harbor demo evidence"
```

---

### Task 5: Complete Full Local Acceptance, Clean Export, and Owner-operable Preview

**Files:**
- Create only repository-external clean export, screenshots, request-origin report, and review package.
- No tracked file is planned. Any finding returns to the owning Task 1–4 implementer, starts with a focused real RED, receives one minimal fix commit, and gets fresh re-review.

**Interfaces:**
- Consumes: all committed Task 1–4 outputs.
- Produces: complete local verification evidence, an exact committed-files-only export, a loopback UI the owner can operate, broad whole-branch review, and a clean retained branch.

- [ ] **Step 1: Verify scope, commits, and test inventory**

Run:

```powershell
git status --short --branch
git diff --check
git log --oneline --decorate -20
git diff --name-status bb4ee0e66ac57a2394afd99b1324f96f7523fad4...HEAD
uv run --no-sync python -m pytest --collect-only -q
```

Expected: tracked tree/index clean; task commits are present; only the two superseded JPEG deletions and planned current files changed; no unrelated repository or historical design file changed. Compare collection with the pre-change 452-test suite and account by exact renamed/removed test names for any count difference; no skip or unexplained loss is allowed.

- [ ] **Step 2: Run focused, browser, full-suite, and direct gates separately**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py tests/test_demo_assets.py tests/test_browser_parity.py tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_repo_check.py tests/test_readme_language.py tests/test_package_release.py -q
uv run --no-sync python scripts/browser_smoke.py --scenario single-harbor
uv run --no-sync python scripts/browser_smoke.py --scenario single-harbor-failures
uv run --no-sync python scripts/browser_smoke.py --scenario byom-transition
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility
uv run --no-sync python scripts/browser_smoke.py --scenario desktop
uv run --no-sync python scripts/browser_smoke.py --scenario mobile
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
```

Record exact pass counts, durations, harbor detection count/class set/representative check, session create/run/release counters, request origins, and zero-failure results. No slow test may be skipped.

- [ ] **Step 3: Run strict committed-files-only clean export**

```powershell
$cleanExport = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-harbor-clean-export-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $cleanExport) { throw 'Refusing to overwrite clean export' }
uv run --no-sync python scripts/clean_export_check.py --output $cleanExport
uv run --no-sync python scripts/pages_artifact_check.py --root (Join-Path $cleanExport 'demo/web')
```

Do not pass `--skip-browser`. Expected: archive rebuild, package build/install/import, full tests, one genuine harbor inference, privacy/origin/link/license checks, and byte/canonical-text equality pass from the export. Record the archive SHA-256 and exact member inventory.

- [ ] **Step 4: Run current-surface forbidden-artifact, privacy, and origin scans**

```powershell
rg -n -I -i 'airfield\.jpg|sports-complex\.jpg|boats\.jpg|DOTA.*\.(jpg|jpeg|png)|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|[A-Z]:[\\/]Users[\\/]|/(Users|home)/[^/ ]+/|Traceback|stack trace|selectedSampleId|sampleSelector|sample-option' demo/web release scripts tests README.md README.en.md THIRD_PARTY_NOTICES.md RELEASE_CHECKLIST.md CHANGELOG.md
git ls-files demo/web/samples
git ls-files demo/web | Sort-Object
```

Expected: no production/runtime/receipt/evidence/notice/README positive superseded image/selector reference and no private/token/path/raw-stack hit. Negative tests/reject-lists may name forbidden paths only to enforce absence. `git ls-files demo/web/samples` returns only `demo/web/samples/harbor.jpg`. Browser evidence permits same-origin page/harbor/manifest/model files and the pinned jsDelivr ORT/WASM origin only after Detect or BYOM model selection; no runtime source-imagery request occurs.

- [ ] **Step 5: Serve and visually operate the exact clean export**

Create a repository-external evidence directory and serve the export:

```powershell
$evidenceRoot = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-harbor-evidence-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $evidenceRoot | Out-Null
uv run --no-sync python -m http.server 8768 --bind 127.0.0.1 --directory (Join-Path $cleanExport 'demo/web')
```

Open `http://127.0.0.1:8768/` and capture repository-external original/result screenshots at 1280×720 and 390×844. Also inspect a 200%-zoom-equivalent width. Verify initial harbor original, fixed non-interactive title, explicit Detect, genuine annotated result, numeric runtime, original/result toggle, cached filters, collapsed BYOM, notice-before-control order, keyboard/focus, stable labels/names/headings, canvas alternative/table synchronization, reduced motion, forced colors, Source/AGPL/public-domain readability, no page-level overflow, no console/page error, and exact request origins.

- [ ] **Step 6: Run broad whole-branch review with one bounded fix wave**

Give the most capable fresh reviewer the approved spec, this plan, full `bb4ee0e66ac57a2394afd99b1324f96f7523fad4...HEAD` diff, task ledgers/reports/reviews, source receipt, Task 1 harbor digest evidence, Task 2–3 request/session/error outputs, canonical screenshot, full-suite/direct-gate output, clean-export digest/inventory, repository-external desktop/mobile screenshots, and privacy/stale scans.

Critical or Important findings permit one focused test-first fix wave through the responsible original implementer, a minimal fix commit, and fresh scoped re-review. A second product fix wave, source/license ambiguity, changed harbor bytes, artifact mismatch, privacy leak, unexpected origin, model/inference contract failure, or unresolved Important finding stops completion. Record Minor findings without unrelated polish.

- [ ] **Step 7: Verification-before-completion and branch readiness only**

Run fresh final commands after any review fix:

```powershell
git status --short --branch
git diff --check
git diff --quiet
git diff --cached --quiet
git log --oneline --decorate -24
git diff --name-status bb4ee0e66ac57a2394afd99b1324f96f7523fad4...HEAD
```

Use `superpowers:verification-before-completion`. Use `superpowers:finishing-a-development-branch` only to confirm branch readiness and list integration choices; do not select or execute one. Retain the branch, worktree, `.superpowers` task ledger, clean export, evidence directory, screenshots, and loopback preview for owner feedback.

---

## Remote Gates A–E — Separate and Unauthorized in Local Implementation

1. **Gate A — Candidate branch and PR:** re-read origin/main and branch/head race, auth/scopes, template/conventions, run the complete local suite again, non-force push only after explicit authorization, and create exactly one review/candidate PR.
2. **Gate B — CI candidate/artifact and merge:** require all PR checks successful; download the exact head-SHA artifact outside the repo; prove byte/canonical-text equality and rerun artifact/privacy/origin/desktop/mobile review; merge only with a separately authorized repository-supported method.
3. **Gate C — Pages configuration and dispatch:** after exact merged-main automatic release gates succeed, separately authorize Pages source/environment policy and dispatch the reviewed workflow once on that exact main SHA.
4. **Gate D — Live site review:** verify deployed SHA, HTTP/assets, initial zero-model network, one harbor original→Detect→result flow, cache/toggle/failure/BYOM/accessibility/responsive/privacy/notices, and artifact equality from the live URL.
5. **Gate E — About and Portfolio Control receipt:** only after independent Gate D approval, separately authorize the About homepage change and Portfolio Control completion receipt.

Plan approval or local implementation authorizes none of these gates. It also does not authorize force push, auto-merge, Pages enable/configuration/dispatch/deployment, GitHub About, Hugging Face, release, tag, visibility change, branch/worktree deletion, or modification of another repository.

## Plan Self-review Record

- **Spec coverage:** Task 1 closes source/receipt/tooling/manifest and removes the two files; Task 2 removes selector/state and preserves explicit genuine inference; Task 3 covers deterministic image failure, stale-state clearing, BYOM, accessibility, responsive, and privacy behavior; Task 4 freezes current docs/evidence/notices/artifact gates/screenshot; Task 5 performs full local acceptance, clean export, visual operation, and broad review.
- **Interface consistency:** the only runtime sample interface is `DemoAssets.getDemoSample()` returning immutable `harbor`; `app.js` owns `DEMO_SAMPLE`, `loadDemoImage`, one `state.image`, one cached output, and no selector identity. Browser scenario names are `single-harbor` and `single-harbor-failures` everywhere.
- **TDD honesty:** each batch names the earliest expected mismatch and explicitly forbids claiming assertions blocked by an earlier failure. Successful Detect coverage uses real Chromium, exact harbor/model bytes, and actual `session.run`; failure routes are deterministic but cannot substitute for the success path.
- **Claim/privacy/license boundary:** one public-domain harbor derivative is separate from the privacy-sanitized AGPL model and DOTAv1 provenance. No committed annotated result, source tile, candidate screenshot, private model, local path, raw diagnostic, token, or evaluation claim is admitted.
- **Current versus history:** stale scans cover current product/tool/test/release/docs surfaces. Approved historical `docs/superpowers` records and Git commits are preserved and excluded from active-contract scans.
- **No placeholders:** every task names exact paths, tests, expected RED, minimal GREEN, verification commands, staging, commit, review inputs, and stop conditions.
- **Remote boundary:** Gates A–E are sequenced for later independent authorization and are not executable during plan writing or local implementation.

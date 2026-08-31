# Aerial OBB Real-sample Before/After Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public synthetic-first page with two rights-reviewed real aerial Before/After samples whose boxes are genuine precomputed model output, while preserving BYOM as the only live-inference path.

**Architecture:** Add a local-only, reproducible asset-preparation tool that verifies exact public source bytes, creates metadata-free WebP crops, and captures raw `output0 [1,N,7]` through the pinned browser runtime without publishing the model. The public app loads one strict same-origin JSON manifest, validates and hashes a selected image, then sends the precomputed tensor through the existing decode/filter/corners/render path. Synthetic geometry remains test-only; BYOM keeps its lazy ORT loader and atomic session lifecycle.

**Tech Stack:** Static HTML/CSS/JavaScript, ONNX Runtime Web 1.20.1/WASM, Playwright through Python, Pillow, pytest, standard-library release and Pages verifiers, `uv`, Git.

**Plan Status:** Approved design translated to an implementation plan; pending written plan review.

## Global Constraints

- Implementation starts from `24039db9e07e327b55433086241ac574430c4531` or an exact descendant containing the approved accessibility/runtime follow-up. If that dependency is absent, stop.
- Commit exactly two public sample images. Preferred IDs are `aircraft` and `naval`; `port` may replace the first priority candidate that fails. If fewer than two candidates pass, stop for owner review.
- The published crop is selected before inference. Never move, add, delete, relabel, cherry-pick, or hand-tune a detection; reject the complete candidate instead.
- Each accepted candidate must produce 1–20 finite, visually understandable detections at confidence `0.25` using the exact published crop and the owner-authorized external compatible OBB model.
- The model must remain outside every repository and worktree. Do not copy, rename, hash, inspect metadata from, log, screenshot, or commit its path, filename, bytes, weight digest, graph metadata, or private identifier.
- No model, weight, ONNX binary, DOTA pixel, DOTA derivative, original full-resolution source file, private path, raw exception, stack, tensor dump, or model metadata may enter the repository, Pages tree, screenshots, reports, or commits.
- Built-in samples are precomputed illustrative output. They never create an ORT session, request jsDelivr, display inference latency, or claim accuracy, evaluation, correctness, ground truth, or latency evidence.
- Built-in sample runtime is exactly `N/A · precomputed output`; successful BYOM runtime remains a finite rounded millisecond value; loading/reset/error/no-cache states remain `—`.
- BYOM remains the only live inference route. Merely loading the page, opening BYOM, or selecting a built-in sample must make zero ORT requests. The first BYOM model selection remains the only lazy-load trigger.
- Claim notice text, mode badge, status text, source credit, non-endorsement, safe recovery copy, generation-token checks, candidate-session-then-release, and accessibility requirements are copied exactly from the approved spec.
- The two sample WebPs plus `samples.json` and `SAMPLE_ASSETS.md` must total at most 700 KiB. Each WebP is at most 300 KiB and at most 1600 px on its longest edge, with no EXIF/IPTC/XMP.
- The public initial/sample paths may request only same-origin files. External source-record links are user-activated navigation only. jsDelivr remains permitted only after BYOM model selection.
- Use strict TDD for every behavior change: add the specified test/assertion, run it, record the earliest actually reached RED, make the minimum GREEN change, and rerun the covering tests before refactoring or committing.
- Use `apply_patch` for repository file edits. Stage only the paths named by the current task and make the small commit specified by that task.
- Local implementation, tests, exact preview, screenshots, evidence, and commits are the only authorized actions. Do not push, create or merge a PR, dispatch a workflow, enable or deploy Pages, edit GitHub About, perform Hugging Face operations, create a release/tag, change visibility, or delete any worktree/branch.

---

## File structure and interfaces

| Path | Responsibility |
|---|---|
| `scripts/prepare_pages_samples.py` | Local-only source-byte verification, deterministic crop/WebP encoding, pinned browser inference capture, output validation, manifest emission, and privacy-safe diagnostics. |
| `tests/test_sample_assets.py` | Deterministic source/crop/output/schema/privacy tests using generated test images and synthetic tensors only. |
| `demo/web/samples/aircraft-before.webp` | Preferred reviewed aircraft crop; created only if the aircraft candidate passes. |
| `demo/web/samples/naval-before.webp` | Preferred reviewed naval crop; created only if the naval candidate passes. |
| `demo/web/samples/port-before.webp` | Reserve reviewed port crop; created only if one preferred candidate fails and the reserve passes. |
| `demo/web/samples/samples.json` | Sole public runtime data source: exactly two IDs, same-origin image paths and hashes, public source records, and precomputed `output0`. |
| `demo/web/samples/SAMPLE_ASSETS.md` | Public acquisition, rights, crop, transformation, output-contract, digest, and non-endorsement record with no private model identity. |
| `demo/web/index.html` | Plain-language intro, claim boundary, sample controls, Before/After semantics, secondary BYOM disclosure, result/table/source structure. |
| `demo/web/app.js` | Strict manifest/image validation, atomic sample/BYOM transitions, shared cached-output rendering, lazy ORT lifecycle, fixed safe errors. |
| `demo/web/style.css` | Sample-first desktop/mobile layout, Before/After figures, selected controls, disclosed BYOM panel, focus and reduced-motion behavior. |
| `tests/fixtures/browser-showcase.svg` | Test-only synthetic image used by BYOM smoke; never published by Pages. |
| `tests/fixtures/browser_showcase_fixture.js` | Test-only canonical tensor fixture used by parity tests; never referenced by public HTML. |
| `scripts/browser_smoke.py` | Real Playwright assertions for sample-first, filters, atomic failures, BYOM lazy load/session lifecycle, privacy, responsive and accessibility behavior. |
| `scripts/pages_artifact_check.py` | Exact Pages inventory/digest/media/schema/origin/privacy boundary for two WebPs and their manifests. |
| `tests/test_pages_artifact_check.py` | Mutation tests proving the expanded Pages allowlist fails closed. |
| `tests/test_browser_parity.py` | Synthetic decode/corner parity against test-only fixtures. |
| `scripts/clean_export_check.py`, `tests/test_clean_export.py` | Release archive inventory with public samples and test-only synthetic fixtures in their correct boundaries. |
| `release/evidence.json`, `tests/test_release_check.py` | Honest precomputed-sample evidence and exact source-file inventory. |
| `release/artifact-manifest.json`, `THIRD_PARTY_NOTICES.md` | Byte-level public-domain image/screenshot provenance, conditions, credit, and non-endorsement notice. |
| `README.md`, `README.en.md`, `demo/web/README.md`, `CHANGELOG.md` | User-facing instructions and claim boundary aligned with the sample-first page. |
| `docs/assets/browser-workbench.png` | Screenshot generated from the exact local artifact with the first built-in sample selected and no private model. |

The public manifest has exactly two top-level keys: integer `schemaVersion` equal to `1`, and `samples`, an
array of exactly two objects. Every sample has exactly `id`, `label`, `image`, `source`, and `output`.
`image` has exactly `url`, `sha256`, `width`, and `height`; `source` has exactly `agency`, `recordUrl`, and
`rights`; `output` has exactly `kind`, `contract`, `sha256`, and `results`; `results` has only `output0`,
whose only keys are `dims` and `data`. Hashes are generated from the admitted bytes/canonical output JSON,
not copied from this plan. `samples.json` contains the actual captured tensors after Task 2; the plan contains
no illustrative detection that could be mistaken for evidence.

The preparation tool exposes these exact Python interfaces:

```python
@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int

@dataclass(frozen=True)
class CandidateSpec:
    id: str
    label: str
    agency: str
    record_url: str
    rights_url: str
    acquisition_url: str
    original_sha256: str
    original_size: tuple[int, int]
    crop: Crop

```

The module defines `verify_source(path: Path, spec: CandidateSpec) -> None`,
`encode_published_webp(path: Path, spec: CandidateSpec) -> bytes`,
`capture_output(model_path: Path, webp: bytes) -> dict[str, object]`,
`validate_output(results: dict[str, object]) -> None`, and
`build_candidate(model_path: Path, source_path: Path, spec: CandidateSpec) -> dict[str, object]` with the
responsibilities described in Task 1 Steps 3–4.

The public application keeps one cached-output representation:

```javascript
state.cached = {
  results: validatedSample.output.results,
  geometry: OBB.letterboxGeometry(image.naturalWidth, image.naturalHeight, IMGSZ),
  provenance: validatedSample.source.agency,
  elapsedMs: null,
};
```

It does not introduce a presentation cache. `renderCachedOutput()` remains the sole decode/filter/sort/table/description/canvas path for both precomputed samples and completed BYOM output.

## Task 1: Add deterministic, privacy-safe sample preparation tooling

**Files:**
- Create: `scripts/prepare_pages_samples.py`
- Create: `tests/test_sample_assets.py`

**Interfaces:**
- Consumes: three external public source files and one owner-authorized external compatible ONNX path supplied at execution time.
- Produces: an external candidate review package containing a metadata-free WebP, public-only tensor JSON, and preview PNG; later tasks copy only accepted public outputs into `demo/web/samples/`.

- [ ] **Step 1: Write the preparation-tool batch RED**

Create `tests/test_sample_assets.py` with these exact tests:

- `test_candidate_specs_pin_public_sources_sizes_digests_and_crops`
- `test_verify_source_rejects_digest_dimension_and_crop_drift`
- `test_encode_published_webp_is_deterministic_small_and_metadata_free`
- `test_validate_output_rejects_wrong_name_dims_length_nonfinite_and_bad_class`
- `test_build_candidate_emits_only_public_schema_and_never_model_identity`
- `test_cli_rejects_a_model_inside_the_repository_without_echoing_its_name`

Each mutation test must assert the exact fixed failure code returned for its changed field, not merely that
an exception occurred. The public-schema test asserts the full emitted key set and confirms the supplied
model path, basename, bytes and sentinel metadata string are absent from serialized output and captured
stdout/stderr.

Use generated RGB images and synthetic `[1,N,7]` values in tests. Do not load a real model or make a network request. Assert exact candidate constants:

```python
EXPECTED = {
    "aircraft": ((2821, 1885), "de7588b09b184b36ba136eb836cf8585c9242df7d96c2f55ec235fcf0422fe61", (1050, 320, 800, 600)),
    "naval": ((1280, 1224), "15406f875ab3cf74059fd9a554428448e438a7a6001ca0aab4edf258adc1b40a", (560, 250, 650, 600)),
    "port": ((1000, 667), "3a0db266e598cc6e6cea097958277d50dc1ad0e7436c03f79023165f883467fa", (0, 0, 1000, 667)),
}
```

- [ ] **Step 2: Run the focused test and observe the actual RED**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_assets.py::test_candidate_specs_pin_public_sources_sizes_digests_and_crops -q
```

Expected RED: collection fails because `scripts.prepare_pages_samples` does not exist. Record only this actually reached failure; the later batch checkpoints are not independently observed until collection succeeds.

- [ ] **Step 3: Implement the minimum source and WebP boundary**

Implement the three immutable candidate specs with these exact public records:

```python
CANDIDATES = {
    "aircraft": CandidateSpec(
        id="aircraft", label="機場與飛機", agency="U.S. Air Force / National Archives",
        record_url="https://catalog.archives.gov/id/6438938",
        rights_url="https://commons.wikimedia.org/wiki/File:Aircraft_stored_at_the_Aerospace_Maintenance_and_Regeneration_Center,_Davis-Monthan_Air_Force_Base,_Arizona_(USA),_on_1_October_1988_(6438938).jpeg",
        acquisition_url="https://upload.wikimedia.org/wikipedia/commons/e/e1/Aircraft_stored_at_the_Aerospace_Maintenance_and_Regeneration_Center%2C_Davis-Monthan_Air_Force_Base%2C_Arizona_%28USA%29%2C_on_1_October_1988_%286438938%29.jpeg",
        original_sha256="de7588b09b184b36ba136eb836cf8585c9242df7d96c2f55ec235fcf0422fe61",
        original_size=(2821, 1885), crop=Crop(1050, 320, 800, 600),
    ),
    "naval": CandidateSpec(
        id="naval", label="港灣與船艦", agency="U.S. Navy / National Archives",
        record_url="https://www.history.navy.mil/our-collections/photography/numerical-list-of-images/nhhc-series/nh-series/80-G-361000/80-G-361740.html",
        rights_url="https://www.history.navy.mil/our-collections/photography.html",
        acquisition_url="https://www.history.navy.mil/bin/imageDownload?image=/content/dam/nhhc/our-collections/photography/images/80-G-361000/80-G-361740&rendition=cq5dam.web.1280.1280.jpeg",
        original_sha256="15406f875ab3cf74059fd9a554428448e438a7a6001ca0aab4edf258adc1b40a",
        original_size=(1280, 1224), crop=Crop(560, 250, 650, 600),
    ),
    "port": CandidateSpec(
        id="port", label="港區與運輸設施", agency="U.S. Army / DVIDS",
        record_url="https://www.dvidshub.net/image/3156545/16th-cab-black-hawks-soar-port-tacoma",
        rights_url="https://www.dvidshub.net/about/copyright",
        acquisition_url="https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/1702/3156545/1000w_q95.jpg",
        original_sha256="3a0db266e598cc6e6cea097958277d50dc1ad0e7436c03f79023165f883467fa",
        original_size=(1000, 667), crop=Crop(0, 0, 1000, 667),
    ),
}
```

`verify_source()` must compare SHA-256, decoded dimensions, crop bounds, and ordinary-file status before decoding the crop. `encode_published_webp()` must convert to RGB, crop without warp, downscale only if the longest edge exceeds 1600, and save with the exact Pillow arguments `format="WEBP", quality=82, method=6, exact=True` while passing no metadata. It then rejects output above 300 KiB or any Pillow-reported EXIF/ICC/XMP payload. The deterministic test encodes the same input twice and requires byte equality.

- [ ] **Step 4: Implement pinned browser capture and public-only output validation**

Use Playwright to create a temporary `about:blank` harness, inject `demo/web/obb.js`, dynamically append only the exact pinned ORT script with its existing SRI and anonymous CORS, load the WebP and model through in-memory/file-input browser APIs, preprocess at 1024 exactly as `app.js`, and return only serialized `output0.dims` and finite numeric `output0.data`.

The implementation must validate:

```python
assert set(results) == {"output0"}
assert dims[0] == 1 and dims[2] == 7 and dims[1] >= 1
assert len(data) == dims[0] * dims[1] * dims[2]
assert all(math.isfinite(value) for value in data)
assert all(0 <= int(data[i + 5]) < 15 and data[i + 5] == int(data[i + 5]) for i in range(0, len(data), 7))
assert all(data[i + 2] > 0 and data[i + 3] > 0 for i in range(0, len(data), 7))
```

Filter nothing during capture. Acceptance count is evaluated later through the production decoder at confidence `0.25`. All diagnostics are fixed codes such as `[SAMPLE_PREP:SOURCE_DIGEST]`; do not interpolate paths, exceptions, tensor data, or model properties.

- [ ] **Step 5: Run the complete tool tests GREEN**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_assets.py -q
```

Expected GREEN: all six tests pass with no network request and no model inference.

- [ ] **Step 6: Commit the tooling task**

Run:

```powershell
git add scripts/prepare_pages_samples.py tests/test_sample_assets.py
git diff --cached --check
git commit -m "test: add deterministic real-sample preparation"
```

## Task 2: Generate, review, and admit exactly two real precomputed samples

**Files:**
- Create: `demo/web/samples/samples.json`
- Create: `demo/web/samples/SAMPLE_ASSETS.md`
- Create one preferred/approved pair from: `demo/web/samples/aircraft-before.webp`, `demo/web/samples/naval-before.webp`, `demo/web/samples/port-before.webp`
- Modify: `scripts/pages_artifact_check.py`
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `tests/test_sample_assets.py`

**Interfaces:**
- Consumes: Task 1 tool, exact public sources, approved external browser-test model, existing `OBB.selectEndToEndOutput()` and `OBB.decodeDetections()` contract.
- Produces: two reviewed same-origin WebPs and one strict manifest accepted by the Pages verifier; no original or model artifact.

- [ ] **Step 1: Write the Pages/sample-boundary batch RED**

Add these exact tests to `tests/test_sample_assets.py`:

- `test_committed_samples_are_exactly_two_approved_candidates_with_1_to_20_default_detections`
- `test_committed_sample_images_match_manifest_and_have_no_metadata`
- `test_public_sample_text_contains_rights_crop_transformation_and_no_private_model_identity`

Add these exact tests to `tests/test_pages_artifact_check.py`:

- `test_pages_tree_requires_two_reviewed_webp_samples_and_public_manifest`
- `test_pages_tree_rejects_sample_digest_dimension_metadata_and_size_drift`
- `test_pages_tree_rejects_unknown_sample_fields_external_asset_urls_and_bad_output`
- `test_sample_source_urls_are_exact_navigation_records_not_runtime_fetches`

Accept only these ID sets: `{"aircraft", "naval"}`, `{"aircraft", "port"}`, or `{"naval", "port"}`. Require the port only when one preferred candidate is absent. Decode at confidence `0.25` with no class selection and assert 1–20 results per accepted candidate.
The same test must assert each WebP is at most 300 KiB, each longest edge is at most 1600 px, and the two
WebPs plus `samples.json` and `SAMPLE_ASSETS.md` total at most 700 KiB.

- [ ] **Step 2: Run the first focused RED**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_assets.py::test_committed_samples_are_exactly_two_approved_candidates_with_1_to_20_default_detections -q
```

Expected RED: `demo/web/samples/samples.json` is missing. Record this earliest failure only.

- [ ] **Step 3: Acquire the exact public source bytes outside every repository**

Run in PowerShell; do not echo the model variable:

```powershell
$sampleSourceRoot = Join-Path ([IO.Path]::GetTempPath()) "aerial-obb-reviewed-public-sources"
$sampleReviewRoot = Join-Path ([IO.Path]::GetTempPath()) "aerial-obb-reviewed-candidate-output"
New-Item -ItemType Directory -Force -Path $sampleSourceRoot, $sampleReviewRoot | Out-Null
Invoke-WebRequest -Uri "https://upload.wikimedia.org/wikipedia/commons/e/e1/Aircraft_stored_at_the_Aerospace_Maintenance_and_Regeneration_Center%2C_Davis-Monthan_Air_Force_Base%2C_Arizona_%28USA%29%2C_on_1_October_1988_%286438938%29.jpeg" -OutFile (Join-Path $sampleSourceRoot "aircraft.jpeg")
Invoke-WebRequest -Uri "https://www.history.navy.mil/bin/imageDownload?image=/content/dam/nhhc/our-collections/photography/images/80-G-361000/80-G-361740&rendition=cq5dam.web.1280.1280.jpeg" -OutFile (Join-Path $sampleSourceRoot "naval.jpeg")
Invoke-WebRequest -Uri "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/1702/3156545/1000w_q95.jpg" -OutFile (Join-Path $sampleSourceRoot "port.jpg")
```

Verify the individual record and rights pages anonymously before continuing. If any source digest/dimension differs from Task 1, a record no longer supports the public-domain basis, or a page introduces third-party rights, stop; do not update the pinned values casually.

- [ ] **Step 4: Capture priority candidates with the owner-controlled model**

Precondition: the controller sets `$env:AERIAL_OBB_APPROVED_MODEL` to the already-authorized external compatible model. The script must validate existence and that the resolved path is outside the repository/worktree, but never print it.

Run:

```powershell
uv run --no-sync python scripts/prepare_pages_samples.py --candidate aircraft --source (Join-Path $sampleSourceRoot "aircraft.jpeg") --model $env:AERIAL_OBB_APPROVED_MODEL --review-root $sampleReviewRoot
uv run --no-sync python scripts/prepare_pages_samples.py --candidate naval --source (Join-Path $sampleSourceRoot "naval.jpeg") --model $env:AERIAL_OBB_APPROVED_MODEL --review-root $sampleReviewRoot
```

Expected: each run emits only fixed candidate ID/verdict codes and creates an external crop, output JSON, and overlay preview. It never copies or renames the model and never records inference runtime.

- [ ] **Step 5: Apply the fixed acceptance and reserve rule**

For each priority candidate, use the production decoder at confidence `0.25`; inspect the external preview against the exact crop. Pass only when 1–20 finite detections are visually understandable and no private person, sensitive information, misleading branding, or obvious non-scene artifact makes the sample unsuitable.

If one priority candidate fails, reject it as a whole and run exactly once:

```powershell
uv run --no-sync python scripts/prepare_pages_samples.py --candidate port --source (Join-Path $sampleSourceRoot "port.jpg") --model $env:AERIAL_OBB_APPROVED_MODEL --review-root $sampleReviewRoot
```

Do not alter a crop or tensor after seeing model output. If the reserve also fails or both priority candidates fail, stop. If two pass, publish only those two via the tool's `--publish-root demo/web/samples` mode. The committed manifest/asset document records public provenance and digests but calls the generator only `owner-authorized external compatible OBB model`.

- [ ] **Step 6: Extend the Pages verifier minimally**

Update `REQUIRED_FILES`, `ALLOWED_FILES`, `ALLOWED_DIRECTORIES`, and reviewed digests for exactly the accepted pair, `samples/samples.json`, and `samples/SAMPLE_ASSETS.md`. Add standard-library WebP RIFF dimension/metadata checks and strict JSON validation. Permit only:

```javascript
fetch(SAMPLES_MANIFEST_URL, { cache: "no-store", credentials: "omit" })
fetch(validated.image.url, { cache: "no-store", credentials: "same-origin" })
```

Both URLs must be same-origin relative paths. Continue rejecting every external fetch, storage/telemetry API, protocol-relative URL, unknown file, symlink, hard link, model/archive suffix, DOTA path, absolute user path, token shape, and unreviewed binary. Allow each exact source record URL only as inert JSON navigation data and prove app code never fetches it.

- [ ] **Step 7: Run focused GREEN and artifact boundary**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_assets.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: every Task 2 test passes; selected assets satisfy source, metadata, output, size, schema and rights records, and the exact expanded Pages tree passes. Public synthetic removal is not asserted until Task 4, so this task does not commit a known failing test.

- [ ] **Step 8: Commit the asset-boundary task**

Stage only the accepted pair and named manifests/verifier/tests:

```powershell
git add demo/web/samples scripts/pages_artifact_check.py tests/test_pages_artifact_check.py tests/test_sample_assets.py
git diff --cached --check
git commit -m "feat: add reviewed real aerial sample assets"
```

## Task 3: Build the sample-first Before/After experience

**Files:**
- Modify: `demo/web/index.html`
- Modify: `demo/web/app.js`
- Modify: `demo/web/style.css`
- Modify: `scripts/browser_smoke.py`
- Modify: `scripts/pages_artifact_check.py`

**Interfaces:**
- Consumes: strict Task 2 `samples.json`, same-origin WebPs, existing OBB decode/corner helpers and `renderCachedOutput()`.
- Produces: initial sample selector, verified Before/After result, shared filter path, source attribution, and secondary BYOM disclosure.

- [ ] **Step 1: Add the real-browser sample-first batch RED**

In `scripts/browser_smoke.py`, replace the initial synthetic-first expectations with named assertions for:

```text
plain-language-intro
claim-notice-before-first-sample-control
exactly-two-sample-buttons
initial-same-origin-only-zero-ORT
first-sample-before-after-and-precomputed-copy
second-sample-switch-and-zero-ORT
sample-confidence-and-class-filter-shared-render
sample-filtered-empty-shared-render
source-link-user-navigation-only
byom-secondary-disclosure
```

The first sample success must assert exact values:

```python
assert page.locator("#modeBadge").inner_text() == "PRECOMPUTED SAMPLE · NO LIVE INFERENCE"
assert page.locator("#runtimeValue").inner_text() == "N/A · precomputed output"
assert page.locator("#status").inner_text() == "範例已載入 · 本頁沒有執行模型推論。"
```

Prove Before and After use the same base image by selecting a class absent from that sample, comparing the entire After canvas RGBA buffer with an offscreen draw of the Before image, then restoring filters and asserting oriented polygon strokes return. Do not settle for checking `src` strings or source text.

- [ ] **Step 2: Run browser smoke and observe the first reachable RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected RED: the plain-language intro or first sample control is missing. Record only the first reached assertion in this batch.

- [ ] **Step 3: Implement the minimum semantic HTML hierarchy**

Make the first public content sequence:

```html
<p class="plain-intro">看模型如何在航拍影像中辨識有方向的目標。</p>
<section id="claimBoundary" class="claim-boundary" aria-labelledby="claimTitle">
  <h2 id="claimTitle">真實航拍範例 — 預先產生結果</h2>
  <p>以下範例使用經授權審查的公共領域航拍影像與預先產生的 OBB output。本頁沒有執行模型推論；這些結果不是 accuracy、evaluation 或 latency evidence。若要實際推論，請開啟 BYOM 並自行提供相容 ONNX model 與影像。</p>
</section>
<section id="samplePicker" aria-labelledby="samplePickerTitle">
  <h2 id="samplePickerTitle">查看 Detect 前／後</h2>
  <div id="sampleList"></div>
</section>
<button id="byomToggle" type="button" aria-expanded="false" aria-controls="byomWorkspace">開啟 BYOM 工作區</button>
```

Use the exact approved claim notice. Keep the skip link first and `main#mainContent`. Build sample buttons as real buttons with `name="sample"`, `aria-pressed`, scene label, agency and unannotated thumbnail. Put the BYOM model/image/Detect controls in `#byomWorkspace`, initially hidden. Keep confidence/class filters in a shared result-controls section so they apply to samples and BYOM but do not compete with sample selection in the initial viewport.

Before/After markup must be semantic:

```html
<div id="comparison" class="comparison" hidden>
  <figure id="beforeFigure"><img id="beforeImage" alt="" /><figcaption>Detect 前</figcaption></figure>
  <figure id="afterFigure"><div id="canvasFrame" class="canvas-frame"><canvas id="canvas" aria-describedby="canvasDescription"></canvas></div><figcaption>Detect 後</figcaption></figure>
</div>
```

The existing filtered textual description remains non-live and the result table remains the detailed equivalent alternative.

- [ ] **Step 4: Implement strict manifest/image validation and atomic sample activation**

Add exact constants and one memoized manifest promise:

```javascript
const SAMPLES_MANIFEST_URL = "samples/samples.json";
let sampleManifestPromise = null;
```

Validate exact object keys, schema version, two allowed unique IDs, relative asset paths, lowercase SHA-256, positive dimensions, HTTPS allowlisted record URLs, source text, output kind/contract, output dims/data length, finite values, class IDs, positive sizes and no unknown fields. Recompute `output.sha256` from canonical UTF-8 JSON containing only `output.results` before accepting it. Load the selected image as bytes with the approved same-origin fetch, verify SHA-256 using `crypto.subtle.digest`, decode through a blob URL, and revoke it after both the `<img>` and canvas hold decoded pixels.

`activateSample(id)` must perform this sequence:

```javascript
const generation = nextGeneration();
state.mode = "sample";
state.phase = "loading";
await releaseSession();
resetByomReadiness();
clearResultPresentation();
clearComparisonPresentation();
// validate manifest, verify image bytes, decode, check token
state.image = image;
state.cached = { results, geometry, provenance: source.agency, elapsedMs: null };
state.phase = "result";
// render Before, then use renderCachedOutput() for After/table/description/summary
```

In `renderSummary()`, sample runtime is derived only when `mode === "sample"`, `phase === "result"`, and cache is non-null. BYOM numeric behavior remains unchanged. Set exact badge/status/provenance and a user-activated source anchor. Focus `#resultTitle` only for explicit sample selection, not filter changes.

- [ ] **Step 5: Implement the minimum responsive presentation**

Desktop uses two equal figures; at `max-width: 900px`, stack Before then After. Preserve square corners, minimum 44 px controls, 19 px body text, 15 px secondary text, focus-visible outlines, 200% zoom, reduced motion, and no horizontal overflow at 1280×720 or 390×844. Do not add a slider, autoplay, map, animation tour, dense canvas labels, or a gallery beyond two buttons.

- [ ] **Step 6: Refresh exact changed Pages text digests and run GREEN**

First run the current-tree verifier and observe digest RED:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_current_pages_tree_passes -q
```

Update only canonical-LF reviewed digests for `index.html`, `app.js`, and `style.css`, then run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest tests/test_sample_assets.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: both real samples, shared filters, exact claims, same-origin/zero-ORT sample path,
responsive checks, and every Task 3 artifact-boundary assertion pass. Task 4 owns the separately introduced
public-synthetic-removal RED, so Task 3 ends without a known failing test.

- [ ] **Step 7: Commit the sample-first UI task**

Run:

```powershell
git add demo/web/index.html demo/web/app.js demo/web/style.css scripts/browser_smoke.py scripts/pages_artifact_check.py
git diff --cached --check
git commit -m "feat: add real-image Before/After showcase"
```

## Task 4: Preserve atomic BYOM behavior and move synthetic evidence out of Pages

**Files:**
- Create: `tests/fixtures/browser-showcase.svg`
- Create: `tests/fixtures/browser_showcase_fixture.js`
- Delete: `demo/web/fixtures/showcase.svg`
- Delete: `demo/web/showcase-fixture.js`
- Modify: `demo/web/index.html`
- Modify: `demo/web/app.js`
- Modify: `scripts/browser_smoke.py`
- Modify: `scripts/pages_artifact_check.py`
- Modify: `tests/test_browser_parity.py`
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `tests/test_package_release.py`

**Interfaces:**
- Consumes: Task 3 sample state and existing BYOM generation-token/session APIs.
- Produces: deterministic sample/BYOM failure and transition guarantees; synthetic geometry remains tests-only and is absent from Pages.

- [ ] **Step 1: Add browser REDs for atomic transitions and recoveries**

Extend `scripts/browser_smoke.py` with real behavior assertions for:

```text
sample-image-failure-clears-before-after-table-summary-description-source-runtime
sample-manifest-schema-failure-clears-every-result-surface
sample-retry-restores-precomputed-result-with-zero-ORT
selecting-either-BYOM-file-clears-sample-and-comparison
sample-selection-releases-BYOM-session-and-clears-both-file-inputs
late-sample-image-and-late-BYOM-session-cannot-republish-stale-state
opening-BYOM-does-not-load-ORT
first-model-selection-loads-one-pinned-SRI-anonymous-ORT
completed-BYOM-runtime-remains-numeric
BYOM-fixed-errors-remain-private-and-recoverable
```

Use route-controlled failures and explicit generation resolvers, never sleeps or timing races. After every failure assert `runtimeValue === "—"`, `modeBadge === "NO RESULT"`, no rows/polygons/Before image/source, reset description, fixed safe status and fixed console code only.

Also add `test_pages_tree_contains_no_public_synthetic_fixture` to
`tests/test_pages_artifact_check.py`. It copies the current Pages tree, asserts the retired public paths are
absent, then injects each retired path separately and requires an `unexpected Pages file` failure.

- [ ] **Step 2: Run the browser smoke and record the earliest transition RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected RED: the first newly asserted sample failure or transition leaves a stale surface or lacks fixed recovery copy. Record the actual first failure only.

- [ ] **Step 3: Implement the minimum shared clearing and safe error map**

Replace synthetic error wording with exact sample recovery copy:

```javascript
SAMPLE_ASSET: "範例無法載入。請選擇另一張範例，或開啟 BYOM 使用自己的模型與影像。"
```

Add one `clearSamplePresentation()` that clears Before image/blob state, comparison visibility, selected button state and source anchor; call it from the existing result reset rather than duplicating caches. Sample manifest/image/digest/schema/decode/render failures set phase `error`, clear every result surface, and emit only `[AERIAL_OBB:SAMPLE_ASSET]` or the existing fixed output/render code. BYOM selection clears sample state before ORT/image work. Sample selection releases the active BYOM session only after incrementing the generation token and clears file input values/neutral labels.

- [ ] **Step 4: Relocate synthetic fixtures and update parity without weakening it**

Using `apply_patch`, create test-only copies with the same canonical 400×200 SVG and tensor values, update
`SHOWCASE_MODULE` to `tests/fixtures/browser_showcase_fixture.js`, and rename the parity test to
`test_test_only_showcase_fixture_is_canonical`. It must retain the existing exact schema version,
provenance, image size, target size, dims, tensor values and float tolerance assertions.

Update browser smoke's BYOM test image to `tests/fixtures/browser-showcase.svg`. Remove both public synthetic files and every public HTML/app reference. Change `test_ci_runs_a_headless_synthetic_browser_smoke` to `test_ci_runs_a_headless_browser_smoke`; the CI command remains `python scripts/browser_smoke.py` and still uses synthetic stubs internally.

- [ ] **Step 5: Update Pages inventory and close the cross-task blocker**

Remove `showcase-fixture.js` and `fixtures/showcase.svg` from required/reviewed Pages sets and exact-reference checks. Assert `fixtures` is no longer an allowed Pages directory unless another required file uses it. Keep test fixtures outside the Pages root and ensure the verifier rejects attempts to re-add either public path.

- [ ] **Step 6: Run focused and browser GREEN**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_package_release.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: synthetic parity remains, no synthetic asset is publishable, sample failures/transitions are atomic, and BYOM lazy load/session/privacy/numeric-runtime behavior remains intact.

- [ ] **Step 7: Commit the state-boundary task**

Run:

```powershell
git add demo/web/index.html demo/web/app.js scripts/browser_smoke.py scripts/pages_artifact_check.py tests/fixtures tests/test_browser_parity.py tests/test_pages_artifact_check.py tests/test_package_release.py
git add -u demo/web/fixtures/showcase.svg demo/web/showcase-fixture.js
git diff --cached --check
git commit -m "test: move synthetic showcase evidence out of Pages"
```

## Task 5: Align public documentation, release evidence, inventory, and screenshot

**Files:**
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `demo/web/README.md`
- Modify: `CHANGELOG.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `release/evidence.json`
- Modify: `release/artifact-manifest.json`
- Modify: `docs/assets/browser-workbench.png`
- Modify: `scripts/release_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `scripts/clean_export_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `scripts/pages_artifact_check.py`

**Interfaces:**
- Consumes: exact committed Pages tree and accepted sample/source manifests.
- Produces: public explanation and byte-level release evidence consistent with the UI; clean archive contains public samples and test-only synthetic evidence in distinct locations.

- [ ] **Step 1: Write the release/evidence batch RED**

Replace synthetic-public expectations with these exact tests:

- `test_browser_precomputed_samples_are_explicit_model_private_and_not_evaluation_evidence`
- `test_browser_demo_source_inventory_contains_two_samples_and_no_public_synthetic_fixture`
- `test_public_domain_sample_and_screenshot_artifacts_have_exact_provenance_and_digests`
- `test_clean_export_requires_public_samples_and_test_only_synthetic_fixtures`
- `test_release_archive_excludes_public_synthetic_showcase_paths`

Require evidence keys for two sample IDs, `precomputed-illustrative-output`, no page inference, `N/A · precomputed output`, zero external runtime requests, BYOM lazy load, no model identity/digest/bytes, and no accuracy/T4 claim.
Use these exact browser evidence fields: `showcase_enabled: true`,
`showcase_kind: "real-public-domain-precomputed"`, `showcase_sample_ids` equal to the two manifest IDs in
manifest order, `showcase_output_kind: "precomputed-illustrative-output"`,
`showcase_inference_performed: false`, `showcase_runtime_label: "N/A · precomputed output"`, and
`showcase_external_runtime_requests: false`. Retain `runtime_load: "lazy-on-byom-selection"`,
`model_bundled: false`, `represents_fine_tuned_medium_accuracy: false`, and
`represents_t4_latency: false`. Remove `showcase_fixture` and `showcase_image` rather than retaining stale
synthetic semantics.

- [ ] **Step 2: Run the focused RED and record the first actual mismatch**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_release_check.py::test_browser_precomputed_samples_are_explicit_model_private_and_not_evaluation_evidence -q
```

Expected RED: current evidence still describes the authored synthetic SVG. Record that mismatch only.

- [ ] **Step 3: Update public copy without changing historical benchmark claims**

In both READMEs and `demo/web/README.md`, replace the public Synthetic Showcase instructions with:

- two real public-domain Before/After examples with precomputed output;
- no model inference/runtime/accuracy/evaluation/latency claim for samples;
- `開啟 BYOM 工作區` as the real inference route;
- pinned jsDelivr/SRI/WASM non-zero-network disclosure only for BYOM;
- source/rights/non-endorsement and safe-error explanations.

Add an `Unreleased` changelog entry scoped to the sample-first UI and test-fixture relocation. Do not rewrite historical rc.1/rc.2 facts or benchmark numbers.

- [ ] **Step 4: Update evidence and third-party inventory from actual bytes**

Set `release/evidence.json` source files to the exact public Pages files, two accepted WebPs, `samples.json`, `SAMPLE_ASSETS.md`, and screenshot; remove both public synthetic paths. Keep model fields generic and model-free.

Change `release/artifact-manifest.json` distribution mode from `code-only-byom` to the exact
`code-plus-public-domain-samples-byom`; update `scripts/release_check.py` and its tests to require that value.
The browser inference distribution remains BYOM and no model is bundled. Add each accepted WebP and
`docs/assets/browser-workbench.png` to `bundled_third_party_artifacts` with actual byte size/SHA-256, source
record, creator/agency, `Public domain — U.S. federal government work`, crop/derivative provenance, credit
request, and non-endorsement restriction. Keep the IBM font entry and six excluded historical artifacts.
Update `THIRD_PARTY_NOTICES.md` consistently; do not imply that public domain waives trademarks,
privacy/publicity, or endorsement rules.

- [ ] **Step 5: Generate the canonical screenshot from the exact artifact**

Run browser smoke so its final deterministic state is the first built-in sample selected, BYOM closed, exact claim notice visible, exact sample badge/provenance/runtime/status visible, and no stale BYOM-ready labels:

```powershell
uv run --no-sync python scripts/browser_smoke.py --screenshot docs/assets/browser-workbench.png
```

The screenshot must not involve the private model. Inspect the PNG metadata and visible text for paths, filenames, model identifiers, raw errors or stacks. Then refresh its actual manifest size/digest and update only the changed `demo/web/README.md` Pages text digest.

- [ ] **Step 6: Update clean-export membership exactly**

Require both accepted WebPs, `samples.json`, `SAMPLE_ASSETS.md`, `tests/fixtures/browser-showcase.svg`, and `tests/fixtures/browser_showcase_fixture.js`. Remove `demo/web/fixtures/showcase.svg`. Preserve `docs/superpowers/** export-ignore`. Update archive tests to prove public samples are present, design/plan docs remain absent, and test-only fixtures are not in the Pages tree.

- [ ] **Step 7: Run documentation/release GREEN**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_release_check.py tests/test_clean_export.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: claims, bytes, public-domain inventory, screenshot, clean-export membership, privacy and Pages digests all agree.

- [ ] **Step 8: Commit the public-evidence task**

Run:

```powershell
git add README.md README.en.md demo/web/README.md CHANGELOG.md THIRD_PARTY_NOTICES.md release/evidence.json release/artifact-manifest.json docs/assets/browser-workbench.png scripts/release_check.py tests/test_release_check.py scripts/clean_export_check.py tests/test_clean_export.py scripts/pages_artifact_check.py
git diff --cached --check
git commit -m "docs: align release evidence with real samples"
```

## Task 6: Run complete local acceptance and branch-readiness review

**Files:**
- Modify: none unless an earlier task has an approved uncommitted correction.
- Test: all unit/static tests, browser smoke, Pages verifier, repository/release/privacy gates, strict clean export, exact local desktop/mobile review.

**Interfaces:**
- Consumes: Tasks 1–5 committed implementation and exact `demo/web` artifact.
- Produces: local verification package and review verdict only; no remote action.

- [ ] **Step 1: Verify unit/static contracts separately**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_sample_assets.py tests/test_browser_parity.py tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py -q
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: source/crop/output, synthetic parity, exact public artifact, release evidence and archive membership pass independently.

- [ ] **Step 2: Verify the real browser smoke separately**

Save acceptance screenshots outside every repository:

```powershell
$acceptanceRoot = Join-Path ([IO.Path]::GetTempPath()) "aerial-obb-real-sample-acceptance"
New-Item -ItemType Directory -Force -Path $acceptanceRoot | Out-Null
uv run --no-sync python scripts/browser_smoke.py --screenshot (Join-Path $acceptanceRoot "desktop.png") --mobile-screenshot (Join-Path $acceptanceRoot "mobile.png")
```

Expected GREEN: both samples, notice order, Before/After pixels/polygons, filters, source navigation, zero-ORT sample flow, atomic failures/retries, BYOM lazy load/numeric runtime/session lifecycle/privacy, keyboard/focus/labels/headings/description, desktop/mobile layout, and console/origin checks pass.

- [ ] **Step 3: Run complete regression and release/privacy gates**

Run:

```powershell
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: full pytest has zero failures and all repository/release/privacy/Pages checks print their `[OK]` verdicts.

- [ ] **Step 4: Commit any final approved local correction before strict clean export**

`scripts/clean_export_check.py` requires a clean committed HEAD. If Task 6 review found no defect, make no commit. If a permitted local fix was required, complete its own RED/GREEN/re-review loop, stage only its scoped paths, commit it, and rerun Steps 1–3 before continuing.

- [ ] **Step 5: Run the strict clean export with browser verification**

Choose a repo-external output path so no generated archive dirties the worktree:

```powershell
$cleanExport = Join-Path ([IO.Path]::GetTempPath()) "aerial-obb-lab-real-sample-clean-export.zip"
if (Test-Path -LiteralPath $cleanExport) { Remove-Item -LiteralPath $cleanExport }
uv run --no-sync python scripts/clean_export_check.py --output $cleanExport
```

Expected GREEN: committed archive inspection, fresh `uv sync`, full pytest, repository check, release check, browser smoke, no Torch/Ultralytics/HF runtime, wheel/sdist build, clean package install, and version import all pass. Do not use `--skip-browser` for final evidence.

- [ ] **Step 6: Serve and inspect the exact local Pages tree**

Run in a dedicated local terminal:

```powershell
uv run --no-sync python -m http.server 8000 --directory demo/web
```

At `http://127.0.0.1:8000/`, inspect 1280×720 and 390×844: claim notice before controls, two understandable sample cards, Before/After order, readable boxes/table, exact precomputed badge/provenance/runtime/status, filtered empty state, BYOM disclosure, Source/AGPL/agency links, keyboard focus, 200% zoom, reduced motion, and no horizontal overflow. DevTools must show only same-origin initial/sample requests and zero ORT until a model is selected. Do not use the private model during this visual pass.

- [ ] **Step 7: Run artifact/privacy/origin scans against the exact clean-export tree**

Extract the repo-external archive to a repo-external temporary directory and run:

```powershell
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/release_check.py
git grep -n -I -E "([A-Za-z]:\\\\Users\\\\|/Users/|/home/|\.onnx|\.pt|DOTA)" HEAD -- demo/web docs/assets release README.md README.en.md THIRD_PARTY_NOTICES.md
```

Review every grep hit in context: contract copy such as `.onnx` and explicit DOTA exclusions are allowed; any path/model/image artifact or DOTA visual is a blocker. Confirm browser network logs include sample site origin only, with pinned jsDelivr appearing only in the stubbed BYOM selection scenario.

- [ ] **Step 8: Final whole-branch review and hygiene**

Use a fresh reviewer against the approved spec and this plan. Resolve Critical/Important findings through one scoped TDD fix wave; record Minors without silently expanding scope. Then run:

```powershell
git status --short
git diff --check 24039db9e07e327b55433086241ac574430c4531...HEAD
git log --oneline 24039db9e07e327b55433086241ac574430c4531..HEAD
git diff --name-status 24039db9e07e327b55433086241ac574430c4531...HEAD
```

Expected GREEN: worktree clean, no whitespace errors, only approved paths, small task commits, no original/model/private/generated acceptance files, and reviewer verdict has no unresolved Critical/Important finding.

Use `verification-before-completion`, then `finishing-a-development-branch` only to report branch readiness and integration options. Stop before selecting or executing an integration/remote option.

## Remote gates after local review — explicitly not authorized by this plan

1. **Remote Gate A — publication preflight and candidate PR:** first confirm the accessibility dependency is integrated or receive an exact central rebase instruction; then a separately authorized non-force push and PR may be considered. This plan authorizes neither.
2. **Remote Gate B — candidate artifact review and integration:** download CI candidate, prove byte/canonical-text equality with reviewed `demo/web`, repeat privacy/desktop/mobile/sample/BYOM checks, then use only a separately authorized repository-supported merge method.
3. **Remote Gate C — Pages operation:** configure or dispatch Pages only after exact merged-main CI succeeds and only with separate written authority.
4. **Remote Gate D — deployed live review:** verify HTTPS/assets/origins, both samples, BYOM safe failure, privacy, responsive/accessibility, source credits and exact deployed SHA after a separately authorized deployment.
5. **Remote Gate E — About and Portfolio Control receipt:** change GitHub About and record Portfolio Control receipt only after a separately passing live review and separate authorization.

No task in this plan authorizes push, PR creation, merge, workflow dispatch, Pages enable/deploy/disable, About edits, HF operations, release/tag, visibility change, branch deletion, or worktree cleanup.

## Plan self-review

- **Spec coverage:** Tasks 1–2 enforce exact source rights, deterministic pre-inference crops, real model-captured raw output, rejection instead of box editing, size/metadata/privacy gates and exactly two accepted samples. Tasks 3–4 implement the sample-first hierarchy, one shared pipeline, exact claims, atomic transitions, BYOM lifecycle, safe failures, accessibility, responsive behavior and test-only synthetic migration. Task 5 aligns all public claims, screenshot and release inventories. Task 6 separates static, browser, full regression, artifact, privacy/origin, clean export and local responsive acceptance.
- **Placeholder scan:** every step names its concrete path, command, assertion, generated-value source and stop condition; no unresolved marker or illustrative detection remains. Actual image/output hashes and tensors are deterministically generated and admitted by Task 2 tests rather than guessed or copied from planning text.
- **Type/name consistency:** `CandidateSpec`, `Crop`, `state.mode === "sample"`, `SAMPLES_MANIFEST_URL`, manifest `schemaVersion`, `samples`, `output.results.output0`, exact badge/runtime/status and file paths are consistent across tasks.
- **Scope:** no hosted inference, bundled model, DOTA content, third sample, slider, gallery, analytics, storage, framework, remote setting, deployment, About edit or other repository is included.
- **Stop conditions:** dependency drift, source rights/digest drift, private-path exposure, output-contract failure, fewer than two acceptable candidates, size/metadata failure, artifact mismatch, unresolved Critical/Important review finding or any remote race stops execution rather than weakening a gate.

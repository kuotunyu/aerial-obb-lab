# Aerial OBB Curated Real-sample Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense single harbor example with three switchable, public-domain real aerial examples that remain unannotated until the visitor explicitly runs genuine local-browser OBB inference.

**Architecture:** Add a repository-external NAIP candidate-admission pipeline first, then atomically publish one approved image for each fixed category into a closed three-entry manifest. Extend the existing single-result state machine with only `selectedSampleId`; reuse the one verified demo session, clear the active result on selection, and keep all presentation derived from the current image, cached output, and filters. Finish by freezing exact asset/text evidence and exercising all three images in real Chromium before any remote gate.

**Tech Stack:** Static HTML, CSS, and JavaScript; ONNX Runtime Web 1.20.1/WASM; Python 3.11; Pillow; Playwright through Python; pytest; USGS NAIP Plus ArcGIS REST ImageServer; standard-library release/Pages/clean-export gates; `uv`; Git.

**Plan Status:** Approved for local implementation; the source-admission refinement was approved in writing on 2026-09-01. Remote Gates A–E remain unauthorized.

## Global Constraints

- Start from accepted product commit `40a6eb130b6e2cf46b89469750eb10f9133d8a83`; the approved design commit `a7b6fb14fd97c72c92c97709e8f3ba23fde299b2` must remain an ancestor of implementation HEAD.
- Execute in the retained isolated worktree `D:\AI-Portfolio\.worktrees\aerial-obb-live-real-image-demo` on `feat/pages-live-real-image-demo`. Preserve untracked `.superpowers/` and stage only the exact task paths.
- The public catalog has exactly `airfield`, `sports-complex`, and `harbor`, at `samples/airfield.jpg`, `samples/sports-complex.jpg`, and `samples/harbor.jpg`; the initial ID is `airfield`.
- Every admitted image is a `1280×800` metadata-stripped sRGB JPEG encoded at quality `90`, derived from one locked NAIP raster in the contiguous United States and recorded by official source identity, crop/export rectangle, year, acquisition date, agency, bytes, and SHA-256.
- The authoritative imagery service is exactly `https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer`; admission requires one five-point-locked source row whose `agency` identifies USDA/FSA and whose official HTTPS `download_url` path contains a distinct `NAIP` segment. `Name` and `raster_name` must be present and must not identify HRO/commercial imagery, but need not repeat the literal `NAIP` token. Query, fragment, credentials, multiple-source, HRO, commercial, and ambiguous records fail closed.
- Candidate evaluation uses the exact reviewed model `demo/web/models/yolo26n-obb-privacy-sanitized.onnx`, exact browser pipeline, and shared confidence `0.25`. Never tune threshold, class filter, NMS, preprocessing, or model per image.
- The current `boats.jpg` leaves the public catalog and Pages artifact. Do not use DOTA pixels/annotations/renders, commercial basemap screenshots, private imagery, precomputed output, manually edited boxes, or an annotated committed image.
- Initial navigation may request the three same-origin JPEGs, HTML/CSS/JavaScript, and self-hosted font. It must request zero `demo-model.json`, ORT script, WASM, or ONNX resources before explicit Detect.
- `state.cached` remains the only inference-result cache. Do not add per-sample result caches, stored result markup, browser storage, service worker, upload, analytics, telemetry, hosted inference, or another presentation cache.
- Switching samples increments the generation token, clears cached output/runtime/canvas/table/description/toggle/error/completion state, displays the new original, and leaves an active verified demo session reusable. Returning to a sample requires another real `session.run`.
- BYOM preserves the existing lazy runtime, candidate-session-before-release, generation-token, safe-error, filename/path/metadata privacy, and one-active-session contracts.
- The static claim notice remains before the first workbench control and states that examples are curated for clarity but are not accuracy, evaluation, or latency-benchmark evidence.
- UI, console, screenshots, reports, receipts, and commits must omit local paths, user filenames, private model identity/metadata, raw exceptions, stacks, response bodies, tokens, signed queries, rejected candidate identities, and browser profile paths.
- Workflow action references remain exactly `actions/checkout@v7`, `actions/setup-python@v7`, `actions/setup-node@v7`, `actions/upload-artifact@v7`, `actions/configure-pages@v6`, `actions/upload-pages-artifact@v5`, and `actions/deploy-pages@v5` wherever used.
- Apply strict test-driven development to every behavior change: write the named failing test, run it, record only the earliest failure actually reached, then implement the minimum GREEN and refactor under passing coverage.
- Local files, repository-external temporary candidate review, tests, loopback preview, screenshots, review, and commits are authorized only after this plan receives written approval.
- Push, PR, merge, workflow dispatch, Pages configuration/deployment, GitHub About, Hugging Face, release, tag, visibility changes, branch deletion, and worktree cleanup remain separate and unauthorized.

---

## Execution Workspace and SDD Ledger

Before Task 1, read `superpowers:using-git-worktrees`, `superpowers:test-driven-development`, `superpowers:subagent-driven-development`, and `superpowers:frontend-design`. Detect the existing linked worktree; do not create a second one.

```powershell
$worktree = 'D:\AI-Portfolio\.worktrees\aerial-obb-live-real-image-demo'
$branch = (git -C $worktree branch --show-current).Trim()
$spec = (git -C $worktree rev-parse a7b6fb14fd97c72c92c97709e8f3ba23fde299b2).Trim()
$product = (git -C $worktree rev-parse 40a6eb130b6e2cf46b89469750eb10f9133d8a83).Trim()
if ($branch -ne 'feat/pages-live-real-image-demo') { throw 'Unexpected implementation branch' }
if ($spec -ne 'a7b6fb14fd97c72c92c97709e8f3ba23fde299b2') { throw 'Approved design commit is unavailable' }
if ($product -ne '40a6eb130b6e2cf46b89469750eb10f9133d8a83') { throw 'Accepted product commit is unavailable' }
git -C $worktree merge-base --is-ancestor $product HEAD
git -C $worktree merge-base --is-ancestor $spec HEAD
git -C $worktree diff --quiet
if ($LASTEXITCODE -ne 0) { throw 'Tracked worktree is dirty' }
git -C $worktree diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Index is dirty' }
```

If `node` is not already available in the local Codex terminal, prepend the bundled runtime for commands that execute browser-parity tests; never commit this host-specific path:

```powershell
$workspaceNodeBin = 'C:\Users\3Hml\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin'
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { $env:Path = "$workspaceNodeBin;$env:Path" }
node --version
```

Create the ignored ledger at:

```text
.superpowers/sdd/2026-09-01-aerial-obb-curated-real-sample-gallery/progress.md
```

Record starting HEAD, fresh implementer/reviewer identities, exact candidate-review root, source-service metadata digest, each first reached RED, GREEN commands/output, approved public sample IDs only, exact staged paths, commit SHAs, review verdicts/fix rounds, cross-task artifact-freeze blockers, screenshot locations, and final verification. Never stage `.superpowers/`; never write private paths or rejected candidate identities into tracked reports.

## File Structure and Interfaces

| Path | Responsibility |
| --- | --- |
| `scripts/prepare_sample_gallery.py` | Exact nine-recipe NAIP candidate-pool acquisition, per-recipe locked-raster admission, deterministic JPEG derivation, public-safe receipt validation, and atomic publication of three approved files. |
| `scripts/sample_gallery_smoke.py` | Repository-external real-browser inference, public-safe observation report, bounded guardrail calculation, and screenshot capture for candidate admission. |
| `tests/test_sample_gallery.py` | Source allowlist, raster identity, deterministic derivation, metadata/privacy, report schema, three-category approval, containment, and atomic publication tests. |
| `release/sample-gallery-sources.json` | Exact final three-image source/derivation/digest/guardrail receipt; no rejected candidate or local path. |
| `demo/web/samples/{airfield,sports-complex,harbor}.jpg` | The only three public real-aerial inputs. |
| `demo/web/demo-model.json` | Closed schema version 2: three sample records, one default ID, unchanged model/input/output/license/sanitization contracts. |
| `demo/web/demo-assets.js` | Frozen expected catalog and fail-closed manifest/model validation. |
| `scripts/prepare_demo_assets.py` / `tests/test_demo_assets.py` | Reproduce and publish the exact sample gallery plus existing sanitized model/license batch without stale managed leaves. |
| `demo/web/index.html` / `demo/web/style.css` | Notice copy, three compact semantic sample buttons, selected state, retained workbench, responsive and accessible presentation. |
| `demo/web/app.js` | `selectedSampleId`, sample decode/switch/reset, unchanged one-session inference pipeline, safe sample-specific recovery. |
| `scripts/browser_smoke.py` | Real initial/select/inference/cache/race/error/BYOM/accessibility/desktop/mobile/origin/privacy behavior for all three samples. |
| `scripts/pages_artifact_check.py` / `tests/test_pages_artifact_check.py` | Exact three-image inventory, digests, sources, reviewed text, and fail-closed Pages artifact mutation tests. |
| `scripts/release_check.py`, `scripts/clean_export_check.py` and tests | Three-image evidence/source/license/clean-export boundary with the one existing model exception. |
| `release/artifact-manifest.json` / `release/evidence.json` | Final public bytes, sample catalog evidence, limitations, and screenshot identity. |
| `README.md`, `README.en.md`, `demo/web/README.md`, `THIRD_PARTY_NOTICES.md`, `demo/web/THIRD_PARTY_NOTICES.md`, `RELEASE_CHECKLIST.md`, `CHANGELOG.md` | Current three-sample journey, public-domain derivation, model/license/privacy/claim boundaries, and release checklist. |
| `docs/assets/browser-workbench.png` | Canonical default-sample result screenshot from the exact final artifact. |

Python admission interfaces are exact. `CandidateRecipe` is a frozen dataclass with
`candidate_id: str`, `category: Literal["airfield", "sports-complex", "harbor"]`, and
`bbox_wgs84: tuple[float, float, float, float]`. The module exposes
`acquire_candidate(recipe, review_root, transport) -> dict`,
`validate_candidate_record(record, review_root) -> None`,
`validate_approved_gallery(report, review_root) -> tuple[dict, dict, dict]`, and
`publish_approved_gallery(report, review_root, pages_root, receipt_path) -> None` with those exact names.

The immutable candidate recipes are:

```python
CANDIDATE_RECIPES = (
    CandidateRecipe("airfield-watsonville", "airfield", (-121.797682, 36.929906, -121.785682, 36.937406)),
    CandidateRecipe("airfield-reid-hillview", "airfield", (-121.825300, 37.330133, -121.813300, 37.337633)),
    CandidateRecipe("airfield-santa-monica", "airfield", (-118.456705, 34.012771, -118.444705, 34.020271)),
    CandidateRecipe("sports-big-league-manteca", "sports-complex", (-121.265210, 37.784685, -121.253210, 37.792185)),
    CandidateRecipe("sports-twin-creeks", "sports-complex", (-122.006126, 37.411869, -121.994126, 37.419369)),
    CandidateRecipe("sports-ken-mercer", "sports-complex", (-121.897930, 37.677402, -121.885930, 37.684902)),
    CandidateRecipe("harbor-port-hueneme", "harbor", (-119.216719, 34.144170, -119.200719, 34.154170)),
    CandidateRecipe("harbor-redwood-city", "harbor", (-122.216577, 37.508270, -122.200577, 37.518270)),
    CandidateRecipe("harbor-stockton", "harbor", (-121.334614, 37.946035, -121.318614, 37.956035)),
)
```

The browser manifest catalog has this exact structural interface; measured values replace no fake literals because Task 2 writes them from the approved receipt:

```javascript
{
  schemaVersion: 2,
  id: "ultralytics-yolo26n-obb-demo",
  defaultSampleId: "airfield",
  samples: [
    {
      id, title, path, bytes, sha256, mediaType: "image/jpeg", width: 1280, height: 800,
      source: {service, productId, year, acquisitionDate, agency, publicDomainRecord},
      derivation: {bboxWgs84, outputSize: [1280, 800], color: "sRGB", jpegQuality: 90, metadata: "stripped"},
      guardrails: {classIds, countMin, countMax, representative: {classId, cx, cy, w, h, tolerance}},
      alt,
    },
  ],
  model, input, output, provenance, sanitization, license, notice,
}
```

Production sample-state interfaces are exact:

```javascript
getSampleById(sampleId)                 // returns the frozen admitted record or throws DEMO_MANIFEST
selectDemoSample(sampleId)              // async; generation-protected original switch, no inference
loadSelectedDemoImage(sample, token)    // async; returns decoded #demoOriginalImage only for current token
setSampleSelection(sampleId)            // updates the three aria-pressed states and selected title/state copy
runDemo()                               // uses state.image for state.selectedSampleId; never a hard-coded file
```

`state.selectedSampleId` is the only new durable state field. `state.cached`, `state.session`, `state.sessionSource`, `state.image`, `state.imageSource`, `state.view`, and `state.generation` keep their current meanings.

---

### Task 1: Build the Reproducible NAIP Candidate-admission Gate

**Files:**
- Create: `scripts/prepare_sample_gallery.py`
- Create: `scripts/sample_gallery_smoke.py`
- Create: `tests/test_sample_gallery.py`

**Interfaces:**
- Produces the exact Python interfaces and closed nine-recipe pool above, a source-valid subset containing two or three candidates per category, a repository-external `approved-gallery.json`, and repository-external before/result screenshots.
- Does not create or modify any tracked public image, manifest, UI file, release evidence, or current sample.

- [ ] **Step 1: Write the source/derivation/report batch RED**

Create these named tests before either script exists:

```python
def test_candidate_recipes_are_exact_conus_naip_only() -> None:
    assert tuple(recipe.candidate_id for recipe in CANDIDATE_RECIPES) == (
        "airfield-watsonville", "airfield-reid-hillview", "airfield-santa-monica",
        "sports-big-league-manteca", "sports-twin-creeks", "sports-ken-mercer",
        "harbor-port-hueneme", "harbor-redwood-city", "harbor-stockton",
    )
    assert {recipe.category for recipe in CANDIDATE_RECIPES} == {
        "airfield", "sports-complex", "harbor"
    }

def test_derivation_is_deterministic_1280_by_800_srgb_jpeg_without_metadata(
    tmp_path: Path, fake_naip_transport: FakeNaipTransport
) -> None:
    first = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "one", fake_naip_transport)
    second = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "two", fake_naip_transport)
    assert first["image"]["sha256"] == second["image"]["sha256"]
    with Image.open(tmp_path / "one" / first["image"]["reviewName"]) as image:
        assert image.size == (1280, 800)
        assert image.mode == "RGB"
        assert image.getexif() == {}
        assert image.info.get("icc_profile") is None

def test_admission_rejects_non_naip_source_hidden_tuning_and_private_fields(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    for mutation in invalid_source_tuning_and_privacy_mutations(valid_candidate_record):
        with pytest.raises(GalleryError, match="GALLERY_RECORD"):
            validate_candidate_record(mutation, tmp_path)

def test_approved_gallery_requires_one_visually_approved_record_per_fixed_category(
    tmp_path: Path, approved_gallery_report: dict
) -> None:
    records = validate_approved_gallery(approved_gallery_report, tmp_path)
    assert tuple(record["category"] for record in records) == (
        "airfield", "sports-complex", "harbor"
    )
```

The invalid mutation set must cover wrong service host/path, non-USDA agency, HRO/non-NAIP product, two raster IDs for one crop, changed bbox, output size/quality/threshold/class filter, missing source digest, extra field, absolute path, query/header/token, raw error/stack, rejected candidate in approved output, duplicate category, and missing visual approval.

Add `test_admission_accepts_official_usda_fsa_record_with_naip_download_path` as the regression for the
approved source refinement. Its fixture keeps `Name` and `raster_name` nonblank and free of HRO/commercial
markers without a literal `NAIP` token, identifies USDA/FSA in `agency`, and uses an HTTPS source URL with a
distinct `NAIP` path segment. Observe the old name-token rule fail before changing production validation.
Mutation coverage must independently reject a missing/ambiguous NAIP path segment, credentials, query,
fragment, a blank product name, and any HRO/commercial marker.

Add `test_acquisition_pool_keeps_two_or_three_source_valid_candidates_per_category` as the regression for
the approved pool ruling. Attempt all nine immutable recipes, fail closed only the individual recipe whose
source gate raises exact `GalleryError("GALLERY_SOURCE_REJECTED")`, and assert the written batch contains
only source-valid records with category counts in `[2, 3]`. Assert that fewer than two retained records in any
category aborts the whole batch, and that network, parse, derivation, containment, or write failures are fatal
rather than silently treated as candidate rejection. Neither the tracked report nor the committed files may
identify a rejected recipe.

- [ ] **Step 2: Run the batch and record the earliest actual RED**

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py -q
```

Expected earliest RED: import failure because `scripts.prepare_sample_gallery` does not exist. Record only that first reached failure; do not claim the later record/approval assertions were independently observed.

- [ ] **Step 3: Implement the minimal closed acquisition and derivation contract**

Use only Python standard library plus Pillow. The source contract is:

```python
NAIP_SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
NAIP_PUBLIC_DOMAIN_RECORD = "https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39"
OUTPUT_SIZE = (1280, 800)
JPEG_QUALITY = 90
DEFAULT_CONFIDENCE = 0.25
SOURCE_FIELDS = (
    "OBJECTID", "Name", "Year", "raster_name", "download_url", "acquisition_date",
    "agency", "resolution_value", "resolution_units", "band_count", "sensor_type",
)
```

For every recipe:

1. Query the exact bbox against the service and require one raster identity to cover the centre plus four inset corners.
2. Require `agency` to identify USDA/FSA and the parsed official HTTPS `download_url` path to contain a distinct case-insensitive `NAIP` segment. Require nonblank `Name` and `raster_name`; reject HRO/commercial markers in either product field or agency, plus blank, multiple-source, ambiguous, or out-of-CONUS results.
3. Send every centre/inset identify point as JSON geometry with `spatialReference: {"wkid": 4326}`. Lock `exportImage` to that one `OBJECTID` with an ArcGIS `esriMosaicLockRaster` rule; request the recipe bbox in EPSG:4326 and `1280,800` output.
4. Cap service JSON at 256 KiB and raster response at 25 MiB; accept service responses only from HTTPS `imagery.nationalmap.gov`. Preserve the exact official HTTPS `download_url` host and normalized path recorded by the selected source item, rejecting credentials, query, fragment, traversal, or a missing distinct `NAIP` path segment; the source URL is provenance, not a runtime browser request.
5. Decode the response, composite alpha over black only if present, convert to RGB, resize with Pillow LANCZOS when the service response differs, and save deterministic JPEG quality `90`, `subsampling=0`, `optimize=False`, `progressive=False`, with no EXIF/ICC/comment.
6. Write a public-safe record containing source values, service-response SHA-256, bbox, derivation, final bytes/SHA-256, and the opaque review filename produced by `f"{candidate_id}.jpg"`. Never serialize the local review root, response headers/body, exception, or temporary export URL.

Containment checks reject a review root inside any Git worktree, symlink/reparse component, traversal, existing unrelated leaf, and overwrite of a prior review batch. `acquire_all` attempts every immutable recipe, catches only exact `GalleryError("GALLERY_SOURCE_REJECTED")` per recipe, requires two or three retained candidates per category, and writes the complete batch only after the pool passes. Every other failure remains fatal. Publishing is not implemented in this task.

- [ ] **Step 4: Implement the real-browser candidate smoke and report schema**

`scripts/sample_gallery_smoke.py` serves the current tracked `demo/web`, selects the existing public sanitized model through BYOM, selects each source-valid repository-external candidate image through the real file input, clicks BYOM Detect, and reads the actual table/canvas state. Its report has exact top-level keys `schemaVersion`, `threshold`, `modelSha256`, and `candidates`. Each candidate has exact keys `candidateId`, `category`, `runCompleted`, `numericRuntime`, `detections`, and `visualReview`; each detection has finite measured `classId`, `confidence`, `cx`, `cy`, `w`, `h`, and `angle`. `candidateId` must be one of the fixed nine and present in the source-valid acquisition batch, `category` must match that recipe, every category must contribute two or three candidates, `threshold` is exactly `0.25`, the model digest is exactly `a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97`, and the initial review state is exactly `unreviewed`.

It captures `f"{candidate_id}-original.png"` and `f"{candidate_id}-result.png"` outside the repository. UI/console/page errors fail the run; the report and screenshot metadata must contain no input path, filename other than fixed candidate ID, model metadata, raw error, stack, browser profile, or rejected source URL.

- [ ] **Step 5: Reach GREEN for unit/privacy behavior**

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py -q
uv run --no-sync python -m pytest tests/test_demo_assets.py tests/test_browser_parity.py -q
git diff --check
```

Expected: the new unit suite and unchanged asset/parity suites pass; `git status --short` lists only the three new code/test paths plus preserved `.superpowers/`.

- [ ] **Step 6: Attempt the exact nine-recipe pool and evaluate every source-valid candidate outside the repository**

```powershell
$reviewRoot = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-naip-review-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $reviewRoot) { throw 'Refusing to overwrite candidate review root' }
uv run --no-sync python scripts/prepare_sample_gallery.py acquire --review-root $reviewRoot
uv run --no-sync python scripts/sample_gallery_smoke.py --review-root $reviewRoot --model demo/web/models/yolo26n-obb-privacy-sanitized.onnx --report (Join-Path $reviewRoot 'observations.json') --screenshot-dir (Join-Path $reviewRoot 'screenshots')
```

The acquisition command attempts all nine immutable recipes. Exact `GalleryError("GALLERY_SOURCE_REJECTED")` excludes only that recipe; every other error aborts the batch. Require two or three source-valid candidates per category, run smoke for every retained candidate, and inspect every resulting original/result pair. Approve exactly one ID per category only if it passes every visual-suitability rule in the spec. Record category pass counts and the three public IDs in the ignored ledger, then use the script's explicit approval command:

```powershell
$approvalPointer = Join-Path $worktree '.superpowers\sdd\2026-09-01-aerial-obb-curated-real-sample-gallery\approved-gallery-location.txt'
uv run --no-sync python scripts/prepare_sample_gallery.py approve --review-root $reviewRoot --observations (Join-Path $reviewRoot 'observations.json') --pointer $approvalPointer
uv run --no-sync python scripts/prepare_sample_gallery.py verify-approved --review-root $reviewRoot
```

The `approve` command presents the two or three source-valid candidates in each category and requires the reviewer to select one fixed ID after inspecting the screenshot pairs; it writes the selection into repository-external `approved-gallery.json` and writes only the review-root location to the ignored pointer. If any category retains fewer than two source-valid candidates or lacks a visually passing image at confidence `0.25`, exit without writing either file and stop implementation; do not replace a recipe, change the threshold/category/source/model, or approve a weak result.

- [ ] **Step 7: Review and commit Task 1**

Fresh spec and quality reviewers inspect the source allowlist, per-recipe fail-closed behavior, deterministic bytes, privacy/containment, real BYOM inference, every source-valid screenshot pair, the two-or-three-per-category pool invariant, and exactly three approvals. Resolve findings through the Task 1 implementer and rerun Steps 5–6. Stage exactly:

```powershell
git add scripts/prepare_sample_gallery.py scripts/sample_gallery_smoke.py tests/test_sample_gallery.py
git diff --cached --check
git commit -m "feat: add reproducible NAIP sample admission"
```

Keep `$reviewRoot` intact and record it only in the ignored ledger for Task 2.

---

### Task 2: Publish the Closed Catalog and Three-option Workbench

**Files:**
- Modify: `scripts/prepare_sample_gallery.py`
- Modify: `tests/test_sample_gallery.py`
- Modify: `scripts/prepare_demo_assets.py`
- Modify: `tests/test_demo_assets.py`
- Modify: `demo/web/demo-model.json`
- Modify: `demo/web/demo-assets.js`
- Modify: `demo/web/index.html`
- Modify: `demo/web/style.css`
- Modify: `demo/web/app.js`
- Modify: `scripts/browser_smoke.py`
- Create: `release/sample-gallery-sources.json`
- Create: `demo/web/samples/airfield.jpg`
- Create: `demo/web/samples/sports-complex.jpg`
- Create: `demo/web/samples/harbor.jpg`
- Delete: `demo/web/samples/boats.jpg`

**Interfaces:**
- Consumes the Task 1 approved review root and exact interfaces in this plan.
- Produces schema version 2, immutable `DemoAssets.getSampleCatalog()`, the three selector buttons, `state.selectedSampleId`, and successful original-before-Detect switching.

- [ ] **Step 1: Write the atomic publication/catalog/selector batch RED**

Add these named tests before copying or changing public assets:

```python
def test_publish_writes_exact_three_images_and_public_safe_receipt_atomically(
    tmp_path: Path, approved_gallery_report: dict
) -> None:
    publish_approved_gallery(report, review_root, pages_root, receipt_path)
    assert sorted(path.name for path in (pages_root / "samples").iterdir()) == [
        "airfield.jpg", "harbor.jpg", "sports-complex.jpg"
    ]
    assert_no_private_or_rejected_fields(json.loads(receipt_path.read_text("utf-8")))

def test_demo_manifest_declares_exact_sample_catalog_and_default() -> None:
    manifest = json.loads((ROOT / "demo/web/demo-model.json").read_text("utf-8"))
    assert manifest["schemaVersion"] == 2
    assert manifest["defaultSampleId"] == "airfield"
    assert [item["id"] for item in manifest["samples"]] == [
        "airfield", "sports-complex", "harbor"
    ]
```

In `scripts/browser_smoke.py`, add `assert_sample_gallery_initial(page, requests, messages)` and a `sample-gallery` scenario. Its initial checks are live DOM/network assertions:

```python
options = page.locator("#sampleSelector .sample-option")
if options.count() != 3:
    raise RuntimeError("real sample selector does not expose exactly three options")
if page.locator('.sample-option[aria-pressed="true"]').get_attribute("data-sample-id") != "airfield":
    raise RuntimeError("airfield is not the exact initial sample")
if page.locator("#demoOriginalImage").get_attribute("src") != "samples/airfield.jpg":
    raise RuntimeError("initial original is not the admitted airfield image")
if any(path in _request_paths(requests) for path in (DEMO_MANIFEST_PATH, DEMO_MODEL_PATH)):
    raise RuntimeError("sample gallery loaded model resources before Detect")
```

The scenario entry point is exactly
`run_sample_gallery(executable_path: Path | None = None, base_url: str | None = None, screenshot: Path | None = None) -> None`.
Register it in CLI choices, the no-argument full run, and dispatch `--screenshot` to that function so Task 4 can capture the exact successful default result.

- [ ] **Step 2: Run and record only the earliest actual RED**

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py::test_publish_writes_exact_three_images_and_public_safe_receipt_atomically -q
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery
```

Expected first unit RED: publication is not implemented. Expected first browser RED after registering the scenario: selector count differs from three. Later catalog/default/network checkpoints are not claimed as observed until earlier assertions become reachable.

- [ ] **Step 3: Atomically publish the approved images and receipt**

Extend `publish_approved_gallery` to validate the complete approved batch before staging. Copy each approved derived image to its fixed category filename, write canonical JSON to `release/sample-gallery-sources.json`, verify decoded dimensions/signature/digest against the record, then atomically replace only the managed three leaves. A failure preserves the previous public batch and receipt. It rejects an existing `boats.jpg`, extra sample leaf, hard link, symlink/reparse point, review root inside Git, and pages root outside the current worktree.

```powershell
$approvalPointer = Join-Path $worktree '.superpowers\sdd\2026-09-01-aerial-obb-curated-real-sample-gallery\approved-gallery-location.txt'
$reviewRoot = [IO.Path]::GetFullPath((Get-Content -Raw -LiteralPath $approvalPointer).Trim())
if (-not (Test-Path -LiteralPath (Join-Path $reviewRoot 'approved-gallery.json'))) { throw 'Approved gallery review batch is unavailable' }
uv run --no-sync python scripts/prepare_sample_gallery.py publish --review-root $reviewRoot --pages-root demo/web --receipt release/sample-gallery-sources.json --manifest demo/web/demo-model.json --loader demo/web/demo-assets.js
```

The ledger substitution above uses the exact Task 1 runtime path and is never committed. After publication, prove raw binary equality between approved derived bytes and the three fixed public files, then remove tracked `demo/web/samples/boats.jpg` through the publication transaction.

- [ ] **Step 4: Replace the single image manifest with the exact catalog**

Change `demo-model.json` from `image` to `defaultSampleId` plus `samples`. Populate measured values only from `release/sample-gallery-sources.json`. Each sample object has exactly the structural fields defined above; sample order is fixed.

The `publish` command generates complete canonical `demo-model.json` and the literal sample portion of `demo-assets.js` from the approved receipt while preserving and revalidating every unchanged model/input/output/provenance/sanitization/license value. The generated JavaScript defines exact `SAMPLE_IDS = Object.freeze(["airfield", "sports-complex", "harbor"])`, schema version `2`, default `airfield`, default confidence `0.25`, all three measured frozen records, and `getSampleCatalog()` returning `EXPECTED.samples`. The public API becomes exactly `Object.freeze({validateManifest, fetchVerifiedModel, getSampleCatalog})`; no runtime JSON generation or local receipt fetch is added.

`validateManifest` remains exact-key fail-closed and deep-freezes every sample/source/derivation/guardrail object. Add mutation tests for unknown ID/path/source, duplicate ID/path, wrong order/default, changed bbox/digest/dimensions/guardrails, external URL, and extra key.

Update `prepare_demo_assets.py` so its admitted/publish batch consumes the exact gallery receipt and public files instead of acquiring `boats-image`. It continues to acquire/validate the pinned upstream model and license, creates the sanitized derivative, rejects stale managed sample leaves, and reproduces the same schema-2 manifest and notice inputs.

- [ ] **Step 5: Implement the minimum three-button semantic UI**

Replace the single sample card interior with:

```html
<fieldset id="sampleSelector" class="sample-selector">
  <legend>選擇範例</legend>
  <button class="sample-option" type="button" name="demo-sample" value="airfield"
          data-sample-id="airfield" aria-pressed="true">
    <img src="samples/airfield.jpg" alt="小型機場航拍原圖縮圖" width="96" height="60" />
    <span><strong>小型機場航拍範例</strong><small>真實航拍原圖</small></span>
  </button>
  <button class="sample-option" type="button" name="demo-sample" value="sports-complex"
          data-sample-id="sports-complex" aria-pressed="false">
    <img src="samples/sports-complex.jpg" alt="運動場館航拍原圖縮圖" width="96" height="60" />
    <span><strong>運動場館航拍範例</strong><small>真實航拍原圖</small></span>
  </button>
  <button class="sample-option" type="button" name="demo-sample" value="harbor"
          data-sample-id="harbor" aria-pressed="false">
    <img src="samples/harbor.jpg" alt="低密度港區航拍原圖縮圖" width="96" height="60" />
    <span><strong>低密度港區航拍範例</strong><small>真實航拍原圖</small></span>
  </button>
</fieldset>
<p id="sampleState">Original · ready</p>
```

Keep the primary Detect and result toggle below the fieldset. Set `#demoOriginalImage` to the exact airfield path, dimensions `1280×800`, and the admitted plain-language alt. The static notice before the selector is exactly:

```html
<strong id="claimTitle">真實航拍範例 · 實際瀏覽器推論</strong>
<p>選擇一張 USGS／USDA NAIP 公領域航拍範例，再按下 Detect。頁面才會載入 OBB 模型並在你的瀏覽器中執行推論；影像不會上傳。範例經挑選以便清楚展示操作，不是 accuracy、evaluation 或 latency benchmark。</p>
```

The left-rail instruction is exactly `先選擇一張真實航拍原圖，再由目前的 browser 執行 Detect。`.

CSS keeps the existing rectangular workbench and adds a compact vertical selector: 96×60 contained thumbnails, visible `aria-pressed` border/check treatment, no color-only selection, 44px minimum button target, no horizontal rail overflow, and mobile stacking without carousel or animation.

- [ ] **Step 6: Implement minimal successful selection state**

Initialize:

```javascript
const sampleOptions = Array.from(document.querySelectorAll(".sample-option"));
const SAMPLE_CATALOG = DemoAssets.getSampleCatalog();
state.selectedSampleId = "airfield";
```

`selectDemoSample(sampleId)` validates the fixed ID, calls `nextGeneration()`, sets `source="demo"`, clears the active result through existing reset helpers, updates `aria-pressed`, sets the matching `src`, decodes the shared image, stores it as the current demo image, restores exact neutral summary/provenance/status, and performs no inference. It preserves `state.session` only when it remains a verified demo session; it never restores former per-sample output.

Change `runDemo()` to require the decoded image for `state.selectedSampleId` and to use `state.image`, not a hard-coded original. Manifest validation verifies that the active sample ID/path/digest is admitted before inference. Initial provenance is exactly `USGS／USDA NAIP · 尚未執行`; successful provenance is `${DEMO_PROVENANCE} · ${sample.title}` and never includes correctness language.

- [ ] **Step 7: Reach Task 2 GREEN**

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py tests/test_demo_assets.py -q
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery
uv run --no-sync python scripts/browser_smoke.py --scenario workbench-layout
uv run --no-sync python scripts/browser_smoke.py --scenario real-demo-success
uv run --no-sync python scripts/browser_smoke.py --scenario stubbed-cache
git diff --check
```

Expected: three selectors and default original pass; selecting each option shows the matching original and clears result state; first Detect remains genuine/lazy; repeated Detect reuses the session. Record the expected cross-task failure that reviewed Pages/release/clean-export inventories still describe `boats.jpg`; Task 4 owns that freeze. No other failure is accepted.

- [ ] **Step 8: Review and commit Task 2**

Fresh reviewers inspect the exact approved bytes, receipt, closed schema, atomic publication, selector UI, single-result state, no early model request, claim copy, and absence of hidden tuning. Fix through the original implementer and re-review. Stage exactly the Task 2 file list:

```powershell
git add scripts/prepare_sample_gallery.py tests/test_sample_gallery.py scripts/prepare_demo_assets.py tests/test_demo_assets.py demo/web/demo-model.json demo/web/demo-assets.js demo/web/index.html demo/web/style.css demo/web/app.js scripts/browser_smoke.py release/sample-gallery-sources.json demo/web/samples/airfield.jpg demo/web/samples/sports-complex.jpg demo/web/samples/harbor.jpg
git add -u -- demo/web/samples/boats.jpg
git diff --cached --check
git commit -m "feat: add curated real-image sample gallery"
```

---

### Task 3: Prove Races, Safe Failures, Accessibility, and Three Real Runs

**Files:**
- Modify: `scripts/browser_smoke.py`
- Modify: `demo/web/app.js`
- Modify only if a reached browser assertion requires it: `demo/web/index.html`
- Modify only if a reached browser assertion requires it: `demo/web/style.css`

**Interfaces:**
- Consumes the Task 2 catalog, selector, `selectDemoSample`, generation state, and existing session/cache/failure/BYOM helpers.
- Produces complete `sample-gallery` and `sample-gallery-failures` real-browser scenarios with safe sample decode recovery and no stale cross-sample publication.

- [ ] **Step 1: Add the failure/race/accessibility batch RED**

Extend `sample-gallery` to run the exact public model once on each fixed sample. For every success assert:

- selected ID/path/title and `aria-pressed` are exact;
- runtime is numeric, mode is `LOCAL BROWSER INFERENCE`, and run counter increments once;
- the filtered output satisfies that sample's manifest guardrails;
- summary, polygons, table, and non-live description represent one identical filtered list;
- switching to the next sample immediately restores `runtime=—`, count `0`, empty table/description, hidden canvas/toggle, enabled Detect after decode, and zero stale title/provenance;
- ORT/model request counts and session creation stay at one while the real run counter reaches three.

Add `run_sample_gallery_failures` with the exact first batch:

```python
def assert_safe_sample_failure(page: object) -> None:
    if page.locator("#status").inner_text() != (
        "這張範例影像目前無法顯示。請選擇其他範例，或重新整理後重試。"
    ):
        raise RuntimeError("sample decode failure lacks fixed recovery copy")
    if page.locator("#runtimeValue").inner_text() != "—":
        raise RuntimeError("sample decode failure retained numeric runtime")
    if page.locator("#resultsBody tr:not([data-empty='true'])").count() != 0:
        raise RuntimeError("sample decode failure retained stale table rows")
    if page.locator(".sample-option:not([disabled])").count() < 2:
        raise RuntimeError("sample decode failure disabled unrelated examples")
```

The scenario intercepts one non-default admitted image with invalid JPEG bytes, selects it after a successful default result, and then selects a healthy third image. A separate deterministic delayed-image route selects A, then B before A resolves; resolving A must not change B's selected state/image/status. A delayed inference from A followed by selection of B must not publish A's canvas/table/runtime after resolution. Both cases assert zero raw error/path/filename/stack in UI and console.

- [ ] **Step 2: Observe the earliest real browser RED**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery-failures
```

Expected earliest RED after Task 2: the existing generic image-decode message differs from the exact sample recovery copy. Record only the actual earliest failure; stale-state and unrelated-selector checkpoints blocked by it are not separately claimed.

- [ ] **Step 3: Implement minimal generation-protected sample recovery**

Add one safe code and no raw diagnostics:

```javascript
ERROR_COPY.DEMO_IMAGE_DECODE =
  "這張範例影像目前無法顯示。請選擇其他範例，或重新整理後重試。";
```

`loadSelectedDemoImage(sample, token)` attaches one-shot load/error listeners before assigning `src`, resolves only if `isCurrentGeneration(token)`, and throws only `DEMO_IMAGE_DECODE`. `selectDemoSample` catches that fixed code, clears complete result presentation, keeps the failed option selected with Detect disabled, keeps every other selector operable, and never releases a valid demo session. A later healthy selection clears the error. Late decode/run completions test the token before every DOM/state write.

Do not add sample caches, extra canvases, per-sample session objects, or console detail. Continue using `clearResultState`, `resetResult`, `setInitialSummary`, `showOriginalSource`, and the single status region.

- [ ] **Step 4: Extend accessibility and responsive contracts**

In the existing scenarios assert:

- the skip link remains first and the claim notice precedes all three sample buttons;
- `#sampleSelector` has one legend and exactly three stable `name="demo-sample"` buttons;
- keyboard Tab/Shift+Tab reaches the options in visual order, Space/Enter selects, focus stays on the activated option, and Detect follows the group;
- each thumbnail has meaningful alt, fixed dimensions, and no duplicate live announcement;
- selected state is distinguishable under forced-colors/high contrast and focus-visible is at least 3 CSS pixels;
- desktop keeps the selector in the left rail; 390×844 and 200%-zoom-equivalent widths have no page-level horizontal overflow and preserve notice → selector → Detect → viewport → summary → filters/table → BYOM order;
- all three originals use `object-fit: contain`, remain readable, and do not cause layout shift outside the fixed thumbnail/viewport boxes.

- [ ] **Step 5: Reach focused and full browser GREEN**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery-failures
uv run --no-sync python scripts/browser_smoke.py --scenario stale-generation
uv run --no-sync python scripts/browser_smoke.py --scenario byom-transition
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility
uv run --no-sync python scripts/browser_smoke.py --scenario desktop
uv run --no-sync python scripts/browser_smoke.py --scenario mobile
uv run --no-sync python scripts/browser_smoke.py
git diff --check
```

Expected: three genuine runs, one active demo session across gallery switches, deterministic failure/race recovery, unchanged BYOM lifecycle, all browser scenarios pass, and requests remain same-origin except pinned jsDelivr ORT/WASM after Detect.

- [ ] **Step 6: Review and commit Task 3**

Fresh spec/quality reviewers receive the exact Task 3 diff, browser output, request log, public-safe screenshot pairs, and Task 1 admission evidence. Resolve findings through the Task 3 implementer and re-review. Stage only reached changes:

```powershell
$task3Paths = @('scripts/browser_smoke.py','demo/web/app.js')
if (git diff --name-only -- demo/web/index.html) { $task3Paths += 'demo/web/index.html' }
if (git diff --name-only -- demo/web/style.css) { $task3Paths += 'demo/web/style.css' }
git add -- $task3Paths
git diff --cached --check
git commit -m "test: harden real sample gallery behavior"
```

---

### Task 4: Freeze Public Evidence, Notices, and Release Gates

**Files:**
- Modify: `scripts/pages_artifact_check.py`
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `scripts/release_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `scripts/clean_export_check.py`
- Modify: `tests/test_clean_export.py`
- Modify if required by a named RED: `scripts/repo_check.py`
- Modify if required by a named RED: `tests/test_repo_check.py`
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
- Consumes the reviewed Task 1–3 files and exact approved receipt.
- Produces exact artifact digests/inventory, three-image evidence/notices, strict clean export, and one canonical default-sample result screenshot.

- [ ] **Step 1: Observe the existing fail-closed artifact RED**

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_current_pages_tree_passes -q
```

Expected RED: the exact reviewed inventory and/or text/image digest differs because Tasks 2–3 changed admitted public bytes. Do not weaken exact digest, inventory, path, origin, model, secret, storage, link, or privacy enforcement.

- [ ] **Step 2: Add exact gallery release/evidence RED tests**

Add or replace only stale single-sample contracts with these exact names:

- `test_pages_tree_admits_exact_curated_gallery_inventory` calls the existing Pages checker against a copied current tree, asserts zero errors, then asserts the exact three sample paths and absence of `samples/boats.jpg`.
- `test_browser_demo_evidence_records_exact_curated_gallery` loads `release/evidence.json` and applies every exact assertion below.
- `test_real_demo_manifest_records_exact_public_gallery_artifacts` cross-checks each `demo-model.json` sample against the source receipt and artifact manifest for path, bytes, SHA-256, source, derivation, and guardrails.
- `test_clean_export_keeps_exact_gallery_and_omits_boats` inspects the tracked archive and asserts the three paths are present byte-identically and `boats.jpg` is absent.
- `test_notices_record_public_domain_naip_derivations` asserts all three fixed titles/paths/source product IDs, `Public Domain`, and `crop/resample/metadata removal`, while retaining the separate model AGPL notice.

Require these evidence values:

```python
assert browser["demo_images"] == [
    "demo/web/samples/airfield.jpg",
    "demo/web/samples/sports-complex.jpg",
    "demo/web/samples/harbor.jpg",
]
assert browser["default_demo_image"] == "demo/web/samples/airfield.jpg"
assert browser["sample_count"] == 3
assert browser["sample_selection"] == "explicit-three-option"
assert browser["confidence"] == 0.25
assert browser["per_image_tuning"] is False
assert browser["precomputed_results"] is False
assert browser["demo_inference_performed"] is True
assert browser["model_bundled"] is True
assert browser["represents_accuracy_evaluation"] is False
assert browser["represents_t4_latency"] is False
```

Mutation tests must reject missing/extra/reordered sample, `boats.jpg`, changed image byte/digest/source record, non-public-domain agency, receipt mismatch, different default, hidden threshold/filter, precomputed result, second model, DOTA/private/path/token content, and unreviewed binary.

Run the named tests and record the first stale single-sample failure each actually reaches:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py::test_pages_tree_admits_exact_curated_gallery_inventory -q
uv run --no-sync python -m pytest tests/test_release_check.py::test_browser_demo_evidence_records_exact_curated_gallery -q
uv run --no-sync python -m pytest tests/test_clean_export.py::test_clean_export_keeps_exact_gallery_and_omits_boats -q
```

- [ ] **Step 3: Update exact Pages/release/clean-export contracts**

Change the single `IMAGE_PATH` into the exact three-path tuple. Freeze raw-binary SHA-256 for each JPEG and canonical-LF digests for changed public text. Add the exact USGS service/public-domain record to approved provenance navigation/data fields without permitting it as a runtime resource origin. Keep runtime resource origins unchanged: site origin initially, pinned jsDelivr ORT/WASM only after Detect.

`release/artifact-manifest.json` contains one bundled-third-party entry per sample with:

- fixed public path and final bytes/SHA-256;
- `kind: "public-domain NAIP aerial sample derivative"`;
- exact locked source product identity/year/agency/service/download record from `release/sample-gallery-sources.json`;
- modification status `crop/resample/metadata removal` and exact bbox/output contract;
- license `Public Domain`; and
- restrictions stating curated integration example, no accuracy/evaluation/model-quality evidence, and no endorsement.

The reviewed public inventory contains all three images and no `boats.jpg`. The one privacy-sanitized model/license/sanitization exception remains byte-identical and exact. Clean export requires `release/sample-gallery-sources.json` in repository evidence but does not put the release receipt into `demo/web` unless the existing artifact policy explicitly admits public evidence there; the visible third-party notice carries the necessary source facts.

- [ ] **Step 4: Update truthful public documentation and notices**

Update all current single-sample prose to the exact user journey: choose one of three public-domain real aerial originals, press Detect, genuine local inference, switch original/result, cached filters, advanced BYOM. State that selection was curated for visual clarity at the shared threshold and is not accuracy/evaluation evidence.

The root and demo notices list the three samples separately by title, fixed path, USGS/USDA NAIP source product/year, public-domain record, crop/resample/metadata-removal status, final digest/bytes, and no-endorsement limitation. Preserve the separate Ultralytics derivative/model/DOTAv1/AGPL/commercial-clearance boundaries.

Use this focused stale-reference scan and review any historical changelog hit in context:

```powershell
rg -n 'boats\.jpg|one official sample|sole sample|Ultralytics official sample' demo/web README.md README.en.md THIRD_PARTY_NOTICES.md RELEASE_CHECKLIST.md release scripts/pages_artifact_check.py scripts/release_check.py scripts/clean_export_check.py tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py
```

Expected after GREEN: no current-contract hit. Historical design documents are not rewritten.

- [ ] **Step 5: Generate and visually inspect the canonical screenshot**

Serve exact current tracked `demo/web`, run genuine default-airfield inference, and overwrite only the canonical screenshot:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery --screenshot docs/assets/browser-workbench.png
```

Required pixels: three compact choices in the left rail with airfield selected, genuine annotated airfield result in the shared viewport, numeric runtime, exact mode/provenance, readable result table, claim notice, Source and AGPL context, no open/stale BYOM state, and no visually dominant wrong box. Inspect PNG metadata and byte content for forbidden paths, filenames, private identity, raw error, stack, token, or browser profile. Freeze actual digest/bytes in the artifact manifest.

- [ ] **Step 6: Reach Task 4 GREEN**

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_repo_check.py tests/test_readme_language.py -q
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/browser_smoke.py
git diff --check
```

Expected: focused tests, all direct gates, and full browser smoke pass; current public scans contain exactly three NAIP samples, one sanitized model, no dense boats sample, no DOTA/private/precomputed result, and no unexpected origin.

- [ ] **Step 7: Review and commit Task 4**

Fresh reviewers inspect artifact equality, source/public-domain records, claim language, exact digests, screenshot pixels/metadata, model/license separation, clean-export behavior, stale references, and all task output. Resolve findings through the Task 4 implementer and re-review. Stage exactly the modified subset of the Task 4 list:

```powershell
$task4Paths = @(
  'scripts/pages_artifact_check.py','tests/test_pages_artifact_check.py',
  'scripts/release_check.py','tests/test_release_check.py',
  'scripts/clean_export_check.py','tests/test_clean_export.py',
  'release/artifact-manifest.json','release/evidence.json',
  'README.md','README.en.md','demo/web/README.md','THIRD_PARTY_NOTICES.md',
  'demo/web/THIRD_PARTY_NOTICES.md','RELEASE_CHECKLIST.md','CHANGELOG.md',
  'docs/assets/browser-workbench.png'
)
if (git diff --name-only -- scripts/repo_check.py) { $task4Paths += 'scripts/repo_check.py' }
if (git diff --name-only -- tests/test_repo_check.py) { $task4Paths += 'tests/test_repo_check.py' }
git add -- $task4Paths
git diff --cached --check
git commit -m "release: freeze curated sample gallery evidence"
```

---

### Task 5: Complete Full Local Acceptance and Owner-operable Preview

**Files:**
- Create only repository-external clean export, screenshots, origin report, and review notes.
- No tracked file is planned. A finding returns to the owning Task 1–4 path, begins with a focused RED, and receives one exact fix commit and fresh re-review.

**Interfaces:**
- Consumes all committed task outputs.
- Produces full local evidence, strict clean export, broad review verdict, clean retained branch, and a loopback preview for owner operation.

- [ ] **Step 1: Verify committed scope and test inventory**

```powershell
git status --short
git diff --check
git log --oneline --decorate -16
uv run --no-sync python -m pytest --collect-only -q
```

Expected: tracked worktree/index clean, only preserved `.superpowers/` untracked, and collected tests exceed the current 254-test baseline by the committed gallery unit tests.

- [ ] **Step 2: Run focused, browser, full-suite, and direct gates separately**

```powershell
uv run --no-sync python -m pytest tests/test_sample_gallery.py tests/test_demo_assets.py tests/test_browser_parity.py tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_repo_check.py tests/test_package_release.py -q
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery
uv run --no-sync python scripts/browser_smoke.py --scenario sample-gallery-failures
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
```

Record exact counts, durations, request origins, three sample run counters, session create/release counters, and zero-failure results. No test may be skipped because it is slow.

- [ ] **Step 3: Run strict clean export from tracked files**

```powershell
$cleanExport = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-gallery-clean-export-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $cleanExport) { throw 'Refusing to overwrite clean export' }
uv run --no-sync python scripts/clean_export_check.py --output $cleanExport
uv run --no-sync python scripts/pages_artifact_check.py --root (Join-Path $cleanExport 'demo/web')
```

Do not pass `--skip-browser`. Expected: archive rebuild, package build/install/import, complete tests, three genuine sample paths, same-origin initial network, model/license/sanitization identity, privacy scans, and browser smoke all pass from the exact exported tree. Record the export archive SHA-256.

- [ ] **Step 4: Run forbidden-artifact, privacy, origin, and stale-reference scans**

```powershell
rg -n -I -i 'boats\.jpg|DOTA.*\.(jpg|jpeg|png)|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|[A-Z]:[\\/]Users[\\/]|Traceback|stack trace' demo/web release README.md README.en.md THIRD_PARTY_NOTICES.md RELEASE_CHECKLIST.md
git ls-files demo/web
git diff --name-status 40a6eb130b6e2cf46b89469750eb10f9133d8a83...HEAD
```

Expected: no current boats/DOTA/private/token/path/raw-stack hit; historical release prose is reviewed in context; public inventory has exactly the approved three JPEGs and one approved ONNX model. Browser request evidence permits only loopback/same-origin page/sample/model files plus pinned jsDelivr ORT/WASM after Detect; no third-party imagery request occurs at runtime.

- [ ] **Step 5: Serve and inspect the exact clean export**

Use the exact clean export, not the dirty worktree tree:

```powershell
uv run --no-sync python -m http.server 8766 --bind 127.0.0.1 --directory (Join-Path $cleanExport 'demo/web')
```

Open `http://127.0.0.1:8766/`. At 1280×720, 390×844, and 200%-zoom-equivalent width, operate all three examples: inspect original, press Detect, inspect result, toggle original/result, adjust cached filters, switch sample, and repeat. Verify selection clarity, no visually dominant error, numeric runtime only after inference, no stale cross-sample result, collapsed secondary BYOM, notice priority, keyboard/focus, labels/names/headings, canvas alternative, reduced motion, Source/AGPL/public-domain readability, no overflow, no page/console error, and exact request origins. Store desktop/mobile original/result screenshots outside the repository.

- [ ] **Step 6: Broad whole-branch review and one bounded fix wave**

Provide the most capable fresh reviewer with the approved spec, this plan, the full `40a6eb130b6e2cf46b89469750eb10f9133d8a83...HEAD` diff, all task ledgers/reports, the exact approved receipt, every source-valid admission screenshot pair, category pool counts, three final artifact screenshot pairs, browser/network output, canonical screenshot, full-suite/direct-gate output, strict clean-export digest, and privacy scans.

Critical or Important findings permit one focused test-first fix wave through the responsible original implementer, its own minimal commit, and fresh re-review. A second product fix wave, source/license ambiguity, weak visual admission, artifact mismatch, privacy leak, unexpected origin, model/inference contract failure, or unresolved Important finding stops completion. Record Minor findings without unrelated polish.

- [ ] **Step 7: Verification-before-completion and branch readiness only**

```powershell
git status --short
git diff --check
git log --oneline --decorate -24
git diff --name-status 40a6eb130b6e2cf46b89469750eb10f9133d8a83...HEAD
git diff --quiet
git diff --cached --quiet
```

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch` only to confirm readiness and present integration choices. Keep the feature branch, worktree, `.superpowers/`, candidate review root, clean export, screenshots, and preview available for owner feedback. Do not select or execute an integration/remote option.

---

## Remote Gates A–E — Separate and Unauthorized

1. **Gate A — Candidate PR:** re-check origin/main, branch/head race, auth/scopes, templates, checks, non-force push, and exactly one review-only PR.
2. **Gate B — Candidate artifact and merge:** download exact CI artifact, compare all three image/model/text bytes and canonical digests, repeat visual/privacy/license/origin review, and use only an explicitly authorized repository-supported merge method.
3. **Gate C — Pages dispatch:** configure or dispatch only from the exact reviewed merged-main SHA after its automatic release gates pass.
4. **Gate D — Live review:** validate all three original/Detect/result flows, initial zero-model network, cached filters, races/failures/BYOM, desktop/mobile/accessibility/privacy/notices, deployed SHA, and forbidden-artifact absence.
5. **Gate E — About and Portfolio Control receipt:** change homepage metadata and record completion only after independent Gate D approval.

No task in this plan authorizes push, PR, merge, auto-merge, force push, Pages enable/configuration/dispatch/deployment, GitHub About, Hugging Face, release, tag, visibility change, branch deletion, worktree cleanup, or modification of another repository.

## Self-review Checklist

- **Spec coverage:** Task 1 covers exact public-domain source, deterministic derivation, same-model threshold, privacy, visual admission, and stop conditions. Task 2 covers the closed catalog, three files, selector, default original, switching, one result/session, and genuine Detect. Task 3 covers failures, races, guardrails, BYOM, accessibility, responsive behavior, and all three real runs. Task 4 covers notices, evidence, digests, artifact/clean-export gates, documentation, and canonical screenshot. Task 5 covers full regression, exact export, operable preview, and broad review.
- **Interface consistency:** `CandidateRecipe`, the nine IDs/bboxes, three public IDs/paths, schema version 2, `defaultSampleId`, `getSampleById`, `selectDemoSample`, `loadSelectedDemoImage`, `setSampleSelection`, and `state.selectedSampleId` are defined once and used with the same names and domains.
- **TDD honesty:** Every implementation task names its first expected reachable RED and explicitly forbids claiming later assertions blocked by it. Successful sample inference uses real browser/model bytes; error routes may be stubbed but mock-only coverage cannot satisfy admission.
- **No unresolved implementation gaps:** Candidate selection is a bounded runtime admission from nine exact recipes, permits only dedicated per-recipe source rejection, requires two or three retained candidates per category, and has explicit stop conditions; it is not an unspecified asset search. Measured source/digest/guardrail facts are generated and validated by Task 1–2 rather than represented by fake literals.
- **Asset/privacy boundary:** Exactly three public-domain JPEG derivatives, one existing sanitized ONNX derivative, and existing licenses enter the artifact. Source tiles, rejected candidates, DOTA content, commercial imagery, local paths, raw diagnostics, tokens, and precomputed results remain excluded.
- **Claim boundary:** Curated visual clarity is disclosed and never promoted as accuracy/evaluation/benchmark evidence; confidence stays `0.25` for every candidate and public sample.
- **Remote boundary:** All remote gates are listed separately and are unauthorized by plan approval or local execution.

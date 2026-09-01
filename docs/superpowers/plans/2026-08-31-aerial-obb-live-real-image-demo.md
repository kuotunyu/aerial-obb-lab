# Aerial OBB Live Real-image Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public Synthetic Showcase with one immediately visible real aerial image whose primary `開始 Detect` action performs genuine YOLO26n-OBB inference in the visitor's browser, while retaining BYOM as an advanced local-file path.

**Architecture:** Publish one reviewed official Ultralytics image and one reviewed official AGPL ONNX model as exact same-origin Pages assets. Keep the current pinned ONNX Runtime Web loader and shared letterbox/decode/filter/rotated-corner/render pipeline; add a strict manifest/digest loader, one active session/result state machine, and a progressive real-image-first UI. Acquire and verify assets through a local-only tool, then enforce their exact bytes, licenses, request origins, and privacy boundary through static verifiers plus a real Playwright/ORT success path.

**Tech Stack:** Static HTML/CSS/JavaScript, ONNX Runtime Web 1.20.1/WASM, Web Crypto SHA-256, Playwright through Python, Pillow, pytest, Python standard-library artifact/release gates, `uv`, Git.

**Plan Status:** Owner-approved design translated into an implementation plan. The owner selected the recommended subagent-driven local execution path in advance; this plan-writing commit performs no implementation or remote action.

## Global Constraints

- The clean implementation base is exactly `24039db9e07e327b55433086241ac574430c4531`, the approved accessibility/runtime follow-up. Do not base implementation on the dirty superseded precomputed-sample worktree.
- Cherry-pick only design commit `271716534c9e350a845dbbe226af97805562dbc6` and the documentation-only commit containing this plan onto the clean implementation branch. Do not carry `scripts/prepare_pages_samples.py`, `tests/test_sample_assets.py`, their dirty changes, or the superseded sample spec/plan.
- The primary public path has exactly one image, exact official `boats.jpg`, visible unannotated on initial load. It has exactly one demo model, exact official `yolo26n-obb.onnx` from Ultralytics assets release `v8.4.0`.
- The official image URL is `https://ultralytics.com/images/boats.jpg`. The official model URL is `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx`. These are acquisition/provenance URLs only; the browser must load reviewed copies from `samples/boats.jpg` and `models/yolo26n-obb.onnx` on the Pages origin.
- Read-only preflight observed exact upstream body lengths of 194,872 bytes for `boats.jpg`, 10,207,250 bytes for `yolo26n-obb.onnx`, and 34,523 bytes for the Ultralytics assets `v8.4.0` license. Any length, redirect-host, release-identity, or content drift is a stop condition pending review; do not silently update a pin.
- The image is loaded initially. `demo-model.json`, ONNX Runtime, its WASM, and the ONNX model are not requested until the visitor presses `開始 Detect`.
- The public model's SHA-256 is verified with Web Crypto before `InferenceSession.create`. Its input must be `images [1,3,1024,1024]`; output must be `output0 [1,N,7]` with finite rows `[cx,cy,w,h,confidence,class,angleRadians]` and exactly the reviewed 15-class mapping.
- The real browser admission gate must produce at least one finite `ship` result at confidence `0.25` on the exact reviewed image. Contract mismatch, zero accepted detections, image drift, model drift, or browser failure stops implementation; do not change model, image, threshold, class mapping, tensor, or boxes to rescue it.
- The primary demo and BYOM are both real local-browser inference. Only completed `session.run` intervals display rounded numeric milliseconds. Idle, loading, reset, error, and no-cache states display `—`.
- Mode badge is exactly `DEMO MODEL · LOCAL BROWSER INFERENCE`; provenance is exactly `Ultralytics YOLO26n-OBB · official AGPL model`; successful BYOM keeps `BYOM · LOCAL BROWSER INFERENCE`.
- Synthetic SVG/fixed output leave `demo/web` and the public UI. Deterministic synthetic geometry may remain only under `tests/fixtures` and must not satisfy the real-model browser acceptance gate.
- No precomputed result, flattened annotated image, manually adjusted box, private/owner model, alternate mirror, CORS proxy, DOTA image/annotation/archive, upload, hosted inference, analytics, telemetry, storage, service worker, cookie, model cache, or new framework is allowed.
- The official model's DOTAv1 training provenance and Ultralytics AGPL route are disclosed without claiming commercial clearance or endorsement. Repository code remains AGPL-3.0-or-later; third-party asset terms remain separately recorded.
- User-facing and console errors use fixed codes/copy only. No local path, local filename, private model identity/metadata, raw exception, response body, raw tensor, stack, token, browser-profile path, or signed redirect URL may enter UI, logs, screenshots, reports, commits, or public manifests.
- Use strict TDD for every behavior change: add the named test/assertion, run it, record only the earliest assertion actually reached in a batch RED, make the minimum GREEN change, rerun the focused coverage, then commit exact paths.
- Use `apply_patch` for text edits. The reviewed binary image/model may be written only by the tested publishing command in Task 2 after external receipt validation.
- Local files, tests, loopback preview, repo-external screenshots/evidence, review, and commits are authorized. Push, PR, merge, workflow dispatch, Pages configuration/deployment, About, Hugging Face, release, tag, visibility, branch/worktree deletion, and every Remote Gate A–E remain unauthorized.

---

## Execution workspace bootstrap

Use `superpowers:using-git-worktrees` before Task 1. The existing design worktree must remain untouched because it intentionally contains three uncommitted files from the rejected approach.

Resolve the plan commit rather than inventing its SHA:

```powershell
$sourceWorktree = '<repository-root>\.worktrees\aerial-obb-real-sample-design'
$planPath = 'docs/superpowers/plans/2026-08-31-aerial-obb-live-real-image-demo.md'
$planCommit = (git -C $sourceWorktree log -1 --format=%H -- $planPath).Trim()
$expectedPlanCommitPaths = @(
  'docs/superpowers/plans/2026-08-31-aerial-obb-live-real-image-demo.md',
  'docs/superpowers/specs/2026-08-31-aerial-obb-live-real-image-demo-design.md'
) | Sort-Object
$planCommitPaths = @(git -C $sourceWorktree diff-tree --no-commit-id --name-only -r $planCommit | Sort-Object)
if (($planCommitPaths -join "`n") -ne ($expectedPlanCommitPaths -join "`n")) { throw 'Plan commit scope is not exactly the approved plan and spec status' }
if ((git -C $sourceWorktree rev-parse 24039db9e07e327b55433086241ac574430c4531).Trim() -ne '24039db9e07e327b55433086241ac574430c4531') { throw 'Approved base is unavailable' }
```

Create a new worktree/branch only when neither already exists:

```powershell
$executionWorktree = '<repository-root>\.worktrees\aerial-obb-live-real-image-demo'
git worktree add -b feat/pages-live-real-image-demo $executionWorktree 24039db9e07e327b55433086241ac574430c4531
git -C $executionWorktree cherry-pick 271716534c9e350a845dbbe226af97805562dbc6 $planCommit
```

Stop instead of deleting, resetting, overwriting, or reusing an existing path/branch. After cherry-pick, require a clean worktree and confirm the diff from the base contains only the new design and plan documents. Create the plan-specific local ledger at:

```text
.superpowers/sdd/2026-08-31-aerial-obb-live-real-image-demo/progress.md
```

Record each RED, GREEN, commit, review verdict, fix round, generated public digest, and stop condition in that ignored ledger. Keep one implementer active at a time and use a fresh task-scoped spec reviewer, fresh quality reviewer, and final whole-branch reviewer as required by `superpowers:subagent-driven-development`.

## File structure and interfaces

| Path | Responsibility |
| --- | --- |
| `scripts/prepare_demo_assets.py` | Local-only anonymous acquisition, redirect/size/digest validation, privacy-safe external receipt, and exact publish operation for the three approved third-party files. |
| `tests/test_demo_assets.py` | Offline fake-transport tests for immutable sources, fail-closed acquisition, receipt privacy, and publish-root/digest enforcement. |
| `demo/web/demo-model.json` | Closed runtime manifest for one image/model: exact same-origin paths, hashes, lengths, dimensions, tensor contract, class mapping, source, release, and license. No result tensor. |
| `demo/web/samples/boats.jpg` | Exact unmodified official Ultralytics sample bytes, visible before Detect. |
| `demo/web/models/yolo26n-obb.onnx` | Exact unmodified official release asset, requested only after Detect. |
| `demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt` | Exact reviewed upstream license text shipped with the model/image. |
| `demo/web/THIRD_PARTY_NOTICES.md` | Public image/model provenance, digests, training-origin disclosure, restrictions, and no-endorsement statement. |
| `demo/web/demo-assets.js` | Browser-side closed manifest validation, same-origin fetch, byte-length/SHA-256 verification, and fixed error-code mapping. No DOM rendering or session ownership. |
| `demo/web/index.html` | Real-image-first semantic document, exact notice, initial figure, primary Detect, result summary/table/description, result/original toggle, and advanced BYOM disclosure. |
| `demo/web/app.js` | Single active source/session/result state machine, lazy runtime/session lifecycle, real demo and BYOM actions, shared preprocessing/decode/filter/render pipeline, safe recovery. |
| `demo/web/style.css` | Real-image-first desktop/mobile layout, progressive controls, focus, busy/error states, result overlay, advanced disclosure, 200% zoom and reduced motion. |
| `demo/web/obb.js` | Existing pure geometry/decode API; modified only if a real RED proves a contract bug. |
| `scripts/browser_smoke.py` | Real Playwright checks for initial zero-model network, actual reviewed ONNX inference, repeat/toggle/filter, failures/retry, BYOM lifecycle/privacy, responsive/accessibility, and screenshots. |
| `scripts/pages_artifact_check.py` | Exact Pages inventory, digests, media/manifest/license/model exception, runtime origin, privacy, and no-synthetic/no-extra-model boundary. |
| `tests/test_pages_artifact_check.py` | Mutation coverage proving only the exact reviewed binary/image/text inventory passes. |
| `scripts/clean_export_check.py`, `tests/test_clean_export.py` | Clean archive policy that allows only the manifest-bound demo ONNX while continuing to reject every other model and DOTA visual. |
| `scripts/release_check.py`, `tests/test_release_check.py` | Public distribution/evidence claims, exact bundled third-party entries, privacy scan, and no unsupported accuracy/latency/commercial claim. |
| `release/artifact-manifest.json` | Byte-level inventory for the reviewed model, image, license and existing font; old excluded model remains historical exclusion metadata. |
| `release/evidence.json` | Honest live-demo runtime/source/claim boundary and exact public source-file list. |
| `README.md`, `README.en.md`, `demo/web/README.md`, `CHANGELOG.md`, `THIRD_PARTY_NOTICES.md` | User instructions and repository-wide evidence/license language aligned to live demo plus advanced BYOM. |
| `.github/workflows/release-gates.yml`, `tests/test_package_release.py`, `tests/test_pages_workflow.py` | CI naming and checks for the real browser demo and exact candidate artifact; action majors remain v7/v7/v7/v7/v6/v5/v5. |
| `docs/assets/browser-workbench.png` | Canonical screenshot generated from the exact reviewed local artifact after genuine demo inference. |

The acquisition module exposes these exact callable signatures:

```python
@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    source_url: str
    expected_bytes: int
    allowed_redirect_hosts: tuple[str, ...]
    public_relative_path: str

@dataclass(frozen=True)
class AssetReceipt:
    asset_id: str
    source_url: str
    redirect_hosts: tuple[str, ...]
    bytes: int
    sha256: str
    media_type: str
    width: int | None
    height: int | None

```

- `acquire_assets(review_root: Path, transport: Callable[[AssetSpec], tuple[bytes, tuple[str, ...], str]] = urlopen_transport) -> dict[str, AssetReceipt]`
- `validate_receipts(review_root: Path) -> dict[str, AssetReceipt]`
- `publish_assets(review_root: Path, pages_root: Path) -> None`

`OFFICIAL_ASSETS` contains exactly `boats-image`, `obb-model`, and `ultralytics-license`. `acquire_assets` accepts no model/image override. Its public output is one fixed line, `[OK] DEMO_ASSETS_ACQUIRED`, while the repo-external `receipt.json` contains public source facts and digests but no absolute paths, redirect query strings, raw headers, or exception text. HTTP `Content-Type` is allowlisted as `application/octet-stream` or `image/jpeg` for the image, `application/octet-stream` for the model, and `text/plain` for the license; the receipt records detected public media type (`image/jpeg`, `application/onnx`, or `text/plain`) rather than copying the transport header.

The browser helper exposes exactly:

```javascript
DemoAssets.validateManifest(value)                 // returns frozen validated manifest or throws fixed code
DemoAssets.sha256Hex(arrayBuffer)                  // Promise<string>
DemoAssets.fetchVerifiedModel(manifest, fetchImpl) // Promise<Uint8Array>
```

`fetchVerifiedModel` accepts only the manifest's exact relative path, `credentials: "same-origin"`, and `cache: "no-store"`; it verifies HTTP 200, byte length, and SHA-256 before returning bytes. It never falls back to an external URL.

The application state is exactly one active identity:

```javascript
const state = {
  source: "demo",
  phase: "idle",
  generation: 0,
  session: null,
  sessionKind: null,
  image: null,
  cached: null,
  view: "original",
};
```

`cached` remains the only result cache:

```javascript
state.cached = {
  results,
  geometry,
  provenance: "Ultralytics YOLO26n-OBB · official AGPL model",
  elapsedMs,
};
```

No UI string, row, polygon, description, badge, or toggle state is cached separately.

---

### Task 1: Build the fail-closed official asset preparation tool

**Files:**
- Create: `scripts/prepare_demo_assets.py`
- Create: `tests/test_demo_assets.py`

**Interfaces:**
- Consumes: the exact three `AssetSpec` constants and a test-injected HTTP transport.
- Produces: a repo-external review directory containing exact bytes plus privacy-safe `receipt.json`; publishes nothing during this task.

- [ ] **Step 1: Write the acquisition batch RED**

Create `tests/test_demo_assets.py` with these exact test functions:

- `test_official_asset_specs_are_immutable_and_same_origin_publishable`
- `test_acquire_rejects_status_redirect_host_length_and_content_type_drift`
- `test_acquire_writes_digest_receipt_without_path_query_header_or_raw_error`
- `test_validate_receipts_rejects_missing_extra_or_changed_bytes`
- `test_publish_rejects_review_root_inside_git_and_wrong_pages_root`
- `test_publish_writes_only_three_approved_paths_and_closed_manifest`
- `test_cli_diagnostics_are_fixed_and_do_not_echo_arguments`

Use a fake transport that returns deterministic JPEG-like, ONNX-like, and UTF-8 license bytes. The tests assert these exact immutable specs:

```python
AssetSpec(
    asset_id="boats-image",
    source_url="https://ultralytics.com/images/boats.jpg",
    expected_bytes=194_872,
    allowed_redirect_hosts=("ultralytics.com", "www.ultralytics.com", "github.com", "release-assets.githubusercontent.com"),
    public_relative_path="samples/boats.jpg",
)
AssetSpec(
    asset_id="obb-model",
    source_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx",
    expected_bytes=10_207_250,
    allowed_redirect_hosts=("github.com", "release-assets.githubusercontent.com"),
    public_relative_path="models/yolo26n-obb.onnx",
)
```

The license spec is exactly:

```python
AssetSpec(
    asset_id="ultralytics-license",
    source_url="https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE",
    expected_bytes=34_523,
    allowed_redirect_hosts=("raw.githubusercontent.com",),
    public_relative_path="third_party/ULTRALYTICS-AGPL-3.0.txt",
)
```

The closed generated `demo-model.json` has the required `output` contract object but no precomputed result data, `results`, `detections`, `boxes`, tensor values, or runtime URL field.

- [ ] **Step 2: Run the first focused test and observe the actual RED**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_demo_assets.py::test_official_asset_specs_are_immutable_and_same_origin_publishable -q
```

Expected RED: collection fails because `scripts.prepare_demo_assets` does not exist. Record this earliest reached failure only; do not claim later batch assertions were independently reached.

- [ ] **Step 3: Implement the minimum immutable spec and receipt types**

Add the dataclasses and exact `OFFICIAL_ASSETS`. Use `urllib.request` only in the production transport. A response is acceptable only when every redirect host is allowlisted, final status is 200, body length equals the immutable expectation, image bytes decode through Pillow with positive dimensions, license bytes decode as UTF-8 and include `GNU AFFERO GENERAL PUBLIC LICENSE` plus `Version 3`, and the model body stays within the exact expected length and 15 MiB hard ceiling.

Write receipt JSON with `json.dumps(receipt_payload, sort_keys=True, indent=2) + "\n"`. Store only hostnames, never redirect URLs/query strings. Map all exceptions to fixed codes:

```python
ERROR_CODES = {
    "network": "DEMO_ASSET_NETWORK",
    "redirect": "DEMO_ASSET_REDIRECT",
    "status": "DEMO_ASSET_STATUS",
    "length": "DEMO_ASSET_LENGTH",
    "digest": "DEMO_ASSET_DIGEST",
    "media": "DEMO_ASSET_MEDIA",
    "scope": "DEMO_ASSET_SCOPE",
    "receipt": "DEMO_ASSET_RECEIPT",
}
```

CLI errors print only `[FAIL] <fixed-code>` and exit 1.

- [ ] **Step 4: Implement exact publish validation without acquiring real assets**

`validate_receipts` recomputes each byte length/digest/media fact and rejects missing or extra files. `publish_assets` resolves both roots; rejects a review root within any Git worktree; requires `pages_root` to equal `<repo>/demo/web`; creates only `samples`, `models`, and `third_party`; copies the three reviewed byte streams; and writes `demo-model.json` plus `THIRD_PARTY_NOTICES.md` from receipt facts.

The generated manifest uses exact non-generated fields:

```json
{
  "schemaVersion": 1,
  "id": "ultralytics-yolo26n-obb-demo",
  "image": {"path": "samples/boats.jpg", "mediaType": "image/jpeg"},
  "model": {
    "path": "models/yolo26n-obb.onnx",
    "source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx",
    "release": "v8.4.0",
    "license": "AGPL-3.0-only"
  },
  "input": {"name": "images", "dims": [1, 3, 1024, 1024], "type": "float32", "channelOrder": "RGB", "normalization": "divide-by-255", "letterboxValue": 114},
  "output": {"name": "output0", "rowWidth": 7, "layout": ["cx", "cy", "w", "h", "confidence", "class", "angleRadians"]},
  "classes": ["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"],
  "defaultConfidence": 0.25,
  "notice": "THIRD_PARTY_NOTICES.md"
}
```

The tool inserts literal `bytes`, `sha256`, image `width`/`height`, and the exact license digest from the validated receipt. Tests assert the final complete key sets; the abbreviated JSON above is not copied verbatim until those generated fields are present.

- [ ] **Step 5: Run Task 1 GREEN and privacy scans**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_demo_assets.py -q
uv run --no-sync python -m pytest tests/test_release_check.py::test_redistributed_binaries_contain_no_absolute_user_paths -q
git diff --check
```

Expected GREEN: all fake-transport, scope, receipt, manifest, and fixed-diagnostic tests pass; no real network request or public asset is written.

- [ ] **Step 6: Review and commit Task 1**

Run the task-scoped spec review and quality review. Resolve every Critical/Important finding through the same implementer and fresh re-review. Then stage exactly:

```powershell
git add scripts/prepare_demo_assets.py tests/test_demo_assets.py
git diff --cached --check
git commit -m "test: add official demo asset preparation"
```

---

### Task 2: Acquire, browser-probe, and admit the exact public assets

**Files:**
- Modify: `scripts/pages_artifact_check.py`
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `scripts/browser_smoke.py`
- Create through the tested publisher: `demo/web/demo-model.json`
- Create through the tested publisher: `demo/web/samples/boats.jpg`
- Create through the tested publisher: `demo/web/models/yolo26n-obb.onnx`
- Create through the tested publisher: `demo/web/THIRD_PARTY_NOTICES.md`
- Create through the tested publisher: `demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt`

**Interfaces:**
- Consumes: Task 1 acquisition tool and the existing BYOM production path.
- Produces: reviewed exact public bytes, a closed manifest, a real-browser contract/detection admission result, and a transitional artifact verifier that still permits the old Synthetic UI only until Task 3.

- [ ] **Step 1: Write the Pages/model admission batch RED**

Add these exact tests to `tests/test_pages_artifact_check.py`:

- `test_pages_tree_requires_exact_reviewed_demo_image_model_manifest_and_license`
- `test_pages_tree_allows_only_the_manifest_bound_onnx_path`
- `test_pages_tree_rejects_model_image_digest_length_media_and_manifest_drift`
- `test_pages_tree_rejects_second_model_external_model_url_and_unlisted_binary`
- `test_pages_tree_rejects_precomputed_results_in_demo_manifest`
- `test_pages_tree_scans_the_reviewed_model_for_absolute_user_paths`

Mutations copy the site to `tmp_path` and alter one field or byte at a time. The exact reviewed ONNX is the only exception to the general model-suffix ban. Every other `.onnx`, `.pt`, `.engine`, `.tflite`, archive, DOTA visual/path, symlink, hard link, storage API, telemetry API, external runtime model URL, or unlisted file remains rejected.

- [ ] **Step 2: Run the batch and record only the earliest real RED**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py -q
```

Expected earliest RED: the current verifier/tree has no required `demo-model.json` and still categorically forbids `.onnx`. Record the first actual failure reported by pytest, not every later assertion hidden behind it.

- [ ] **Step 3: Add the transitional exact-model verifier rule**

Update the verifier so `models/yolo26n-obb.onnx` is allowed only when:

```python
relative == "models/yolo26n-obb.onnx"
and manifest["model"]["path"] == relative
and path.stat().st_size == manifest["model"]["bytes"] == 10_207_250
and sha256(path.read_bytes()).hexdigest() == manifest["model"]["sha256"]
```

Add directories `models`, `samples`, and `third_party`, plus the five new files, to the transitional exact inventory. Retain the old `showcase-fixture.js` and `fixtures/showcase.svg` only until Task 3 so the current public UI remains functional between commits. Parse `demo-model.json` as a closed schema and verify all source/path/license/class/input/output fields. Treat its exact official `model.source` URL as inert validated provenance and exclude only that field span from the runtime-origin scan; browser code may consume only `model.path`. Remove `fetch` from the blanket browser-API ban, but continue to reject XMLHttpRequest, WebSocket, EventSource, sendBeacon, storage, service workers, protocol-relative URLs, absolute remote resource URLs, and any model URL other than the exact same-origin relative path proven by browser tests.

- [ ] **Step 4: Acquire into a repo-external review directory**

Use a freshly created directory outside every repository/worktree:

```powershell
$reviewRoot = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-official-demo-assets-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $reviewRoot | Out-Null
uv run --no-sync python scripts/prepare_demo_assets.py acquire --review-root $reviewRoot
uv run --no-sync python scripts/prepare_demo_assets.py verify --review-root $reviewRoot
```

Expected output is fixed `[OK] DEMO_ASSETS_ACQUIRED` and `[OK] DEMO_ASSETS_VERIFIED`. Inspect the external receipt and bytes read-only. Require image length 194,872; model length 10,207,250; exact allowed redirect-host sequences; JPEG decode with positive dimensions; official release/tag/source identity; AGPL v3 license text; no local path, signed query, token, raw header, stack, or private marker. Record only the receipt SHA-256 in the SDD ledger.

- [ ] **Step 5: Publish exact bytes and freeze literal digests**

Run the tested publisher once:

```powershell
uv run --no-sync python scripts/prepare_demo_assets.py publish --review-root $reviewRoot --pages-root demo/web
```

Expected output: `[OK] DEMO_ASSETS_PUBLISHED`. Copy the resulting literal hashes, sizes, dimensions, and canonical text hashes into the verifier constants and mutation-test expectations via `apply_patch`. Do not edit binary bytes, sanitize metadata, recompress the image, or change filenames.

- [ ] **Step 6: Add and run a real browser admission probe through existing BYOM**

Extend `scripts/browser_smoke.py` with a dedicated command mode:

```powershell
uv run --no-sync python scripts/browser_smoke.py --admission-model demo/web/models/yolo26n-obb.onnx --admission-image demo/web/samples/boats.jpg
```

The mode serves the current `demo/web`, selects the exact repo-public model and image through the existing file inputs, waits up to 120 seconds, presses existing BYOM Detect, and asserts:

- ORT loads only after model selection;
- session contract is accepted;
- completion badge is `BYOM · LOCAL BROWSER INFERENCE`;
- runtime matches `^[1-9][0-9]* ms$`;
- at least one visible result row has class `ship` and finite confidence/width/height/angle;
- canvas pixels differ between base image and annotated result;
- UI and console contain no absolute path, filename, raw error, stack, or model metadata; and
- no origin other than loopback plus pinned jsDelivr occurs.

Expected admission GREEN. If it fails, stop Task 2 without modifying image, model, confidence, decoder, or output.

- [ ] **Step 7: Run Task 2 GREEN and exact artifact scan**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_demo_assets.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/browser_smoke.py --admission-model demo/web/models/yolo26n-obb.onnx --admission-image demo/web/samples/boats.jpg
git diff --check
```

Expected GREEN: exact transitional Pages tree passes, mutations fail closed, and real official BYOM inference detects at least one ship.

- [ ] **Step 8: Review and commit Task 2**

Run fresh task-scoped spec/quality reviews. Then stage exactly:

```powershell
git add scripts/pages_artifact_check.py tests/test_pages_artifact_check.py scripts/browser_smoke.py demo/web/demo-model.json demo/web/samples/boats.jpg demo/web/models/yolo26n-obb.onnx demo/web/THIRD_PARTY_NOTICES.md demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt
git diff --cached --check
git commit -m "feat: admit reviewed browser demo assets"
```

Keep the repo-external review directory until final acceptance; never commit it.

---

### Task 3: Replace Synthetic with the one-click real-image inference experience

**Files:**
- Create: `demo/web/demo-assets.js`
- Modify: `demo/web/index.html`
- Modify: `demo/web/app.js`
- Modify: `demo/web/style.css`
- Delete: `demo/web/showcase-fixture.js`
- Delete: `demo/web/fixtures/showcase.svg`
- Modify: `scripts/browser_smoke.py`
- Modify: `scripts/pages_artifact_check.py`
- Modify: `tests/test_pages_artifact_check.py`

**Interfaces:**
- Consumes: Task 2 exact manifest/image/model and existing `OBB` geometry API.
- Produces: initial real image, lazy real demo session, shared result rendering, original/result toggle, repeated-run reuse, progressive filters, and final no-synthetic Pages inventory.

- [ ] **Step 1: Write the real initial/success batch browser RED**

Replace Synthetic-first assertions in `scripts/browser_smoke.py` with these named scenario helpers:

- `assert_real_demo_initial(page, requests, messages) -> None`
- `exercise_real_demo_success(page, requests, messages) -> None`
- `assert_original_result_toggle(page, requests) -> None`
- `assert_demo_cached_filters(page, stubbed_run_counter) -> None`

The batch must assert exact user-visible behavior:

- first focusable item remains `跳至主要工作區`;
- heading intro is `先看真實航拍原圖，再按下 Detect；模型會在你的瀏覽器中找出帶方向的目標。`;
- exact approved notice precedes the first visible control;
- `samples/boats.jpg` is visible with label `原圖 · 尚未 Detect`;
- initial summary is count `0`, top/runtime `—`, mode `尚未 Detect`, provenance `官方範例 · 尚未執行`;
- primary button is `開始 Detect`;
- there is no Synthetic text/control and advanced BYOM is collapsed;
- initial requests contain no `demo-model.json`, ORT, WASM, or ONNX;
- click performs a real ORT session and yields the exact badge/provenance/numeric runtime, at least one ship row, polygon pixels, matching description, success status, `再次 Detect`, and `查看原圖`;
- view switching issues no request/run and toggles exact labels;
- repeated Detect on the real page issues no second model/runtime request and keeps a numeric runtime; a separate deterministic ORT-stub page proves exactly one additional `session.run`; and
- filters update the same cached canvas/table/summary/description without a new run, proven by the stubbed run counter while the separate real page preserves the genuine inference result.

In the same harness refactor, add `--scenario` with initial choices `full`, `real-demo-success`, and `stubbed-cache`, plus `--network-report <path>`. A network report contains only phase (`initial`, `detect`, `result`), origin, pathname, method, status, and count; it strips query/fragment and never records a local file path, signed redirect, response body, header, error, or stack. Task 4 extends the scenario choices with its failure cases.

- [ ] **Step 2: Run the browser smoke and record the earliest reached RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
```

Expected earliest RED: the current page still exposes `Synthetic Showcase` and does not contain the real-image-first introduction. Record only that assertion; later demo inference checkpoints are not yet reachable.

- [ ] **Step 3: Implement strict browser manifest and binary verification**

Create `demo-assets.js` with the existing UMD pattern used by `obb.js`. `validateManifest` compares exact key sets recursively, freezes its return value, and maps every validation issue to `DEMO_MODEL_CONTRACT`. `sha256Hex` uses `crypto.subtle.digest("SHA-256", arrayBuffer)`. `fetchVerifiedModel` uses:

```javascript
const response = await fetch(manifest.model.path, {
  credentials: "same-origin",
  cache: "no-store",
});
if (!response.ok) throw new Error("DEMO_MODEL_LOAD");
const bytes = await response.arrayBuffer();
if (bytes.byteLength !== manifest.model.bytes) throw new Error("DEMO_MODEL_INTEGRITY");
if (await sha256Hex(bytes) !== manifest.model.sha256) throw new Error("DEMO_MODEL_INTEGRITY");
return new Uint8Array(bytes);
```

The caller fetches `demo-model.json` with the same options only after Detect. No source URL is fetched.

- [ ] **Step 4: Replace the document hierarchy with exact approved copy**

Update `index.html` so the header is followed by:

```html
<section id="claimBoundary" class="claim-boundary" aria-labelledby="claimTitle">
  <h2 id="claimTitle">真實航拍範例 · 實際瀏覽器推論</h2>
  <p>下方是 Ultralytics 官方航拍範例。按下 Detect 後，頁面才會載入官方 OBB 模型並在你的瀏覽器中執行推論；影像不會上傳。這是操作示範，不是 accuracy、evaluation 或 latency benchmark。</p>
</section>
```

Within `main#mainContent`, place the intro, one `figure` with the exact same-origin image, primary `#demoDetectBtn`, hidden-until-result `#viewToggleBtn`, summary/result canvas/table, collapsed `<details id="byomPanel">` with summary `使用自己的模型與圖片（進階）`, and the existing BYOM controls. Filters move into a collapsed `#resultControls` shown only after success. Preserve skip link, theme color, semantic input names, Source/AGPL, and add `模型與素材來源` pointing to `THIRD_PARTY_NOTICES.md`. Load scripts in order `obb.js`, `demo-assets.js`, `app.js`; remove `showcase-fixture.js`.

- [ ] **Step 5: Implement one active source/session state and real demo success path**

Replace `mode` with `source` and remove `activateShowcase`, `clearSyntheticResult`, `showcaseBtn`, `OBB_SHOWCASE`, and every synthetic copy. On initial image load, assign the image to `state.image`, size the canvas, keep the unannotated `<img>` visible, and leave Detect enabled.

Add exactly `ensureDemoSession(generation)`, `runActiveInference(source, generation)`, `setResultView(view)`, and `resetToDemoOriginal()`; the following paragraphs define their complete state and side-effect contracts.

`ensureDemoSession` returns the current session only when `state.sessionKind === "demo"`. Otherwise it lazily loads ORT, fetches/validates the manifest and model, creates a candidate with `executionProviders: ["wasm"]`, validates input/output names, checks generation, atomically assigns `{session, sessionKind: "demo"}`, then releases the previous session. A stale candidate is released and never published.

`runActiveInference("demo", generation)` preprocesses the exact loaded image, measures only `await session.run({images: tensor})`, assigns `cached`, sets phase `result`, calls the existing shared renderer, sets exact badge/provenance/status, focuses `#resultTitle`, exposes result controls/toggle, and changes the primary label to `再次 Detect`.

`renderSummary` displays numeric runtime when either source is `demo` or `byom`, phase is `result`, cached exists, and elapsed time is finite. All other states display `—`.

`setResultView("original")` hides the annotated canvas and shows the original `<img>` without changing result state. `setResultView("result")` reverses it. It updates `aria-pressed` and exact button copy and performs no fetch, session creation, run, decode, or cache mutation.

- [ ] **Step 6: Style the simple primary flow and responsive states**

Use the existing visual language and variables. Desktop centers one large 16:9-compatible media frame with a compact summary; BYOM sits below. Mobile stacks media, primary action, summary, result controls, table, then advanced BYOM. Keep minimum 44 px targets, visible focus, square corners, no horizontal overflow, and no essential animation. The initial viewport must not expose file pickers or the class wall.

Use `.media-frame[data-view="original"]` and `.media-frame[data-view="result"]` to switch the `<img>` and canvas without duplicating pixels. Under reduced motion, remove result reveal animation.

- [ ] **Step 7: Remove Synthetic assets and close the final Pages inventory**

Delete the two public synthetic files. Update `scripts/pages_artifact_check.py` and tests so the final exact inventory requires `demo-assets.js` and the five reviewed asset/notice files, rejects any `showcase-fixture.js`, `fixtures/showcase.svg`, `OBB_SHOWCASE`, or `Synthetic Showcase`, and no longer allows the `fixtures` directory. Update canonical text hashes from the exact final files only after the browser GREEN.

- [ ] **Step 8: Run Task 3 GREEN**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest tests/test_pages_artifact_check.py tests/test_browser_parity.py -q
uv run --no-sync python scripts/pages_artifact_check.py
git diff --check
```

Expected GREEN: real image initial state, real model inference, repeat/toggle/filter, exact claims, no Synthetic public path, geometry parity, and exact Pages inventory all pass.

- [ ] **Step 9: Review and commit Task 3**

Complete task-scoped spec/quality review and any fix/re-review. Stage exactly:

```powershell
git add demo/web/demo-assets.js demo/web/index.html demo/web/app.js demo/web/style.css scripts/browser_smoke.py scripts/pages_artifact_check.py tests/test_pages_artifact_check.py
git add -u -- demo/web/showcase-fixture.js demo/web/fixtures/showcase.svg
git diff --cached --check
git commit -m "feat: add one-click real-image OBB demo"
```

---

### Task 4: Harden failures, transitions, privacy, and accessibility

**Files:**
- Modify: `demo/web/app.js`
- Modify: `demo/web/demo-assets.js`
- Modify: `demo/web/index.html` only if a real semantic RED requires it
- Modify: `demo/web/style.css` only if a real responsive/focus RED requires it
- Modify: `scripts/browser_smoke.py`

**Interfaces:**
- Consumes: Task 3 success path and existing fixed BYOM session lifecycle.
- Produces: deterministic safe recovery for every documented failure and complete demo/BYOM transition/accessibility behavior.

- [ ] **Step 1: Write the failure/transition batch RED**

Add these scenario helpers and call them from the full smoke:

- `exercise_demo_runtime_retry(browser, entry_url) -> None`
- `exercise_demo_manifest_and_integrity_failures(browser, entry_url) -> None`
- `exercise_demo_contract_inference_and_render_failures(browser, entry_url) -> None`
- `exercise_demo_byom_transitions(browser, entry_url) -> None`
- `assert_demo_accessibility_and_responsive(browser, entry_url) -> None`

Each scenario begins from a fresh context and deterministic route/session counter. Assert:

- failed ORT request: fixed runtime copy, original visible, all result state cleared, retry requests runtime once and succeeds;
- malformed manifest and changed/truncated model: session create count zero, exact model-safe copy, original visible, retry succeeds only with reviewed bytes;
- wrong candidate input/output names: candidate released, old valid session retained until a valid replacement, fixed model copy;
- image decode failure, session.run rejection, output schema mutation, and rotated-corner/render exception: no stale overlay/cache/table/description/numeric runtime/toggle/completed badge;
- second Detect made stale by switching to BYOM: late completion does not republish demo state;
- opening BYOM is network-silent; selecting either BYOM file clears demo result; model selection lazy-loads/reuses ORT, creates candidate before releasing old session, hides filename, and successful BYOM remains numeric;
- returning to demo shows original/no result; next Detect recreates or reuses only a valid demo session;
- every error has an enabled `重新 Detect` or documented BYOM recovery path;
- no UI/console string contains sentinel filename, absolute path, model metadata, response body, raw error, or stack; and
- no unexpected external origin occurs.

For the batch RED, run the first runtime-retry scenario and record its earliest actual failure only.

- [ ] **Step 2: Observe the runtime-retry RED**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario demo-runtime-retry
```

Expected RED: Task 3 does not yet fully clear/recover the documented runtime failure state or expose the exact retry behavior. Record the actual first failed assertion.

- [ ] **Step 3: Implement one fixed error table and atomic clearing**

Replace obsolete Synthetic copy with:

```javascript
const ERROR_COPY = Object.freeze({
  RUNTIME_LOAD: "偵測元件無法載入。請檢查網路後重試，或開啟進階 BYOM。",
  DEMO_MODEL_LOAD: "Demo 模型目前無法使用。請稍後重試，或開啟進階 BYOM。",
  DEMO_MODEL_INTEGRITY: "Demo 模型目前無法使用。請稍後重試，或開啟進階 BYOM。",
  DEMO_MODEL_CONTRACT: "Demo 模型目前無法使用。請稍後重試，或開啟進階 BYOM。",
  IMAGE_DECODE: "範例影像無法讀取。請重新整理後重試，或開啟進階 BYOM。",
  INFERENCE_RUN: "這次 Detect 未完成。請重試，或開啟進階 BYOM。",
  OUTPUT_SCHEMA: "這次 Detect 未完成。請重試，或開啟進階 BYOM。",
  RENDER_RESULT: "結果無法顯示。請重試，或開啟進階 BYOM。",
});
```

Add `clearActiveResult({keepOriginal: true})` as the sole presentation-clearing path. It never changes `state.generation`; every user action increments generation before starting async work, and async error handlers call it only after confirming their captured token is still current. The function sets `cached` null and phase error/reset, clears overlay/table/summary/description/toggle/result controls, sets runtime `—`, returns the media view to original, and never deletes the decoded official image. `reportFailure` accepts only a known fixed code and logs only `[AERIAL_OBB:<code>]`.

- [ ] **Step 4: Preserve atomic candidate-session replacement across demo and BYOM**

Factor both session paths through:

```javascript
async function installCandidateSession(candidate, kind, generation) {
  validateSessionContract(candidate);
  if (!isCurrentGeneration(generation)) {
    await candidate.release?.();
    return false;
  }
  const previous = state.session;
  state.session = candidate;
  state.sessionKind = kind;
  await previous?.release?.();
  return isCurrentGeneration(generation);
}
```

Candidate validation failure releases only the candidate. Generation-stale candidates and outputs are released/dropped. File selections use neutral labels only. Selecting either BYOM file clears the demo cached result immediately, but a valid active session is not discarded until its replacement is ready unless an explicit source reset requires release.

- [ ] **Step 5: Complete filtered descriptions and semantics**

Preserve the approved non-live description format for every filtered detection: class, confidence, centre x/y, width/height, angle. Filtered-empty text is exactly `目前沒有符合篩選條件的偵測結果。` Reset/error text contains no prior class/coordinates. The status region remains the sole aria-live region.

Assert the skip link is keyboard-first; notice is before the first visible control; theme-color is `#edf1f4`; model/image/confidence/class checkbox names are stable; Detect uses `aria-busy`; result focus moves only after explicit completion; original/result toggle exposes `aria-pressed`; headings are ordered; all controls have labels.

- [ ] **Step 6: Run focused scenarios to GREEN one at a time**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario demo-runtime-retry
uv run --no-sync python scripts/browser_smoke.py --scenario demo-integrity-failures
uv run --no-sync python scripts/browser_smoke.py --scenario demo-runtime-failures
uv run --no-sync python scripts/browser_smoke.py --scenario demo-byom-transitions
uv run --no-sync python scripts/browser_smoke.py --scenario accessibility-responsive
```

Expected GREEN: each isolated scenario passes with deterministic request/session/release counters and safe text.

- [ ] **Step 7: Run cumulative Task 4 GREEN**

Run:

```powershell
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest tests/test_browser_parity.py tests/test_pages_artifact_check.py tests/test_package_release.py -q
uv run --no-sync python scripts/pages_artifact_check.py
git diff --check
```

Expected GREEN: full real demo, all failures/recovery, BYOM, privacy, origins, accessibility, and artifact rules pass together.

- [ ] **Step 8: Review and commit Task 4**

Complete task-scoped spec/quality review and re-review. Stage only paths actually changed from the allowed list:

```powershell
git add demo/web/app.js demo/web/demo-assets.js scripts/browser_smoke.py
git add demo/web/index.html demo/web/style.css
git diff --cached --check
git commit -m "fix: harden demo recovery and BYOM transitions"
```

Before committing, remove unchanged staged paths with `git restore --staged <path>`; do not reset their working-tree content.

---

### Task 5: Align release evidence, clean export, CI, docs, and screenshot

**Files:**
- Modify: `scripts/clean_export_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `scripts/release_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `release/artifact-manifest.json`
- Modify: `release/evidence.json`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `demo/web/README.md`
- Modify: `CHANGELOG.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `.github/workflows/release-gates.yml`
- Modify: `tests/test_package_release.py`
- Modify: `tests/test_pages_workflow.py` only if its exact candidate assertion needs the new inventory
- Modify: `docs/assets/browser-workbench.png`

**Interfaces:**
- Consumes: Tasks 1–4 exact public artifact and behavior.
- Produces: honest distribution claims, exact binary inventory, archive policy, Linux CI path, and canonical real-result screenshot.

- [ ] **Step 1: Write the release/archive/CI batch RED**

Replace obsolete synthetic/code-only assertions with exact tests:

- `test_browser_demo_evidence_describes_one_live_public_model_and_advanced_byom`
- `test_artifact_manifest_binds_official_image_model_license_and_font_bytes`
- `test_release_allows_only_manifest_bound_public_demo_model`
- `test_release_rejects_second_model_dota_visual_and_unlisted_large_binary`
- `test_clean_export_requires_real_demo_assets_and_excludes_synthetic_public_fixture`
- `test_clean_export_allows_only_the_manifest_bound_demo_onnx`
- `test_ci_runs_live_real_model_browser_smoke_on_ubuntu_cpu`

The CI test keeps official action majors exactly:

```text
actions/checkout@v7
actions/setup-python@v7
actions/setup-node@v7
actions/upload-artifact@v7
actions/configure-pages@v6
actions/upload-pages-artifact@v5
actions/deploy-pages@v5
```

- [ ] **Step 2: Run the batch and record the earliest actual RED**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_release_check.py tests/test_clean_export.py tests/test_package_release.py tests/test_pages_workflow.py -q
```

Expected earliest RED: current evidence still declares `code-only-byom`/Synthetic and the archive still rejects every `.onnx`. Record the first reported assertion only.

- [ ] **Step 3: Make the exact manifest-bound binary exception**

Set `release/artifact-manifest.json` `distribution_mode` to `public-agpl-demo-model-plus-byom`, keep `commercial_use_cleared: false`, and add exact bundled entries for:

- `demo/web/samples/boats.jpg`;
- `demo/web/models/yolo26n-obb.onnx`;
- `demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt`;
- the existing reviewed IBM font; and
- `docs/assets/browser-workbench.png`, described as a local interface screenshot derived from the exact official sample and public demo-model output, not ground truth or model-quality evidence.

Each entry uses actual bytes/digest from Task 2, immutable public source/release, modification status `unmodified`, license, DOTAv1 training disclosure for the model, restrictions, and non-endorsement. Keep the historical old `demo/web/yolo26n-obb.onnx` exclusion record because it is a different path/digest/provenance; do not relabel it as the new model.

Change archive/release policy so only `demo/web/models/yolo26n-obb.onnx` is permitted, and only when listed with matching digest/size in `bundled_third_party_artifacts`. Manifest listing cannot waive the exact path constant. All other model suffixes remain blocked before archive acceptance.

- [ ] **Step 4: Update browser evidence and public documentation**

Set `release/evidence.json` browser facts to:

```json
{
  "distribution_mode": "public-demo-model-plus-byom",
  "model": "Ultralytics yolo26n-obb.onnx v8.4.0 (public demo only)",
  "runtime": "ONNX Runtime Web 1.20.1 WASM",
  "runtime_load": "lazy-on-demo-detect-or-byom-model-selection",
  "model_bundled": true,
  "live_demo_enabled": true,
  "synthetic_showcase_enabled": false,
  "sample_image": "demo/web/samples/boats.jpg",
  "demo_model": "demo/web/models/yolo26n-obb.onnx",
  "represents_fine_tuned_medium_accuracy": false,
  "represents_t4_latency": false
}
```

Complete the object with the existing exact runtime URL/integrity/CORS, layout/accessibility facts, source-file inventory, and limitations. Update both READMEs and `demo/web/README.md` to explain: original image → Detect → local result → original/result switch; first Detect downloads the same-origin official model plus pinned jsDelivr runtime; no upload; not accuracy/evaluation/T4 latency evidence; BYOM advanced; model/image sources and AGPL/DOTAv1 restrictions. Keep historical benchmark/training claims unchanged. Replace the `Unreleased` Synthetic entry in `CHANGELOG.md`; do not rewrite released rc.1/rc.2 history.

Update root `THIRD_PARTY_NOTICES.md` consistently with public Pages notices and literal manifest digests. Do not claim commercial clearance.

- [ ] **Step 5: Update CI naming and retain exact release/deploy separation**

Rename the browser job display from `Synthetic browser smoke / Ubuntu CPU` to `Live demo browser smoke / Ubuntu CPU`, and the execution step to `Exercise the real-image browser demo and BYOM safety paths`. Keep `CUDA_VISIBLE_DEVICES: "-1"`, locked dependencies, Playwright Chromium install, exact action majors, Pages candidate upload, manual-only deploy workflow, and no About mutation. Do not dispatch either workflow.

- [ ] **Step 6: Generate the canonical screenshot from the exact artifact**

Run the real full smoke so its screenshot state is: exact notice visible, official real image annotated after genuine demo inference, exact badge/provenance/numeric runtime, no advanced BYOM ready labels, no local filename/path, and no private model.

```powershell
uv run --no-sync python scripts/browser_smoke.py --screenshot docs/assets/browser-workbench.png
```

Inspect visible pixels and PNG metadata. Add its actual bytes/digest to the required `docs/assets/browser-workbench.png` manifest entry and describe it as a local screenshot of official sample plus official model output, not ground truth or accuracy evidence.

- [ ] **Step 7: Run Task 5 GREEN**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_release_check.py tests/test_clean_export.py tests/test_package_release.py tests/test_pages_workflow.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/browser_smoke.py
git diff --check
```

Expected GREEN: claims, manifest bytes, model exception, workflow majors, real browser path, exact Pages inventory, privacy, and no unsupported benchmark/commercial claim agree.

- [ ] **Step 8: Review and commit Task 5**

Complete task-scoped reviews and fixes. Stage exactly changed paths from the task list, verify the binary screenshot intentionally changed, then commit:

```powershell
git add scripts/clean_export_check.py tests/test_clean_export.py scripts/release_check.py tests/test_release_check.py release/artifact-manifest.json release/evidence.json README.md README.en.md demo/web/README.md CHANGELOG.md THIRD_PARTY_NOTICES.md .github/workflows/release-gates.yml tests/test_package_release.py tests/test_pages_workflow.py docs/assets/browser-workbench.png
git diff --cached --check
git commit -m "docs: align release evidence with live demo"
```

Remove unchanged paths from the index before commit.

---

### Task 6: Complete local acceptance and leave an operable preview

**Files:**
- Modify: none unless a scoped review finding is resolved through its own RED/GREEN/fix/re-review cycle.
- Evidence: repo-external desktop/mobile screenshots, network log, asset receipt, clean archive, and review reports only.

**Interfaces:**
- Consumes: all committed Tasks 1–5.
- Produces: clean branch, exact local preview URL, final verification evidence, and branch-readiness verdict; no remote action.

- [ ] **Step 1: Run unit/static contracts separately**

Run:

```powershell
uv run --no-sync python -m pytest tests/test_demo_assets.py tests/test_browser_parity.py tests/test_pages_artifact_check.py tests/test_release_check.py tests/test_clean_export.py tests/test_package_release.py tests/test_pages_workflow.py -q
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: asset, geometry, artifact, release, archive, CI, and Pages contracts pass with zero failures.

- [ ] **Step 2: Run the complete real browser smoke with repo-external evidence**

```powershell
$acceptanceRoot = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-live-demo-acceptance-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $acceptanceRoot | Out-Null
uv run --no-sync python scripts/browser_smoke.py --screenshot (Join-Path $acceptanceRoot 'desktop.png') --mobile-screenshot (Join-Path $acceptanceRoot 'mobile.png') --network-report (Join-Path $acceptanceRoot 'network.json')
```

Expected GREEN: real initial image, exact one-click inference, at least one ship, numeric runtime, repeat/session reuse, original/result switch, filters, safe failures/retry, BYOM transitions, privacy/origins, desktop/mobile/accessibility, and no Synthetic public experience.

- [ ] **Step 3: Run the complete regression and repository/release gates**

```powershell
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected GREEN: pytest has zero failures and all three scripts print their `[OK]` verdict.

- [ ] **Step 4: Perform artifact, license, origin, and privacy scans separately**

Run:

```powershell
uv run --no-sync python scripts/release_check.py
rg -n -I -i "Synthetic Showcase|OBB_SHOWCASE|showcase-fixture|fixtures/showcase|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}" demo/web README.md README.en.md release THIRD_PARTY_NOTICES.md
uv run --no-sync python scripts/pages_artifact_check.py
```

Expected: no Synthetic/private-token hit in current public text; exact model/image/license digests and notices pass. Review every allowed historical mention outside current public UI in context. Compare network report origins against loopback plus the pinned jsDelivr runtime only; initial phase must be loopback-only, and ONNX request must start only after Detect.

- [ ] **Step 5: Commit any approved fix before strict clean export**

If Steps 1–4 expose a defect, use the original task implementer when available, write/observe a focused RED, apply one minimal fix, run task-scoped reviews, commit exact paths, and repeat Steps 1–4. Do not stack speculative workarounds. If no defect exists, make no commit.

- [ ] **Step 6: Run strict clean export with browser verification**

Require a clean committed HEAD first. Write the archive outside the repository:

```powershell
$cleanExport = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-live-demo-' + (git rev-parse --short HEAD) + '.zip')
if (Test-Path -LiteralPath $cleanExport) { throw 'Refusing to overwrite existing clean-export evidence' }
uv run --no-sync python scripts/clean_export_check.py --output $cleanExport
```

Do not use `--skip-browser`. Expected GREEN: committed snapshot, full tests, repo/release/Pages checks, real browser smoke, package build/install/version import, exact manifest-bound model allowance, and no unlisted model/private/DOTA artifact.

- [ ] **Step 7: Serve the exact verified UI for owner operation**

Start a dedicated loopback server from the committed `demo/web` tree and leave it running:

```powershell
uv run --no-sync python -m http.server 8765 --bind 127.0.0.1 --directory demo/web
```

Open `http://127.0.0.1:8765/` in the in-app browser. Verify at 1280×720 and 390×844: real original visible before Detect, notice and CTA understandable, first Detect completes, result/toggle/filters work, BYOM remains secondary, no horizontal overflow, Source/AGPL/asset notice links are readable, keyboard focus works, 200% zoom works, reduced motion works, console has no unexpected errors, and network origins match the acceptance report. This is the preview the owner will use for UI feedback.

- [ ] **Step 8: Run final whole-branch review and one allowed fix wave**

Dispatch the most capable fresh reviewer against the approved spec, this plan, the base diff, exact artifact, screenshots, and verification logs. Resolve any Critical/Important finding through one scoped TDD fix wave and fresh re-review. Record Minors separately; do not silently expand scope.

- [ ] **Step 9: Verify branch hygiene and readiness**

Run:

```powershell
git status --short
git diff --check 24039db9e07e327b55433086241ac574430c4531...HEAD
git log --oneline 24039db9e07e327b55433086241ac574430c4531..HEAD
git diff --name-status 24039db9e07e327b55433086241ac574430c4531...HEAD
git ls-files demo/web
```

Expected: clean worktree, no whitespace errors, only approved docs/tool/tests/UI/assets/evidence/workflow paths, small task commits, no superseded sample-preparation tool, no public Synthetic files, one exact model, one exact image, and no unresolved Critical/Important review finding.

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch` only to confirm readiness and report options. Do not select or execute an integration/remote option.

---

## Remote Gates A–E — separately controlled and not authorized

1. **Remote Gate A — preflight and candidate PR:** compare origin/main, checks, branch/PR races and reviewed HEAD; only a later explicit authorization may allow non-force push and one PR.
2. **Remote Gate B — candidate artifact and integration:** download the exact CI candidate, verify model/image/text byte equality, serve it repo-external, repeat real demo/privacy/license/desktop/mobile review, then use only an authorized repository-supported merge method.
3. **Remote Gate C — Pages deployment:** configure/dispatch only from the exact reviewed merged-main SHA after main CI succeeds and only under fresh written authorization.
4. **Remote Gate D — live review:** verify HTTPS/assets, initial zero-model requests, real Detect and numeric runtime, same-origin ONNX, pinned jsDelivr, failures/recovery, BYOM, accessibility, licenses, privacy, and exact deployed SHA.
5. **Remote Gate E — About and Portfolio Control receipt:** change About and record the Portfolio Control receipt only after an independently passing live review and separate authorization.

No local task authorizes push, PR, merge, workflow dispatch, Pages enable/deploy/disable, About, Hugging Face, release/tag, visibility, remote branch deletion, or worktree cleanup.

## Plan self-review

- **Spec coverage:** Task 1 creates the immutable, privacy-safe acquisition boundary. Task 2 admits the one official image/model/license, verifies real browser compatibility, and opens only one exact binary exception. Task 3 implements the real-image-first UI, lazy real inference, shared pipeline, view comparison, filters, and removes Synthetic. Task 4 covers every safe failure, atomic transition, BYOM lifecycle, privacy and accessibility state. Task 5 aligns release/archive/CI/docs/screenshot evidence. Task 6 separates unit, browser, full regression, artifact/license/privacy/origin, clean export, local desktop/mobile operation, and broad review.
- **Generated facts:** the plan contains no invented digest, image dimension, output count, or detection tensor. Task 2 deterministically obtains those literal facts from the exact immutable sources, freezes them before publication, and stops on upstream length/identity drift.
- **Type/name consistency:** `AssetSpec`, `AssetReceipt`, `DemoAssets.validateManifest`, `fetchVerifiedModel`, `state.source`, `state.sessionKind`, `state.cached`, exact paths, exact badge/provenance/copy, and scenario names remain consistent across tasks.
- **TDD honesty:** every batch records only its earliest actually reached RED; later checkpoints are rerun after the earlier block becomes reachable. The successful production path uses the exact reviewed ONNX in real Chromium/ORT and cannot be satisfied by mocks, greps, or precomputed output.
- **License/privacy:** one official AGPL model and one official sample are separately inventoried, unchanged and digest-bound; DOTAv1 training origin is disclosed without commercial-clearance claims; private models/files and DOTA pixels remain prohibited.
- **Scope:** one image, one model, one primary Detect flow, one original/result switch, advanced BYOM, required evidence and no gallery/webcam/server/framework/telemetry/remote mutation.
- **Workspace safety:** implementation starts from the exact approved base in a new worktree and cherry-picks only the two current docs commits, preserving all rejected-plan commits and dirty files in the old worktree without deletion or reset.
- **Stop conditions:** base/commit drift, existing worktree collision, source/redirect/length/digest/license drift, model contract or real-detection failure, unexpected origin, privacy leak, artifact mismatch, unresolved Critical/Important review finding, or remote race stops execution instead of weakening a gate.

# Aerial OBB Privacy-sanitized Real-image Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a one-click browser OBB demo that shows the exact official aerial image before Detect and performs genuine local inference with a structurally identical, privacy-sanitized derivative of the exact official YOLO26n-OBB model.

**Architecture:** Acquire the immutable upstream image/model/license only into a repository-external review root, remove exactly one admitted `ModelProto.metadata_props[0].value` entry with a pinned ONNX tool, and publish only the deterministic derivative after structural, privacy, license, and byte-identical browser-output gates pass. The static UI lazy-loads the same-origin derivative and pinned ONNX Runtime Web only after Detect, then shares the existing preprocess/decode/filter/rotated-corner/render pipeline with the retained advanced BYOM path.

**Tech Stack:** Python 3.11, `onnx==1.22.0`, locked protobuf, pytest, Playwright Chromium, static HTML/CSS/JavaScript, Web Crypto SHA-256, ONNX Runtime Web 1.20.1 WASM, Pillow, `uv`, Git.

**Plan Status:** Written spec approved. The owner delegated execution-mode choice; the recommended subagent-driven path is preselected after this documentation commit.

## Global Constraints

- Implementation branch is `feat/pages-live-real-image-demo` in `<repository-root>\.worktrees\aerial-obb-live-real-image-demo`; prerequisite HEAD is the plan commit whose parent is spec commit `582a88960b9a699d3adedfd3a918b09f8e5b128b`.
- Completed Task 1 commits `64aad4f`, `601d91a`, `214bf90`, and `8116e79` are authoritative; do not restart or rewrite their reviewed acquisition boundary.
- The immutable upstream model is `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx`, 10,207,250 bytes, SHA-256 `02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38`.
- The immutable image is `https://ultralytics.com/images/boats.jpg`, 194,872 bytes, 1920×1080 JPEG, SHA-256 `8c5ada657cf8110a9f8aaac954c1dd96cde0187315b581276c32b0d1863e756f`.
- The immutable license is `https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE`, 34,523 bytes, SHA-256 `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
- The source ONNX remains outside every repository/worktree and is never staged, committed, archived, screenshotted, or published. The only public model path is `demo/web/models/yolo26n-obb-privacy-sanitized.onnx`.
- Sanitization removes exactly the complete `ModelProto.metadata_props[0]` entry whose value matches the absolute-user-path rule. Zero, multiple, wrong-index, wrong-field, or post-sanitize matches stop the task.
- Sanitization may not optimize, simplify, quantize, retrain, rename graph values, rewrite operators, change tensors, change opsets, add external data, purge all metadata, or replace a substring.
- Source and derivative clones with `metadata_props` cleared must serialize identically; initializers and every non-metadata field remain identical. The same browser ORT run must yield byte-identical complete `output0` Float32 bytes.
- The real image/model admission must retain `images [1,3,1024,1024]`, `output0 [1,N,7]`, finite rows, the existing 15-class order, and at least one `ship` at confidence `0.25`. No threshold/model/image/tensor/box rescue is allowed.
- Initial page load requests no manifest, ORT, WASM, or ONNX. First `開始 Detect` performs genuine browser inference; completed runs alone show numeric milliseconds. Loading/reset/error/no-cache shows `—`.
- Public provenance is exactly `Ultralytics YOLO26n-OBB · privacy-sanitized AGPL derivative`; notices say one non-inference metadata entry was removed on 2026-08-31 and graph/weights were verified unchanged.
- No DOTA image/annotation/archive/derived render, private or owner model, precomputed detection, alternate mirror, proxy, hosted inference, upload, analytics, telemetry, storage, cookie, service worker, or new frontend framework is allowed.
- UI, console, network reports, screenshots, receipts, reports, and commits never include the removed metadata, local path/filename, private model identity, signed query, raw header/body/exception/stack, raw tensor, token, or browser-profile path.
- Follow strict TDD for every behavior change. Record only the earliest assertion genuinely reached in a batch RED; do not claim later blocked assertions as separate RED evidence.
- Use exact staging per task, small non-amended commits, a fresh implementer per task, fresh task-scoped spec and quality review, original-implementer fix rounds, and a broad final review.
- Existing dirty worktrees and the former PR #6 worktree remain untouched. Do not reset, delete, clean, or reuse them.
- Local files, tests, asset preparation, browser preview, evidence, and commits are authorized. Remote Gates A–E are not: no push, PR, merge, workflow dispatch, Pages enable/disable/deploy, About, Hugging Face, release/tag, visibility, force push, remote branch deletion, or worktree cleanup.
- GitHub Actions majors remain `actions/checkout@v7`, `actions/setup-python@v7`, `actions/setup-node@v7`, `actions/upload-artifact@v7`, `actions/configure-pages@v6`, `actions/upload-pages-artifact@v5`, and `actions/deploy-pages@v5`.

---

## File Responsibility Map

- `scripts/sanitize_demo_model.py`: pure ONNX inspection/sanitization, structural comparison, deterministic serialization, closed receipt, fixed CLI diagnostics.
- `tests/test_sanitize_demo_model.py`: in-memory ONNX fixtures and sanitizer TDD; no real model bytes.
- `scripts/prepare_demo_assets.py`: immutable acquisition, Unicode-safe Git worktree discovery, admitted-review orchestration, exact public publication.
- `tests/test_demo_assets.py`: acquisition/publish contracts, admitted derivative layout, Unicode subprocess regression.
- `scripts/model_parity_smoke.py`: local-only Playwright/ORT source-versus-derivative raw-output parity and public-safe report.
- `tests/test_model_parity_smoke.py`: parity harness argument/report/privacy and failure behavior with deterministic local fixtures/mocks; real-model success cannot be mocked.
- `demo/web/demo-assets.js`: frozen manifest validation and digest-verified same-origin derivative fetch.
- `demo/web/index.html`, `style.css`, `app.js`: real-image-first UI, lazy demo inference, BYOM, state/error/accessibility behavior.
- `scripts/browser_smoke.py`: genuine demo, deterministic failure, BYOM, responsive, accessibility, privacy, and origin scenarios.
- `scripts/pages_artifact_check.py`, `scripts/repo_check.py`, `scripts/release_check.py`, `scripts/clean_export_check.py`: exact derivative exception and otherwise closed public/release boundary.
- `release/artifact-manifest.json`, `release/evidence.json`, third-party notices, READMEs, changelog, workflows, and `docs/assets/browser-workbench.png`: truthful public/release evidence.

---

### Task 1: Deterministic ONNX Sanitizer and Unicode-safe Acquisition

**Files:**
- Create: `scripts/sanitize_demo_model.py`
- Create: `tests/test_sanitize_demo_model.py`
- Modify: `scripts/prepare_demo_assets.py:369-380`
- Modify: `tests/test_demo_assets.py:281-306,347-384`
- Modify: `pyproject.toml:18-23`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: exact source model identity from Global Constraints; reviewed `checked_child`, `is_reparse_point`, and fixed-diagnostic patterns from `prepare_demo_assets.py`.
- Produces:

```python
@dataclass(frozen=True)
class SanitizationReceipt:
    source_bytes: int
    source_sha256: str
    output_bytes: int
    output_sha256: str
    onnx_version: str
    protobuf_version: str
    removed_metadata_entries: int
    modified_field: str
    modification_date: str
    structural_equivalent: bool
    checker_passed: bool
    privacy_passed: bool
    deterministic: bool

def sanitize_model_bytes(source: bytes, *, expected_source_sha256: str) -> tuple[bytes, SanitizationReceipt]: ...
def sanitize_official_model(source: Path, output: Path, receipt: Path) -> SanitizationReceipt: ...
def validate_sanitized_model(source: Path, output: Path, receipt: Path) -> SanitizationReceipt: ...
```

- CLI success is exactly `[OK] DEMO_MODEL_SANITIZED` or `[OK] DEMO_MODEL_VERIFIED`; failure is `[FAIL] <fixed-code>` with exit 1.

- [ ] **Step 1: Add the locked tooling dependency**

Use `apply_patch` to add only `onnx==1.22.0` to `[dependency-groups].dev`, then run:

```powershell
uv lock
uv sync --frozen --no-install-project
uv run --no-sync python -c "import onnx, google.protobuf; print(onnx.__version__); print(google.protobuf.__version__)"
```

Expected: ONNX is exactly `1.22.0`; record the lock-resolved protobuf version in the task report, not as an unverified constant.

- [ ] **Step 2: Write the sanitizer batch RED**

Create these exact tests with in-memory `onnx.helper` models and sentinel path fragments assembled at runtime so repository scanners do not contain a literal absolute path:

```python
def test_sanitize_requires_exact_digest_and_one_admitted_metadata_field() -> None: ...
def test_sanitize_removes_only_metadata_entry_zero_and_preserves_all_other_fields() -> None: ...
def test_sanitize_rejects_zero_multiple_wrong_index_and_nonmetadata_matches() -> None: ...
def test_sanitize_is_deterministic_checker_valid_private_and_transactional(tmp_path: Path) -> None: ...
def test_validate_sanitized_model_rejects_graph_tensor_opset_receipt_and_privacy_mutations(tmp_path: Path) -> None: ...
def test_sanitizer_cli_diagnostics_are_fixed_and_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None: ...
```

Assertions separately label and prove the graph, initializer raw SHA, input/output, opset, metadata-preservation, exact-key receipt, deterministic rerun, no external data, and no-partial-output contracts. The wrong-index case creates a safe metadata entry before the sensitive entry and must fail rather than removing index 1.

- [ ] **Step 3: Observe the earliest sanitizer RED**

```powershell
uv run --no-sync python -m pytest tests/test_sanitize_demo_model.py -q
```

Expected earliest failure: import error for `scripts.sanitize_demo_model`. Record only that failure for the batch.

- [ ] **Step 4: Implement the minimal pure sanitizer**

Use the official ONNX API and deterministic protobuf serialization. The core must follow this structure:

```python
SOURCE_SHA256 = "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"
MODIFIED_FIELD = "ModelProto.metadata_props[0].value"
MODIFICATION_DATE = "2026-08-31"

def sanitize_model_bytes(source: bytes, *, expected_source_sha256: str) -> tuple[bytes, SanitizationReceipt]:
    require_source_digest(source, expected_source_sha256)
    model = onnx.load_model_from_string(source)
    matches = inspect_sensitive_fields(model)
    require_exact_match(matches, MODIFIED_FIELD)
    original_without_metadata = clone_without_metadata(model)
    del model.metadata_props[0]
    derived = model.SerializeToString(deterministic=True)
    validated = onnx.load_model_from_string(derived)
    onnx.checker.check_model(validated)
    require_structural_identity(original_without_metadata, clone_without_metadata(validated))
    require_private_bytes(derived)
    return derived, make_receipt(source, derived)
```

Every exception maps to a fixed internal code. Never include a protobuf value, file argument, URL, or exception string in stdout/stderr.

- [ ] **Step 5: Add and observe the Unicode Git subprocess RED**

Add:

```python
def test_git_worktree_roots_decodes_utf8_bytes_independent_of_host_locale(monkeypatch: pytest.MonkeyPatch) -> None: ...
```

The fake `subprocess.run` returns UTF-8 bytes containing a non-ASCII worktree path. Run only this test. Expected RED: current `text=True` contract receives/handles the wrong type or decodes through the host code page.

- [ ] **Step 6: Make Git decoding explicitly UTF-8**

Change `_git_worktree_roots` to request bytes and decode once:

```python
completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
porcelain = completed.stdout.decode("utf-8", errors="strict")
```

Map decode/process failures to the existing fixed path error without exposing output.

- [ ] **Step 7: Run Task 1 GREEN and privacy checks**

```powershell
uv run --no-sync python -m pytest tests/test_sanitize_demo_model.py tests/test_demo_assets.py -q
uv run --no-sync python -m pytest tests/test_release_check.py::test_redistributed_binaries_contain_no_absolute_user_paths -q
uv run --no-sync python -m compileall -q scripts tests
git diff --check
```

Expected: all pass; no real upstream model/image/license is read or written in this task.

- [ ] **Step 8: Review and commit Task 1**

Run fresh task-scoped spec and quality reviews. Resolve findings through original-implementer RED/GREEN fix rounds. Stage exactly:

```powershell
git add pyproject.toml uv.lock scripts/sanitize_demo_model.py scripts/prepare_demo_assets.py tests/test_sanitize_demo_model.py tests/test_demo_assets.py
git diff --cached --check
git commit -m "feat: add deterministic demo model sanitizer"
```

---

### Task 2: Acquire, Sanitize, Prove Parity, and Admit Exact Public Assets

**Files:**
- Modify: `scripts/prepare_demo_assets.py`
- Modify: `tests/test_demo_assets.py`
- Create: `scripts/model_parity_smoke.py`
- Create: `tests/test_model_parity_smoke.py`
- Modify: `scripts/pages_artifact_check.py:20-109,355-448`
- Modify: `tests/test_pages_artifact_check.py`
- Create: `demo/web/samples/boats.jpg`
- Create: `demo/web/models/yolo26n-obb-privacy-sanitized.onnx`
- Create: `demo/web/demo-model.json`
- Create: `demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt`
- Create: `demo/web/third_party/yolo26n-obb-privacy-sanitization.json`
- Create: `demo/web/THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: Task 1 `sanitize_official_model` and `validate_sanitized_model`; exact upstream facts; existing BYOM preprocess/session behavior.
- Produces:

```python
@dataclass(frozen=True)
class AdmittedAssets:
    receipts: dict[str, AssetReceipt]
    sanitization: SanitizationReceipt

def validate_source_receipts(review_root: Path) -> dict[str, AssetReceipt]: ...
def validate_admitted_assets(review_root: Path) -> AdmittedAssets: ...
def publish_assets(review_root: Path, pages_root: Path) -> None: ...
def run_parity(review_root: Path, report: Path) -> None: ...
```

- External review layout is exactly `samples/boats.jpg`, `models/yolo26n-obb.onnx`, `licenses/ULTRALYTICS-AGPL-3.0.txt`, `receipt.json`, `sanitized/yolo26n-obb-privacy-sanitized.onnx`, and `sanitized/sanitization-receipt.json`.
- Public `demo-model.json` is closed schema containing derivative path/digest/size, source digest/release, input/output/class contract, image path/digest/dimensions, sanitization record path, license path, and exact provenance.

- [ ] **Step 1: Write admitted-layout and closed-manifest RED tests**

Add:

```python
def test_validate_admitted_assets_requires_exact_source_and_sanitized_layout(tmp_path: Path) -> None: ...
def test_publish_never_copies_upstream_model_and_writes_exact_derivative_set(tmp_path: Path) -> None: ...
def test_manifest_and_sanitization_record_are_closed_and_privacy_safe(tmp_path: Path) -> None: ...
def test_pages_tree_admits_exact_derivative_assets_during_local_ui_transition(tmp_path: Path) -> None: ...
def test_pages_tree_rejects_upstream_model_digest_alternate_model_and_binary_path_leak(tmp_path: Path) -> None: ...
```

Run each independently so missing APIs/assets produce honest RED evidence instead of one masking batch assertion.

Task 2 is an explicitly local, non-deployable transition: the verifier allows the already reviewed current
Synthetic UI files together with the new exact derivative assets so the branch remains testable while no
remote gate is authorized. Task 3 must delete Synthetic and tighten the verifier to the final real-demo-only
inventory in the same commit as the new UI; Task 2 must not claim the transitional tree is the final Pages
candidate.

- [ ] **Step 2: Implement admitted review and publish orchestration**

The CLI sequence becomes exactly:

```text
acquire  -> [OK] DEMO_ASSETS_ACQUIRED
sanitize -> [OK] DEMO_MODEL_SANITIZED
verify   -> [OK] DEMO_ASSETS_VERIFIED
publish  -> [OK] DEMO_ASSETS_PUBLISHED
```

`publish` validates source and derivative receipts, rejects unknown/stale/link/hard-link members, and stages
all six public paths before replacing any destination. Each destination leaf is replaced atomically; if any
replacement fails, the complete prior six-file set is restored. This offline local preparation transaction
does not promise lock-free, all-at-once visibility to a concurrent reader across the six fixed paths. The
consistent deployment boundary is the completed Git tree and its verified Pages artifact; no workflow deploys
from the local mutation loop. It never copies `models/yolo26n-obb.onnx`.

- [ ] **Step 3: Write the parity harness RED**

Create exact tests:

```python
def test_parity_report_schema_never_contains_paths_metadata_or_tensor_values(tmp_path: Path) -> None: ...
def test_parity_rejects_shape_name_type_byte_and_ship_mismatch(tmp_path: Path) -> None: ...
def test_parity_cli_uses_fixed_diagnostics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None: ...
```

The report schema is exactly `{runtime, input, output, output_bytes_equal, detections_equal, accepted_ship, verdict}` with public version/shape/boolean/count facts only. Expected batch RED: missing `scripts.model_parity_smoke`.

- [ ] **Step 4: Implement local-only real ORT parity**

Use Playwright Chromium and opaque loopback routes `/source-model`, `/derived-model`, and `/sample-image`. Load exact `demo/web/obb.js` and pinned ORT 1.20.1. Both sessions receive the same preprocessed Float32 input. Compare output name/type/shape, then compare full `Float32Array` bytes before decoding. Decode/sort both through the same OBB functions, require exact detections and at least one `ship` at `0.25`, release both sessions, and write the closed report atomically.

The script accepts `--review-root` and `--report`, but argparse/runtime failures print only fixed codes and never echo arguments.

- [ ] **Step 5: Acquire into a fresh external root without encoding overrides**

```powershell
$reviewRoot = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-official-demo-assets-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $reviewRoot | Out-Null
uv run --no-sync python scripts/prepare_demo_assets.py acquire --review-root $reviewRoot
```

Do not set `PYTHONUTF8`. Require exact lengths, redirect hosts, source SHA values, JPEG decode, ONNX media, AGPL text, and private receipt. Any drift stops the task.

- [ ] **Step 6: Sanitize twice and freeze deterministic facts**

Run the sanitizer into two fresh external output directories, require byte-identical output and receipt facts, then keep one admitted derivative under the review layout:

```powershell
uv run --no-sync python scripts/prepare_demo_assets.py sanitize --review-root $reviewRoot
uv run --no-sync python scripts/prepare_demo_assets.py verify --review-root $reviewRoot
```

Record only source/derivative hashes, sizes, tool versions, booleans, match count/field, and public source identities in the SDD report. Never record the removed key/value or external path.

- [ ] **Step 7: Observe the real parity RED, then run GREEN**

First run the parity CLI before its complete real route/session logic and record the actual fixed failure. Complete only the minimum missing behavior, then run:

```powershell
$parityReport = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-parity-' + [guid]::NewGuid().ToString('N') + '.json')
uv run --no-sync python scripts/model_parity_smoke.py --review-root $reviewRoot --report $parityReport
```

Expected GREEN: exact output bytes equal, exact detections equal, at least one ship, no path/metadata/tensor in the report. Incompatibility is a stop condition; do not seek another model.

- [ ] **Step 8: Publish exact admitted bytes and freeze verifier literals**

```powershell
uv run --no-sync python scripts/prepare_demo_assets.py publish --review-root $reviewRoot --pages-root demo/web
```

Use `apply_patch` to freeze generated derivative size/SHA, sanitization-record canonical-text SHA, manifest canonical-text SHA, notice/license SHA, and image facts into the verifier/tests. Do not edit/recompress/re-export any asset.

- [ ] **Step 9: Run Task 2 GREEN**

```powershell
uv run --no-sync python -m pytest tests/test_demo_assets.py tests/test_sanitize_demo_model.py tests/test_model_parity_smoke.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/model_parity_smoke.py --review-root $reviewRoot --report $parityReport
uv run --no-sync python scripts/pages_artifact_check.py
git diff --check
```

Inspect `git status --short`; only listed Task 2 paths may be present. Keep the external source until Task 6 acceptance, never inside Git.

- [ ] **Step 10: Review and commit Task 2**

Fresh reviewers inspect the real binary only through digest, closed manifest, scanner, and parity evidence; they must not print metadata. Resolve findings via the original implementer. Stage exact listed paths and commit:

```powershell
git add scripts/prepare_demo_assets.py tests/test_demo_assets.py scripts/model_parity_smoke.py tests/test_model_parity_smoke.py scripts/pages_artifact_check.py tests/test_pages_artifact_check.py demo/web/samples/boats.jpg demo/web/models/yolo26n-obb-privacy-sanitized.onnx demo/web/demo-model.json demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt demo/web/third_party/yolo26n-obb-privacy-sanitization.json demo/web/THIRD_PARTY_NOTICES.md
git diff --cached --check
git commit -m "feat: admit privacy-sanitized demo assets"
```

---

### Task 3: Real-image-first Demo and Genuine Detect Path

**Files:**
- Create: `demo/web/demo-assets.js`
- Modify: `demo/web/README.md`
- Modify: `demo/web/index.html`
- Modify: `demo/web/style.css`
- Modify: `demo/web/app.js`
- Delete: `demo/web/showcase-fixture.js`
- Delete: `demo/web/fixtures/showcase.svg`
- Modify: `scripts/browser_smoke.py`
- Modify: `scripts/pages_artifact_check.py`
- Modify: `tests/js/browser_parity_runner.js`
- Modify: `tests/test_browser_parity.py`
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `tests/test_package_release.py`

**Interfaces:**
- Consumes: Task 2 manifest/image/derivative; existing `OBB` preprocess/decode/filter/corners API; pinned ORT loader.
- Produces: `DemoAssets.validateManifest(value)`, `DemoAssets.fetchVerifiedModel(manifest, signal)`, `ensureDemoSession(generation)`, `runActiveInference(source, generation)`, `setResultView(view)`, and `resetToDemoOriginal()`.

- [ ] **Step 1: Write the real initial/success batch browser RED**

Add these named scenarios before changing UI:

```python
def assert_real_demo_initial(page, requests, messages) -> None: ...
def exercise_real_demo_success(page, requests, messages) -> None: ...
def assert_original_result_toggle(page, requests, run_counter) -> None: ...
def assert_demo_cached_filters(page, run_counter) -> None: ...
```

Assert the official original is visible; summary is count `0`, top/runtime `—`, mode `尚未 Detect`, provenance `官方範例 · 尚未執行`; primary button is `開始 Detect`; Synthetic is absent; BYOM/filters are collapsed; and initial requests omit manifest/ORT/WASM/ONNX. Success must show the exact derivative provenance, numeric runtime, at least one ship row, oriented polygon pixels, synchronized non-live description, `再次 Detect`, and `查看原圖`.

- [ ] **Step 2: Observe honest RED**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario real-demo-success
```

Expected earliest RED: current page still exposes Synthetic-first UI. Record only the first reached assertion.

- [ ] **Step 3: Implement manifest validation and verified fetch**

`demo-assets.js` rejects unknown/missing keys, wrong fixed path/source/release/license/shape/classes/sanitization record, unsafe URLs, size/digest/type drift, and non-frozen results. `fetchVerifiedModel` fetches only the same-origin derivative, enforces a 15 MiB streamed cap, verifies exact length and Web Crypto SHA-256 before returning an `ArrayBuffer`, and throws fixed codes only.

- [ ] **Step 4: Replace the page structure**

Within `main#mainContent`, render the real-image intro/figure first, then `#demoDetectBtn`, hidden `#viewToggleBtn`, summary/canvas/table, hidden-until-result `#resultControls`, and collapsed `<details id="byomPanel">`. Preserve the skip link, claim-before-control order, theme color, stable input names, canvas description, Source/AGPL links, and add `模型與素材來源` to `THIRD_PARTY_NOTICES.md`.

- [ ] **Step 5: Implement one source/session/result state machine**

Replace `mode` with `source` (`demo`, `byom`, or `null`), remove every Synthetic function/copy/cache, and keep one generation token. Demo Detect lazily loads manifest/runtime/model, creates a candidate session, validates input/output, then replaces the previous session only after candidate success. It runs the same preprocess/decode/filter/corners/render pipeline as BYOM. Repeated Detect reuses a valid demo session; filter/toggle changes never run inference.

In the same GREEN, remove the transitional Synthetic paths from the Pages allowlist and make the exact
real-demo JS/image/derivative/manifest/license/notice inventory authoritative. A current-tree verifier pass
must now fail if `showcase-fixture.js`, `fixtures/showcase.svg`, or any Synthetic reference remains.

Because `demo/web/README.md` ships inside the Pages artifact, update its current usage instructions in Task 3
and include Markdown in the Synthetic-reference scan. First add a verifier regression that fails on a
Synthetic reference in a staged Markdown file; after the scanner is GREEN, observe the current-tree failure
against the stale README, then replace only its current Synthetic-first instructions with the real-image
original → Detect → result flow. Task 5 may refine broader documentation but must not reintroduce the stale
mode.

`tests/test_browser_parity.py` contains one obsolete production-Synthetic contract that imports
`showcase-fixture.js`. Remove only that test and its now-dead dedicated helper/constant while preserving all
pure OBB preprocessing, decode, schema, geometry, and Python/browser parity tests. This scoped removal is
required so deleting the production fixture and running the complete parity file are not contradictory.
Its Node helper `tests/js/browser_parity_runner.js` also directly imports that production fixture. Remove only
that import and the duplicate showcase-derived branch, and point the preserved end-to-end literal assertions
at the runner's existing fixture-derived `detections` and `corners` outputs. Do not change
`tests/fixtures/browser_parity.json` or weaken any pure OBB assertion.

- [ ] **Step 6: Run Task 3 GREEN**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario real-demo-success
uv run --no-sync python scripts/browser_smoke.py --scenario stubbed-cache
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest tests/test_package_release.py tests/test_browser_parity.py tests/test_pages_artifact_check.py -q
git diff --check
```

- [ ] **Step 7: Review and commit Task 3**

Complete fresh spec/quality review and fix rounds, then stage exactly Task 3 files and commit:

```powershell
git add demo/web/demo-assets.js demo/web/README.md demo/web/index.html demo/web/style.css demo/web/app.js demo/web/showcase-fixture.js demo/web/fixtures/showcase.svg scripts/browser_smoke.py scripts/pages_artifact_check.py tests/js/browser_parity_runner.js tests/test_browser_parity.py tests/test_pages_artifact_check.py tests/test_package_release.py
git diff --cached --check
git commit -m "feat: add real-image browser OBB demo"
```

---

### Task 4: Failure Recovery, BYOM Transitions, Privacy, and Accessibility

**Files:**
- Modify: `demo/web/app.js`
- Modify: `demo/web/demo-assets.js`
- Modify: `demo/web/index.html` only for semantic failures proved by browser RED
- Modify: `demo/web/style.css` only for responsive/focus failures proved by browser RED
- Modify: `scripts/browser_smoke.py`
- Modify: `scripts/pages_artifact_check.py` only to freeze reviewed Task 4 text digests after all browser behavior is GREEN

**Interfaces:**
- Consumes: Task 3 real demo and existing candidate-session lifecycle.
- Produces: deterministic `manifest-failure`, `model-digest-failure`, `runtime-failure`, `session-failure`, `run-failure`, `output-failure`, `render-failure`, `stale-generation`, `byom-transition`, `accessibility`, `desktop`, and `mobile` scenarios.

- [ ] **Step 1: Write one failure/transition browser batch RED**

Add real-browser assertions for manifest fetch/status/schema failure; truncated/changed model; runtime request failure and retry; candidate contract failure; image decode/run/output/render failure; stale demo completion after BYOM selection; BYOM selection clearing demo results; candidate-before-old-session release; return-to-demo; and fixed actionable recovery. Every failure must show original image, runtime `—`, and clear cache/canvas/table/description/toggle/completed badge.

Privacy assertions scan UI, console, page errors, network report, and screenshot metadata for runtime-assembled sentinel local filenames/paths, response bodies, model metadata, raw exceptions/stacks, tokens, and signed queries.

- [ ] **Step 2: Observe the earliest actual RED**

```powershell
uv run --no-sync python scripts/browser_smoke.py --scenario manifest-failure
```

Run later scenarios independently only after the first blocker is reachable; record their real first failures.

- [ ] **Step 3: Implement minimal fixed recovery**

Use one `clearResultState({keepImage: true})` path for all demo failures. Abort/ignore stale fetch/run work using generation tokens. Never release the current valid session until a candidate is fully validated. Runtime retry must make one new pinned request and return to demo original before retrying Detect.

- [ ] **Step 4: Write and satisfy accessibility/responsive RED**

Assert skip-link focus/target; notice before first control; labels/names; heading order; non-live synchronized canvas description; empty/reset/error text; aria-live status without duplicate detection announcements; visible focus; reduced motion; 200% zoom; no horizontal overflow at 1280×720 and 390×844; readable Source/AGPL/sanitization links; and advanced BYOM remaining secondary.

- [ ] **Step 5: Run Task 4 GREEN**

```powershell
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest tests/test_package_release.py tests/test_browser_parity.py tests/test_pages_artifact_check.py -q
git diff --check
```

- [ ] **Step 6: Review and commit Task 4**

Fresh reviewers inspect every failure cleanup and privacy assertion. Resolve via original implementer, stage exact changed paths, and commit:

```powershell
git add demo/web/app.js demo/web/demo-assets.js demo/web/index.html demo/web/style.css scripts/browser_smoke.py scripts/pages_artifact_check.py
git diff --cached --check
git commit -m "fix: harden live demo recovery and accessibility"
```

---

### Task 5: Release Boundary, License Notices, CI, Documentation, and Evidence

**Files:**
- Modify: `scripts/pages_artifact_check.py`
- Modify: `scripts/repo_check.py`
- Modify: `scripts/release_check.py`
- Modify: `scripts/clean_export_check.py`
- Modify: `tests/test_pages_artifact_check.py`
- Modify: `tests/test_repo_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `release/artifact-manifest.json`
- Modify: `release/evidence.json`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `demo/web/README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `CHANGELOG.md`
- Modify: `.github/workflows/release-gates.yml`
- Modify: `.github/workflows/pages.yml` only if exact artifact path/verification requires it
- Modify: `docs/assets/browser-workbench.png`

**Interfaces:**
- Consumes: committed Task 2 bytes/digests/records and Task 3–4 visible behavior.
- Produces: exact derivative-only release/archive exception, truthful AGPL modification record, candidate workflow, public docs, and screenshot evidence.

- [ ] **Step 1: Write repository/release RED tests**

Assert exactly one allowed model path/digest/size; source digest is forbidden as a published file; modification status is `metadata-only`; source/derivative hashes differ; notices include modification date/source/sanitizer/AGPL/DOTAv1/non-endorsement/no-commercial-clearance; manifest/evidence paths agree; clean export includes the derivative and rejects any second model; workflows use current action majors and preserve manual-only Pages deployment.

Run focused tests and record the earliest actual old code-only/Synthetic assertion failure.

- [ ] **Step 2: Implement the exact binary exception**

`pages_artifact_check`, `repo_check`, `release_check`, and `clean_export_check` allow only `demo/web/models/yolo26n-obb-privacy-sanitized.onnx` when its exact digest/size and transformation record match. Every other model suffix/path remains forbidden. Binary scanning still checks the allowed model for absolute paths, tokens, private markers, and source-binary digest.

- [ ] **Step 3: Update manifests and legal/source notices**

Set distribution mode to `public-agpl-privacy-sanitized-demo-model-plus-byom`, keep `commercial_use_cleared: false`, and record image, derivative, AGPL license, sanitization JSON, font, and final screenshot with literal bytes/digests. State modified on 2026-08-31, graph/weights structurally unchanged, DOTAv1 training provenance, source/sanitizer links, no endorsement, and no commercial-clearance claim. Preserve the excluded historical model record as historical; do not relabel it as the current artifact.

- [ ] **Step 4: Update README/CHANGELOG/UI documentation**

Explain the exact journey: real original → first Detect downloads same-origin derivative plus pinned jsDelivr runtime → local result → original/result switch; no upload; advanced BYOM; not accuracy/evaluation/T4-latency evidence. Remove current Synthetic-first instructions without rewriting released rc history.

- [ ] **Step 5: Update CI labels without deploying**

Rename `Synthetic browser smoke / Ubuntu CPU` to `Live demo browser smoke / Ubuntu CPU` and its step to `Exercise the real-image browser demo and BYOM safety paths`. Preserve CPU-only settings, locked sync, Playwright install, Pages candidate upload, manual-only deploy, exact action majors, and no About mutation. Do not run or dispatch workflows.

- [ ] **Step 6: Generate canonical screenshot from the exact committed artifact**

```powershell
uv run --no-sync python scripts/browser_smoke.py --screenshot docs/assets/browser-workbench.png
```

Inspect pixels and PNG metadata. Required state: real image annotated after genuine derivative inference, exact provenance/numeric runtime, no advanced BYOM ready labels, no path/filename/private metadata, and notice/source/license links visible. Freeze actual screenshot bytes/digest in the manifest.

- [ ] **Step 7: Run Task 5 GREEN**

```powershell
uv run --no-sync python -m pytest tests/test_repo_check.py tests/test_release_check.py tests/test_pages_artifact_check.py tests/test_clean_export.py tests/test_readme_language.py -q
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
uv run --no-sync python scripts/browser_smoke.py
git diff --check
```

- [ ] **Step 8: Review and commit Task 5**

Review artifact/license/privacy/workflow/docs consistency, fix through RED/GREEN, stage exact Task 5 paths, and commit:

```powershell
git add scripts/pages_artifact_check.py scripts/repo_check.py scripts/release_check.py scripts/clean_export_check.py tests/test_pages_artifact_check.py tests/test_repo_check.py tests/test_release_check.py tests/test_clean_export.py release/artifact-manifest.json release/evidence.json README.md README.en.md demo/web/README.md THIRD_PARTY_NOTICES.md CHANGELOG.md .github/workflows/release-gates.yml .github/workflows/pages.yml docs/assets/browser-workbench.png
git diff --cached --check
git commit -m "release: document privacy-sanitized live demo"
```

---

### Task 6: Complete Local Acceptance, Broad Review, and Operable Preview

**Files:**
- Modify only files justified by a focused acceptance RED and approved one-wave review fix.
- Create only repo-external verification logs, network reports, clean export, and temporary screenshots.

**Interfaces:**
- Consumes: Tasks 1–5 committed branch and retained external source review root.
- Produces: final local verification evidence, clean branch, broad review verdict, and loopback preview for owner operation.

- [ ] **Step 1: Establish clean committed state**

```powershell
git status --short
git diff --check
git log --oneline --decorate -12
```

Expected: empty status and only reviewed task commits after the docs plan commit.

- [ ] **Step 2: Re-run sanitizer and real parity gates**

Use the retained external review root without printing its path:

```powershell
uv run --no-sync python scripts/prepare_demo_assets.py verify --review-root $reviewRoot
uv run --no-sync python scripts/model_parity_smoke.py --review-root $reviewRoot --report $parityReport
```

Expected: fixed OK lines; exact source/derivative output bytes equal; accepted ship true; reports privacy-clean.

- [ ] **Step 3: Run focused and full regression**

```powershell
uv run --no-sync python -m pytest tests/test_sanitize_demo_model.py tests/test_demo_assets.py tests/test_model_parity_smoke.py tests/test_package_release.py tests/test_browser_parity.py tests/test_pages_artifact_check.py -q
uv run --no-sync python scripts/browser_smoke.py
uv run --no-sync python -m pytest -q
```

Expected: focused, full browser smoke, and all collected tests pass; full count remains at least the prior 133-test baseline plus new tests.

- [ ] **Step 4: Run artifact, license, privacy, and origin gates**

```powershell
uv run --no-sync python scripts/repo_check.py
uv run --no-sync python scripts/release_check.py
uv run --no-sync python scripts/pages_artifact_check.py
rg -n -I -i "Synthetic Showcase|OBB_SHOWCASE|showcase-fixture|fixtures/showcase|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}" demo/web README.md README.en.md release THIRD_PARTY_NOTICES.md
```

Expected: verifier OK; no current public Synthetic/private-token hit; every historical mention outside current UI reviewed in context. Network report origins are loopback plus pinned jsDelivr only; initial phase is loopback-only.

- [ ] **Step 5: Run strict clean export**

```powershell
$cleanExport = Join-Path ([IO.Path]::GetTempPath()) ('aerial-obb-clean-export-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $cleanExport) { throw 'Refusing to overwrite existing clean-export evidence' }
uv run --no-sync python scripts/clean_export_check.py --output $cleanExport
```

Do not use `--skip-browser`. Expected: committed snapshot, package build/install/import, tests, links, derivative/artifact/privacy/license gates, and real browser smoke pass from the clean export.

- [ ] **Step 6: Serve and inspect exact local UI**

Start and leave running:

```powershell
uv run --no-sync python -m http.server 8765 --bind 127.0.0.1 --directory demo/web
```

Open `http://127.0.0.1:8765/` in the in-app browser. At 1280×720 and 390×844, verify original image before Detect, genuine successful Detect, result/toggle/filters, secondary BYOM, no overflow, notice/source/AGPL/sanitization readability, keyboard/focus/labels/headings/aria-live/description, 200% zoom, reduced motion, console errors, and request origins. Store any extra screenshots outside the repository.

- [ ] **Step 7: Run broad whole-branch review and one fix wave**

Dispatch the most capable fresh reviewer against both approved specs, this plan, base diff, exact binary/manifest/notices, task reports, screenshots, test outputs, and privacy/origin evidence. Resolve Critical/Important findings through one focused RED/GREEN original-implementer fix wave and fresh re-review. Record Minors; do not expand scope or weaken a stop condition.

- [ ] **Step 8: Verification-before-completion and branch readiness**

```powershell
git status --short
git diff --check
git log --oneline --decorate -20
git diff --name-status 582a88960b9a699d3adedfd3a918b09f8e5b128b...HEAD
git ls-files demo/web
```

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch` only to confirm readiness and report integration options. Keep the feature branch, worktree, loopback preview, and external review root for feedback. Do not select or execute a remote/integration option.

---

## Remote Gates A–E — Later and Separately Authorized

1. **Gate A — candidate PR:** origin/main drift, checks, branch/PR race, non-force push, one PR.
2. **Gate B — candidate/integration:** download exact CI artifact, compare derivative/image/text bytes, repeat privacy/license/real-browser review, then repository-supported merge.
3. **Gate C — Pages:** configure/dispatch only from exact reviewed merged-main SHA after main CI succeeds.
4. **Gate D — live review:** HTTPS/assets, zero-model initial network, real Detect, exact provenance, failures/BYOM/accessibility/privacy, deployed SHA.
5. **Gate E — About/Portfolio Control:** only after independently passing live review.

No task in this plan authorizes any remote gate.

## Plan Self-review

- **Spec coverage:** deterministic one-entry sanitization, structural identity, raw browser parity, real image/demo, BYOM, errors, accessibility, AGPL modification notice, artifact exception, clean export, local preview, and remote separation each map to a named task and command.
- **Generated facts:** derivative/protobuf/screenshot hashes and sizes are produced only after the exact transformation and then frozen; the plan does not invent them.
- **Type consistency:** `SanitizationReceipt`, `AdmittedAssets`, `sanitize_official_model`, `validate_sanitized_model`, `validate_admitted_assets`, and `run_parity` have one definition and one direction of dependency.
- **TDD honesty:** each batch names its likely earliest blocker, requires independent reruns for later failures, and reserves the real-model acceptance for genuine Chromium/ORT rather than a mock or source grep.
- **Privacy/license:** removed metadata is never printed; only the derivative is published; modification/date/source/sanitizer/AGPL/DOTAv1/non-endorsement/no-commercial-clearance remain consistent.
- **Scope:** one derivative, one image, one primary Detect path, one advanced BYOM path, and their necessary evidence; no gallery, webcam, server inference, optimizer, framework, or remote mutation.

# YOLO26 OBB Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Subagents are prohibited by the approved brief.

**Goal:** Produce a clean, locally committed `v1.0.0rc1` release candidate whose claims, browser geometry, licensing, privacy, package, CI, and clean-export gates run without GPU, DOTA, secrets, or downloaded weights.

**Architecture:** A committed release evidence registry and artifact manifest are the source of truth for public claims and shipped binaries. Small standard-library Python gates plus pure JavaScript geometry functions enforce those contracts; Ubuntu/Windows CI and a Git-archive-based clean export run the same CPU-only checks.

**Tech Stack:** Python 3.11 standard library, pytest, Node.js built-ins, vanilla browser JavaScript, uv/hatchling, GitHub Actions.

## Global Constraints

- Do not train, run full validation, initialize CUDA, run GPU inference, or build TensorRT.
- Preserve accepted A100/T4 evidence and the negative matched deltas.
- Do not download DOTA or model weights during core gates.
- Do not create or mutate remotes, tags, releases, Spaces, model repositories, or uploads.
- Stage explicit files only; never use `git add -A`.
- Author and committer must be `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailer.
- Keep `notes.private.md`, interview material, `.env`, datasets, weights, and runtime artifacts ignored and outside archives.

## File Structure

- `release/evidence.json`: accepted result values, provenance, evidence strength, and limitations.
- `release/artifact-manifest.json`: hashes, sizes, origins, and redistribution constraints.
- `scripts/release_check.py`: offline claim, evidence, manifest, privacy, and snapshot checks.
- `scripts/clean_export.py`: deterministic committed-tree archive creation and extraction checks.
- `demo/space-static/obb.js`: pure, browser/Node-compatible preprocessing and OBB geometry.
- `tests/fixtures/browser_parity.json`: fixed synthetic contract values; contains no DOTA data.
- `tests/js/browser_parity_runner.js`: Node adapter for the pure browser functions.
- `tests/test_release_check.py`: evidence, manifest, and privacy gate tests.
- `tests/test_browser_parity.py`: independent Python-to-JavaScript contract comparison.
- `tests/test_clean_export.py`: archive selection and dirty-tree guard tests.
- `.github/workflows/release-gates.yml`: Ubuntu/Windows CPU-only CI-equivalent workflow.
- `THIRD_PARTY_NOTICES.md`, `CITATION.cff`, `CHANGELOG.md`, `RELEASE_CHECKLIST.md`: release metadata and owner actions.

---

### Task 1: Evidence Registry and Claim Gate

**Files:**
- Create: `release/evidence.json`
- Create: `scripts/release_check.py`
- Create: `tests/test_release_check.py`
- Modify: `README.md`
- Modify: `README.zh-TW.md`
- Modify: `docs/training_results.md`
- Modify: `docs/analysis_results.md`
- Modify: `docs/model_card.md`
- Modify: `demo/space-static/README.md`
- Modify: `demo/space/README.md`

**Interfaces:**
- Consumes: `docs/per_class_metrics.json`, `docs/analysis_results.json`, and public Markdown files.
- Produces: `verify_evidence(root: Path) -> list[str]`, `verify_claims(root: Path) -> list[str]`, and CLI exit status 0 only when every error list is empty.

- [ ] **Step 1: Write the failing evidence tests**

```python
def test_matched_fine_tuning_is_a_negative_delta(repo_root):
    errors = release_check.verify_evidence(repo_root)
    assert errors == []
    evidence = json.loads((repo_root / "release/evidence.json").read_text("utf-8"))
    assert evidence["matched_evaluation"]["delta_percentage_points"] == {
        "mAP50": -0.05,
        "mAP50_95": -0.13,
    }

def test_dota8_and_t4_claims_are_scoped(repo_root):
    evidence = json.loads((repo_root / "release/evidence.json").read_text("utf-8"))
    assert evidence["export_smoke"]["dataset"] == "DOTA8 val"
    assert evidence["export_smoke"]["production_certification"] is False
    assert evidence["t4_benchmark"]["batch"] == 1
    assert evidence["t4_benchmark"]["imgsz"] == 1024
```

- [ ] **Step 2: Run the focused tests and observe the missing-registry failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_release_check.py -q`

Expected: FAIL because `release/evidence.json` and `scripts/release_check.py` do not exist.

- [ ] **Step 3: Add the minimal evidence registry and verifier**

Record `0.7816142568252565 / 0.6314216725888359` as the checksum-gated fine-tuned aggregate,
`-0.05 / -0.13` percentage points as the accepted rounded matched deltas, `0.9950` for each DOTA8
export-smoke backend, and the complete T4 latency rows. Every section must include `source_files`,
`evidence_kind`, `limitations`, and whether a raw log is committed.

```python
def verify_evidence(root: Path) -> list[str]:
    evidence = load_json(root / "release/evidence.json")
    metrics = load_json(root / "docs/per_class_metrics.json")
    errors: list[str] = []
    if evidence["matched_evaluation"]["fine_tuned"]["mAP50"] != metrics["aggregate"]["mAP50"]:
        errors.append("fine-tuned mAP50 differs from per_class_metrics.json")
    if evidence["export_smoke"]["production_certification"] is not False:
        errors.append("DOTA8 smoke must not claim production certification")
    return errors
```

- [ ] **Step 4: Replace ambiguous public claims with bounded claim markers**

Use stable markers such as `<!-- claim:matched-evaluation -->` and verify the numeric values and
required limitation phrases inside each marker block. Replace “no accuracy loss” with “identical
to four reported decimals on DOTA8 export smoke”; state that the raw benchmark console log is not
committed and that the accepted T4 table is historical evidence, not a fresh run or universal SLA.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_release_check.py -q`

Run: `.venv/Scripts/python.exe -m pytest -q`

Expected: all tests PASS and both README variants agree with `release/evidence.json`.

- [ ] **Step 6: Commit the evidence blocker**

```powershell
git add -- release/evidence.json scripts/release_check.py tests/test_release_check.py README.md README.zh-TW.md docs/training_results.md docs/analysis_results.md docs/model_card.md demo/space-static/README.md demo/space/README.md
git commit -m "fix: bind release claims to evidence"
```

### Task 2: Browser Preprocess and OBB Decode Parity

**Files:**
- Create: `demo/space-static/obb.js`
- Create: `tests/fixtures/browser_parity.json`
- Create: `tests/js/browser_parity_runner.js`
- Create: `tests/test_browser_parity.py`
- Modify: `demo/space-static/app.js`
- Modify: `demo/space-static/index.html`
- Modify: `scripts/repo_check.py`

**Interfaces:**
- Consumes: a fixture with image dimensions, RGBA bytes, one `[N,7]` output, threshold, and literal expected geometry.
- Produces: `letterboxGeometry(width, height, size)`, `rgbaToChw(rgba)`, `decodeDetections(output, geometry, confidence, classIds, classCount)`, and `rotatedCorners(detection)` on `globalThis.OBB` and `module.exports`.

- [ ] **Step 1: Write a fixture-based failing parity test**

```python
def test_javascript_matches_independent_synthetic_reference(repo_root):
    completed = subprocess.run(
        ["node", "tests/js/browser_parity_runner.js", "tests/fixtures/browser_parity.json"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(completed.stdout)
    assert actual["geometry"] == {"scale": 1.6, "newWidth": 1024, "newHeight": 512,
                                  "padX": 0, "padY": 256}
    assert actual["chw"] == [1.0, 0.0, 128 / 255, 0.0, 1.0, 64 / 255]
```

- [ ] **Step 2: Run the parity test and observe the missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_browser_parity.py -q`

Expected: FAIL because `demo/space-static/obb.js` is absent.

- [ ] **Step 3: Implement the pure functions and strict schema validation**

Reject non-finite values, output lengths not divisible by seven, non-positive scale, negative box
dimensions, and class IDs outside `0..14`. Derive corners directly from literal angle radians;
do not run NMS because the exported end-to-end model already emits final detections.

```javascript
function decodeDetections(output, geometry, confidence, classIds, classCount) {
  if (output.length % 7 !== 0) throw new Error("expected flattened [N,7] output");
  // validate, unletterbox, filter, sort, and return final detections
}
```

- [ ] **Step 4: Refactor the DOM app to consume `globalThis.OBB`**

Load `obb.js` before `app.js`, use `letterboxGeometry` for canvas placement,
`rgbaToChw` for normalization, `decodeDetections` for `[N,7]`, and `rotatedCorners` for drawing.
Keep the existing UI and model unchanged.

- [ ] **Step 5: Extend static checks and run browser parity**

Run: `node --check demo/space-static/obb.js`

Run: `.venv/Scripts/python.exe -m pytest tests/test_browser_parity.py -q`

Run: `.venv/Scripts/python.exe scripts/repo_check.py`

Expected: synthetic preprocess/decode/corner checks PASS; loopback static assets return 200.

- [ ] **Step 6: Commit the browser blocker**

```powershell
git add -- demo/space-static/obb.js demo/space-static/app.js demo/space-static/index.html scripts/repo_check.py tests/fixtures/browser_parity.json tests/js/browser_parity_runner.js tests/test_browser_parity.py
git commit -m "fix: verify browser OBB decode parity"
```

### Task 3: Artifact, Privacy, and License Boundary

**Files:**
- Create: `release/artifact-manifest.json`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `CITATION.cff`
- Create: `CHANGELOG.md`
- Create: `RELEASE_CHECKLIST.md`
- Modify: `scripts/release_check.py`
- Modify: `scripts/repo_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `README.md`
- Modify: `README.zh-TW.md`
- Modify: `demo/space-static/index.html`
- Modify: `demo/space/app.py`

**Interfaces:**
- Consumes: committed paths plus manifest entries `{path, bytes, sha256, provenance, license, restrictions}`.
- Produces: `verify_artifacts(root: Path) -> list[str]` and `verify_privacy(root: Path, paths: Iterable[str]) -> list[str]`.

- [ ] **Step 1: Write failing manifest and privacy tests**

```python
def test_manifest_hashes_every_redistributed_binary(repo_root):
    assert release_check.verify_artifacts(repo_root) == []

def test_private_and_runtime_names_are_rejected():
    errors = release_check.verify_privacy(Path("."), ["notes.private.md", ".env", "runs/x.pt"])
    assert len(errors) == 3
```

- [ ] **Step 2: Run the focused tests and observe missing-manifest failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_release_check.py -q`

Expected: FAIL because `release/artifact-manifest.json` and verifier functions are absent.

- [ ] **Step 3: Add hashes and fail-closed privacy checks**

Include the 10,153,722-byte ONNX model with SHA-256
`12572cc068417329f3e1874e3e1013368820cb48ae6b143356cdfccdb6b9511e` and all five DOTA-derived
comparison images. Reject secret-shaped text, local user paths, notebook outputs, private filename
patterns, checkpoints/datasets, archives, caches, and unexpected tracked files larger than 25 MB.

- [ ] **Step 4: Add legal and release metadata**

State that code is AGPL-3.0-or-later; Ultralytics offers AGPL and Enterprise routes; DOTA images
and annotations are academic-only/non-commercial and Google Earth source terms may also apply;
bundled official/fine-tuned weights remain constrained by their training-data and upstream license
boundary. List an owner action to obtain legal/rights-holder approval before commercial,
closed-source, or asset-redistribution use. Add DOTA citations and project release metadata.

- [ ] **Step 5: Run privacy, license-link, artifact, and full gates**

Run: `.venv/Scripts/python.exe scripts/release_check.py`

Run: `.venv/Scripts/python.exe scripts/repo_check.py`

Run: `.venv/Scripts/python.exe -m pytest -q`

Expected: manifest hashes match; ignored private files are absent from the candidate path set;
notebooks remain output-free; local Markdown links resolve.

- [ ] **Step 6: Commit the legal/privacy blocker**

```powershell
git add -- release/artifact-manifest.json THIRD_PARTY_NOTICES.md CITATION.cff CHANGELOG.md RELEASE_CHECKLIST.md scripts/release_check.py scripts/repo_check.py tests/test_release_check.py README.md README.zh-TW.md demo/space-static/index.html demo/space/app.py
git commit -m "fix: enforce artifact and license boundaries"
```

### Task 4: Package, Cross-Platform CI, and Clean Export

**Files:**
- Create: `scripts/clean_export.py`
- Create: `tests/test_clean_export.py`
- Create: `.github/workflows/release-gates.yml`
- Modify: `pyproject.toml`
- Modify: `scripts/repo_check.py`
- Modify: `README.md`
- Modify: `README.zh-TW.md`
- Modify: `RELEASE_CHECKLIST.md`

**Interfaces:**
- Consumes: clean Git `HEAD`, explicit output directory, and the committed-tree file list.
- Produces: `create_archive(root: Path, output: Path) -> Path`, `inspect_archive(path: Path) -> list[str]`, and a CLI that refuses a dirty tree.

- [ ] **Step 1: Write failing clean-export tests**

```python
def test_archive_rejects_private_or_runtime_members(tmp_path):
    archive = tmp_path / "bad.zip"
    write_zip(archive, ["README.md", "notes.private.md", "runs/best.pt"])
    assert clean_export.inspect_archive(archive) == [
        "private release member: notes.private.md",
        "runtime release member: runs/best.pt",
    ]
```

- [ ] **Step 2: Run the focused tests and observe the missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clean_export.py -q`

Expected: FAIL because `scripts/clean_export.py` does not exist.

- [ ] **Step 3: Implement deterministic committed-tree export**

Use `git status --porcelain`, `git archive --format=zip HEAD`, and `zipfile` inspection. Reject
absolute/traversal paths, private/runtime members, missing required release files, and a manifest
hash mismatch. Never copy the working directory recursively.

- [ ] **Step 4: Make snapshot checks work without `.git` and set release version**

When `.git` is absent, `repo_check.py` must scan the extracted snapshot, run working-file privacy
checks, and report history/ignore checks as not applicable rather than failing. Set package version
to `1.0.0rc1`; build a wheel and source distribution and inspect both for only package/release
metadata expected by hatchling.

- [ ] **Step 5: Add Ubuntu and Windows CPU CI jobs**

Use Python 3.11, Node 22, `astral-sh/setup-uv`, `uv sync --frozen --no-editable`, pytest,
`repo_check.py`, `release_check.py`, `uv build --no-sources`, and clean-export creation. Do not
install optional ML/demo groups and do not reference secrets.

- [ ] **Step 6: Run package and clean-export gates locally**

Run: `uv sync --frozen --no-editable`

Run: `uv build --no-sources`

Run: `.venv/Scripts/python.exe scripts/clean_export.py --output dist/yolo26-dota-obb-1.0.0rc1.zip`

Extract to a temporary directory, create a locked non-editable CPU environment, then run pytest,
`repo_check.py`, `release_check.py`, Node syntax/parity, and `uv build --no-sources` there.

- [ ] **Step 7: Commit the package/CI blocker**

```powershell
git add -- scripts/clean_export.py tests/test_clean_export.py .github/workflows/release-gates.yml pyproject.toml scripts/repo_check.py README.md README.zh-TW.md RELEASE_CHECKLIST.md
git commit -m "ci: gate package and clean export"
```

### Task 5: Final Release-Candidate Audit

**Files:**
- Modify only files that fail a verified release gate; create a separate small commit per blocker.

**Interfaces:**
- Consumes: committed `HEAD` and the freshly created clean archive.
- Produces: final evidence for status, identity, trailers, files, tests, package, browser, links, privacy, licensing, refs, and remote non-mutation.

- [ ] **Step 1: Run every gate from the committed tree**

```powershell
uv sync --frozen --no-editable
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/repo_check.py
.venv/Scripts/python.exe scripts/release_check.py
node tests/js/browser_parity_runner.js tests/fixtures/browser_parity.json
uv build --no-sources
```

- [ ] **Step 2: Run an actual headless browser smoke over loopback HTTP**

Stub ONNX Runtime before page load, return the synthetic `[1,1,7]` output, upload the synthetic
fixture image, click Detect, and assert the table class/confidence/angle plus zero console errors.
Do not fetch the ONNX model and do not perform inference.

- [ ] **Step 3: Build and fully verify a fresh clean committed export**

Create the archive only after all required commits exist. Extract to a new temporary directory and
repeat install, tests, repo/release checks, browser parity, local-link/static HTTP smoke, and package
build from that extraction.

- [ ] **Step 4: Audit Git identity, trailers, refs, and status**

Run `git status --short --branch`, `git show-ref --head`, `git remote -v`, and a full-history
author/committer/trailer scan. Confirm the branch is `portfolio/obb-v1.0-release-hardening`, the
worktree is clean, every identity matches, no `Co-authored-by` exists, and no remote exists or was
changed.

- [ ] **Step 5: Mark the goal complete and report the release candidate**

Report Feature Freeze / Release Candidate verdict, commits and files, test/CI/browser/export
evidence, claim changes, privacy/license/artifact audit, limitations, suggested GitHub repository
metadata, exact owner actions for tomorrow, final branch/HEAD/status/identity/trailers, and explicit
confirmation that no remote mutation occurred.

# YOLO26 OBB v1.0 Release-Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The approved brief prohibits subagents, so execution is inline.

**Goal:** Produce a clean, locally verified, unpushed `1.0.0rc1` release candidate that preserves the real negative fine-tuning result and binds every public claim and redistributed artifact to evidence.

**Architecture:** Keep immutable A100/T4 observations in explicit evidence JSON and verify public documents against them with CPU-only code. Extract the browser math into matching Python and JavaScript cores driven by one synthetic fixture, then execute the same gates in normal checkout, CI matrices, and a clean `git archive` export.

**Tech Stack:** Python 3.11, pytest, standard-library release tooling, NumPy, Node.js built-in test runner, vanilla JavaScript modules, ONNX Runtime Web, uv, hatchling, GitHub Actions.

## Global Constraints

- Work only on `portfolio/obb-v1.0-release-hardening`; never rewrite, amend, squash, or rebase history.
- Commit only as `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` and add no `Co-authored-by` trailer.
- Never use `git add -A`; stage an explicit file list for every commit.
- Use CPU-only local checks. Do not invoke CUDA, GPU inference/training, full DOTAv1 validation, or TensorRT build.
- Do not create or modify remotes, push, open a PR, create a tag/release, or write to Hugging Face.
- Do not read, stage, or publish `notes.private.md`; require it to remain ignored.
- Keep the feature set frozen: change only release blockers, tests, evidence, privacy, licensing, packaging, CI, and reproducibility material.
- Treat existing A100/T4 results as historical evidence and record missing provenance as a limitation rather than rerunning them.

---

### Task 1: Claim-to-evidence registry and verifier

**Files:**
- Create: `docs/evidence/training_evaluation.json`
- Create: `docs/evidence/deployment_benchmark_t4.json`
- Create: `scripts/verify_release.py`
- Create: `tests/test_release_evidence.py`
- Modify: `README.md`
- Modify: `README.zh-TW.md`
- Modify: `docs/training_results.md`
- Modify: `docs/analysis_results.md`
- Modify: `docs/model_card.md`

**Interfaces:**
- Consumes: committed JSON evidence and Markdown files rooted at `Path`.
- Produces: `verify_release(root: Path) -> list[str]`, where an empty list means all claim gates pass; CLI exits nonzero and prints every violation otherwise.

- [ ] **Step 1: Write failing evidence tests**

```python
def test_release_claims_match_machine_evidence() -> None:
    assert verify_release(ROOT) == []

def test_matched_finetune_result_is_not_an_improvement_claim() -> None:
    evidence = load_json("docs/evidence/training_evaluation.json")
    assert evidence["matched_comparison"]["delta_map50_percentage_points"] < 0
    assert evidence["matched_comparison"]["delta_map50_95_percentage_points"] < 0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_release_evidence.py -q`

Expected: collection fails because `scripts.verify_release` and the evidence files do not exist.

- [ ] **Step 3: Add minimal evidence and verifier**

Implement table parsing and required disclosure checks for the two READMEs, model
card, training results, analysis results, DOTA8 smoke scope, T4 context, and demo
model separation. Record missing raw baseline precision and missing T4 runtime
versions as limitations.

- [ ] **Step 4: Correct overclaims and verify GREEN**

Change the HBB/NMS text from observed suppression to a geometry-based suppression
risk. State that DOTA8 is not DOTAv1 production certification and that the browser
demo is neither the fine-tuned checkpoint nor the T4 benchmark path.

Run: `python -m pytest tests/test_release_evidence.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```text
docs: bind public claims to recorded evidence
```

### Task 2: Artifact, privacy, and repository policy gate

**Files:**
- Create: `docs/artifact_manifest.json`
- Create: `tests/test_artifact_policy.py`
- Modify binary metadata: `demo/space-static/yolo26n-obb.onnx`
- Modify: `scripts/repo_check.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Git tracked/untracked/ignored paths, artifact manifest entries, binary metadata bytes, notebook JSON.
- Produces: `check_artifacts(files: list[Path]) -> None` and `check_privacy(files: list[Path]) -> None`; each raises `RuntimeError` with exact paths on violation.

- [ ] **Step 1: Write failing artifact/privacy tests**

```python
def test_browser_model_contains_no_absolute_build_path() -> None:
    data = (ROOT / "demo/space-static/yolo26n-obb.onnx").read_bytes()
    assert b"/home/" not in data and b"C:\\\\Users\\\\" not in data

def test_declared_local_artifact_hashes_match() -> None:
    assert artifact_errors(ROOT) == []
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_artifact_policy.py -q`

Expected: failure reports the embedded upstream home-directory path in the ONNX metadata and the absent manifest.

- [ ] **Step 3: Sanitize and inventory**

Clear the ONNX `description` metadata value while preserving its graph, input/output
schema, producer, opset, and other license metadata. Populate SHA-256/size/source/use
and restriction records for the ONNX, five DOTA-derived comparison images, and pinned
external fine-tuned checkpoint.

- [ ] **Step 4: Expand repository policy checks and verify GREEN**

Reject unexpected model/data/archive extensions, unallowlisted files over 5 MiB,
absolute personal paths, notebook output, private materials, and unignored local
runtime paths. Assert `notes.private.md` is ignored and untracked without reading it.

Run: `python -m pytest tests/test_artifact_policy.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```text
fix: sanitize and inventory release artifacts
```

### Task 3: Browser/Python OBB pipeline parity

**Files:**
- Create: `src/obbkit/browser_pipeline.py`
- Create: `demo/space-static/obb_core.mjs`
- Create: `tests/fixtures/browser_pipeline.json`
- Create: `tests/test_browser_pipeline.py`
- Create: `tests/js/browser_pipeline.test.mjs`
- Modify: `demo/space-static/app.js`
- Modify: `demo/space-static/index.html`
- Modify: `scripts/repo_check.py`

**Interfaces:**
- Python produces `letterbox_geometry(width, height, size)`, `rgba_to_chw(rgba)`, `decode_detections(...)`, and `obb_corners(...)`.
- JavaScript exports `letterboxGeometry`, `rgbaToChw`, `decodeDetections`, and `obbCorners` with numerically equivalent arguments and return fields.

- [ ] **Step 1: Write the shared fixture and failing tests**

```python
def test_fixture_decode_and_corners_match_reference() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert decode_detections(**fixture["decode_input"]) == fixture["expected_detections"]
```

```javascript
test("synthetic fixture preserves letterbox, CHW, decode, angle, and corners", () => {
  assert.deepEqual(decodeDetections(...fixture.decode_args), fixture.expected_detections);
});
```

- [ ] **Step 2: Verify RED in both runtimes**

Run: `python -m pytest tests/test_browser_pipeline.py -q`

Run: `node --test tests/js/browser_pipeline.test.mjs`

Expected: both fail because the reference modules do not exist.

- [ ] **Step 3: Implement the pure cores and wire the page**

Implement only the fixture-covered operations. Keep model loading and DOM rendering
in `app.js`; import the pure math from `obb_core.mjs` through a module script.

- [ ] **Step 4: Verify GREEN and static HTTP behavior**

Run: `python -m pytest tests/test_browser_pipeline.py -q`

Run: `node --test tests/js/browser_pipeline.test.mjs`

Run: `python scripts/repo_check.py`

Expected: all commands pass and local HTTP serves both JavaScript modules.

- [ ] **Step 5: Commit**

```text
test: lock browser OBB decoding to Python reference
```

### Task 4: Licensing and release documentation

**Files:**
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `CITATION.cff`
- Create: `CHANGELOG.md`
- Create: `docs/RELEASE_CHECKLIST.md`
- Modify: `README.md`
- Modify: `README.zh-TW.md`
- Modify: `docs/model_card.md`
- Modify: `demo/space-static/README.md`
- Modify: `demo/space/README.md`

**Interfaces:**
- Consumes: official DOTA terms, Ultralytics v8.4.93 AGPL license and current licensing page, artifact manifest.
- Produces: a human-readable distribution boundary and exact owner actions; no legal authorization is inferred.

- [ ] **Step 1: Write the notices and citation metadata**

Record DOTA image sources/academic-only terms and required citations, Ultralytics
code/model licensing links, ONNX Runtime Web attribution, and the repository's own
AGPL-3.0-or-later choice.

- [ ] **Step 2: Write changelog and release checklist**

Record `1.0.0-rc.1` changes, historical-evidence limits, all local gates, prohibited
contents, and owner-only remote/legal actions.

- [ ] **Step 3: Align user-facing licensing language**

Make both READMEs, demo cards, and model card link to the notices and avoid presenting
the repository license as permission to use DOTA commercially.

- [ ] **Step 4: Verify documents**

Run: `python scripts/verify_release.py`

Run: `python scripts/repo_check.py`

Expected: claims, local links, privacy, and artifact gates pass.

- [ ] **Step 5: Commit**

```text
docs: define release licensing and owner actions
```

### Task 5: Cross-platform CPU CI and package gates

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_package_metadata.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `scripts/repo_check.py`

**Interfaces:**
- Consumes: Python 3.11, uv `0.11.18`, Node bundled on GitHub-hosted runners.
- Produces: Ubuntu/Windows jobs that install only default/dev dependencies and run pytest, release preflight, JavaScript tests, sdist/wheel build, and wheel import smoke.

- [ ] **Step 1: Write failing package metadata tests**

```python
def test_package_is_the_release_candidate() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == "1.0.0rc1"
    assert data["project"]["license"] == "AGPL-3.0-or-later"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_package_metadata.py -q`

Expected: failure because the package version is still `0.1.0`.

- [ ] **Step 3: Update package metadata and lock**

Set version `1.0.0rc1`, SPDX license expression, classifiers, URLs that do not invent
an uncreated GitHub repository, and add the minimal build check dependency. Regenerate
`uv.lock` without optional GPU groups.

- [ ] **Step 4: Add and locally emulate the CI matrix**

Run: `uv sync --locked --no-install-project`

Run: `python -m pytest -q`

Run: `python scripts/repo_check.py`

Run: `node --test tests/js/browser_pipeline.test.mjs`

Run: `python -m build`

Expected: all commands pass without importing Torch or requiring a secret.

- [ ] **Step 5: Commit**

```text
ci: gate the CPU release on Windows and Ubuntu
```

### Task 6: Clean committed export and browser smoke

**Files:**
- Create: `scripts/clean_export_check.py`
- Create: `tests/test_clean_export.py`
- Modify: `docs/RELEASE_CHECKLIST.md`

**Interfaces:**
- Consumes: `git archive HEAD`, uv, Node, and the current CPU-only Python interpreter.
- Produces: a temporary ASCII-path export that runs tests, repository/release checks, JS tests, package build/install/import, and emits a JSON summary; it never copies ignored files.

- [ ] **Step 1: Write failing archive-safety tests**

```python
def test_archive_policy_rejects_private_and_runtime_paths() -> None:
    errors = archive_policy_errors(["README.md", "notes.private.md", "runs/x/best.pt"])
    assert errors == ["private path: notes.private.md", "runtime/model path: runs/x/best.pt"]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_clean_export.py -q`

Expected: collection fails because `scripts.clean_export_check` does not exist.

- [ ] **Step 3: Implement archive and package verification**

Use `git archive --format=tar HEAD`, safe tar extraction, exact command exit-code
capture, wheel content inspection, fresh wheel installation, and JSON output. Refuse
private/runtime paths before any command runs.

- [ ] **Step 4: Run clean export and headless browser smoke**

Run: `python scripts/clean_export_check.py`

Run the webapp-testing helper with the exported static directory and a native Python
Playwright script that waits for network idle, verifies headings/controls/module
loading, checks console/page errors, and captures a screenshot.

Expected: the archive contains only committed files; all CPU/package/browser gates pass.

- [ ] **Step 5: Commit**

```text
test: verify the clean committed release export
```

### Task 7: Final release audit

**Files:**
- Modify only if a verified gate exposes a release blocker.

**Interfaces:**
- Consumes: final `HEAD`, all refs, all commits, status, ignore rules, and remote configuration.
- Produces: a clean release-candidate verdict or an exact remaining limitation.

- [ ] **Step 1: Run the full verification suite fresh**

Run pytest, repository/release checks, Node tests, package build/install, clean export,
external anonymous link checks, and browser smoke from committed `HEAD`.

- [ ] **Step 2: Audit Git invariants**

Verify branch name, clean status, refs, no remotes, all author/committer identities,
no collaboration trailers, ignored `notes.private.md`, and no untracked release files.

- [ ] **Step 3: Record final owner actions and limitations**

List legal review, GitHub creation/push/CI enablement, and optional later HF sync as
owner-only actions. Preserve missing benchmark runtime versions and lack of full
DOTAv1 deployment certification as limitations.

- [ ] **Step 4: Mark the goal complete**

Only after every local gate and clean-status audit succeeds, update the active goal to
`complete` and report the branch, HEAD, commits, changed files, evidence, and absence
of remote mutations.

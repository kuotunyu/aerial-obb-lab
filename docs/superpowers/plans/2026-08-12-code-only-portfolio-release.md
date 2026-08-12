# Code-Only Portfolio Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a `v1.0.0rc2` public GitHub portfolio candidate that keeps the engineering evidence and BYOM demos while distributing no DOTA-derived raster image, dataset content, trained weight, or model binary.

**Architecture:** A schema-v2 distribution manifest records zero bundled third-party artifacts and preserves excluded historical artifacts as audit metadata. Static browser inference accepts local ONNX bytes through the File API; Python demos require an explicit local `MODEL_PATH`. Offline release, repository, browser, package, and clean-export gates enforce the code-only boundary from committed files.

**Tech Stack:** Python 3.11 standard-library release gates, pytest, vanilla JavaScript, ONNX Runtime Web 1.20.1 loaded from its pinned CDN, Playwright/Chromium synthetic smoke, uv 0.11.18, Hatchling, Git.

## Global Constraints

- Execute inline in this task; do not create subagents or additional threads.
- CPU only. Do not initialize CUDA, run model inference, train, validate, export models, or build TensorRT.
- Do not download DOTA, weights, or model binaries.
- Do not read, stage, or publish `notes.private.md`, interview material, secrets, or ignored runtime files.
- Do not create or mutate remotes, push, create a PR, tag, Release, or modify Hugging Face.
- Do not use `git add -A`; stage only exact reviewed paths.
- Do not amend, squash, rebase, or rewrite history.
- Author and committer must remain `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no `Co-authored-by` trailer.
- Use red-green TDD for each behavior change and preserve the accepted negative and bounded historical claims.
- Public package and release version is `1.0.0rc2`; documentation spelling is `v1.0.0-rc.2`.
- The final committed tree and release archive must contain no `.pt`, `.onnx`, `.engine`, `.torchscript`, `.tflite`, `.mlpackage`, DOTA dataset file, or DOTA-derived raster image.

---

## File responsibility map

- `release/artifact-manifest.json`: code-only distribution policy, empty bundled inventory, excluded historical artifact audit.
- `scripts/release_check.py`: claims, manifest, prohibited-path, owner-HF-link, and committed privacy enforcement.
- `scripts/repo_check.py`: general syntax, notebook, link, secret, and static BYOM-demo checks.
- `scripts/clean_export_check.py`: committed ZIP inventory, extraction, package, browser, and isolated-install verification.
- `demo/space-static/index.html`: model and image selection controls.
- `demo/space-static/app.js`: local model byte loading, session lifecycle, preprocess/inference/render orchestration.
- `demo/space-static/obb.js`: unchanged pure geometry and output-contract API.
- `scripts/browser_smoke.py`: headless synthetic model-selection and image-detection smoke.
- `demo/model_source.py`: pure explicit-local-model path validation shared by Python demos.
- `demo/app.py`, `demo/space/app.py`: optional local UI entry points with no download or fallback.
- `tests/test_release_check.py`: code-only policy and public-link regression tests.
- `scripts/browser_smoke.py`: executable BYOM behavior contract in headless Chromium.
- `tests/test_model_source.py`: explicit local model path behavior without ML imports.
- `tests/test_clean_export.py`: archive policy and required-member regressions.
- `README.md`, `README.zh-TW.md`, `docs/*.md`, demo READMEs: public code-only narrative and bounded evidence.
- `docs/OWNER_ACTIONS.md`: exact authenticated Hugging Face and GitHub steps reserved for the owner.
- `pyproject.toml`, `src/obbkit/__init__.py`, `uv.lock`, `CHANGELOG.md`, `CITATION.cff`, `RELEASE_CHECKLIST.md`: rc2 metadata.

---

### Task 1: Enforce and apply the code-only artifact boundary

**Files:**
- Modify: `tests/test_release_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `scripts/release_check.py`
- Modify: `scripts/clean_export_check.py`
- Modify: `release/artifact-manifest.json`
- Modify: `.gitignore`
- Delete: `demo/space-static/yolo26n-obb.onnx`
- Delete: `assets/hbb_vs_obb_1_P0706_ship.jpg`
- Delete: `assets/hbb_vs_obb_2_P2726_ship.jpg`
- Delete: `assets/hbb_vs_obb_3_P2124_large-vehicle.jpg`
- Delete: `assets/hbb_vs_obb_4_P1957_large-vehicle.jpg`
- Delete: `assets/hbb_vs_obb_5_P2781_large-vehicle.jpg`

**Interfaces:**
- Produces: `verify_code_only_paths(relative_paths: list[str], manifest: dict) -> list[str]`.
- Produces: manifest keys `bundled_third_party_artifacts` and `excluded_historical_artifacts`.
- Consumes later: clean-export and documentation tasks rely on the code-only manifest schema.

- [ ] **Step 1: Write failing release-policy tests**

Add tests equivalent to:

```python
def test_code_only_manifest_bundles_no_third_party_artifacts() -> None:
    manifest = json.loads((ROOT / "release/artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["distribution_mode"] == "code-only-byom"
    assert manifest["bundled_third_party_artifacts"] == []
    assert len(manifest["excluded_historical_artifacts"]) == 6


def test_committed_tree_contains_no_model_or_dota_visual() -> None:
    checker = load_release_check()
    manifest = checker.load_json(ROOT / "release" / "artifact-manifest.json")
    assert checker.verify_code_only_paths(checker.committed_paths(ROOT), manifest) == []
```

Change the clean-export policy test so `.onnx` and known `assets/hbb_vs_obb_*.jpg` members are rejected rather than accepted.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_release_check.py tests/test_clean_export.py -q
```

Expected: FAIL because schema v1 still bundles six restricted artifacts and `verify_code_only_paths` does not exist.

- [ ] **Step 3: Implement the manifest and path policy**

In `scripts/release_check.py`, define explicit suffix and historical-path policies:

```python
FORBIDDEN_MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlpackage"}
DOTA_DERIVED_VISUAL_RE = re.compile(r"^assets/hbb_vs_obb_.*\.(?:jpg|jpeg|png)$", re.I)


def verify_code_only_paths(relative_paths: list[str], manifest: dict) -> list[str]:
    errors = []
    normalized = {path.replace("\\", "/") for path in relative_paths}
    for path in sorted(normalized):
        if Path(path).suffix.casefold() in FORBIDDEN_MODEL_SUFFIXES:
            errors.append(f"code-only release contains model binary: {path}")
        if DOTA_DERIVED_VISUAL_RE.match(path):
            errors.append(f"code-only release contains DOTA-derived visual: {path}")
    for entry in manifest["excluded_historical_artifacts"]:
        if entry["path"] in normalized:
            errors.append(f"excluded historical artifact is still distributed: {entry['path']}")
    return errors
```

Replace the v1 manifest with schema v2. Move all six current artifact records, including accepted
hashes and provenance, into `excluded_historical_artifacts`; set
`bundled_third_party_artifacts` to `[]`; set `distribution_mode` to `code-only-byom`; state that
the historical artifacts are not distributed. Replace the former owner-hosted Space URL on the
ONNX record with the official Ultralytics OBB documentation URL and
`"historical_distribution_location_redacted": true`; the audit does not need to advertise the
owner's remote artifact location.

Update `verify_artifacts` to validate schema v2, require excluded entries to be absent, and hash only entries in `bundled_third_party_artifacts`. Call `verify_code_only_paths` from `main` and from archive inspection. Remove the ONNX exception from `.gitignore`.

- [ ] **Step 4: Remove the six tracked restricted artifacts explicitly**

Run one `git rm --` command with all six exact paths. Do not use a recursive wildcard and do not remove the `assets` directory itself.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_release_check.py tests/test_clean_export.py -q
```

Expected: PASS; `git ls-files '*.onnx' '*.pt' '*.engine'` returns no paths and `git ls-files 'assets/hbb_vs_obb_*'` returns no paths.

- [ ] **Step 6: Commit exact files**

Stage the two tests, two checkers, manifest, `.gitignore`, and the six explicit deletions. Commit:

```text
fix: exclude restricted release artifacts
```

---

### Task 2: Require a user-supplied ONNX model in the browser

**Files:**
- Modify: `demo/space-static/index.html`
- Modify: `demo/space-static/app.js`
- Modify: `demo/space-static/style.css`
- Modify: `demo/space-static/README.md`
- Modify: `scripts/browser_smoke.py`
- Modify: `scripts/repo_check.py`
- Modify: `release/evidence.json`
- Modify: `tests/test_release_check.py`

**Interfaces:**
- Produces: `loadModelFile(file: File) -> Promise<void>` in browser code.
- Produces: `updateDetectEnabled() -> void`, enabling detection only for ready model plus image.
- Consumes: unchanged `OBB.selectEndToEndOutput` and `OBB.decodeDetections` APIs.

- [ ] **Step 1: Turn the headless smoke into a failing BYOM behavior contract**

Update `scripts/browser_smoke.py` before changing the demo. The smoke must select an in-memory
model file, observe that Detect remains disabled until an image is also selected, and require the
ORT stub to receive non-empty `Uint8Array` bytes. Record every browser request and fail if any URL
other than loopback assets and the intercepted pinned ORT runtime is requested. Update the browser
evidence test to expect `distribution_mode == "bring-your-own-model"`, no model hash/size, and no
inherited accuracy or T4 latency.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.venv/Scripts/python.exe scripts/browser_smoke.py
.venv/Scripts/python.exe -m pytest tests/test_release_check.py -q
```

Expected: browser smoke FAIL because `#modelInput` is absent; evidence test FAIL because the
current browser record is tied to the former bundled model.

- [ ] **Step 3: Implement local model-byte loading**

Add a model file input and label above the image input. Implement:

```javascript
async function loadModelFile(file) {
  setStatus("loading local model...");
  const modelBytes = new Uint8Array(await file.arrayBuffer());
  const nextSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  if (session && typeof session.release === "function") await session.release();
  session = nextSession;
  modelLabel.textContent = file.name;
  setStatus("model ready; choose an image");
  updateDetectEnabled();
}
```

On failure, set `session = null`, show `model load failed: ...`, and keep Detect disabled. Remove `MODEL_URL`, `ensureSession`, and background loading. The detection handler must use the existing `session` directly and must return when either model or image is missing.

- [ ] **Step 4: Update headless smoke for in-memory model selection**

Make the ORT stub reject any `InferenceSession.create` argument that is not a non-empty `Uint8Array`. In Playwright:

```python
page.locator("#modelInput").set_input_files(
    files=[{
        "name": "synthetic-model.onnx",
        "mimeType": "application/octet-stream",
        "buffer": b"synthetic-not-a-real-model",
    }]
)
```

Assert Detect is disabled before the model, remains disabled before the image, becomes enabled after both selections, renders the expected ship row, and produces no console/page error. Continue intercepting the CDN runtime so the smoke has no external request and performs no inference.

- [ ] **Step 5: Update repository and evidence gates**

Remove the ONNX file from `check_static_demo.required`. Require `modelInput`, `fileInput`, `obb.js`, and `app.js`. Reject `MODEL_URL`, known bundled model names, and model-fetch fallbacks. Update `release/evidence.json.browser_demo` to describe BYOM source bytes, remove model hash/size/revision fields, and retain the no-accuracy/no-T4 limitations.

- [ ] **Step 6: Run focused and browser tests**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_browser_parity.py tests/test_release_check.py -q
.venv/Scripts/python.exe scripts/browser_smoke.py --screenshot dist/browser-smoke-rc2.png
.venv/Scripts/python.exe scripts/repo_check.py
```

Expected: all commands exit 0; the smoke reports synthetic local model bytes and one rendered detection.

- [ ] **Step 7: Commit exact files**

Commit:

```text
feat: require user-supplied browser model
```

---

### Task 3: Require explicit local models in Python demos

**Files:**
- Create: `demo/model_source.py`
- Create: `tests/test_model_source.py`
- Modify: `demo/app.py`
- Modify: `demo/space/app.py`
- Modify: `demo/space/requirements.txt`
- Modify: `demo/space/README.md`

**Interfaces:**
- Produces: `require_model_path(raw: str | None = None, *, allowed_suffixes: tuple[str, ...] = (".pt", ".onnx")) -> Path`.
- Consumes: `MODEL_PATH` and optional `MODEL_DEVICE`, defaulting to CPU.

- [ ] **Step 1: Write failing pure path-resolution tests**

Test missing configuration, nonexistent path, unsupported suffix, and a valid temporary `.onnx` file:

```python
def test_model_path_is_required(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_PATH is required"):
        require_model_path()


def test_existing_supported_model_is_resolved(tmp_path) -> None:
    model = tmp_path / "owner-model.onnx"
    model.write_bytes(b"fixture")
    assert require_model_path(str(model), allowed_suffixes=(".onnx",)) == model.resolve()
```

The path tests exercise only the pure helper. The repository release policy separately scans the
published demo entry points for forbidden remote acquisition/fallback constructs; do not add
source-text assertions to this unit test.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_model_source.py -q
```

Expected: import failure because `demo/model_source.py` does not exist.

- [ ] **Step 3: Implement pure model-path validation**

Implement `require_model_path` using only `os` and `pathlib`. Expand and resolve the path, require a regular file, normalize allowed suffixes case-insensitively, and raise actionable `RuntimeError` messages.

Update both demos to call the helper before constructing `YOLO`. Remove Hugging Face download, official-name fallback, implicit export, and CUDA auto-selection. Default `MODEL_DEVICE` to `cpu`; a user may explicitly override it outside the release verification environment. Remove `huggingface_hub` from `demo/space/requirements.txt`.

- [ ] **Step 4: Run and verify GREEN**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_model_source.py -q
.venv/Scripts/python.exe -m py_compile demo/model_source.py demo/app.py demo/space/app.py
```

Expected: PASS without importing Gradio, Torch, Ultralytics, or an ML model in the tests.

- [ ] **Step 5: Commit exact files**

Commit:

```text
fix: require explicit local demo model
```

---

### Task 4: Remove public owner-artifact dependencies and document handoff

**Files:**
- Modify: `tests/test_release_check.py`
- Modify: `scripts/release_check.py`
- Modify: `README.md`
- Modify: `README.zh-TW.md`
- Modify: `docs/training_results.md`
- Modify: `docs/analysis_results.md`
- Modify: `docs/model_card.md`
- Modify: `docs/per_class_metrics.json`
- Modify: `release/evidence.json`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `demo/space-static/README.md`
- Modify: `demo/space/README.md`
- Modify: `notebooks/03_recover_per_class_metrics_colab.py`
- Regenerate: `notebooks/03_recover_per_class_metrics_colab.ipynb`
- Create: `docs/OWNER_ACTIONS.md`

**Interfaces:**
- Produces: `verify_public_links(root: Path = ROOT) -> list[str]`.
- Produces: a recovery notebook requiring an owner-supplied checkpoint instead of the current public HF repository.

- [ ] **Step 1: Write failing public-link and recovery tests**

Add a test that `verify_public_links(ROOT)` returns no errors and a direct regression that the recovery notebook source contains neither the owner's HF namespace nor `hf_hub_download`.

The verifier scans an explicit `PUBLIC_PRESENTATION_FILES` tuple containing the two root READMEs,
training/analysis/model-card documents, demo HTML/JS/Markdown, and the recovery notebook source.
Design and implementation records under `docs/superpowers/` are historical process documents and
are not scanned as product presentation. Reject URLs matching:

```python
OWNER_HF_URL_RE = re.compile(
    r"https://huggingface\.co/(?:spaces/)?steven0226/(?:yolo26m-obb-dota|yolo26-obb-aerial-detection)",
    re.I,
)
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_release_check.py -q
```

Expected: FAIL listing the current README, model-card, training-result, demo, and recovery-notebook references.

- [ ] **Step 3: Remove owner-hosted download dependencies**

- Replace README live-demo/model links with links to the local BYOM demo folder and instructions.
- Remove the DOTA-derived image embed and state that comparison renders were excluded; keep quantitative tables.
- Replace model-card download snippets with explicit `MODEL_PATH` examples.
- Replace the recovery notebook's fixed HF repository download with a required `/content/best.pt` upload, followed by the existing checksum gate.
- Update the `.py` source first, then regenerate the paired `.ipynb` using Jupytext; do not hand-edit notebook JSON.
- Preserve model revision/hash evidence while replacing owner repository identifiers in machine-readable evidence with neutral historical-source wording.
- In `docs/per_class_metrics.json`, retain the accepted checkpoint hash/revision while replacing the public repository field with `"distribution": "not included; owner-supplied checkpoint required"`.

- [ ] **Step 4: Add exact owner instructions**

Create `docs/OWNER_ACTIONS.md` with these authenticated actions:

1. Hugging Face model `steven0226/yolo26m-obb-dota`: Settings -> Visibility -> Private; confirm the page is inaccessible anonymously.
2. Hugging Face Space `steven0226/yolo26-obb-aerial-detection`: Settings -> Visibility -> Private until its files are replaced by the BYOM folder; confirm anonymous access is denied.
3. GitHub: create an empty public repository named `yolo26-dota-obb` without generated files.
4. Add the remote and push only after copying/reviewing the URL; require all release-gate jobs before merge.
5. Create a tag/Release only after hosted CI is green.

State explicitly that these are not completed locally and require the owner's login.

- [ ] **Step 5: Run documentation, link, and notebook gates**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_release_check.py -q
.venv/Scripts/python.exe -m jupytext --sync notebooks/03_recover_per_class_metrics_colab.py
.venv/Scripts/python.exe scripts/repo_check.py
rg -n "https://huggingface.co/(spaces/)?steven0226/(yolo26m-obb-dota|yolo26-obb-aerial-detection)" README.md README.zh-TW.md docs demo notebooks/03_recover_per_class_metrics_colab.py
```

Expected: tests and repository gate pass; `rg` returns no public owner-artifact URL.

- [ ] **Step 6: Commit exact files**

Commit:

```text
docs: publish code-only portfolio guidance
```

---

### Task 5: Cut and verify the local rc2 candidate

**Files:**
- Modify: `tests/test_package_release.py`
- Modify: `tests/test_clean_export.py`
- Modify: `pyproject.toml`
- Modify: `src/obbkit/__init__.py`
- Modify: `uv.lock`
- Modify: `scripts/clean_export_check.py`
- Modify: `CHANGELOG.md`
- Modify: `CITATION.cff`
- Modify: `RELEASE_CHECKLIST.md`

**Interfaces:**
- Produces: package version `1.0.0rc2`.
- Produces: default clean export `dist/yolo26-dota-obb-v1.0.0rc2.zip`.

- [ ] **Step 1: Write failing rc2 and clean-export tests**

Change `EXPECTED_VERSION` to `1.0.0rc2`. Require the clean-export default filename to contain `v1.0.0rc2`, require no ONNX member, and require the BYOM HTML/JS, manifest, owner actions, browser fixture, and all release gates.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_package_release.py tests/test_clean_export.py -q
```

Expected: FAIL because project metadata and clean-export defaults remain rc1.

- [ ] **Step 3: Bump metadata and update final release records**

- Set `pyproject.toml` and `src/obbkit/__init__.py` to `1.0.0rc2`.
- Run `uv lock` and verify `uv lock --check`.
- Add `1.0.0-rc.2` to `CHANGELOG.md` describing code-only distribution and BYOM demos.
- Update `CITATION.cff`, manifest, evidence, and checklist release identifiers to rc2.
- Change the clean-export default path and required-member set; remove the ONNX member and require `docs/OWNER_ACTIONS.md`.
- Mark only local automated/manual gates complete; leave authenticated owner actions unchecked.

- [ ] **Step 4: Run the complete pre-commit verification**

Set `CUDA_VISIBLE_DEVICES=-1`, then run:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/repo_check.py
.venv/Scripts/python.exe scripts/release_check.py
.venv/Scripts/python.exe scripts/browser_smoke.py --screenshot dist/browser-smoke-rc2.png
uv lock --check
uv build --no-sources --out-dir dist/package-rc2-check
git diff --check
```

Expected: all commands exit 0; no Torch/CUDA/model runtime is imported by release tests.

- [ ] **Step 5: Commit exact rc2 files**

Commit:

```text
test: verify code-only clean export
```

- [ ] **Step 6: Build and verify the committed clean export**

Confirm `git status --short` is empty. Remove only the exact ignored stale file `dist/yolo26-dota-obb-v1.0.0rc2.zip` if it already exists, after resolving it inside the workspace. Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe scripts/clean_export_check.py
```

Expected: archive inspection,  full pytest, repository gate, release gate, headless browser smoke, wheel/sdist build, and isolated wheel import all pass from committed files. Record the final ZIP byte size and SHA-256 in the handoff report.

- [ ] **Step 7: Perform final Git, privacy, identity, and remote audit**

Verify:

- Branch is `portfolio/obb-v1.0-release-hardening`.
- Working tree, staged set, and untracked set are empty.
- No tracked private/interview path exists and ignore rules still match them.
- Every author and committer is the allowed identity.
- No `Co-authored-by` trailer exists.
- No remote, remote ref, tag, push, PR, Release, or HF mutation exists.
- `git ls-files` contains no model suffix or DOTA-derived visual path.

- [ ] **Step 8: Handoff owner-only operations**

Report completed local work, exact commits, changed/deleted files, test evidence, archive hash, remaining historical evidence limitations, and the numbered instructions in `docs/OWNER_ACTIONS.md`. Do not mark the external actions complete.

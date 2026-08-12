# Aerial OBB Lab Release Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan inline. Subagents are prohibited by the approved brief. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** Give the unpushed release candidate one coherent `Aerial OBB Lab` identity across its
public presentation, package metadata, owner handoff, notebooks, and clean export.

**Architecture:** Treat release identity as metadata and policy, not as a refactor of the stable
`obbkit` import or historical experiment provenance. Tests protect machine-consumed names and the
private-HF boundary; human-facing documents carry the approved zh-TW-first brand.

**Tech Stack:** Python 3.11, pytest, TOML metadata, JSON evidence, Markdown, Jupytext, GitHub Actions.

## Global Constraints

- CPU only with `CUDA_VISIBLE_DEVICES=-1`; do not run training, model inference, validation,
  TensorRT build, or export.
- Keep version `1.0.0rc2`, module import `obbkit`, and historical experiment run names unchanged.
- Do not read or stage `notes.private.md`, weights, datasets, screenshots, or runtime outputs.
- Use exact-path staging only; never use `git add -A`.
- Author and committer must be `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no
  trailers.
- Do not add a remote, push, create a PR/tag/Release, create a Space, or mutate Hugging Face.

---

### Task 1: Define the release-identity contract

**Files:**

- Modify: `tests/test_package_release.py`
- Modify: `tests/test_clean_export.py`
- Modify: `tests/test_release_check.py`

**Interfaces:**

- Consumes: `[project]` metadata from `pyproject.toml`, `DEFAULT_OUTPUT` from
  `scripts.clean_export_check`, and `verify_public_links()` from `scripts.release_check`.
- Produces: failing contracts for the `aerial-obb-lab` distribution, GitHub repository URL,
  release archive filename, and private-HF public-link rejection.

- [ ] Add `test_release_identity_matches_aerial_obb_lab()` that parses `pyproject.toml` and asserts
  `project.name == "aerial-obb-lab"` and
  `project.urls.Repository == "https://github.com/kuotunyu/aerial-obb-lab"`.
- [ ] Change the existing clean-export filename expectation to
  `aerial-obb-lab-v1.0.0rc2.zip`.
- [ ] Add a temporary-root `verify_public_links()` test that injects
  `https://huggingface.co/steven0226/aerial-obb-lab-model-archive` into one public presentation
  file and expects it to be rejected.
- [ ] Run `uv run --no-sync python -m pytest tests/test_package_release.py
  tests/test_clean_export.py tests/test_release_check.py -q` and confirm failures name only the
  missing new identity behavior.
- [ ] Stage only the three test files and commit `test: define aerial obb lab release identity`.

### Task 2: Apply the bounded release identity

**Files:**

- Modify: `pyproject.toml`
- Modify: `scripts/clean_export_check.py`
- Modify: `scripts/release_check.py`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `CITATION.cff`
- Modify: `docs/model_card.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: the tests from Task 1.
- Produces: `aerial-obb-lab` distribution metadata, repository URL, clean-export filename, current
  public brand, and rejection of the current private HF archive from public presentation.

- [ ] Change `[project].name` to `aerial-obb-lab`, use an engineering-lab description, and add
  `[project.urls] Repository = "https://github.com/kuotunyu/aerial-obb-lab"`.
- [ ] Change `DEFAULT_OUTPUT` to
  `ROOT / "dist" / "aerial-obb-lab-v1.0.0rc2.zip"`.
- [ ] Expand `OWNER_HF_ARTIFACT_RE` so the current private archive and both historical owner
  artifact names remain forbidden in public presentation files.
- [ ] Change the README titles to `Aerial OBB Lab` plus the approved zh-TW/English subtitle while
  preserving every bounded claim block verbatim.
- [ ] Change the citation and model-card display titles without changing model or dataset facts.
- [ ] Add a changelog bullet recording the release-identity change.
- [ ] Run the focused tests from Task 1 and confirm they pass.
- [ ] Stage exactly the eight implementation files and commit
  `docs: adopt aerial obb lab release identity`.

### Task 3: Reconcile owner handoff and notebook archive naming

**Files:**

- Modify: `docs/OWNER_ACTIONS.md`
- Modify: `RELEASE_CHECKLIST.md`
- Modify: `tests/test_readme_language.py`
- Modify: `notebooks/01_train_dotav1_a100.py`
- Modify: `notebooks/01_train_dotav1_a100.ipynb`
- Modify: `notebooks/02_benchmark_colab.py`
- Modify: `notebooks/02_benchmark_colab.ipynb`

**Interfaces:**

- Consumes: owner-confirmed HF and GitHub state.
- Produces: an exact remaining-action list and notebook archive target
  `aerial-obb-lab-model-archive`, with paired `.py`/`.ipynb` sources synchronized.

- [ ] Update the README-language/owner-copy expectation to the approved zh-TW GitHub description.
- [ ] Rewrite `docs/OWNER_ACTIONS.md` into completed, not-applicable, and remaining sections; do
  not instruct creation of a Space.
- [ ] Mark the HF privacy/rename and empty GitHub creation items complete in
  `RELEASE_CHECKLIST.md`; retain push, hosted CI, tag, and Release as incomplete.
- [ ] Replace only the notebook HF repository-name variables/comments with
  `aerial-obb-lab-model-archive`; do not change run names or execute cells.
- [ ] Run `uv run --no-sync jupytext --sync notebooks/01_train_dotav1_a100.py
  notebooks/02_benchmark_colab.py` and verify notebook outputs/execution counts remain empty.
- [ ] Run `uv run --no-sync python -m pytest tests/test_readme_language.py
  tests/test_repo_check.py tests/test_release_check.py -q`.
- [ ] Stage only the seven files and commit `docs: record aerial obb lab owner handoff`.

### Task 4: Rebuild and audit the unpushed candidate

**Files:**

- Generated but ignored: `dist/aerial-obb-lab-v1.0.0rc2.zip`
- No tracked file changes expected.

**Interfaces:**

- Consumes: the committed branch from Tasks 1–3.
- Produces: fresh CPU-only test/package/browser/clean-export evidence and final Git audit.

- [ ] Run full pytest, `scripts/repo_check.py`, `scripts/release_check.py`, static browser smoke,
  Gradio UI smoke, `uv lock --check`, `uv build --no-sources`, and `git diff --check` with GPU
  visibility disabled.
- [ ] Run `scripts/clean_export_check.py` from the clean committed tree and record archive bytes and
  SHA-256.
- [ ] Verify the final branch, HEAD, clean status, all refs, zero remotes/tags/remote refs, allowed
  authors/committers, no trailers, and ignored private/runtime paths without reading them.
- [ ] Verify the empty GitHub repository read-only and state explicitly that no code, branch, tag,
  PR, Release, or HF mutation was performed by the local workflow.


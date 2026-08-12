# zh-TW README and Gradio UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline to implement this plan task-by-task. Do not delegate it to subagents or additional threads. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GitHub presentation zh-TW-first and replace the duplicated Gradio layouts with one readable, wide, model-free-previewable workbench.

**Architecture:** `README.md` becomes the canonical Traditional Chinese document while `README.en.md` preserves the complete English claims. A standard-library UI contract feeds a shared Gradio 6.20 builder used by both real BYOM entry points and a loopback-only UI preview; release and Playwright gates verify language, interaction, responsive layout, privacy, and the absence of ML-runtime imports in preview mode.

**Tech Stack:** Python 3.11, Gradio 6.20.0, vanilla CSS, Playwright/Chromium, pytest, uv 0.11.18, GitHub Actions, standard-library release gates.

## Global Constraints

- Execute inline in this task; do not create subagents or additional threads.
- Continue the unpushed `v1.0.0-rc.2` candidate on `portfolio/obb-v1.0-release-hardening`.
- Public prose is Traditional Chinese; established technical terms remain in their original form.
- Use the approved `96vw`, `1720px`, `38% / 62%`, 32px title, 18px body/action, 16px label, and 900px responsive breakpoint values exactly.
- The real demos continue requiring explicit local `MODEL_PATH`; preview mode imports no Torch, Ultralytics, model binary, or Hugging Face client.
- Set `CUDA_VISIBLE_DEVICES=-1` for every verification command. Do not train, validate, export, build TensorRT, or run model inference.
- Do not download DOTA, weights, or model binaries and do not use a secret or HF token.
- Do not create or mutate a remote, push, tag, PR, Release, or Hugging Face repository/Space.
- Do not read, stage, or publish `notes.private.md`, interview material, ignored screenshots, or runtime output.
- Do not use `git add -A`, amend, rebase, squash, or rewrite history.
- Author and committer remain `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no `Co-authored-by` trailer.
- Use red-green TDD for behavior changes and commit exact reviewed paths only.

---

## File responsibility map

- `README.md`: canonical complete Traditional Chinese GitHub/package landing page.
- `README.en.md`: complete English presentation and bounded claim blocks.
- `README.zh-TW.md`: compatibility pointer to the canonical README.
- `docs/OWNER_ACTIONS.md`: zh-TW GitHub About description and topics for authenticated owner use.
- `scripts/release_check.py`: bilingual claim targets, public presentation inventory, demo acquisition and interaction policy.
- `scripts/clean_export_check.py`: required bilingual/UI/smoke members and committed-snapshot rebuild.
- `demo/ui_contract.py`: standard-library-only labels, safe model display, status, safe error text, and detection summaries.
- `demo/gradio_ui.py`: one Gradio component tree and event flow shared by both real demos and preview.
- `demo/gradio.css`: approved desktop and responsive visual system.
- `demo/gradio_preview.py`: loopback-only UI preview with no ML runtime or fabricated detections.
- `demo/app.py`, `demo/space/app.py`: model-specific inference adapters that call `build_demo`.
- `scripts/gradio_ui_smoke.py`: Playwright desktop/narrow UI contract using the real Gradio preview.
- `tests/test_readme_language.py`: canonical-language and About guidance behavior.
- `tests/test_gradio_ui_contract.py`: pure UI state/summary/error behavior.
- `tests/test_release_check.py`: published source-policy regressions.
- `tests/test_package_release.py`: dependency-group and CI matrix contract.
- `tests/test_clean_export.py`: required release member contract.
- `.github/workflows/release-gates.yml`: CPU-only UI smoke in the existing browser job.
- `release/evidence.json`, `CHANGELOG.md`, `RELEASE_CHECKLIST.md`: rc2 UI evidence and completed local gates.

---

### Task 1: Make Traditional Chinese the canonical repository presentation

**Files:**
- Create: `README.en.md`
- Create: `tests/test_readme_language.py`
- Modify: `README.md`
- Modify: `README.zh-TW.md`
- Modify: `docs/OWNER_ACTIONS.md`
- Modify: `pyproject.toml`
- Modify: `scripts/release_check.py`
- Modify: `scripts/clean_export_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `tests/test_package_release.py`

**Interfaces:**
- Produces: canonical `README.md`, secondary `README.en.md`, compatibility `README.zh-TW.md`.
- Produces: `verify_readme_language_structure(root: Path = ROOT) -> list[str]`.
- Consumes later: clean export and final release checks require all three README paths.

- [ ] **Step 1: Write failing language-structure and About tests**

Create `tests/test_readme_language.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_release_check():
    path = ROOT / "scripts" / "release_check.py"
    spec = importlib.util.spec_from_file_location("release_check_language", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_language_structure_is_zh_tw_first() -> None:
    checker = load_release_check()
    assert checker.verify_readme_language_structure(ROOT) == []

    canonical = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")
    compatibility = (ROOT / "README.zh-TW.md").read_text(encoding="utf-8")

    assert canonical.startswith("正體中文 | [English](README.en.md)")
    assert english.startswith("[正體中文](README.md) | English")
    assert len(compatibility) < 500
    assert "[README.md](README.md)" in compatibility
    assert "<!-- claim:" not in compatibility


def test_owner_actions_recommends_zh_tw_about_metadata() -> None:
    text = (ROOT / "docs" / "OWNER_ACTIONS.md").read_text(encoding="utf-8")
    assert "Code-only YOLO26 OBB × DOTA 作品集" in text
    assert "deployment benchmark" in text
    assert "BYOM demo" in text
    assert "`zh-tw`" in text
    assert "Website field" in text
```

Extend `tests/test_clean_export.py::test_clean_export_keeps_its_own_gate_and_browser_fixture` with
`"README.en.md"`.

Update `tests/test_package_release.py::test_sdist_explicitly_excludes_demo_models_and_dota_visuals`
so its exact include set also requires `"/README.en.md"`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_readme_language.py tests/test_release_check.py tests/test_clean_export.py -q
```

Expected: FAIL because `README.en.md` and `verify_readme_language_structure` do not exist, the root
README is English, and the clean-export and sdist inventories lack the English README.

- [ ] **Step 3: Move the two complete READMEs without copying content through the shell**

Run exactly:

```powershell
git mv -- README.md README.en.md
git mv -- README.zh-TW.md README.md
```

Use `apply_patch` to add this first line to `README.md`:

```markdown
正體中文 | [English](README.en.md)
```

Change its final language link to:

```markdown
*English version: [README.en.md](README.en.md)*
```

Use `apply_patch` to add this first line to `README.en.md`:

```markdown
[正體中文](README.md) | English
```

Change its final language link to:

```markdown
*正體中文版本：[README.md](README.md)*
```

Create the compatibility pointer with `apply_patch`:

```markdown
# 正體中文版已移至 README.md

請閱讀 canonical [正體中文 README](README.md)。

English readers can use [README.en.md](README.en.md).
```

- [ ] **Step 4: Implement bilingual claim and presentation routing**

In `scripts/release_check.py`, change `CLAIM_FILES` so every former English `README.md` tuple points
to `README.en.md`, every former Chinese `README.zh-TW.md` tuple points to `README.md`, and the exact
required tokens stay unchanged for their language.

Change `PUBLIC_PRESENTATION_FILES` to include all three README files. Add:

```python
def verify_readme_language_structure(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    canonical = (root / "README.md").read_text(encoding="utf-8")
    english = (root / "README.en.md").read_text(encoding="utf-8")
    compatibility = (root / "README.zh-TW.md").read_text(encoding="utf-8")
    if not canonical.startswith("正體中文 | [English](README.en.md)"):
        errors.append("README.md: canonical zh-TW language navigation is missing")
    if not english.startswith("[正體中文](README.md) | English"):
        errors.append("README.en.md: English language navigation is missing")
    if (
        len(compatibility) >= 500
        or "[README.md](README.md)" not in compatibility
        or "<!-- claim:" in compatibility
    ):
        errors.append("README.zh-TW.md: expected a short canonical-README pointer")
    return errors
```

Call it from `main()` immediately after `verify_claims(ROOT)`.

Add `"README.en.md"` to `scripts/clean_export_check.py::REQUIRED_MEMBERS`.
Add `"/README.en.md"` to `tool.hatch.build.targets.sdist.include` in `pyproject.toml`; keep
`readme = "README.md"` unchanged so package metadata renders the canonical zh-TW document.

- [ ] **Step 5: Update the owner-only GitHub About instructions**

Replace the suggested description and topics in `docs/OWNER_ACTIONS.md` with:

```markdown
3. Suggested description: `Code-only YOLO26 OBB × DOTA 作品集：誠實評估、deployment benchmark、BYOM demo 與可重現 release gates。`
4. Suggested topics: `computer-vision`, `object-detection`, `oriented-bounding-box`, `obb`,
   `yolo`, `dota`, `onnx`, `tensorrt`, `onnxruntime`, `gradio`, `byom`, `mlops`,
   `reproducibility`, `portfolio`, `zh-tw`.
5. Leave the Website field empty until a reviewed BYOM site is deliberately published.
```

- [ ] **Step 6: Run focused gates and verify GREEN**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_readme_language.py tests/test_release_check.py tests/test_clean_export.py tests/test_package_release.py -q
.venv/Scripts/python.exe scripts/repo_check.py
.venv/Scripts/python.exe scripts/release_check.py
git diff --check
```

Expected: all commands exit 0; local Markdown links resolve and both full READMEs retain every
bounded claim block.

- [ ] **Step 7: Commit exact language files**

Stage only the paths listed in this task and commit:

```text
docs: make zh-tw the primary presentation
```

---

### Task 2: Define a pure Gradio UI contract

**Files:**
- Create: `demo/ui_contract.py`
- Create: `tests/test_gradio_ui_contract.py`

**Interfaces:**
- Produces: `detection_summary(rows: Sequence[Sequence[object]]) -> str`.
- Produces: `detect_enabled(has_image: bool, *, preview: bool = False) -> bool`.
- Produces: `image_status(has_image: bool, *, preview: bool = False) -> str`.
- Produces: `safe_model_label(value: object) -> str`.
- Produces: `safe_detection_error() -> str`.
- Consumes later: `demo/gradio_ui.py` uses these functions without duplicating copy or exposing exceptions.

- [ ] **Step 1: Write failing pure contract tests**

Create `tests/test_gradio_ui_contract.py`:

```python
from __future__ import annotations

from demo.ui_contract import (
    detect_enabled,
    detection_summary,
    image_status,
    safe_detection_error,
    safe_model_label,
)


def test_detection_summary_handles_empty_and_ranked_rows() -> None:
    assert detection_summary([]) == "偵測數量：**0** · Top confidence：**—**"
    assert detection_summary(
        [["ship", 0.91, 100.0, 50.0, 90.0], ["plane", 0.73, 80.0, 40.0, 12.0]]
    ) == "偵測數量：**2** · Top confidence：**0.910**"


def test_image_status_is_explicit_and_preview_is_labeled() -> None:
    assert image_status(False) == "Model ready；請先選擇圖片。"
    assert image_status(True) == "圖片已就緒；調整設定後按 Detect。"
    assert image_status(False, preview=True) == "UI-only preview；不會執行 inference。"


def test_detect_requires_an_image_and_preview_stays_disabled() -> None:
    assert detect_enabled(False) is False
    assert detect_enabled(True) is True
    assert detect_enabled(True, preview=True) is False


def test_safe_detection_error_never_echoes_exception_details() -> None:
    message = safe_detection_error()
    assert message == "Detect failed；請檢查 local model 與 OBB output contract 後重試。"
    assert "C:\\" not in message
    assert "Traceback" not in message


def test_safe_model_label_hides_paths_and_escapes_html() -> None:
    assert safe_model_label(r"C:\private\model<script>.onnx") == "model&lt;script&gt;.onnx"
    assert safe_model_label("/private/model.onnx") == "model.onnx"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_gradio_ui_contract.py -q
```

Expected: import failure because `demo/ui_contract.py` does not exist.

- [ ] **Step 3: Implement the standard-library contract**

Create `demo/ui_contract.py`:

```python
"""Pure copy and summary helpers for the zh-TW Gradio workbench."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape


def detection_summary(rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "偵測數量：**0** · Top confidence：**—**"
    top = max(float(row[1]) for row in rows)
    return f"偵測數量：**{len(rows)}** · Top confidence：**{top:.3f}**"


def detect_enabled(has_image: bool, *, preview: bool = False) -> bool:
    return has_image and not preview


def image_status(has_image: bool, *, preview: bool = False) -> str:
    if preview:
        return "UI-only preview；不會執行 inference。"
    if has_image:
        return "圖片已就緒；調整設定後按 Detect。"
    return "Model ready；請先選擇圖片。"


def safe_model_label(value: object) -> str:
    basename = str(value).replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return escape(basename)


def safe_detection_error() -> str:
    return "Detect failed；請檢查 local model 與 OBB output contract 後重試。"
```

- [ ] **Step 4: Run the pure tests and compilation**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_gradio_ui_contract.py -q
.venv/Scripts/python.exe -m py_compile demo/ui_contract.py
```

Expected: five tests pass without importing Gradio, Torch, Ultralytics, or a model.

- [ ] **Step 5: Commit the pure contract**

Stage the two exact files and commit:

```text
test: define gradio ui contract
```

---

### Task 3: Build the shared wide Gradio workbench and model-free preview

**Files:**
- Create: `demo/gradio_ui.py`
- Create: `demo/gradio.css`
- Create: `demo/gradio_preview.py`
- Modify: `demo/app.py`
- Modify: `demo/space/app.py`
- Modify: `demo/space/README.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `scripts/release_check.py`
- Modify: `tests/test_release_check.py`
- Modify: `tests/test_package_release.py`

**Interfaces:**
- Consumes: `detect_enabled`, `detection_summary`, `image_status`, `safe_model_label`, and
  `safe_detection_error` from Task 2.
- Produces: `build_demo(*, detect_fn, class_names, model_name, device, imgsz, preview=False) -> gr.Blocks`.
- Produces: CLI `python demo/gradio_preview.py --port PORT` bound to `127.0.0.1` with `share=False`.
- Consumes later: the Playwright smoke targets `#app-header`, `#input-panel`, `#result-panel`, and `#detect-button`.

- [ ] **Step 1: Write failing source-policy and dependency tests**

Add to `tests/test_release_check.py`:

```python
def test_gradio_sources_use_shared_explicit_detect_flow() -> None:
    checker = load_release_check()
    assert checker.verify_gradio_interaction_sources(ROOT) == []


def test_gradio_source_policy_rejects_upload_inference() -> None:
    checker = load_release_check()
    assert checker.gradio_interaction_source_errors(
        {"legacy.py": "inp.upload(detect, [inp], [out])"}
    ) == ["legacy.py: upload-triggered inference"]


def test_preview_source_has_no_ml_or_remote_model_import() -> None:
    text = (ROOT / "demo" / "gradio_preview.py").read_text(encoding="utf-8")
    for forbidden in ("torch", "ultralytics", "huggingface_hub", "MODEL_PATH"):
        assert forbidden not in text
```

Add to `tests/test_package_release.py`:

```python
def test_ui_preview_dependency_group_excludes_ml_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = project["dependency-groups"]
    assert groups["ui-preview"] == ["gradio==6.20.0"]
    assert {item["include-group"] for item in groups["demo"] if isinstance(item, dict)} == {
        "local-ml",
        "ui-preview",
    }
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_release_check.py tests/test_package_release.py -q
```

Expected: FAIL because the shared builder, preview, source-policy functions, and `ui-preview` group
do not exist.

- [ ] **Step 3: Add the interaction source policy**

In `scripts/release_check.py`, add:

```python
GRADIO_ENTRYPOINT_FILES = ("demo/app.py", "demo/space/app.py")
GRADIO_UI_SOURCE_FILES = ("demo/gradio_ui.py",)


def gradio_interaction_source_errors(sources: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for relative in sorted(sources):
        if re.search(r"\.upload\s*\(", sources[relative]):
            errors.append(f"{relative}: upload-triggered inference")
    return errors


def verify_gradio_interaction_sources(root: Path = ROOT) -> list[str]:
    sources = {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in GRADIO_ENTRYPOINT_FILES + GRADIO_UI_SOURCE_FILES
    }
    errors = gradio_interaction_source_errors(sources)
    for relative in GRADIO_ENTRYPOINT_FILES:
        text = sources[relative]
        if "build_demo(" not in text or "gr.Blocks(" in text:
            errors.append(f"{relative}: shared Gradio builder is not used")
    ui_text = sources["demo/gradio_ui.py"]
    if (
        ".click(" not in ui_text
        or ".failure(" not in ui_text
        or "interactive=False" not in ui_text
    ):
        errors.append("demo/gradio_ui.py: explicit disabled Detect flow is missing")
    return errors
```

Call `verify_gradio_interaction_sources(ROOT)` from `main()` after
`verify_demo_model_sources(ROOT)`.

- [ ] **Step 4: Add the exact UI dependency groups and refresh the lock**

In `pyproject.toml`, replace the direct Gradio entry in `demo` with:

```toml
ui-preview = [
    "gradio==6.20.0",
]
demo = [
    { include-group = "local-ml" },
    { include-group = "ui-preview" },
]
```

Run:

```powershell
uv lock
uv lock --check
```

Expected: the lock remains on Gradio 6.20.0 and the project metadata records both groups.

- [ ] **Step 5: Create the approved CSS contract**

Create `demo/gradio.css` with these exact structural rules, followed by component-specific spacing,
border, focus, dark-result-canvas, and semantic status rules using the same selectors:

```css
.gradio-container {
  width: 96vw !important;
  max-width: 1720px !important;
  margin-inline: auto !important;
  font-size: 18px !important;
}
#app-header { padding: 20px 26px; }
#app-header h1 { font-size: 32px; line-height: 1.15; }
#workbench-grid { gap: 20px; }
#input-panel { flex: 38 1 0%; min-width: 360px; }
#result-panel { flex: 62 1 0%; min-width: 0; }
#detect-button button { min-height: 50px; font-size: 18px; font-weight: 800; }
.form label, .block label { font-size: 16px !important; }
#input-image, #result-image { min-height: 360px; }
button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible {
  outline: 3px solid #60a5fa !important;
  outline-offset: 2px;
}
@media (max-width: 900px) {
  .gradio-container { width: calc(100vw - 24px) !important; font-size: 18px !important; }
  #workbench-grid { flex-direction: column !important; }
  #input-panel, #result-panel { min-width: 100% !important; }
}
```

Do not use remote fonts, images, animations, or external CSS.

- [ ] **Step 6: Implement the shared builder**

Create `demo/gradio_ui.py`. Load CSS through
`Path(__file__).with_name("gradio.css").read_text(encoding="utf-8")`; render `model_name` through
`safe_model_label`, and escape `device` and `imgsz` with `html.escape` before inserting them into
`gr.HTML`.

Implement `build_demo` with this component/event shape:

```python
def build_demo(*, detect_fn, class_names, model_name, device, imgsz, preview=False):
    css = Path(__file__).with_name("gradio.css").read_text(encoding="utf-8")
    with gr.Blocks(
        title="YOLO26 OBB 航拍旋轉框偵測",
        css=css,
        fill_width=True,
        analytics_enabled=False,
    ) as app:
        gr.HTML(render_header(model_name, device, imgsz, preview), elem_id="app-header")
        status = gr.Markdown(image_status(False, preview=preview), elem_id="app-status")
        with gr.Row(elem_id="workbench-grid"):
            with gr.Column(scale=38, min_width=360, elem_id="input-panel"):
                inp = gr.Image(type="numpy", label="Input image", elem_id="input-image")
                conf = gr.Slider(
                    0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold"
                )
                classes = gr.Dropdown(
                    choices=list(class_names),
                    multiselect=True,
                    label="Class filter（留空 = 全部）",
                )
                detect_button = gr.Button(
                    "開始 Detect",
                    variant="primary",
                    interactive=False,
                    elem_id="detect-button",
                )
            with gr.Column(scale=62, min_width=0, elem_id="result-panel"):
                out_img = gr.Image(label="Detection result", elem_id="result-image")
                summary = gr.Markdown(detection_summary([]), elem_id="detection-summary")
                out_table = gr.Dataframe(
                    headers=["class", "conf", "w(px)", "h(px)", "angle(°)"],
                    label="Detection list",
                    interactive=False,
                )

        def set_image_state(image):
            ready = detect_enabled(image is not None, preview=preview)
            return gr.Button(value="開始 Detect", interactive=ready), image_status(
                image is not None, preview=preview
            )

        def mark_stale(image):
            return image_status(image is not None, preview=preview)

        def run_detection(image, threshold, selected):
            if image is None:
                raise gr.Error("請先選擇圖片。")
            try:
                annotated, rows = detect_fn(image, threshold, selected)
            except Exception:
                traceback.print_exc()
                raise gr.Error(safe_detection_error(), print_exception=False) from None
            return annotated, rows, detection_summary(rows), "Detect 完成。"

        inp.input(set_image_state, inp, [detect_button, status], queue=False)
        inp.clear(set_image_state, inp, [detect_button, status], queue=False)
        conf.change(mark_stale, inp, status, queue=False)
        classes.change(mark_stale, inp, status, queue=False)
        detection_event = detect_button.click(
            run_detection,
            [inp, conf, classes],
            [out_img, out_table, summary, status],
            trigger_mode="once",
            concurrency_limit=1,
            show_progress="full",
        )
        detection_event.failure(
            lambda: safe_detection_error(), inputs=None, outputs=status, queue=False
        )
    return app
```

`render_header` must label preview as `UI-only preview` and real mode as `Model ready`, call
`safe_model_label` so it shows only an HTML-escaped basename, and contain no absolute path.
The `.failure(...)` continuation is required because raising `gr.Error` halts the primary callback;
it updates the persistent status line with the same safe message while the modal reports the error.

- [ ] **Step 7: Refactor both real entry points to use the builder**

In both `demo/app.py` and `demo/space/app.py`:

- remove `import gradio as gr`;
- import `build_demo` from `gradio_ui` after ensuring the `demo` directory is on `sys.path`;
- keep `require_model_path`, `YOLO`, `DEVICE`, `IMGSZ`, and each existing `detect` adapter;
- replace the inline `gr.Blocks` tree and all `.upload(...)` bindings with:

```python
app = build_demo(
    detect_fn=detect,
    class_names=[str(name) for name in NAMES.values()],
    model_name=MODEL_PATH.name,
    device=DEVICE,
    imgsz=IMGSZ,
)
```

Use `demo = app` in `demo/space/app.py` if its Space frontmatter still expects that variable. Remove
the broken late `Path`-based `sys.path` mutation from `demo/app.py`. Keep `app.launch()` or
`demo.launch()` only inside `if __name__ == "__main__":`.

- [ ] **Step 8: Add the model-free preview entry point**

Create `demo/gradio_preview.py` using only `argparse`, `sys`, `gradio_ui`, and fixed DOTA class-name
strings. It must parse `--port` as an integer and launch exactly:

```python
app.launch(
    server_name="127.0.0.1",
    server_port=args.port,
    share=False,
    show_error=False,
    inbrowser=args.open,
)
```

Call `build_demo(..., model_name="preview-model.onnx", device="CPU", imgsz=1024, preview=True)`.
Pass a `preview_detect` callable that raises `RuntimeError("preview mode has no inference")`; the
button remains disabled because `preview=True`, so the callable is a fail-closed guard.

- [ ] **Step 9: Update the Gradio README**

In `demo/space/README.md`, lead in Traditional Chinese, retain technical terms in original form,
document the real command with `MODEL_PATH`, and add the preview command:

```powershell
uv sync --frozen --no-install-project --group ui-preview
.venv/Scripts/python.exe demo/gradio_preview.py --open
```

State explicitly that preview mode loads no model and performs no inference.

- [ ] **Step 10: Run focused tests and no-ML preview import verification**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_gradio_ui_contract.py tests/test_release_check.py tests/test_package_release.py -q
.venv/Scripts/python.exe -m py_compile demo/ui_contract.py demo/gradio_ui.py demo/gradio_preview.py demo/app.py demo/space/app.py
uv sync --frozen --no-install-project --group ui-preview
.venv/Scripts/python.exe -c "import runpy,sys; sys.argv=['demo/gradio_preview.py','--help']; runpy.run_path('demo/gradio_preview.py',run_name='__main__'); assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
uv lock --check
git diff --check
```

Expected: all tests/compilation pass; the help command exits 0 after argparse output and neither ML
runtime appears in `sys.modules`.

- [ ] **Step 11: Commit the shared UI implementation**

Stage only the files listed in this task and commit:

```text
feat: redesign gradio workbench
```

---

### Task 4: Add a real Gradio UI smoke and CPU-only CI gate

**Files:**
- Create: `scripts/gradio_ui_smoke.py`
- Modify: `.github/workflows/release-gates.yml`
- Modify: `scripts/clean_export_check.py`
- Modify: `tests/test_clean_export.py`
- Modify: `tests/test_package_release.py`

**Interfaces:**
- Produces: `scripts/gradio_ui_smoke.py --screenshot PATH` with loopback-only network policy.
- Consumes: `demo/gradio_preview.py --port PORT` and DOM IDs from Task 3.
- Produces: clean-export browser phase that runs both static BYOM and Gradio UI smokes.

- [ ] **Step 1: Write failing clean-export and CI tests**

Extend `tests/test_clean_export.py` so `REQUIRED_MEMBERS` must include:

```python
{
    "README.en.md",
    "demo/gradio.css",
    "demo/gradio_preview.py",
    "demo/gradio_ui.py",
    "demo/ui_contract.py",
    "scripts/gradio_ui_smoke.py",
}
```

Update `tests/test_package_release.py::test_ci_runs_core_cpu_gates_on_ubuntu_and_windows` to require
`CUDA_VISIBLE_DEVICES` and remove `"cuda"` from the forbidden tuple. Add:

```python
def test_ci_runs_model_free_gradio_ui_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )
    for token in (
        "uv sync --frozen --no-install-project --group ui-preview",
        "python scripts/gradio_ui_smoke.py",
        'CUDA_VISIBLE_DEVICES: "-1"',
    ):
        assert token in workflow
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_clean_export.py tests/test_package_release.py -q
```

Expected: FAIL because the Gradio smoke is absent from the archive inventory and CI workflow.

- [ ] **Step 3: Implement the Playwright smoke**

Create `scripts/gradio_ui_smoke.py` with these behaviors:

1. Reserve an available loopback port with `socket.bind(("127.0.0.1", 0))`.
2. Start `[sys.executable, "demo/gradio_preview.py", "--port", str(port)]` with
   `CUDA_VISIBLE_DEVICES=-1`, captured stdout/stderr, and no shell.
3. Poll `http://127.0.0.1:{port}` for at most 30 seconds using `urllib.request.urlopen`.
4. Launch Chromium with Playwright and record requests, console errors, and page errors.
5. At viewport `1920x1080`, assert:
   - `#app-header` contains `UI-only preview` and `CPU`;
   - `.gradio-container` computed width is between 1600px and 1740px;
   - the header `h1` computed font size is `32px`;
   - `#detect-button button` is disabled;
   - input and result panels share the same top coordinate and result width exceeds input width.
6. Upload an in-memory synthetic PNG through `#input-panel input[type=file]`; assert the input
   preview appears, the preview-only status remains explicit, Detect stays disabled, and the
   detection summary remains at zero without invoking `preview_detect`.
7. At viewport `820x1100`, assert the result panel top is below the input panel bottom.
8. Fail on any non-loopback request, console error, or page error.
9. Write the optional screenshot path after desktop assertions.
10. In `finally`, terminate the preview subprocess, wait five seconds, then kill it only if it did
   not exit.
11. Print `[OK] Gradio UI smoke: zh-TW wide workbench, responsive stack, no model inference`.

Do not import Gradio, Torch, or Ultralytics in the smoke process itself.

- [ ] **Step 4: Add the smoke to clean export**

Add the six Task 4 required members to `scripts/clean_export_check.py::REQUIRED_MEMBERS`. In
`verify_snapshot`, when `run_browser` is true, run in this exact order:

```python
steps.append(_run([uv, "sync", "--frozen", "--no-install-project", "--group", "ui-preview"], export))
steps.append(
    _run(
        [str(python), "scripts/gradio_ui_smoke.py", "--screenshot", str(temp_root / "gradio-ui.png")],
        export,
    )
)
steps.append(
    _run(
        [str(python), "-c", "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('ultralytics') is None"],
        export,
    )
)
```

Keep `--skip-browser` skipping both browser smokes in the core CI job because the browser job runs
them explicitly.

- [ ] **Step 5: Extend the existing browser CI job**

In `.github/workflows/release-gates.yml`:

- add `CUDA_VISIBLE_DEVICES: "-1"` to both job environments;
- change the browser job dependency command to
  `uv sync --frozen --no-install-project --group ui-preview`;
- retain the existing static smoke;
- add `uv run --no-sync python scripts/gradio_ui_smoke.py` immediately after it.

Do not add a Docker service, model, DOTA, Torch, Ultralytics, token, or GPU runner.

- [ ] **Step 6: Run the real local UI smoke and inspect its screenshot**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe scripts/gradio_ui_smoke.py --screenshot dist/gradio-ui-rc2.png
.venv/Scripts/python.exe -m pytest tests/test_clean_export.py tests/test_package_release.py -q
```

Expected: smoke and tests exit 0. Inspect `dist/gradio-ui-rc2.png` visually and confirm the desktop
layout matches the approved A workbench, uses the full width, and has readable type before commit.

- [ ] **Step 7: Commit the UI smoke and CI gate**

Stage the exact Task 4 paths and commit:

```text
test: add gradio ui smoke
```

---

### Task 5: Refresh rc2 evidence and committed clean export

**Files:**
- Modify: `release/evidence.json`
- Modify: `CHANGELOG.md`
- Modify: `RELEASE_CHECKLIST.md`
- Modify: `tests/test_release_check.py`

**Interfaces:**
- Produces: machine-readable `gradio_ui` evidence scoped to model-free UI verification.
- Produces: regenerated ignored `dist/yolo26-dota-obb-v1.0.0rc2.zip` from final committed files.

- [ ] **Step 1: Write the failing UI-evidence assertion**

Add to `tests/test_release_check.py`:

```python
def test_gradio_ui_evidence_is_model_free_and_zh_tw() -> None:
    evidence = load_evidence()["gradio_ui"]
    assert evidence["language"] == "zh-TW"
    assert evidence["layout"] == "wide-workbench-38-62"
    assert evidence["preview_model_loaded"] is False
    assert evidence["model_inference_run"] is False
    assert evidence["desktop_max_width_px"] == 1720
    assert evidence["responsive_breakpoint_px"] == 900
```

- [ ] **Step 2: Run the evidence test and verify RED**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest tests/test_release_check.py -q
```

Expected: FAIL with missing `gradio_ui` evidence.

- [ ] **Step 3: Add bounded UI evidence and release notes**

Add this object to `release/evidence.json`:

```json
"gradio_ui": {
  "language": "zh-TW",
  "layout": "wide-workbench-38-62",
  "desktop_max_width_px": 1720,
  "responsive_breakpoint_px": 900,
  "preview_model_loaded": false,
  "model_inference_run": false,
  "network_scope": "loopback only",
  "limitations": [
    "UI smoke validates layout and interaction state, not model correctness, accuracy, or latency.",
    "Real inference requires an owner-supplied compatible local model."
  ]
}
```

Add an rc2 bullet to `CHANGELOG.md` for the zh-TW canonical README and model-free Gradio workbench.
Update `RELEASE_CHECKLIST.md` to record the zh-TW README/English navigation, explicit Detect flow,
responsive Gradio preview smoke, and unchanged owner-only external actions.

- [ ] **Step 4: Run the complete pre-commit verification**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/repo_check.py
.venv/Scripts/python.exe scripts/release_check.py
.venv/Scripts/python.exe scripts/browser_smoke.py --screenshot dist/browser-smoke-rc2.png
.venv/Scripts/python.exe scripts/gradio_ui_smoke.py --screenshot dist/gradio-ui-rc2.png
uv lock --check
uv build --no-sources --out-dir dist/package-rc2-ui-check
git diff --check
```

Expected: all commands exit 0, no ML runtime is imported by preview/release tests, and both browser
screenshots match their bounded synthetic/UI-only purposes.

- [ ] **Step 5: Commit the final evidence refresh**

Stage only the four Task 5 files and commit:

```text
docs: refresh zh-tw ui release evidence
```

- [ ] **Step 6: Rebuild the final committed clean export**

Confirm `git status --short` is empty. Resolve
`dist/yolo26-dota-obb-v1.0.0rc2.zip`, verify it is inside the workspace, and remove only that exact
ignored stale archive if it exists. Then run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe scripts/clean_export_check.py
```

Expected: archive inspection, full pytest, repository/release gates, both browser smokes, package
build, and isolated wheel import pass from committed files. Record the ZIP byte size and SHA-256.

- [ ] **Step 7: Perform the final Git and remote audit**

Verify:

- branch is `portfolio/obb-v1.0-release-hardening`;
- status, staged set, and untracked set are empty;
- `.superpowers/`, preview screenshots, model binaries, DOTA visuals, private/interview files, and
  runtime files are not tracked;
- `notes.private.md` remains ignored without reading it;
- every reachable author and committer is the allowed identity;
- no `Co-authored-by` trailer exists;
- refs contain no unexpected tag or remote ref;
- no remote, push, PR, Release, or Hugging Face mutation occurred.

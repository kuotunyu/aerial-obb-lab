# zh-TW-First README and Gradio UX Design

**Status:** Approved for implementation on 2026-08-12  
**Release:** Continue the unpushed `v1.0.0-rc.2` candidate  
**Branch:** `portfolio/obb-v1.0-release-hardening`

## Objective

Make the public repository presentation primarily Traditional Chinese (`zh-TW`) while preserving
original technical terms, and redesign the optional Gradio reference demo as a space-efficient,
readable wide workbench that can be reviewed without a model or GPU.

The work remains code-only and CPU-safe. It does not add a model, dataset, benchmark, training run,
validation run, remote deployment, or artifact upload.

## Approved language direction

Public prose is written in natural Traditional Chinese. Product names, standards, libraries,
hardware, file formats, metrics, and established ML terms stay in their original form where that is
clearer, including `YOLO26`, `OBB`, `DOTA`, `Colab`, `A100`, `T4`, `ONNX`, `TensorRT`, `Gradio`,
`BYOM`, `CPU`, `GPU`, `inference`, `fine-tuning`, `baseline`, `benchmark`, `release`, `artifact`,
`checkpoint`, `mAP`, `CI`, `Docker`, `Hugging Face`, and `GitHub`.

Do not mechanically translate these terms or mix Simplified Chinese into public copy.

## README structure

- `README.md` becomes the canonical, complete Traditional Chinese README and therefore the default
  GitHub landing page and package long description.
- The current complete English README moves to `README.en.md`.
- `README.zh-TW.md` becomes a short compatibility pointer to `README.md`; it must not duplicate the
  full Chinese document.
- The canonical README begins with `正體中文 | [English](README.en.md)`.
- The English README begins with `[正體中文](README.md) | English`.
- Critical claim blocks remain complete in both full READMEs. The release verifier checks Chinese
  tokens in `README.md` and English tokens in `README.en.md`; the compatibility pointer carries no
  evidence claims.
- Repository, clean-export, package, local-link, and public-owner-link gates include the new English
  filename and continue requiring the canonical README.

## GitHub About guidance

No remote is created or changed. The owner instructions will recommend this GitHub About text:

> Code-only YOLO26 OBB × DOTA 作品集：誠實評估、deployment benchmark、BYOM demo 與可重現 release gates。

The suggested topics remain technical and language-neutral, with `zh-tw` added:

`computer-vision`, `object-detection`, `oriented-bounding-box`, `obb`, `yolo`, `dota`, `onnx`,
`tensorrt`, `onnxruntime`, `gradio`, `byom`, `mlops`, `reproducibility`, `portfolio`, `zh-tw`.

The Website field remains empty until the owner deliberately publishes a reviewed BYOM site.

## Considered alternatives

1. **Canonical zh-TW README plus English secondary file — selected.** GitHub opens directly in
   Traditional Chinese, English remains one click away, and each full document has one owner.
2. **One bilingual README — rejected.** It doubles page length, weakens visual hierarchy, and makes
   claim maintenance harder.
3. **Keep English `README.md` with a Chinese intro — rejected.** It does not satisfy a genuinely
   zh-TW-first repository.

For Gradio, the selected layout is the wide workbench. A horizontal toolbar was rejected because it
compresses controls on smaller screens; a result-first viewer was rejected because it creates a
large empty canvas before input and weakens input/output comparison.

## Gradio architecture

Create one shared UI builder used by both Python demo entry points so their layout, Traditional
Chinese copy, states, and CSS cannot drift.

```python
build_demo(
    *,
    detect_fn,
    class_names,
    model_name,
    device,
    imgsz,
    preview=False,
) -> gr.Blocks
```

Responsibilities are separated as follows:

- `demo/app.py` and `demo/space/app.py` own explicit local model acquisition and inference only.
- `demo/ui_contract.py` contains standard-library-only labels, detection-summary formatting, and
  safe user-error mapping so core tests do not import Gradio or an ML runtime.
- `demo/gradio_ui.py` owns component construction, event wiring, safe status summaries, and
  user-facing error handling.
- `demo/gradio.css` owns the approved visual system and responsive layout.
- `demo/gradio_preview.py` supplies fixed class names and preview-only state so the actual Gradio
  interface can launch without Ultralytics, Torch, a model, or inference.

The preview is a development/review surface, not a fake detector. Detect remains disabled or returns
an explicit preview-only message; it never fabricates model results.

Add an `ui-preview` dependency group containing Gradio without the `local-ml` group. The existing
`demo` group includes both `ui-preview` and `local-ml`. This allows the preview and its smoke test to
run without installing Torch or Ultralytics. The preview binds to loopback only and never enables a
public share URL. Remove the currently broken late `Path`-based `sys.path` mutation in
`demo/app.py`; shared imports are resolved before app construction.

## Approved wide-workbench layout

- Use `width: 96vw` with a desktop `max-width: 1720px`.
- Use a compact single-row header with title/subtitle on the left and status chips on the right.
- Show `Model ready`, the escaped local model filename, `CPU`, and `imgsz` as concise chips.
- Use a `38% / 62%` input/result desktop grid.
- Use a 32px page title, 18px body/action text, and 16px labels and helper copy.
- Keep page padding between 24px and 32px on desktop; avoid decorative empty sections.
- Input contains image preview/upload, `Confidence threshold`, `Class filter`, and the primary
  `Detect` action.
- Result contains the annotated image, a compact detection summary, and the existing sortable
  detection table. The summary may report detection count and top confidence; it must not report or
  imply a benchmark latency.
- Below `900px`, stack input and result into one column, preserve readable type, and
  keep the action full width.
- Use restrained slate, blue, and semantic status colors with visible keyboard focus and adequate
  contrast. Do not add a decorative framework, animation system, or new UI feature.

## Interaction and data flow

1. The process validates `MODEL_PATH` before the real demo is constructed. Invalid or missing paths
   remain actionable command-line startup errors.
2. A successfully loaded model provides class names to the shared builder; the header reports ready
   state without exposing an absolute path.
3. Before an image exists, `Detect` is disabled.
4. Selecting an image renders its preview and enables `Detect`. Upload does not trigger inference.
5. The user adjusts `Confidence threshold` and `Class filter`, then explicitly starts Detect.
6. Gradio's pending state prevents duplicate submission while the event is running.
7. Success returns the annotated image, rows sorted by confidence, detection count, and top
   confidence. No elapsed-time or cross-device performance claim is added.
8. Changing image or settings marks prior output as needing another Detect instead of silently
   rerunning the model.

The underlying `detect_fn(image, conf, selected_classes)` contract remains model-specific and
returns the annotated image plus detection rows. The shared UI wrapper derives presentation-only
summary values from those rows.

## Error handling and privacy

- A missing image cannot enqueue Detect.
- A model/output/runtime failure becomes a short actionable `gr.Error` and status message.
- The UI preserves the selected image and settings so the user can retry.
- User-visible errors never include a traceback, secret, token, ignored path, or absolute local
  model path. Detailed exceptions may remain in the local process console for the owner.
- Preview mode is clearly labeled and never presents synthetic output as inference evidence.
- Image and model data stay local; no network upload or automatic model acquisition is introduced.

## TDD and verification

Implementation begins with failing tests and release-policy checks for:

- canonical `README.md` being zh-TW-first with an English link;
- `README.en.md` carrying the complete English claims;
- `README.zh-TW.md` being a compatibility pointer rather than a duplicate;
- the About recommendation containing Traditional Chinese and the `zh-tw` topic;
- clean-export and presentation-file inventories including `README.en.md` and the shared Gradio UI;
- no `.upload(detect...)` or equivalent automatic inference binding;
- an explicit Detect click path, disabled-before-image behavior, and sanitized error copy;
- pure detection-summary behavior for zero and nonzero rows;
- a preview entry point that imports no Torch, Ultralytics, model binary, or Hugging Face client.

Verification then runs:

- focused tests for README claims, UI contract, and clean export;
- Python compilation and repository/release gates;
- a real local Gradio preview served on loopback and inspected with Playwright at 1920px desktop and
  820px narrow viewports for computed width, typography, status text, responsive stacking, and
  disabled Detect behavior;
- full CPU-only pytest, package build, and committed clean-export rebuild.

All verification sets `CUDA_VISIBLE_DEVICES=-1`. No model inference, retraining, validation,
TensorRT build, DOTA download, weight download, secret, or remote write is permitted.

## Release and Git boundaries

- Continue `v1.0.0-rc.2`; no version bump is needed before its first push.
- Use small English commits for the language structure, Gradio UI, and final gate refresh.
- Do not stage `.superpowers/`, ignored private material, preview screenshots, or runtime files.
- Regenerate the committed clean export only after the implementation commits and a clean status.
- Do not create a remote, push, tag, PR, Release, or modify Hugging Face.

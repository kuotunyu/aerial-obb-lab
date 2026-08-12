# Browser-Native UI Redesign

**Status:** Approved on 2026-08-12
**Release:** Continue the `v1.0.0-rc.2` candidate
**Branch:** `portfolio/obb-v1.0-release-hardening`

## Objective

Replace the repository's Gradio presentation path with one distinctive browser-native workbench
that demonstrates a different deployment method from the owner's other portfolio projects. The
interface must feel deliberately designed without becoming decorative, use larger readable type,
and use the available viewport efficiently.

The implementation remains a code-only, CPU-safe, bring-your-own-model (`BYOM`) release. It does
not add or download a model, dataset, benchmark, hosted service, or restricted image. It does not
retrain, validate, export, or publish a model.

## Product decision

The existing vanilla HTML, CSS, and JavaScript application becomes the only interactive demo. It
runs a user-selected compatible ONNX model with ONNX Runtime Web's WASM execution provider and
processes both model bytes and image pixels locally in the browser.

Remove the Gradio implementation instead of retaining a hidden legacy path. One interface avoids
duplicate copy, tests, dependency groups, privacy explanations, and behavior that can drift. The
browser deployment path also makes the portfolio's engineering range clearer than replacing
Gradio with another Python UI framework.

The selected implementation is preferred over these alternatives:

1. **Vanilla browser app — selected.** Reuses the verified preprocessing and OBB decode contract,
   has no build step, supports static hosting, and allows a fully custom visual system.
2. **React and Vite — rejected.** They would offer component tooling but add a Node build pipeline
   and dependency surface that a single workbench does not need.
3. **Streamlit or NiceGUI — rejected.** They would replace one server-side Python UI framework
   with another and would not meaningfully diversify the portfolio.

## Source and dependency structure

Rename `demo/space-static/` to `demo/web/` so the maintained product path describes the technology
rather than a former hosting provider. The application keeps four focused files:

- `index.html`: semantic content, component order, accessible labels, and loading/error regions;
- `style.css`: tokens, layout, typography, responsive behavior, and interaction states;
- `app.js`: file selection, ONNX Runtime Web session lifecycle, preprocessing, inference, state,
  canvas rendering, and result-table presentation;
- `obb.js`: DOM-free geometry, tensor conversion, output validation, and OBB decoding.

Delete the Gradio entry points, shared builder, CSS, UI contract, Space reference folder, and
Gradio-only smoke test. Remove the `ui-preview` and combined `demo` dependency groups. Preserve the
optional `local-ml` dependency group because it supports local Python inference and export tooling
outside the public UI, but remove Gradio from the lockfile and public instructions.

Repository docs, artifact inventories, release checks, third-party notices, and tests must refer to
the browser app as the single demo. No public URL is added and no GitHub Pages or Hugging Face Space
is changed as part of this work.

## Visual direction

The visual world is a restrained aerial-systems workbench: warm off-white page surfaces, deep navy
text and image canvas, a single cobalt interaction color, cool slate borders, and quiet semantic
green/amber/red states. Avoid gradients, glass effects, neon, decorative animation, oversized
empty hero sections, illustrations, and excessive cards.

Typography and spacing targets:

- system-oriented Traditional Chinese stack with reliable Windows fallbacks;
- 18px base text, 17–18px labels and controls, 18–20px supporting copy;
- 38–42px desktop title with compact line height and 32px mobile title;
- 44px minimum interactive height and visible keyboard focus;
- 24–32px desktop page padding and 16–20px mobile page padding;
- content width up to approximately 1600px so the result canvas receives useful space.

The page may contain borders and subtle surface changes for grouping, but it must not nest cards
inside cards or use decoration to manufacture hierarchy.

## Information architecture

Use only three visible levels:

1. **Compact product header.** Show `Aerial OBB Lab`, a short Traditional Chinese description, and
   three concise trust signals: `Browser`, `WASM`, and `Local-only`. Do not repeat runtime details in
   a second banner.
2. **Single workbench.** Use an approximately 34/66 desktop grid. The left control rail contains
   the ONNX model picker, image picker, confidence threshold, class filter, and one primary Detect
   action. The right workspace contains the result summary, canvas, and result table.
3. **Quiet legal and contract footer.** State the expected `output0 [1,N,7]` contract, local file
   handling, and code license in one compact line or wrapped block without competing with the task.

Do not create separate panels for instructions, privacy, runtime status, and model metadata. Their
essential information belongs in the relevant control, header trust signals, or live status line.

## Workbench behavior

The control rail behaves as a short guided sequence without introducing a wizard:

- model and image pickers show numbered labels, selected filenames, and clear ready states;
- Detect stays disabled until both inputs are ready;
- confidence is displayed as a large numeric value beside the slider;
- the 15 DOTA classes use compact selectable chips in a bounded region; none selected means all;
- state changes do not trigger inference automatically;
- progress, success, and errors share one live region directly above the primary action.

The result workspace prioritizes the image:

- a compact summary row reports detection count and top confidence;
- elapsed browser time may appear only as session-local diagnostic text, never as a benchmark or
  comparison with the recorded T4 result;
- the canvas fills the available width and uses a dark neutral empty state before image selection;
- rotated boxes remain visible, but dense class/confidence labels are removed from the canvas;
- the complete class, confidence, width, height, and angle values remain in a fixed-height,
  scrollable table sorted by confidence;
- an empty result is a valid state and must not look like an application failure.

At widths below 900px, the controls stack above results, all primary controls remain full width,
and the table scrolls horizontally without shrinking its text below the readability target.

## Data flow and boundaries

1. The user selects a local `.onnx` file.
2. JavaScript reads its bytes through the File API and creates an ONNX Runtime Web WASM session.
3. The user selects an image; the browser creates an object URL and renders the original image.
4. On explicit Detect, the existing deterministic letterbox and RGBA-to-CHW preprocessing create
   the `images [1,3,1024,1024]` float tensor.
5. The application requires the documented end-to-end `output0 [1,N,7]` output.
6. The existing OBB decoder maps detections back to original-image coordinates, filters them, and
   returns confidence-sorted results.
7. Presentation code draws rotated polygons without dense text labels and fills the summary/table.

`obb.js` remains independently testable and has no DOM, ONNX Runtime, or framework dependency.
Selected model bytes and image data must never be uploaded, persisted, or sent to analytics.

## Error handling and accessibility

- Use actionable Traditional Chinese messages for incompatible files, model-load failures,
  missing `output0`, invalid tensor dimensions, inference failures, and zero detections.
- Never expose a local absolute path, stack trace, token, or ignored filename beyond the basename
  the user explicitly selected.
- Mark the live status with an appropriate `aria-live` policy; retain native file and range inputs
  behind accessible labels.
- Provide visible focus, sufficient contrast, keyboard-operable class chips, and reduced-motion-safe
  state transitions.
- Disable repeat submission while inference is active and restore the current inputs after errors.

## Portfolio presentation

The canonical Traditional Chinese README links to `demo/web/` as the sole demo source and explains
that it is a custom browser-native deployment, not a Gradio app. The English README mirrors the
same factual claim.

Add one committed desktop screenshot generated by the browser smoke path from an original synthetic
fixture. Label it as a synthetic UI fixture, not model-quality evidence. Do not include the local
marina photograph, DOTA-derived imagery, a model binary, or a fabricated production result.

## TDD and verification

Implementation begins with failing tests for:

- absence of Gradio source paths, dependencies, and public documentation references;
- presence of the renamed `demo/web/` files in committed and clean-export inventories;
- Traditional Chinese primary UI copy and required local-only/model-contract disclosures;
- minimum typography, interactive-size, compact-layout, and responsive CSS contracts;
- disabled-before-ready behavior, explicit Detect, sanitized failures, and valid empty results;
- dense-result canvas behavior that does not draw a text label for every detection;
- the unchanged preprocessing, output selection, OBB geometry, and confidence-sorted table contract.

Browser verification uses the existing synthetic local model bytes and deterministic output stub.
Playwright captures desktop and mobile views, checks keyboard focus and responsive stacking, and
rejects unexpected network requests other than the version-pinned ONNX Runtime Web asset. Core CI
continues to run on Ubuntu and Windows CPU without Torch, CUDA, DOTA, weights, HF token, or secrets.

Final verification includes focused tests, full CPU tests, repository and release gates, package
builds, and a clean committed export rebuild. No GPU inference, model training, full validation,
TensorRT build, external publication, PR, tag, or Release is part of this redesign.

## Acceptance criteria

- The public source tree contains one custom browser-native demo and no Gradio implementation.
- At 1440px and wider, the workbench uses the majority of the viewport without crowded controls or
  a large unused lower/right area.
- Default text and controls are comfortably readable without browser zoom.
- The first viewport communicates input, local processing, result canvas, and summary without
  repeated explanatory panels.
- Dense OBB results remain visually legible because canvas labels do not overlap.
- Desktop and mobile screenshots show a consistent, restrained visual system.
- Existing model/output geometry remains unchanged and all release gates pass from committed files.

# Aerial OBB Lab Release Checklist

**State:** The historical v1.0.0 publication was republished from a clean code-only root after explicit
owner authorization. The current local Pages candidate adds one exact privacy-sanitized AGPL demo
model and one official sample image; it remains pending separate remote review gates. Historical
Hugging Face artifacts remain Private. No tag, GitHub Release, replacement Space, DOTA data, or
fine-tuned checkpoint was published.

## Historical v1.0.0 completed gates

- [x] Historical v1.0.0: locked Python 3.11 CPU environment and wheel install passed in a clean
  Windows export; the
  same commands are defined in the Ubuntu/Windows CI matrix.
- [x] Historical v1.0.0: the full pytest suite passed without Torch, CUDA, DOTA, tokens, or
  downloaded weights.
- [x] Historical v1.0.0: wheel and source distribution built and contained only intended package files.
- [x] Historical v1.0.0: clean committed export repeated tests, package, link, privacy, artifact,
  and browser gates.

## Current Pages candidate local gates

- [x] Claim/evidence arithmetic and bounded Markdown claim blocks pass.
- [x] Browser preprocess, output schema, decode, angle, and rotated corners match Python parity fixtures.
- [x] Headless Chromium shows the official original first, runs genuine local inference only after
  explicit Detect, verifies numeric runtime/results, cached filters/toggle, and advanced BYOM safety.
- [x] The canonical `README.md` is complete Traditional Chinese and `README.en.md` retains the
  complete English claims; no duplicate language-pointer file is published.
- [x] The production browser page pins ONNX Runtime Web 1.20.1 and verifies the jsDelivr response
  with SHA-384 SRI plus anonymous CORS.
- [x] The browser-native workbench uses an explicit Detect action, a 31/69 desktop layout, 19px
  base type, 15px minimum secondary text, square-edged working surfaces, visible keyboard focus,
  and a responsive single-column layout below 960px.
- [x] Initial navigation is same-origin only; Detect lazy-loads the pinned SRI JavaScript/WASM and
  same-origin manifest-bound derivative without Torch, Ultralytics Python, GPU, or upload.
- [x] The distributable inventory contains one official sample, one exact metadata-only sanitized
  derivative, its unmodified AGPL text and sanitization record, plus one self-hosted OFL display font.
- [x] Committed and archived paths admit only that exact derivative model and contain no DOTA-derived render.
- [x] Notebooks have zero outputs/execution counts and remain synchronized with Jupytext sources.
- [x] Tracked files and Git history pass token-pattern/privacy checks.
- [x] Local Markdown links, static assets, loopback HTTP, and JavaScript syntax pass.
- [ ] Current candidate: run the complete pytest suite without Torch, CUDA, DOTA, tokens, or
  downloaded weights.
- [ ] Current candidate: build and verify the strict committed clean export, including wheel and
  source-distribution rebuild, install, import, link, privacy, artifact, and browser gates.
- [ ] Current candidate: repeat the complete local CPU, browser, artifact, license, privacy, and
  clean-export gates from the final committed bytes; official JavaScript actions must retain their
  current Node 24-compatible majors.
- [ ] Hosted Ubuntu CPU, Windows CPU, and live-demo browser checks remain a separate authorized remote gate.

## Manual release audit

- [x] Fine-tuning is described as about -0.05pt mAP50 and -0.13pt mAP50-95, not an improvement.
- [x] DOTA8 0.9950 is described only as export-smoke parity.
- [x] TensorRT 20.22 ms / 49.4 FPS is limited to the recorded T4/batch-1/1024px environment.
- [x] The browser demo uses one exact privacy-sanitized nano derivative for genuine local inference;
  advanced BYOM accepts a user-supplied compatible model/image without upload.
- [x] The demo and UI evidence inherit no fine-tuned-medium accuracy, evaluation, historical T4
  latency, production SLA, or certification claim.
- [x] DOTA academic-only, underlying image-source, AGPL/Enterprise, and external-artifact boundaries are
  visible in `THIRD_PARTY_NOTICES.md`.
- [x] Historical v1.0.0: branch, HEAD, status, refs, author, committer, trailers, ignored/private
  paths, and remotes were audited after its final commit.
- [ ] Current candidate: complete the final branch audit and whole-branch review after the Task 5
  verification commit.
- [x] The historical public branch began at a clean code-only root with no DOTA-derived image,
  private document, owner handle, or private Hugging Face repo identifier.
- [x] The current candidate adds only the reviewed derivative/sample/license/sanitization inventory;
  no second/source model, DOTA pixels, annotations, or derived render is admitted.

No Dockerfile or service was added: the release paths are a static browser site and a Python
package, so Docker would not remove a product-path dependency.

## Publication state

- [x] Rename the historical Hugging Face model to a neutral archive name, keep it Private, and
  verify anonymous access fails; the public tree does not record its owner identifier or repo name.
- [x] Make the historical Hugging Face Space private and verify anonymous API/page access returns
  `401`; do not create a replacement Space for this release candidate.
- [x] Create the public GitHub repository `aerial-obb-lab` and, after explicit authorization, push
  only `portfolio/obb-v1.0-release-hardening`.
- [x] Historical v1.0.0: publish the reviewed code-only tree from a clean root commit after explicit authorization.
- [x] Restore branch protection with admin enforcement and linear history enabled; force pushes and
  branch deletion are disabled.
- [x] Historical v1.0.0: confirm the published tip passed Ubuntu/Windows CPU and browser gates, then
  review the public file inventory; the release branch became the default branch.
- [ ] Current Pages candidate: perform push/PR/merge/CI/Pages/live review only through the separately
  authorized remote gates; local planning and implementation do not authorize those mutations.
- [x] No tag or GitHub Release is part of this release-candidate scope; future publication remains
  a separate, explicit owner decision.

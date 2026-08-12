# v1.0.0-rc.2 Release Checklist

**State:** Public feature-frozen, code-only release candidate. The default GitHub branch was
republished from a clean root after explicit owner authorization; historical Hugging Face artifacts
remain Private. No tag, GitHub Release, replacement Space, model weight, or dataset was published.

## Automated gates

- [x] Locked Python 3.11 CPU environment and wheel install pass in a clean Windows export; the
  same commands are defined in the Ubuntu/Windows CI matrix.
- [x] Full pytest suite passes without Torch, CUDA, DOTA, tokens, or downloaded weights.
- [x] Claim/evidence arithmetic and bounded Markdown claim blocks pass.
- [x] Browser synthetic preprocess, output schema, decode, angle, and corners match Python.
- [x] Headless Chromium exercises upload, preprocess, canvas drawing, and result rendering with a
  deterministic inline runtime/output stub; it performs no model inference or external network request.
- [x] The canonical `README.md` is complete Traditional Chinese and `README.en.md` retains the
  complete English claims; no duplicate language-pointer file is published.
- [x] The production browser page pins ONNX Runtime Web 1.20.1 and verifies the jsDelivr response
  with SHA-384 SRI plus anonymous CORS.
- [x] The browser-native workbench uses an explicit Detect action, a 34/66 desktop layout, 19px
  base type, 15px minimum secondary text, square-edged working surfaces, visible keyboard focus,
  and a responsive single-column layout below 900px.
- [x] The loopback-only browser smoke loads the real UI with a synthetic SVG and stubbed output;
  it performs no Torch, Ultralytics, GPU, or model inference.
- [x] The distributable inventory contains exactly one self-hosted OFL display font; excluded
  historical model and DOTA-derived hashes remain audit-only.
- [x] Committed and archived paths contain no model binary or DOTA-derived comparison render.
- [x] Notebooks have zero outputs/execution counts and remain synchronized with Jupytext sources.
- [x] Tracked files and Git history pass token-pattern/privacy checks.
- [x] Local Markdown links, static assets, loopback HTTP, and JavaScript syntax pass.
- [x] Wheel and source distribution build and contain only intended package files.
- [x] Clean committed export repeats tests, package, link, privacy, artifact, and browser gates.
- [x] Hosted Ubuntu CPU, Windows CPU, and synthetic browser checks pass on the clean publication;
  official JavaScript actions use their current Node 24-compatible v7 majors.

## Manual release audit

- [x] Fine-tuning is described as about -0.05pt mAP50 and -0.13pt mAP50-95, not an improvement.
- [x] DOTA8 0.9950 is described only as export-smoke parity.
- [x] TensorRT 20.22 ms / 49.4 FPS is limited to the recorded T4/batch-1/1024px environment.
- [x] The browser demo requires a user-supplied local model and inherits no checkpoint accuracy or
  T4 latency claim.
- [x] Browser UI evidence is limited to layout, interaction state, deterministic decode, and
  rendering; it makes no model accuracy, production latency, or certification claim.
- [x] DOTA academic-only, underlying image-source, AGPL/Enterprise, and external-artifact boundaries are
  visible in `THIRD_PARTY_NOTICES.md`.
- [x] Branch, HEAD, status, refs, author, committer, trailers, ignored/private paths, and remotes are
  audited after the final commit.
- [x] The public branch begins at a clean code-only root and contains no model binary, DOTA-derived
  comparison image, private document, owner handle, or private Hugging Face repo identifier.

No Dockerfile or service was added: the release paths are a static browser site and a Python
package, so Docker would not remove a product-path dependency.

## Publication state

- [x] Rename the historical Hugging Face model to a neutral archive name, keep it Private, and
  verify anonymous access fails; the public tree does not record its owner identifier or repo name.
- [x] Make the historical Hugging Face Space private and verify anonymous API/page access returns
  `401`; do not create a replacement Space for this release candidate.
- [x] Create the public GitHub repository `aerial-obb-lab` and, after explicit authorization, push
  only `portfolio/obb-v1.0-release-hardening`.
- [x] Publish the reviewed code-only tree from a clean root commit after explicit authorization.
- [x] Restore branch protection with admin enforcement and linear history enabled; force pushes and
  branch deletion are disabled.
- [x] Confirm the published tip passes Ubuntu/Windows CPU and synthetic browser gates, then review
  the public file inventory; the release branch is already the default branch.
- [x] No tag or GitHub Release is part of this release-candidate scope; future publication remains
  a separate, explicit owner decision.

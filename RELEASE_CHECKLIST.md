# v1.0.0-rc.2 Release Checklist

**State:** Local feature-frozen release candidate with one external owner blocker. The Private HF
archive rename is complete, and the public GitHub repository already contains an earlier release
branch tip. The historical Hugging Face Space is still public and running. The local hardening
commands performed no tag, GitHub Release, pull request, Space creation, or Hugging Face mutation.

## Automated gates

- [x] Locked Python 3.11 CPU environment and wheel install pass in a clean Windows export; the
  same commands are defined in the Ubuntu/Windows CI matrix.
- [x] Full pytest suite passes without Torch, CUDA, DOTA, tokens, or downloaded weights.
- [x] Claim/evidence arithmetic and bounded Markdown claim blocks pass.
- [x] Browser synthetic preprocess, output schema, decode, angle, and corners match Python.
- [x] Headless Chromium exercises upload, preprocess, canvas drawing, and result rendering with a
  deterministic output stub; it performs no model inference or external network request.
- [x] The canonical `README.md` is complete Traditional Chinese, `README.en.md` retains the complete
  English claims, and `README.zh-TW.md` is only a compatibility pointer.
- [x] The shared Gradio workbench uses an explicit Detect action, a 38/62 desktop layout, readable
  type, and a responsive single-column layout below 900px.
- [x] The loopback-only Gradio preview smoke loads the real UI with a synthetic PNG while Detect
  remains disabled; it imports no Torch or Ultralytics and performs no model inference.
- [x] The distributable artifact inventory is empty; excluded historical hashes remain audit-only.
- [x] Committed and archived paths contain no model binary or DOTA-derived comparison render.
- [x] Notebooks have zero outputs/execution counts and remain synchronized with Jupytext sources.
- [x] Tracked files and Git history pass token-pattern/privacy checks.
- [x] Local Markdown links, static assets, loopback HTTP, and JavaScript syntax pass.
- [x] Wheel and source distribution build and contain only intended package files.
- [x] Clean committed export repeats tests, package, link, privacy, artifact, and browser gates.

## Manual release audit

- [x] Fine-tuning is described as about -0.05pt mAP50 and -0.13pt mAP50-95, not an improvement.
- [x] DOTA8 0.9950 is described only as export-smoke parity.
- [x] TensorRT 20.22 ms / 49.4 FPS is limited to the recorded T4/batch-1/1024px environment.
- [x] Browser and Python demos require user-supplied local models and inherit no checkpoint accuracy
  or T4 latency claim.
- [x] Gradio UI evidence is limited to layout and interaction state; it makes no accuracy, latency,
  model-correctness, or production-deployment claim.
- [x] DOTA academic-only, underlying image-source, AGPL/Enterprise, and external-artifact boundaries are
  visible in `THIRD_PARTY_NOTICES.md`.
- [x] Branch, HEAD, status, refs, author, committer, trailers, ignored/private paths, and remotes are
  audited after the final commit.

No Dockerfile or service was added: the release paths are a static browser site and a Python
package, so Docker would not remove a product-path dependency. The local Docker daemon was
unavailable during final verification; hosted Ubuntu execution remains part of the checked-in CI
matrix.

## Owner actions required before public release

- [x] Rename the historical Hugging Face model to `aerial-obb-lab-model-archive`, keep it Private,
  and verify anonymous access fails.
- [ ] Make the historical Hugging Face Space private and verify anonymous access fails; do not
  create a replacement Space for this release candidate.
- [x] Create the public GitHub repository `aerial-obb-lab`; a shared-checkout action has already
  pushed release-branch tip `381f676d3f01afa401d34b839b03256adf2597a4`.
- [ ] Review and push only the final corrected branch tip, then require Ubuntu/Windows CPU gates.
- [ ] Add the reviewed GitHub topics; the anonymous API currently reports no topics.
- [ ] Create a tag or GitHub Release only after hosted CI and the public file inventory are green.

Exact authenticated instructions are in [`docs/OWNER_ACTIONS.md`](docs/OWNER_ACTIONS.md). None of
these external actions is completed by the local release workflow.

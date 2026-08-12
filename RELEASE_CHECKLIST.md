# v1.0.0-rc.2 Release Checklist

**State:** Local feature-frozen release candidate. Historical Hugging Face artifacts were made
Private and the public GitHub repository already exists, but the current local hardening commits
remain intentionally unpushed. This workflow performs no tag, GitHub Release, pull request, Space
creation, or Hugging Face mutation.

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
- [x] The clean tree/archive and the pre-existing public Git history are distinguished explicitly;
  private notes and interview material never entered any commit.

No Dockerfile or service was added: the release paths are a static browser site and a Python
package, so Docker would not remove a product-path dependency. The local Docker daemon was
unavailable during final verification; hosted Ubuntu execution remains part of the checked-in CI
matrix.

## Owner actions required before public release

- [x] Rename the historical Hugging Face model to a neutral archive name, keep it Private, and
  verify anonymous access fails; the public tree does not record its owner identifier or repo name.
- [x] Make the historical Hugging Face Space private and verify anonymous API/page access returns
  `401`; do not create a replacement Space for this release candidate.
- [x] Create the public GitHub repository `aerial-obb-lab` and, after explicit authorization, push
  only `portfolio/obb-v1.0-release-hardening`.
- [ ] Push the reviewed release-candidate commits only after the final local branch audit.
- [ ] Decide whether discoverable legacy non-secret Git history is acceptable. Any clean-history
  migration requires separate explicit authorization and is outside this no-rewrite candidate.
- [ ] Clear `enforce_admins` temporarily for the existing protected branch, retry the ordinary
  push without rewriting history, then immediately restore the protection.
- [ ] Confirm the pushed tip passes Ubuntu/Windows CPU and synthetic browser gates, then review the
  public file inventory against that exact SHA; the release branch is already the default branch.
- [ ] Create a tag or GitHub Release only after hosted CI and the public file inventory are green.

Exact authenticated instructions are in [`docs/OWNER_ACTIONS.md`](docs/OWNER_ACTIONS.md). None of
these external actions is completed by the local release workflow.

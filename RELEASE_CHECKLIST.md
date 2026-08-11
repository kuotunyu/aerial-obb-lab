# v1.0.0-rc.1 Release Checklist

**State:** Local feature-frozen release candidate. No push, tag, GitHub Release, pull request, or
Hugging Face mutation is part of this checklist run.

## Automated gates

- [x] Locked Python 3.11 CPU environment and wheel install pass in a clean Windows export; the
  same commands are defined in the Ubuntu/Windows CI matrix.
- [x] Full pytest suite passes without Torch, CUDA, DOTA, tokens, or downloaded weights.
- [x] Claim/evidence arithmetic and bounded Markdown claim blocks pass.
- [x] Browser synthetic preprocess, output schema, decode, angle, and corners match Python.
- [x] Bundled artifact size/SHA-256 values match `release/artifact-manifest.json`.
- [x] Notebooks have zero outputs/execution counts and remain synchronized with Jupytext sources.
- [x] Tracked files and Git history pass token-pattern/privacy checks.
- [x] Local Markdown links, static assets, loopback HTTP, and JavaScript syntax pass.
- [x] Wheel and source distribution build and contain only intended package files.
- [x] Clean committed export repeats tests, package, link, privacy, artifact, and browser gates.

## Manual release audit

- [x] Fine-tuning is described as about -0.05pt mAP50 and -0.13pt mAP50-95, not an improvement.
- [x] DOTA8 0.9950 is described only as export-smoke parity.
- [x] TensorRT 20.22 ms / 49.4 FPS is limited to the recorded T4/batch-1/1024px environment.
- [x] Browser demo is identified as official `yolo26n-obb`, not fine-tuned `yolo26m-obb` evidence.
- [x] DOTA academic-only, underlying image-source, AGPL/Enterprise, and external HF boundaries are
  visible in `THIRD_PARTY_NOTICES.md`.
- [x] Branch, HEAD, status, refs, author, committer, trailers, ignored/private paths, and remotes are
  audited after the final commit.

## Owner actions required before public release

1. Decide whether the repository should include the five DOTA-derived JPEGs and bundled ONNX model.
   For commercial or unrestricted redistribution, obtain written DOTA/underlying-image permission
   or remove the restricted artifacts and their claims.
2. Confirm the intended distribution satisfies AGPL-3.0, or obtain an applicable Ultralytics
   Enterprise license for closed-source/commercial use.
3. Have qualified counsel review the combined DOTA, underlying imagery, weights, and AGPL boundary.
4. After reviewing the final local diff, create the repository, push the branch, configure branch
   protection/CI, and optionally create a signed tag and Release. None of these actions is automated.
5. If the local model card/demo changes are desired on Hugging Face, update the model repository and
   static Space manually only after the source release decision; verify the pinned revisions again.

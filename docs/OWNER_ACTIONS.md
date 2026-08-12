# Owner actions after local release-candidate approval

Authenticated owner actions are recorded separately from the local hardening workflow. The local
workflow did not create or mutate a remote, push code, publish a Space, upload artifacts, or create
a tag or Release.

## Completed owner actions — 2026-08-12

### Private historical model archive

- Renamed the historical model repository to
  `steven0226/aerial-obb-lab-model-archive`.
- Kept repository visibility **Private**.
- Anonymous HTTP verification returned `401`; the model page and files are not a public release.

### Historical Space

- The previously recorded historical Space URL does not resolve and the Space is absent from the
  owner's visible repositories.
- No replacement Space is required for this release candidate. Do not create
  `aerial-obb-lab-browser` unless a separate public-hosting decision is made later.

### Empty public GitHub repository

- Created `https://github.com/kuotunyu/aerial-obb-lab` as an empty Public repository.
- It was not initialized with a README, license, `.gitignore`, template, or generated file.
- About description:
  `Aerial OBB 工程實驗室：以 YOLO26 與 DOTA 實驗為核心，涵蓋可重現訓練、誠實 baseline 評估、ONNX／TensorRT export、效能分析、Browser BYOM demo 與 CPU CI release gates。`
- Suggested topics: `computer-vision`, `object-detection`, `oriented-bounding-box`, `obb`,
   `yolo`, `dota`, `onnx`, `tensorrt`, `onnxruntime`, `gradio`, `byom`, `mlops`,
   `reproducibility`, `portfolio`, `zh-tw`.
- Leave the Website field empty until a reviewed BYOM site is deliberately published.

## Remaining owner actions

1. Review the final local branch, clean-export SHA-256, and exact tracked-file inventory.
2. Review `https://github.com/kuotunyu/aerial-obb-lab.git` character-for-character, then add it as
   `origin` only after local approval.
3. Push only `portfolio/obb-v1.0-release-hardening`; do not push ignored files, runtime artifacts,
   weights, datasets, or an automatic tag.
4. On GitHub, require all release-gate jobs to pass on Ubuntu and Windows CPU before merging.

## Optional tag or Release after hosted CI

After reviewing the GitHub file inventory and successful hosted CI, decide whether to merge and
create `v1.0.0-rc.2`. Create a GitHub Release only after that decision. Do not upload model weights,
DOTA-derived visuals, datasets, or the historical clean-export archive as release assets without a
new rights and privacy review.

These controls reduce avoidable exposure but do not eliminate legal responsibility. The repository
is a code-only portfolio engineering artifact, not a grant of dataset, model-weight, underlying
image-source, or Ultralytics Enterprise rights.

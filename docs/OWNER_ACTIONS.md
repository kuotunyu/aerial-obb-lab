# Owner actions after local release-candidate approval

Authenticated owner actions are recorded separately from the local hardening workflow. The local
hardening commands did not publish a Space, upload artifacts, or create a tag or Release. At
2026-08-12 15:14:55 +08:00, however, this shared checkout recorded an external `update by push` for
the release branch. The remote state below must therefore be treated as already public, not as a
future owner action.

## Completed owner actions — 2026-08-12

### Private historical model archive

- Renamed the historical model repository to
  `steven0226/aerial-obb-lab-model-archive`.
- Kept repository visibility **Private**.
- Anonymous HTTP verification returned `401`; the model page and files are not a public release.

### Historical Space

- Anonymous read-only verification of
  `steven0226/yolo26-obb-aerial-detection` on 2026-08-12 returned `private: false`, SDK `static`,
  runtime stage `RUNNING`, and revision `23211d473c0aa9f424f19a7a3c40fc1931356a0d`.
- **Owner blocker:** back up the Space privately if needed, then set the existing Space to **Private**
  in Hugging Face settings. Confirm from a signed-out session that the page and files
  return `401` or `404` before treating the code-only release boundary as complete.
- No replacement Space is required for this release candidate. Do not create
  `aerial-obb-lab-browser` unless a separate public-hosting and rights review is completed later.

### Public GitHub repository

- Created `https://github.com/kuotunyu/aerial-obb-lab` as a Public repository.
- Read-only `git ls-remote` verification now shows
  `portfolio/obb-v1.0-release-hardening` at
  `381f676d3f01afa401d34b839b03256adf2597a4`; the repository is no longer empty.
- About description:
  `Aerial OBB 工程實驗室：以 YOLO26 與 DOTA 實驗為核心，涵蓋可重現訓練、誠實 baseline 評估、ONNX／TensorRT export、效能分析、Browser BYOM demo 與 CPU CI release gates。`
- Suggested topics: `computer-vision`, `object-detection`, `oriented-bounding-box`, `obb`,
   `yolo`, `dota`, `onnx`, `tensorrt`, `onnxruntime`, `gradio`, `byom`, `mlops`,
   `reproducibility`, `portfolio`, `zh-tw`.
- Leave the Website field empty until a reviewed BYOM site is deliberately published.

## Remaining owner actions

1. Make the still-public historical Hugging Face Space private and verify anonymous denial.
2. Review the final local branch, clean-export SHA-256, exact tracked-file inventory, and the branch
   that is already public on GitHub.
3. After local approval, push only the final corrected
   `portfolio/obb-v1.0-release-hardening` tip; do not push ignored files, runtime artifacts,
   weights, datasets, or an automatic tag.
4. Add the suggested GitHub topics; the anonymous API currently reports an empty topics list.
5. On GitHub, require all release-gate jobs to pass on Ubuntu and Windows CPU before merging.

## Optional tag or Release after hosted CI

After reviewing the GitHub file inventory and successful hosted CI, decide whether to merge and
create `v1.0.0-rc.2`. Create a GitHub Release only after that decision. Do not upload model weights,
DOTA-derived visuals, datasets, or the historical clean-export archive as release assets without a
new rights and privacy review.

These controls reduce avoidable exposure but do not eliminate legal responsibility. The repository
is a code-only portfolio engineering artifact, not a grant of dataset, model-weight, underlying
image-source, or Ultralytics Enterprise rights.

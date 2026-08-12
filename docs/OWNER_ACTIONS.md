# Owner actions after local release-candidate approval

Authenticated owner actions are recorded separately from the local hardening workflow. With the
owner's explicit authorization, the workflow added the reviewed GitHub `origin` and pushed only
`portfolio/obb-v1.0-release-hardening`. It did not create a pull request, tag, GitHub Release, or
Space, upload an artifact, or mutate Hugging Face.

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
- The public repository now contains `portfolio/obb-v1.0-release-hardening`; final handoff audits
  must compare its exact remote SHA with the clean local branch tip.
- About description:
  `Aerial OBB 工程實驗室：以 YOLO26 與 DOTA 實驗為核心，涵蓋可重現訓練、誠實 baseline 評估、ONNX／TensorRT export、效能分析、Browser BYOM demo 與 CPU CI release gates。`
- Suggested topics: `computer-vision`, `object-detection`, `oriented-bounding-box`, `obb`,
   `yolo`, `dota`, `onnx`, `tensorrt`, `onnxruntime`, `gradio`, `byom`, `mlops`,
   `reproducibility`, `portfolio`, `zh-tw`.
- Leave the Website field empty until a reviewed BYOM site is deliberately published.

## Remaining owner actions

1. Make the still-public historical Hugging Face Space private and verify anonymous denial.
2. Add the suggested GitHub topics; the anonymous API currently reports an empty topics list.
3. Confirm the latest pushed tip passes Ubuntu/Windows CPU and synthetic browser gates, then decide
   whether to merge it or make it the repository's default branch.

## Optional tag or Release after hosted CI

After reviewing the GitHub file inventory and successful hosted CI, decide whether to merge and
create `v1.0.0-rc.2`. Create a GitHub Release only after that decision. Do not upload model weights,
DOTA-derived visuals, datasets, or the historical clean-export archive as release assets without a
new rights and privacy review.

These controls reduce avoidable exposure but do not eliminate legal responsibility. The repository
is a code-only portfolio engineering artifact, not a grant of dataset, model-weight, underlying
image-source, or Ultralytics Enterprise rights.

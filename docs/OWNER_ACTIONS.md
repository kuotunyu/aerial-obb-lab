# Owner actions after release-candidate approval

Authenticated owner actions are recorded separately from the reproducible local gates. With the
owner's explicit authorization, the project completed a low-risk clean-history publication to the
public GitHub repository. It did not create a tag, GitHub Release, replacement Space, model upload,
dataset upload, or public Hugging Face artifact.

## Completed owner actions — 2026-08-12

Hugging Face identifiers are intentionally redacted from this public handoff because exact owner
paths and private repo names are unnecessary to use or review this code-only release.

### Private historical model archive

- Renamed the historical model repository to a neutral archive name.
- Kept repository visibility **Private**.
- Anonymous HTTP verification returned `401`; the model page and files are not a public release.

### Historical Space

- The owner changed the historical demo Space to **Private**.
- The Space API and page both returned anonymous HTTP `401`; the formerly public static Space and
  bundled ONNX artifact are no longer anonymously accessible.
- No replacement Space was created for this release candidate.

### Public GitHub repository

- Repository: `https://github.com/kuotunyu/aerial-obb-lab` (**Public**).
- `portfolio/obb-v1.0-release-hardening` is already the default branch.
- The owner explicitly authorized clean-history publication.
- The public branch was replaced by one clean root commit, followed only by reviewed code-only release commits.
- Hosted required checks pass on the published code: `Core CPU / ubuntu-latest`,
  `Core CPU / windows-latest`, and `Synthetic browser smoke / Ubuntu CPU`.
- Admin enforcement is enabled. Linear history and strict required checks remain enabled.
- Force pushes and branch deletion are disabled.
- No pull request, tag, GitHub Release, or release asset was created by this publication.

About description:

`Aerial OBB 工程實驗室：以 YOLO26 與 DOTA 實驗為核心，涵蓋可重現訓練、誠實 baseline 評估、ONNX／TensorRT export、效能分析、Browser BYOM demo 與 CPU CI release gates。`

Topics:

`byom`, `computer-vision`, `dota`, `javascript`, `mlops`, `obb`, `object-detection`, `onnx`,
`onnxruntime`, `oriented-bounding-box`, `portfolio`, `reproducibility`, `tensorrt`, `webassembly`,
`yolo`, `zh-tw`.

Leave the Website field empty until a reviewed BYOM site is deliberately published.

## Public history boundary

The public default branch no longer references the superseded release history that contained a
bundled model, DOTA-derived comparison renders, retired demo surfaces, or internal planning files.
The reachable public branch contains only the audited code-only tree and subsequent clean release
commits. Private notes and interview material never entered local or remote commits.

A force-updated Git ref removes the superseded history from normal repository navigation, but
unreferenced Git objects may remain temporarily recoverable by an already-known object ID or host
retention mechanism. No secret was identified in the superseded history. If permanent host-side
purging becomes necessary, the owner must use GitHub's documented support process or recreate the
repository after a separate destructive-action review.

## Remaining optional action

After reviewing hosted CI and the public inventory, the owner may create `v1.0.0-rc.2` as a tag or
GitHub Release. Do not attach model weights, DOTA-derived visuals, datasets, or historical archives
without a new rights and privacy review.

These controls reduce avoidable exposure but do not eliminate legal responsibility. The repository
is a code-only portfolio engineering artifact, not a grant of dataset, model-weight, underlying
image-source, or Ultralytics Enterprise rights.

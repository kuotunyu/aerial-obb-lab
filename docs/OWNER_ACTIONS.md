# Owner actions after local release-candidate approval

These steps require the owner's authenticated accounts. They are intentionally **not completed**
by the local hardening workflow. Review each target before confirming a destructive or public
action.

## 1. Restrict the historical Hugging Face model

1. Sign in and open `steven0226/yolo26m-obb-dota`.
2. Open **Settings → Repository visibility** and change the repository to **Private**.
3. Sign out or use a private browser window and confirm the model page and files are inaccessible
   anonymously.

Reason: the checkpoint is DOTA-trained and is intentionally outside this code-only public release.
Do not delete it; private storage preserves the historical evidence and is reversible.

## 2. Restrict or replace the historical Hugging Face Space

1. Sign in and open `steven0226/yolo26-obb-aerial-detection`.
2. Open **Settings → Repository visibility** and change the Space to **Private**.
3. Confirm anonymously that access is denied.
4. Keep it private unless every distributed model and DOTA-derived file is removed. If it is later
   made public, replace its contents with the reviewed `demo/space-static/` BYOM files and rerun the
   browser gate first.

## 3. Create the GitHub repository

1. Create an empty public repository named `yolo26-dota-obb`.
2. Do not initialize it with a README, license, `.gitignore`, template, or generated files.
3. Suggested description: `Code-only YOLO26 OBB on DOTA portfolio: honest evaluation, deployment benchmarks, BYOM demos, and reproducible release gates.`
4. Suggested topics: `computer-vision`, `object-detection`, `oriented-bounding-box`, `obb`,
   `yolo`, `dota`, `onnx`, `tensorrt`, `onnxruntime`, `mlops`, `reproducibility`, `portfolio`.

## 4. Review and push the candidate

1. Copy the new repository URL and review it character-for-character before adding a remote.
2. From the clean local candidate, add that URL as `origin`.
3. Push only `portfolio/obb-v1.0-release-hardening`; do not push ignored files, local artifacts,
   weights, datasets, or an automatic tag.
4. On GitHub, require all release-gate jobs to pass on Ubuntu and Windows CPU before merging.

## 5. Tag or publish only after hosted CI is green

After reviewing the GitHub file inventory and successful hosted CI, decide whether to merge and
create `v1.0.0-rc.2`. Create a GitHub Release only after that decision. Do not upload model weights,
DOTA-derived visuals, datasets, or the historical clean-export archive as release assets without a
new rights and privacy review.

These steps reduce avoidable exposure but do not eliminate legal responsibility. The repository
is a portfolio engineering artifact, not a grant of dataset, model-weight, or Enterprise rights.

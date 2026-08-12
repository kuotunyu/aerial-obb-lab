# Owner actions after local release-candidate approval

Authenticated owner actions are recorded separately from the local hardening workflow. With the
owner's explicit authorization, the workflow added the reviewed GitHub `origin` and pushed only
`portfolio/obb-v1.0-release-hardening`. It did not create a pull request, tag, GitHub Release, or
Space, upload an artifact, or mutate Hugging Face.

## Completed owner actions — 2026-08-12

Hugging Face identifiers are intentionally redacted from this public handoff because the exact
owner paths are unnecessary to use or review this code-only release.

### Private historical model archive

- Renamed the historical model repository to a neutral archive name.
- Kept repository visibility **Private**.
- Anonymous HTTP verification returned `401`; the model page and files are not a public release.

### Historical Space

- The owner changed the historical demo Space to **Private**.
- On 2026-08-12, the Space API and page both returned anonymous HTTP `401`; the formerly public
  static Space and bundled ONNX artifact are no longer anonymously accessible.
- No replacement Space is required for this release candidate. Do not create
  `aerial-obb-lab-browser` unless a separate public-hosting and rights review is completed later.

### Public GitHub repository

- Created `https://github.com/kuotunyu/aerial-obb-lab` as a Public repository.
- The public repository contains `portfolio/obb-v1.0-release-hardening`, and that branch is already
  the default branch. Its remote tip remains `74f104f5d1572db2a442cd7614a9778542c2f1ea`; final
  handoff audits must compare it with the clean local branch tip after the reviewed push.
- About description:
  `Aerial OBB 工程實驗室：以 YOLO26 與 DOTA 實驗為核心，涵蓋可重現訓練、誠實 baseline 評估、ONNX／TensorRT export、效能分析、Browser BYOM demo 與 CPU CI release gates。`
- Current topics verified by the GitHub API: `byom`, `dota`, `mlops`, `obb`, `object-detection`,
  `onnx`, `onnxruntime`, `oriented-bounding-box`, `portfolio`, `reproducibility`, `tensorrt`.
- Recommended additions still missing: `computer-vision`, `javascript`, `webassembly`, `yolo`,
  `zh-tw`.
- Leave the Website field empty until a reviewed BYOM site is deliberately published.

## Remaining owner actions

### Existing public Git history boundary

The current clean tree and clean-export archive exclude retired Gradio surfaces, internal planning
files, and exact private Hugging Face owner identifiers. The existing public remote tip predates
that cleanup: those legacy paths and an alternate account association remain visible in earlier
public commits. An ordinary push removes these files from HEAD but does not erase earlier commits.

The audited refs contain no token-shaped secret or personal absolute path. `notes.private.md` and
`interview.md` have never entered any local or remote commit. One internal product brief exists only
in the unpushed local history; pushing this exact no-rewrite history would make that brief reachable
from an earlier commit even though it remains absent from the release tree and archive.

The current instruction forbids rebase, squash, amend, force-push, and other history rewriting.
Therefore the owner must choose one of these paths before publication:

1. Preserve the reviewed history and accept that legacy non-secret metadata remains discoverable;
   or
2. If historical erasure is required, provide separate explicit authorization for a clean-history
   migration and remote ruleset review. Do not improvise this during the ordinary release push.

### Unblock the reviewed branch push

The 2026-08-12 non-force push was rejected with `GH006`; the remote tip remained unchanged. The
target rule currently has `enforce_admins=true`, `Require linear history`, and these three
`required status checks`:

- `Core CPU / ubuntu-latest`
- `Core CPU / windows-latest`
- `Synthetic browser smoke / Ubuntu CPU`

The local history contains the already-recorded merge commit
`71f4358340c5de8a1fab6281666af5eb95b3906c`. Do not rebase, amend, squash, force-push, delete the
remote branch, or manufacture status results to satisfy this rule.

Owner action:

1. Open GitHub **Settings → Branches → Branch protection rule** for
   `portfolio/obb-v1.0-release-hardening`.
2. Temporarily clear **Do not allow bypassing the above settings** (the API field is
   `enforce_admins`), then save the rule. Leave force pushes and deletions disabled.
3. Ask Codex to retry the same ordinary push. Because `kuotunyu` is the repository administrator,
   the one-time push can then bypass the pre-existing linear-history and not-yet-created check
   requirements without changing any commit.
4. After the push starts the three workflows, immediately re-enable **Do not allow bypassing the
   above settings**. Confirm all three checks pass on the exact pushed SHA.

Other remaining actions:

1. Optionally add the five missing recommended topics listed above.
2. After the reviewed push, verify the public file inventory and Actions run against the exact
   local SHA. The release branch is already the default branch; no default-branch change is needed.

## Optional tag or Release after hosted CI

After reviewing the GitHub file inventory and successful hosted CI, decide whether to merge and
create `v1.0.0-rc.2`. Create a GitHub Release only after that decision. Do not upload model weights,
DOTA-derived visuals, datasets, or the historical clean-export archive as release assets without a
new rights and privacy review.

These controls reduce avoidable exposure but do not eliminate legal responsibility. The repository
is a code-only portfolio engineering artifact, not a grant of dataset, model-weight, underlying
image-source, or Ultralytics Enterprise rights.

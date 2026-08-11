# YOLO26 OBB Release-Hardening Design

**Status:** Approved by the feature-frozen implementation brief dated 2026-08-12.

## Objective

Turn the existing YOLO26 OBB on DOTA portfolio project into a clean, locally committed release
candidate without retraining, rerunning full validation, using a GPU, or changing any remote.
Preserve the observed near-tie/slight regression against the matched official baseline and make
every public performance claim traceable to committed evidence with explicit limits.

## Constraints

- Run local gates on CPU only. Do not initialize CUDA, TensorRT, or model inference.
- Treat the accepted A100 validation and T4 deployment results as historical evidence.
- Do not add models, datasets, major UI behavior, or new benchmarks.
- Keep DOTA, checkpoints, secrets, runtime output, private notes, and interview material out of
  version control and release archives.
- Do not create or mutate remotes, tags, releases, Spaces, model repositories, or uploads.
- Use only `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` as author and committer,
  with no co-author trailer.

## Approaches Considered

1. **Committed evidence registry with deterministic gates (selected).** Record accepted metrics,
   provenance, limitations, asset hashes, and redistribution boundaries in machine-readable
   artifacts. Verify them with small CPU-only programs and cross-platform CI.
2. **Documentation-only correction.** This is smaller, but future edits could silently make the
   READMEs disagree with evidence and it would not validate packaged files or browser geometry.
3. **Fresh end-to-end model certification.** This would require weights, DOTA, GPU execution,
   TensorRT builds, and full validation, directly violating the release constraints.

## Architecture

### Evidence and claims

`release/evidence.json` is the release-level registry. It records the matched comparison, the
DOTA8 export smoke, the T4 benchmark environment, analysis counts, browser model identity, and
links to primary committed evidence. Historical claims that lack a raw committed console log are
marked as transcribed accepted results instead of being upgraded into stronger evidence.

`release/artifact-manifest.json` lists redistributed binary/visual artifacts with byte size,
SHA-256, provenance, license, and restrictions. A release verifier checks schemas, arithmetic,
artifact hashes, notebook cleanliness, documentation claim markers, privacy rules, and archive
membership. It operates offline and does not import Torch or Ultralytics.

### Browser parity

Pure browser geometry functions are separated from DOM and ONNX Runtime orchestration. A fixed
synthetic JSON fixture covers letterbox dimensions/padding, RGBA-to-RGB CHW normalization,
`[cx, cy, w, h, confidence, class, angle_radians]` decoding, filtering, and rotated corners. A
Python reference independently computes the same contract and compares its output with a Node
runner. Browser smoke stubs ONNX Runtime, loads the static page over loopback HTTP, and confirms
that a deterministic output reaches the rendered table without fetching a model or using a GPU.

### Packaging and CI

Ubuntu and Windows jobs create Python 3.11 environments from `uv.lock` using a non-editable
install, run unit and repository gates, run JavaScript parity/syntax checks, build the wheel and
source distribution, and inspect their contents. The non-editable install is required on Windows
because CPython 3.11 may decode editable `.pth` paths with the system code page.

A clean-export tool derives an archive only from `git ls-files`/the committed tree, rejects dirty
or unexpected inputs, excludes private and runtime material by construction, and verifies the
archive in a temporary extraction. Docker is not added because neither the static production demo
nor the CPU-only release gates require a service image.

### Public documentation and licensing

Both READMEs, result reports, the model card, and demo copy use the same bounded claims:

- Fine-tuning is a near-tie/slight regression: about -0.05 percentage point mAP50 and -0.13
  percentage point mAP50-95 under matched conditions.
- The three 0.9950 values are DOTA8 export-smoke parity, not DOTAv1 production certification.
- 20.22 ms / 49.4 FPS applies only to the recorded Tesla T4, batch 1, 1024 px, FP16 TensorRT
  environment and is not a browser or general deployment promise.
- The browser demo runs the official lightweight `yolo26n-obb` export, not the fine-tuned medium
  checkpoint, and inherits neither its accuracy evidence nor its T4 latency.

`THIRD_PARTY_NOTICES.md` distinguishes repository code, bundled Ultralytics-derived weights,
DOTA-derived visuals/weights, CDN-fetched ONNX Runtime Web, and non-vendored Python dependencies.
It records the AGPL/Enterprise decision boundary and DOTA's academic-only, non-commercial terms;
commercial or closed-source use remains an explicit owner/legal action rather than a release
claim. `CITATION.cff`, `CHANGELOG.md`, and `RELEASE_CHECKLIST.md` complete the release metadata.

## Failure Handling

All gates fail closed with a nonzero exit code and name the file, claim, or artifact that failed.
Network checks are separate, anonymous, read-only evidence and never a prerequisite for offline
CI. Missing external permission is documented as an owner action while unrelated local work
continues.

## Verification Strategy

Release-blocking behavior follows red-green TDD: add one failing test or fixture assertion, run it
to confirm the intended failure, implement the smallest fix, then rerun the focused and full
suites. Final verification runs in a clean committed export on Windows CPU and uses the same
commands encoded for Ubuntu/Windows CI. The release is a candidate only when the worktree is clean,
all committed author/committer identities and trailers pass, and no remote mutation occurred.

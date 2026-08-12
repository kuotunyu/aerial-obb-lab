# Changelog

All notable changes to this project are documented here.

## [1.0.0-rc.2] - 2026-08-12

### Code-only portfolio candidate

- Removed the bundled ONNX model and five DOTA-derived comparison renders while preserving their
  hashes and provenance as excluded historical audit records.
- Converted browser and Python demos to bring-your-own-model operation with no model download,
  named-weight fallback, implicit export, or GPU default.
- Removed public dependencies on owner-hosted Hugging Face artifacts and added explicit owner-only
  handoff steps for remote visibility and GitHub publication.
- Added release gates that reject model binaries, DOTA-derived visuals, owner artifact links,
  remote demo acquisition, private paths, and runtime output in both Git and clean exports.
- Made the complete Traditional Chinese `README.md` the canonical GitHub and package landing page,
  with a complete `README.en.md` and a short compatibility pointer.
- Replaced duplicated Gradio layouts with a shared 38/62 wide workbench, explicit Detect flow,
  model-free loopback preview, responsive Playwright smoke, and no ML runtime in the preview gate.
- Adopted the `Aerial OBB Lab` release identity across the zh-TW-first presentation, Python
  distribution, citation metadata, and clean-export filename while preserving `obbkit` imports and
  historical experiment identifiers.

### Remaining evidence limitations

- Fine-tuning remains a near-tie/slight regression; no retraining or full validation was run.
- DOTA8 values remain export-smoke parity only, and the T4 latency is limited to its recorded
  batch-1, 1024px historical environment.
- The checkpoint and DOTA-derived imagery are not distributed; independent reproduction requires
  owner-supplied artifacts and the applicable permissions.
- Anonymous verification found the historical Hugging Face Space still public and running; the
  owner must make it private before treating the code-only publication boundary as complete.

## [1.0.0-rc.1] - 2026-08-12

### Release hardening

- Bound public performance claims to a committed machine-readable evidence registry.
- Preserved the matched fine-tuning result as a near-tie/slight regression rather than an
  improvement claim.
- Scoped the DOTA8 result to export-smoke parity and the TensorRT result to its recorded T4
  environment.
- Added synthetic Python/JavaScript parity for browser preprocessing, `[N,7]` decode, angle, and
  rotated corners without model inference.
- Added a headless end-to-end UI smoke with a fixed synthetic SVG and deterministic ONNX output
  stub; it exercises upload, canvas drawing, and result rendering without model inference.
- Added artifact hashes, privacy gates, third-party notices, citation metadata, cross-platform CI,
  package checks, and clean-export verification.

### Known limitations

- Historical baseline and T4 raw console logs are not committed; their accepted transcriptions
  and evidence strength are explicit in `release/evidence.json`.
- DOTA and underlying imagery are not cleared for commercial use.
- The static browser demo uses the official nano model and has no fine-tuned-medium accuracy or T4
  latency claim.

## Historical implementation - 2026-07-13 to 2026-07-15

- Completed the original phases 0-7: environment, resumable Colab training, matched evaluation,
  geometry analysis, ONNX/TensorRT experiment, demos, and model card.

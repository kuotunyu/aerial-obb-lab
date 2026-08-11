# Changelog

All notable changes to this project are documented here.

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

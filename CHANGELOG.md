# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-08-16

### Evidence-first stable release

- Promoted the validated `1.0.0-rc.2` code-only BYOM candidate to the stable `1.0.0` release without changing model, dataset, training, benchmark, or browser inference behavior.
- Preserved the matched fine-tuning result as a near-tie/slight regression: Δ mAP50 **-0.05pt** and Δ mAP50-95 **-0.13pt**.
- Preserved the DOTA8 result as export-smoke parity only, the T4 numbers as historical environment-bound latency evidence, and the geometry analysis as a ground-truth proxy rather than detector/NMS proof.
- Kept the public distribution code-only: no model weights, ONNX/TensorRT binaries, DOTA-derived visual assets, or owner-hosted private artifact links are shipped.
- Promoted Python package metadata, citation metadata, release URL, and version-consistency coverage to stable `1.0.0`.

### Stable-release evidence boundary

- No fresh training, full DOTAv1 validation, GPU benchmark, paid service call, or new model inference was run for the stable promotion.
- Browser BYOM remains a local model/image integration demo; the synthetic browser gate does not establish model accuracy or T4-equivalent latency.
- Historical baseline and T4 raw console logs remain unavailable; their accepted transcriptions and limitations stay explicit in `release/evidence.json`.

## [1.0.0-rc.2] - 2026-08-12

### Code-only portfolio candidate

- Removed the bundled ONNX model and five DOTA-derived comparison renders while preserving their
  hashes and provenance as excluded historical audit records.
- Converted browser and Python demos to bring-your-own-model operation with no model download,
  named-weight fallback, implicit export, or GPU default.
- Removed public dependencies on owner-hosted Hugging Face artifacts and retained anonymous
  visibility findings in the machine-readable evidence registry.
- Added release gates that reject model binaries, DOTA-derived visuals, owner artifact links,
  remote demo acquisition, private paths, and runtime output in both Git and clean exports.
- Republished the public default branch through an explicitly authorized clean-history publication;
  its reachable history begins at a code-only root with no weights or DOTA-derived visuals.
- Replaced private Hugging Face repo names in executable notebooks with explicit owner-supplied
  placeholders and made the privacy gate reject known private repo identifiers across tracked text.
- Updated the official checkout, Python, and Node setup actions to their Node 24-compatible v7
  majors after verifying the upstream releases.
- Made the complete Traditional Chinese `README.md` the canonical GitHub and package landing page,
  with a complete `README.en.md` and no duplicate compatibility-pointer file.
- Removed the duplicate server-side UI and made one 34/66 browser-native workbench canonical,
  with explicit Detect, 19px base type, 15px minimum secondary text, square-edged controls,
  responsive Playwright smoke, safe errors, polygon-only overlays, and no model inference in the
  release gate.
- Adopted the `Aerial OBB Lab` release identity across the zh-TW-first presentation, Python
  distribution, citation metadata, and clean-export filename while preserving `obbkit` imports and
  historical experiment identifiers.
- Recorded anonymous `401` verification for both the Private model archive and the formerly public
  historical Space after the owner completed the visibility change.
- Added default-false acknowledgements before every historical GPU/DOTA/HF workflow, removed the
  smoke test's remote-write path, and removed the unused default Hugging Face client dependency.
- Pinned the browser runtime to ONNX Runtime Web 1.20.1 with SHA-384 SRI while documenting that
  model/image bytes stay local but runtime code is fetched from jsDelivr.
- Removed obsolete planning, one-time owner-instruction, language-pointer, and remote-checkpoint
  helper files from the public release surface.
- Removed the historical GPU inference/export stack from the release dependency graph after its
  obsolete Torch/torchvision bounds blocked automated vulnerability remediation.

### Remaining evidence limitations

- Fine-tuning remains a near-tie/slight regression; no retraining or full validation was run.
- DOTA8 values remain export-smoke parity only, and the T4 latency is limited to its recorded
  batch-1, 1024px historical environment.
- The checkpoint and DOTA-derived imagery are not distributed; independent reproduction requires
  owner-supplied artifacts and the applicable permissions.

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
- The rc.1 static browser demo used the official nano model and carried no fine-tuned-medium
  accuracy or T4 latency claim; rc.2 removed that model and moved to BYOM.

## Historical implementation - 2026-07-13 to 2026-07-15

- Completed the original phases 0-7: environment, resumable Colab training, matched evaluation,
  geometry analysis, ONNX/TensorRT experiment, demos, and model card.

# Aerial OBB Pages Compressed Model Length Design

**Status:** Approved under the owner's standing direction to complete the recommended production fix

## Problem and evidence

The reviewed `demo/web` artifact is byte-identical to the deployed GitHub Pages tree, but the live fixed-harbor Detect path fails with `DEMO_MODEL_SIZE`. GitHub Pages serves the 10,207,127-byte ONNX response with `Content-Encoding: gzip` and a `Content-Length` of 8,771,637 bytes. Fetch exposes that transfer length while its response body yields the decoded model bytes. The current loader incorrectly requires the transfer length to equal the decoded manifest length before it reads the body.

## Considered approaches

1. **Use decoded bytes as the authority (recommended).** Remove the `Content-Length` equality check. Retain the streamed 15 MiB cap, the exact decoded byte-count check, and SHA-256 verification. This works for encoded and identity responses without weakening the artifact contract.
2. **Condition the header check on `Content-Encoding`.** Keep the equality check only for identity responses. This adds brittle HTTP intermediary semantics while duplicating the authoritative decoded-body checks.
3. **Change hosting or the model filename to avoid compression.** This depends on hosting behavior outside the repository and does not make the loader correct for ordinary HTTP content coding.

Approach 1 is selected because the bytes consumed by ONNX Runtime are the security and compatibility boundary. Transfer framing is not.

## Design

`DemoAssets.fetchVerifiedModel` continues to admit only the exact same-origin model URL and continues to require an OK streaming response. It reads decoded response chunks, aborts if the decoded stream exceeds either the manifest byte count or 15 MiB, requires the final decoded length to equal the manifest, and requires the exact SHA-256 digest before returning the buffer. It no longer treats `Content-Length` as an artifact-size claim.

The canonical generator template in `scripts/prepare_sample_gallery.py` and the committed `demo/web/demo-assets.js` output change together. No UI copy, manifest, model, image, provenance, Pages configuration, or BYOM behavior changes.

## Regression coverage

A real Playwright scenario serves the admitted model as a gzip-coded HTTP response whose transfer `Content-Length` is smaller than the decoded model. It stubs only ONNX Runtime execution, then uses the real page, Fetch stream, model-length check, digest check, and Detect state transition. Before the production change it must fail with the fixed model-size recovery; after the change it must reach a successful result with one inference run.

Existing truncated-model and changed-digest scenarios remain green, proving that decoded size and SHA-256 rejection still fail closed. The full browser smoke, unit/contract suite, artifact verifier, repo/release gates, generator replay, and clean export remain required before remote publication.

## Remote gate

The correction ships through a separate pull request. It may update the existing Pages deployment only after the exact merge SHA passes main Release gates. No release, tag, Hugging Face, visibility, archive, or unrelated-repository mutation is in scope.

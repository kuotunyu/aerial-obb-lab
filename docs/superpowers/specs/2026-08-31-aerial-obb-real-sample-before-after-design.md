# Aerial OBB Real-sample Before/After Showcase Design

- **Date:** 2026-08-31
- **Repository:** `kuotunyu/aerial-obb-lab`
- **Design branch:** `docs/pages-real-sample-before-after-design`
- **Base commit:** `24039db9e07e327b55433086241ac574430c4531`
- **Dependency:** the local `fix/pages-live-review-accessibility` branch must be integrated before implementation begins
- **Status:** Conversational design approved; pending written-spec review
- **Scope:** replace the public synthetic-first experience with two reviewed real-image Before/After samples while preserving BYOM as the only live-inference path

## Problem and outcome

The current public workbench is technically honest but difficult for a first-time visitor to understand. Its
primary action loads an abstract authored SVG and a fixed tensor-like result. After that action, a visitor can
filter the result but receives no intuitive explanation of what rotated detection does to a real aerial image.
The natural reaction is “what happens next?”

The approved outcome is a portfolio-first path that a non-specialist can understand within ten seconds:

1. see a real aerial image;
2. see the same image with oriented detections;
3. understand that the shown result was generated earlier and is not live inference; and
4. optionally open BYOM to run a compatible local model on a local image.

The change is explanatory, not a new accuracy claim. It does not publish a model, add a hosted inference
service, or treat a curated example as benchmark evidence.

## Selected approach and alternatives

### Selected: side-by-side real-image Before/After

The home path presents two compact sample cards. Selecting one renders the same reviewed real aerial image
twice: an unannotated Before figure and an After canvas containing the image plus precomputed oriented boxes.
The result summary and table sit directly below the pair. BYOM is a secondary, explicitly opened workspace.

This approach was selected because the difference is visible without learning a comparison control, it works
with keyboard and touch input, and it keeps the claim boundary adjacent to the result.

### Rejected for this phase

- **Drag comparison slider:** visually engaging, but adds a new gesture, keyboard semantics, pointer capture,
  zoom interaction, and mobile edge cases without improving the core explanation.
- **Multi-scene gallery:** useful only after there is evidence that visitors explore more than the first two
  examples; it increases asset, licensing, artifact-size, and review scope now.
- **Bundled model or hosted inference:** would weaken the code-only/privacy boundary, increase download and
  supply-chain risk, and is unnecessary because BYOM already provides genuine browser inference.

## Public asset boundary and candidate selection

### Allowed source class

Only individual images whose description and source record explicitly identify them as U.S. federal
government public-domain works may become committed samples. A general agency policy is supporting evidence,
not a substitute for an image-specific rights record. The implementation must preserve a permanent source
page, originating agency, author/creator when available, public-domain basis, retrieval date, original file
digest, published crop digest, crop rectangle, output dimensions, and transformation record.

Initial candidates are limited to these reviewed source families:

1. **Aircraft boneyard:** U.S. Air Force/NARA aerial photograph of stored aircraft at the Aerospace
   Maintenance and Regeneration Center, NARA identifier 6438938, marked public domain as a U.S. federal
   government work.
2. **Naval shipyard:** U.S. Navy/NHHC aerial view of Pearl Harbor Naval Shipyard, photo 80-G-361740, marked
   public domain as a U.S. federal government work.
3. **Port reserve candidate:** U.S. Army aerial photograph over the Port of Tacoma, DVIDS image 3156545,
   marked public domain as an official-duty U.S. federal government work.

The final release contains exactly two candidates that pass every gate below. The aircraft and naval
candidates have priority; the port candidate is the defined replacement if either priority candidate fails.
No other image source may be substituted without a written design amendment.

### Candidate acceptance gate

For each candidate, implementation must perform these steps in order:

1. save the image-specific source and rights record outside the repository for review;
2. inspect the original image and metadata; reject third-party copyright, ambiguous authorship, visible
   protected logos used as branding, identifiable private persons, or sensitive/private information;
3. create one deterministic crop with no geometric warp, at most 1600 pixels on its longest edge;
4. remove EXIF/IPTC/XMP and encode a same-origin WebP capped at 300 KiB;
5. run the owner-authorized external compatible OBB model locally against that exact published crop;
6. accept only finite, schema-valid output that produces 1–20 visually understandable detections at the
   fixed default confidence `0.25`; and
7. reject the candidate instead of moving, adding, deleting, relabeling, or hand-tuning any box.

The two committed sample images together, their JSON, and attribution text must remain below 700 KiB. No
original full-resolution file, model, weight, ONNX binary, DOTA pixel, DOTA derivative, private model name,
private path, or model metadata may enter the repository or Pages artifact.

## Output provenance and claim presentation

The After boxes are real precomputed outputs produced locally by an owner-authorized compatible OBB model.
The generating model remains outside every repository and artifact. The committed provenance records only:

- `kind: precomputed-illustrative-output`;
- generation date;
- repository commit containing the decoder/output contract;
- published image SHA-256;
- output contract `output0 [1,N,7]`; and
- result JSON SHA-256.

It must not record the model filename, local path, embedded metadata, private identifier, weight digest, or
runtime. The output is never described as ground truth or a verified correct answer.

Before the first sample-selection control, an always-visible, non-collapsible notice states:

> **真實航拍範例 — 預先產生結果**
> 以下範例使用經授權審查的公共領域航拍影像與預先產生的 OBB output。本頁沒有執行模型推論；這些結果不是 accuracy、evaluation 或 latency evidence。若要實際推論，請開啟 BYOM 並自行提供相容 ONNX model 與影像。

The active sample result uses these exact presentation values:

- mode badge: `PRECOMPUTED SAMPLE · NO LIVE INFERENCE`;
- runtime: `N/A · precomputed output`;
- provenance: the originating agency and a user-activated source link; and
- status: `範例已載入 · 本頁沒有執行模型推論。`

No NASA, NARA, DoD, service branch, archive, photographer, or source institution is represented as endorsing
the model, output, repository, or portfolio. Image credit and model-output attribution are separate.

## Files and data contracts

The implementation plan may refine filenames only if the repository already has a stricter adjacent
convention. The intended public layout is:

```text
demo/web/
  samples/
    aircraft-before.webp
    naval-before.webp
    samples.json
    SAMPLE_ASSETS.md
```

`samples.json` is the only application data source for sample selection. Each entry contains:

```json
{
  "id": "aircraft",
  "label": "機場與飛機",
  "image": {
    "url": "samples/aircraft-before.webp",
    "sha256": "64 lowercase hexadecimal characters",
    "width": 1600,
    "height": 1000
  },
  "source": {
    "agency": "originating U.S. federal agency",
    "recordUrl": "https://permanent image-specific source record",
    "rights": "Public domain — U.S. federal government work"
  },
  "output": {
    "kind": "precomputed-illustrative-output",
    "contract": "output0 [1,N,7]",
    "sha256": "64 lowercase hexadecimal characters",
    "results": {
      "output0": {"dims": [1, 1, 7], "data": [512, 512, 100, 50, 0.9, 0, 0]}
    }
  }
}
```

The JSON example illustrates shape only; implementation must replace the example values with generated,
reviewed candidate data and must not copy the example detection. Application code validates IDs, dimensions,
finite values, class indexes, hashes, source URL scheme/host allowlist, and output contract before publishing
any result. An invalid entry fails closed.

`SAMPLE_ASSETS.md` is a human-reviewable manifest containing the complete acquisition and transformation
record. `samples.json` contains only the public runtime subset. The Pages verifier allowlists both images and
both manifest files by exact path, size, media type, and digest.

## User experience and information hierarchy

### Initial state

The page leads with one plain-language sentence: `看模型如何在航拍影像中辨識有方向的目標。` The claim
notice follows immediately and remains before the first interactive control. Two sample cards appear next,
each with a small unannotated thumbnail, scene label, source agency, and `查看 Detect 前／後` action.

No model picker, image picker, tensor contract, or technical class list competes with the sample choice in
the initial viewport. The existing technical workbench is available through one secondary button:
`開啟 BYOM 工作區`.

### Sample selected

Selecting a card performs an atomic transition:

1. increment the existing generation token;
2. set phase to `loading`;
3. clear the previous sample/BYOM table, statistics, descriptions, completion state, and overlay;
4. release any active BYOM session and clear BYOM file selections;
5. fetch the selected same-origin image and manifest entry;
6. verify the entry and render only if both are current and valid; and
7. move focus to the result heading and publish the fixed success status.

Desktop presents two equal `figure` elements side by side. Mobile stacks Before first and After second. Both
figures use the same decoded image. Before is a plain `img`; After is a canvas that draws the image and the
shared decoded/filter result. Labels are always visible and are not encoded by color alone.

The result table and non-live canvas description consume the same confidence-sorted, class-filtered detection
list. The confidence/class controls never run inference. Filtered-empty text explicitly states that no sample
detections match the current filters and the After canvas has no oriented polygons. Runtime remains
`N/A · precomputed output` for populated and filtered-empty sample results.

### BYOM opened

`開啟 BYOM 工作區` reveals the existing model picker, image picker, confidence controls, class controls, and
Detect action without navigating away. Opening the section alone does not load ORT. Selecting a model remains
the only ORT lazy-load trigger. A successful real run uses `BYOM · LOCAL BROWSER INFERENCE` and a finite
numeric runtime. Loading, reset, no-cache, and all errors use `—`.

Selecting a sample while BYOM is active releases its session and clears local file selections. Selecting
either BYOM file clears the sample result. Generation-token checks keep late sample loads, session creation,
image decode, and inference completion from republishing stale state.

## Synthetic fixture disposition

The authored synthetic SVG remains useful as deterministic geometry and browser-test evidence but is no
longer a public primary CTA. Implementation moves the synthetic asset/output from the published `demo/web`
tree into test fixtures, or otherwise proves it is absent from the Pages artifact. Browser tests may inject
or serve it from the test harness only. Public UI copy must not direct ordinary visitors to a synthetic-first
path.

Removing the public synthetic fixture does not weaken parity coverage: letterbox, `[1,N,7]` selection,
confidence/class filtering, rotated corners, rendering, runtime boundaries, and safe failures remain covered
by deterministic tests.

## Error and recovery states

Sample image, manifest, digest, schema, decode, or render failure is atomic. No partial Before image, stale
After overlay, table, statistic, description, source attribution, or prior runtime may remain. The user sees
only:

> 範例無法載入。請選擇另一張範例，或開啟 BYOM 使用自己的模型與影像。

Recovery controls are the other sample card and `開啟 BYOM 工作區`. UI and console must not expose local
paths, private model identity, model metadata, raw exception, response body, tensor data, or stack. Source
URLs are never fetched at runtime; they are ordinary user-activated attribution links.

Existing BYOM fixed recovery messages and candidate-session-then-release lifecycle remain unchanged.

## Accessibility and responsive behavior

- The skip link remains the first focusable element and targets `main#mainContent`.
- The claim notice stays visually before the sample controls.
- Sample cards are real buttons with stable names and programmatic selected state.
- Before and After are `figure`/`figcaption` pairs with visible labels.
- The After canvas uses `aria-describedby` to reference the same current filtered description as the table.
- The description is not `aria-live`; the existing concise status region remains the sole announcement
  channel.
- Explicit sample selection moves focus to the result heading; filter changes do not steal focus.
- Keyboard, touch, 200% zoom, reduced motion, 1280×720 desktop, and 390×844 mobile remain usable without
  horizontal overflow.
- Before/After order is semantic DOM order, so the mobile stack and screen-reader order match.

## Test design and release gates

Implementation uses strict TDD with real browser failures before behavior changes. Required coverage is:

### Static and asset contracts

- exactly two sample entries and two allowlisted same-origin WebP files;
- exact image/result digests, dimensions, size budgets, MIME signatures, and empty EXIF/IPTC/XMP;
- image-specific HTTPS source records on the approved host allowlist and explicit public-domain basis;
- no DOTA names/pixels/derivatives, weights, ONNX, model metadata, source originals, or unreviewed images;
- sample JSON schema rejects duplicate IDs, traversal, external asset URLs, non-finite values, invalid dims,
  bad class indexes, mismatched hash, missing attribution, and unknown fields; and
- synthetic public assets are absent while deterministic test fixtures remain available.

### Real browser behavior

- initial page and both sample paths issue only same-origin requests and zero ORT requests;
- claim notice precedes the first sample control;
- each sample renders one identical Before/After base image, with polygons only on After;
- exact mode, runtime, provenance, notice, source link, status, result table, and canvas description;
- confidence and class filters keep table, summary, description, and After canvas synchronized;
- filtered empty, sample switching, repeated selection, failed asset/schema/decode/render, and retry contain no
  stale result;
- opening BYOM does not request ORT; selecting a model requests the one pinned SRI/anonymous-CORS runtime;
- BYOM candidate/session/release, pending Detect clearing, numeric completed runtime, privacy, and safe failures
  remain green; and
- keyboard focus, labels, heading order, 200% zoom, reduced motion, desktop/mobile screenshots, and overflow.

### Repository and artifact gates

- full pytest, browser smoke, repo check, release/privacy check, Pages artifact verifier, and strict clean
  export;
- downloaded candidate bytes equal the reviewed `demo/web` tree before any later merge/deploy gate;
- anonymous HTTP checks for repository and attribution records; and
- no external request from initial/sample use, with jsDelivr permitted only after BYOM model selection.

## Non-goals

- No bundled or remotely fetched model, weight, ONNX file, DOTA content, upload service, analytics, storage,
  telemetry, hosted inference, or framework.
- No benchmark, accuracy, latency, ground-truth, or correctness claim from the two samples.
- No manual box adjustment, relabeling, cherry-picking of individual detections, or hidden confidence change.
- No drag slider, large gallery, autoplay, animation-heavy tour, map, geolocation, or sample download feature.
- No change to GitHub Pages, GitHub About, repository visibility, Hugging Face, releases, tags, or other repos
  in this design turn.

## Dependency and remote gates

This design branch is intentionally based on local commit `24039db`, which contains the approved
accessibility/runtime follow-up. Product implementation must not start until that dependency is integrated
or central coordination provides an exact replacement base. A stacked product PR is not implicitly
authorized.

All remote actions remain separate written gates:

1. **Gate A — candidate PR:** separately authorized non-force push and PR after local implementation review.
2. **Gate B — artifact review and merge:** download CI candidate, compare exact bytes, review locally, then
   use only an authorized repository-supported merge method.
3. **Gate C — Pages deployment:** configure/dispatch only after exact merged-main CI succeeds.
4. **Gate D — live review:** HTTPS/assets/origins, both samples, BYOM, privacy, desktop/mobile, accessibility,
   and attribution review on the deployed SHA.
5. **Gate E — About and Portfolio Control receipt:** only after a separately passing live review.

This spec, its commit, and later planning authorize none of those gates.

## Self-review record

- **Placeholders:** no behavioral placeholder exists. Three named source candidates and a deterministic
  priority/replacement rule resolve asset selection; implementation stops if fewer than two candidates pass.
- **Consistency:** precomputed samples never display live-inference runtime or load ORT; only completed BYOM
  results do. Sample switching and BYOM transitions share atomic clearing and generation-token protection.
- **Ambiguity:** image rights, transformations, size, metadata, output contract, prohibited manual editing,
  public copy, error copy, focus, and remote gates are explicit.
- **Scope:** two real-image samples, the explanatory hierarchy, synthetic test-fixture disposition, and
  required regressions only; no model distribution, hosted inference, benchmark, deployment, or About edit.

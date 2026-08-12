# Aerial OBB Lab Release Identity Design

## Context

The owner selected an engineering-lab identity so this portfolio is distinguishable from other
YOLO projects without turning DOTA, a non-commercial research dataset, into the product brand.
The empty public GitHub repository now exists at `kuotunyu/aerial-obb-lab`, and the historical
Hugging Face model repository was renamed to `steven0226/aerial-obb-lab-model-archive` and kept
private.

## Selected approach

Use a complete but bounded release identity:

- Human-facing brand: **Aerial OBB Lab**.
- GitHub and Python distribution slug: `aerial-obb-lab`.
- Private historical model archive: `aerial-obb-lab-model-archive`.
- Primary subtitle: **YOLO26 Oriented Detection：從誠實評估到 Browser Deployment**.

This updates the current public presentation, package metadata, citation metadata, release archive
name, owner handoff, and notebook archive target. It does not rename the stable `obbkit` Python
import, historical experiment run directories, model identifiers, evidence keys, or old planning
records whose original names are part of their historical context.

## Alternatives considered

1. **Display-name-only rename.** Smallest change, but wheel/sdist, clean export, and owner
   instructions would retain the ambiguous old identity.
2. **Complete but bounded release identity — selected.** Gives the current release one coherent
   identity while preserving reproducibility-sensitive identifiers.
3. **Rewrite every historical identifier.** Rejected because it would blur provenance and create
   unnecessary notebook, evidence, and commit-history churn.

## Public presentation

`README.md` remains the canonical zh-TW landing page and leads with the new brand and subtitle.
`README.en.md` carries the matching English identity. Technical claims remain unchanged: the
fine-tuned model is still a near-tie/slight regression, DOTA8 remains export-smoke evidence only,
T4 latency remains environment-specific, and demos remain BYOM without bundled weights.

The GitHub repository URL may appear in package and owner metadata because the empty public
repository now exists. The private Hugging Face archive must not appear in public presentation
files as a download or demo link.

## Package and export behavior

The distribution name becomes `aerial-obb-lab`; its import package remains `obbkit`. The default
clean committed export becomes `dist/aerial-obb-lab-v1.0.0rc2.zip`. Package version remains
`1.0.0rc2` because nothing has been published and this is release-candidate hardening, not a new
feature release.

## Owner handoff state

The owner-action record distinguishes completed authenticated actions from remaining actions:

- completed: private HF archive rename/visibility and empty public GitHub repository creation;
- not applicable: the historical Space URL does not resolve and no replacement Space is needed;
- remaining: local remote review/addition, branch push, hosted CPU CI, and optional tag/Release.

The local workflow performs none of the remaining remote actions.

## Verification and safety

- Add failing tests for distribution identity, repository URL, clean-export name, and rejection of
  private owner-HF links from public presentation.
- Update the minimum metadata and documentation needed to satisfy those contracts.
- Synchronize edited notebook sources without executing cells, training, validation, export, or
  inference.
- Run the full CPU-only test, repository, release, browser, package, privacy, and clean-export
  gates with `CUDA_VISIBLE_DEVICES=-1`.
- Stage exact paths only, preserve ignored private/runtime files, and perform no remote mutation.

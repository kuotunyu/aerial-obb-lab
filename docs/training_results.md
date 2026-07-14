# Phase 2 fine-tuning results — raw output

`yolo26m-obb.pt` fine-tuned on a DOTAv1 re-split (`split_dota` rates `[0.8, 1.2]`, gap 500,
62,030 train / 21,271 val tiles at 1024px) on a Colab A100, evaluated against the same
official checkpoint on the identical val split. Full run config in `args.yaml` (archived to
the private HF model repo); training/eval scripts in `notebooks/01_train_dotav1_a100.ipynb`.

## Aggregate comparison

| model | split | mAP50 | mAP50-95 |
|---|---|---:|---:|
| yolo26m-obb.pt (official, published) | DOTAv1 **test** | 81.0 | 55.3 |
| yolo26m-obb.pt (official, our reproduction) | DOTAv1 **val** (this repo's tiling) | 78.2 | 63.3 |
| fine-tuned `best.pt` (epoch 13) | DOTAv1 **val** (this repo's tiling) | 78.2 | 63.1 |

**Only the bottom two rows are apples-to-apples** (same val split, same tiling, same
`ultralytics` eval call, same conditions). The top row is the officially published number on
the **test** split, evaluated with DOTA's own toolkit after stitching tile predictions back to
full-resolution images — a different pipeline, included for context only, not for direct
comparison.

**Methodology caveat worth stating plainly**: our val-split mAP50-95 (63.3) is ~8 points
*higher* than the official test-split number (55.3), while mAP50 is ~3 points *lower*
(78.2 vs 81.0). If this were simple val/train leakage we'd expect both metrics to move the
same direction. The more likely explanation is that the two numbers come from different
evaluation pipelines (official: full-image stitching + DOTA's own AP tool; ours: tile-level
`ultralytics` `val()`) rather than the model having seen the val data — but this is exactly
the kind of number that deserves a footnote instead of being reported at face value.

**Fine-tuning result**: Δ mAP50 = **-0.05pt**, Δ mAP50-95 = **-0.13pt** — within noise,
essentially a tie. Expected: `yolo26m-obb.pt` is already the official DOTAv1-trained
checkpoint, so continuing to fine-tune it on our own re-tiled version of the same dataset is
closer to "keep training an already-converged model" than "adapt to a new domain." The value
of this run is demonstrating (and quantifying) that ceiling, not chasing a score.

## Per-class comparison (fine-tuned − baseline)

12 of 15 classes below; `plane` / `ship` / `storage tank` fine-tuned per-class numbers weren't
captured before the Colab session ended (baseline-only for those three).

| class | baseline mAP50 | fine-tuned mAP50 | Δ mAP50 | baseline mAP50-95 | fine-tuned mAP50-95 | Δ mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| helicopter | 0.748 | 0.833 | **+0.085** | 0.530 | 0.618 | **+0.088** |
| soccer ball field | 0.629 | 0.660 | +0.031 | 0.527 | 0.555 | +0.028 |
| small vehicle | 0.697 | 0.721 | +0.024 | 0.538 | 0.571 | +0.033 |
| large vehicle | 0.818 | 0.824 | +0.006 | 0.669 | 0.686 | +0.017 |
| ground track field | 0.744 | 0.750 | +0.006 | 0.657 | 0.636 | -0.021 |
| bridge | 0.640 | 0.626 | -0.014 | 0.412 | 0.407 | -0.005 |
| baseball diamond | 0.828 | 0.815 | -0.013 | 0.627 | 0.604 | -0.023 |
| swimming pool | 0.723 | 0.706 | -0.017 | 0.473 | 0.456 | -0.017 |
| tennis court | 0.938 | 0.918 | -0.020 | 0.909 | 0.890 | -0.019 |
| roundabout | 0.689 | 0.668 | -0.021 | 0.531 | 0.528 | -0.003 |
| harbor | 0.876 | 0.848 | -0.028 | 0.637 | 0.591 | -0.046 |
| basketball court | 0.716 | 0.641 | -0.075 | 0.660 | 0.587 | -0.073 |
| plane | 0.953 | — | — | 0.863 | — | — |
| ship | 0.900 | — | — | 0.754 | — | — |
| storage tank | 0.833 | — | — | 0.705 | — | — |

**Note on scale-choice hypothesis**: `SPLIT_RATES=[0.8, 1.2]` was chosen expecting the `0.8`
(downscale) rate to help large/elongated classes (bridge, harbor, large vehicle — the ones
[analysis_results.md](analysis_results.md) flags with the highest HBB/OBB area inflation) by
letting them fit within a single tile. That hypothesis **only partially held**: `large vehicle`
improved marginally (+0.006 / +0.017), but `harbor` and `bridge` both got slightly *worse*.
The classes that improved most (`helicopter`, `small vehicle`, `soccer ball field`) look more
consistent with the `1.2` (upscale) rate's benefit for small/fine-detail objects. Recorded
honestly rather than reframed after the fact — see
[DESIGN_NOTES.md](DESIGN_NOTES.md) for the full reasoning trail.

## Training curve

Per-epoch `metrics/mAP50-95(B)` from `results.csv` (`EarlyStopping(patience=15)`, ran 28/30
epochs):

| epoch | 1 | 2 | 5 | 9 | 13 | 18 | 23 | 28 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mAP50-95 | 0.571 | 0.499 | 0.590 | 0.620 | **0.631** | 0.627 | 0.620 | 0.614 |

**Epoch 13 is the true peak** (`best.pt` saved there) — matches the trainer's own
`EarlyStopping` message (`Best results observed at epoch 13`). Epoch 2's dip (0.571 → 0.499)
recovers by epoch 5; classic behavior for continuing to train an already-converged checkpoint
with a freshly-initialized `optimizer=auto` schedule — the first few epochs perturb the
weights out of the sharp minimum before re-converging.

After epoch 13, `val/box_loss` stays essentially flat (~0.86), but `val/cls_loss` climbs
steadily from 1.56 (epoch 5) to 1.82 (epoch 28) — a mild, gradual classification-head overfit
that the localization head doesn't share. That's the actual mechanism behind the slow mAP
decline from epoch 13 to 28, and it's exactly the pattern `patience=15` (not something
shorter) was set up to distinguish from ordinary noise before stopping.

## Confusion matrix (fine-tuned `best.pt`)

Two patterns stand out, and neither is "classes get confused with each other" — cross-class
off-diagonal confusion is uniformly ≤3%. Instead:

- **Miss rate** (true object predicted as background): worst on `bridge` (33%), `roundabout`
  (39%), `soccer ball field` (38%) — bridge's high miss rate lines up with it also having the
  highest HBB/OBB area inflation in [analysis_results.md](analysis_results.md): thin,
  elongated, easily missed structures are hard on both counts.
- **False-positive rate** (background predicted as an object): worst on `small vehicle` (30%)
  and `ship` (24%) — consistent with small, numerous objects being easy to hallucinate in
  visual clutter (parking lots, docks).

## Actual run config (vs. planned)

`args.yaml` shows `batch: 40`, not the `batch: 36` this session had settled on as the
VRAM-safety-margin choice for the A100 80GB tier (a mismatch between the uploaded notebook and
what was actually run — worth knowing, not worth re-running to fix). It completed all 28
epochs without OOM, so `batch=40` is now an empirically-validated safe setting for this
data/model combination, not just an estimate.

Other settings of note from `args.yaml`: `close_mosaic: 10` (mosaic augmentation turned off for
the last 10 configured epochs, i.e. from epoch 21 on — matches the ~24min vs ~29min/epoch
speed-up observed in the later epochs), `multi_scale: 0.0` (no runtime scale-jitter augmentation,
all scale diversity came from the `[0.8, 1.2]` dataset-level split), `degrees: 0.0` (no extra
rotation augmentation beyond what the OBB task already handles).

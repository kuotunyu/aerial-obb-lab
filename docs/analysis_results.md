# HBB vs OBB on DOTAv1 val — raw analysis output

<!-- claim:analysis -->
Objects analyzed: **28853** across **456** images (ground-truth labels).

This is **ground-truth geometry** evidence, not detector accuracy or production performance. The
machine-readable values are in [`analysis_results.json`](analysis_results.json) and the bounded
release claim is in [`../release/evidence.json`](../release/evidence.json).

## Area inflation: axis-aligned box area / oriented box area

| class | n objects | mean | median | p90 | max |
|---|---:|---:|---:|---:|---:|
| bridge | 464 | 2.43x | 1.90x | 3.71x | 21.1x |
| harbor | 2090 | 2.15x | 1.80x | 3.66x | 26.7x |
| large vehicle | 4387 | 2.14x | 2.13x | 3.21x | 5.4x |
| ship | 8960 | 1.95x | 2.07x | 2.69x | 6.7x |
| ground track field | 144 | 1.62x | 1.60x | 2.18x | 6.1x |
| small vehicle | 5438 | 1.61x | 1.50x | 2.31x | 3.2x |
| tennis court | 760 | 1.52x | 1.32x | 2.27x | 2.6x |
| swimming pool | 440 | 1.52x | 1.42x | 2.06x | 2.5x |
| basketball court | 132 | 1.51x | 1.49x | 2.08x | 2.2x |
| baseball diamond | 214 | 1.50x | 1.47x | 1.99x | 2.0x |
| soccer ball field | 153 | 1.47x | 1.35x | 2.07x | 2.4x |
| plane | 2531 | 1.42x | 1.24x | 2.00x | 2.2x |
| helicopter | 73 | 1.40x | 1.21x | 2.00x | 2.1x |
| roundabout | 179 | 1.02x | 1.00x | 1.00x | 2.0x |
| storage tank | 2888 | 1.00x | 1.00x | 1.00x | 1.3x |

## Dense-scene neighbor overlap (same-class pairs within an image)

`phantom rate` = pairs that a horizontal-box view sees as heavily overlapping
(IoU>=0.3) while the true oriented boxes barely overlap (IoU<0.1) — exactly the
detections an HBB NMS would wrongly suppress.

| class | touching pairs | HBB IoU>=0.3 | ... of which OBB IoU<0.1 | phantom rate |
|---|---:|---:|---:|---:|
| large vehicle | 3688 | 810 | 810 | 100% |
| ship | 6457 | 736 | 732 | 99% |
| harbor | 509 | 43 | 42 | 98% |
| small vehicle | 2341 | 42 | 39 | 93% |
| bridge | 44 | 12 | 12 | 100% |
| plane | 1367 | 1 | 0 | 0% |
| storage tank | 219 | 0 | 0 | - |
| baseball diamond | 14 | 0 | 0 | - |
| tennis court | 257 | 0 | 0 | - |
| basketball court | 49 | 0 | 0 | - |
| ground track field | 1 | 0 | 0 | - |
| helicopter | 18 | 0 | 0 | - |
| soccer ball field | 18 | 0 | 0 | - |
| swimming pool | 15 | 0 | 0 | - |

## Rendered comparisons (assets/)

- `hbb_vs_obb_1_P0706_ship.jpg` — P0706, dominant class: ship (score 393)
- `hbb_vs_obb_2_P2726_ship.jpg` — P2726, dominant class: ship (score 95)
- `hbb_vs_obb_3_P2124_large-vehicle.jpg` — P2124, dominant class: large vehicle (score 82)
- `hbb_vs_obb_4_P1957_large-vehicle.jpg` — P1957, dominant class: large vehicle (score 75)
- `hbb_vs_obb_5_P2781_large-vehicle.jpg` — P2781, dominant class: large vehicle (score 68)
<!-- /claim:analysis -->

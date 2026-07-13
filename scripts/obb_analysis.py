"""Phase 4: run the HBB-vs-OBB analysis on DOTAv1 val and produce README material.

- Downloads DOTAv1 (2 GB, original images) via ultralytics if not present
- Computes per-class area-inflation and dense-scene neighbor-overlap statistics
- Renders 5 side-by-side HBB vs OBB comparison crops into assets/
- Writes docs/analysis_results.md with ready-to-paste markdown tables

Run:  uv run python scripts/obb_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obbkit.analysis import (  # noqa: E402
    DOTA_CLASSES,
    inflation_stats,
    load_split,
    neighbor_overlap_stats,
    top_dense_images,
)
from obbkit.viz import render_hbb_vs_obb  # noqa: E402

N_VIZ = 5


def md_inflation(stats: dict) -> str:
    lines = [
        "| class | n objects | mean | median | p90 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls, s in sorted(stats.items(), key=lambda kv: -kv[1]["mean"]):
        lines.append(
            f"| {cls} | {s['n']} | {s['mean']:.2f}x | {s['median']:.2f}x | {s['p90']:.2f}x | {s['max']:.1f}x |"
        )
    return "\n".join(lines)


def md_overlap(stats: dict) -> str:
    lines = [
        "| class | touching pairs | HBB IoU>=0.3 | ... of which OBB IoU<0.1 | phantom rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls, s in sorted(stats.items(), key=lambda kv: -kv[1].get("hbb_iou>=0.3", 0)):
        n03 = s.get("hbb_iou>=0.3", 0)
        phantom = s.get("hbb_iou>=0.3_but_obb_iou<0.1", 0)
        if s["touching_pairs"] == 0:
            continue
        rate = f"{phantom / n03 * 100:.0f}%" if n03 else "-"
        lines.append(f"| {cls} | {s['touching_pairs']} | {n03} | {phantom} | {rate} |")
    return "\n".join(lines)


def main() -> int:
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset("DOTAv1.yaml")  # auto-downloads on first run
    data_root = Path(data["path"])
    print("DOTAv1 root:", data_root)

    print("parsing val labels ...")
    objects = load_split(data_root, "val")
    n_img = len({o.image for o in objects})
    print(f"{len(objects)} objects across {n_img} val images")

    print("computing area inflation ...")
    infl = inflation_stats(objects)
    print("computing neighbor overlaps (this walks polygon pairs, ~minutes) ...")
    over = neighbor_overlap_stats(objects)

    print("rendering comparison crops ...")
    by_image: dict[str, list] = {}
    for o in objects:
        by_image.setdefault(o.image, []).append(o)
    images_dir = data_root / "images" / "val"
    rendered = []
    for img_name, score in top_dense_images(objects, k=N_VIZ * 3):
        if len(rendered) >= N_VIZ:
            break
        objs = by_image[img_name]
        # focus on the dominant crowded class in this image
        cls_counts = Counter(o.cls for o in objs)
        dom_cls, dom_n = cls_counts.most_common(1)[0]
        if dom_n < 4:
            continue
        focus = [o for o in objs if o.cls == dom_cls]
        src = next((p for p in (images_dir / f"{img_name}{ext}" for ext in (".jpg", ".png")) if p.exists()), None)
        if src is None:
            continue
        out = ROOT / "assets" / f"hbb_vs_obb_{len(rendered) + 1}_{img_name}_{DOTA_CLASSES[dom_cls].replace(' ', '-')}.jpg"
        render_hbb_vs_obb(src, objs, out, focus=focus)
        rendered.append((out.name, img_name, DOTA_CLASSES[dom_cls], score))
        print("  ->", out.name, f"(dense pairs score={score})")

    doc = ROOT / "docs" / "analysis_results.md"
    doc.write_text(
        "# HBB vs OBB on DOTAv1 val — raw analysis output\n\n"
        f"Objects analyzed: **{len(objects)}** across **{n_img}** images (ground-truth labels).\n\n"
        "## Area inflation: axis-aligned box area / oriented box area\n\n"
        f"{md_inflation(infl)}\n\n"
        "## Dense-scene neighbor overlap (same-class pairs within an image)\n\n"
        "`phantom rate` = pairs that a horizontal-box view sees as heavily overlapping\n"
        "(IoU>=0.3) while the true oriented boxes barely overlap (IoU<0.1) — exactly the\n"
        "detections an HBB NMS would wrongly suppress.\n\n"
        f"{md_overlap(over)}\n\n"
        "## Rendered comparisons (assets/)\n\n"
        + "\n".join(f"- `{n}` — {img}, dominant class: {c} (score {s})" for n, img, c, s in rendered)
        + "\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "analysis_results.json").write_text(
        json.dumps({"inflation": infl, "overlap": over}, indent=2), encoding="utf-8"
    )
    print("wrote", doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

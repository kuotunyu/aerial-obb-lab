"""Historical Phase 4 HBB-vs-OBB analysis for academic-use DOTAv1 data.

- Downloads DOTAv1 (2 GB, original images) via ultralytics if not present
- Computes per-class area-inflation and dense-scene neighbor-overlap statistics
- Renders 5 side-by-side HBB vs OBB comparison crops into assets/
- Writes docs/analysis_results.md with ready-to-paste markdown tables

This is not a release gate. It requires an explicit acknowledgement before any dataset lookup or
automatic download:
  .venv/Scripts/python.exe scripts/obb_analysis.py --acknowledge-dota-academic-use
Don't use `uv run` on a non-ASCII repo path -- see docs/DESIGN_NOTES.md T6.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the historical DOTAv1 geometry analysis (academic use only)."
    )
    parser.add_argument(
        "--acknowledge-dota-academic-use",
        action="store_true",
        help="confirm that the operator has reviewed and accepts the DOTA academic-use terms",
    )
    return parser.parse_args()


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


def main(*, acknowledge_dota_academic_use: bool = False) -> int:
    if not acknowledge_dota_academic_use:
        raise SystemExit(
            "Refusing dataset access: review the DOTA academic-use terms, then rerun with "
            "--acknowledge-dota-academic-use."
        )

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
    inflation_count = sum(int(row["n"]) for row in infl.values())
    overall_mean = sum(int(row["n"]) * float(row["mean"]) for row in infl.values()) / inflation_count
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
        "<!-- claim:analysis -->\n"
        f"Objects analyzed: **{len(objects)}** across **{n_img}** images (ground-truth labels).\n\n"
        f"Overall weighted mean across all objects: **{overall_mean:.2f}x**.\n\n"
        "This is **ground-truth geometry** evidence, not detector accuracy or production "
        "performance. The machine-readable values are in `analysis_results.json`; the bounded "
        "release claim is in `../release/evidence.json`.\n\n"
        "## Area inflation: axis-aligned box area / oriented box area\n\n"
        f"{md_inflation(infl)}\n\n"
        "## Dense-scene neighbor overlap (same-class pairs within an image)\n\n"
        "`phantom rate` = pairs that a horizontal-box view sees as heavily overlapping\n"
        "(IoU>=0.3) while the true oriented boxes barely overlap (IoU<0.1). This is a\n"
        "ground-truth geometry proxy for potential suppression risk, not a detector/NMS result.\n\n"
        f"{md_overlap(over)}\n\n"
        "## Rendered-comparison boundary\n\n"
        f"{len(rendered)} local comparison renders were derived from DOTA validation imagery. "
        "They must remain untracked and are excluded from the public code-only release. The "
        "aggregate values above do not depend on redistributing those images.\n"
        "<!-- /claim:analysis -->\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "analysis_results.json").write_text(
        json.dumps({"inflation": infl, "overlap": over}, indent=2), encoding="utf-8"
    )
    print("wrote", doc)
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        main(acknowledge_dota_academic_use=args.acknowledge_dota_academic_use)
    )

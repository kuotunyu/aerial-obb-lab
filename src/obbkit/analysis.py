"""HBB vs OBB quantitative analysis on DOTA ground-truth labels.

Answers "why do aerial scenes need oriented boxes?" with two measurements
computed on the DOTAv1 val split (original images + YOLO-OBB labels):

1. Area inflation: area(axis-aligned box of the polygon) / area(OBB polygon).
   How much background a horizontal box is forced to swallow, per class.

2. Neighbor overlap: for same-class object pairs in the same image, IoU of
   their horizontal boxes vs IoU of their oriented boxes. High HBB-IoU pairs
   with near-zero OBB-IoU are exactly the detections a horizontal-box NMS
   would wrongly suppress in dense scenes (ships in harbors, parked vehicles).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

DOTA_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]


@dataclass
class ObbObject:
    image: str
    cls: int
    poly: np.ndarray  # (4, 2) absolute pixel coords

    @property
    def obb_area(self) -> float:
        x, y = self.poly[:, 0], self.poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @property
    def hbb(self) -> tuple[float, float, float, float]:
        x, y = self.poly[:, 0], self.poly[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    @property
    def hbb_area(self) -> float:
        x0, y0, x1, y1 = self.hbb
        return (x1 - x0) * (y1 - y0)


def load_split(data_root: str | Path, split: str = "val") -> list[ObbObject]:
    """Parse YOLO-OBB labels (cls + 8 normalized coords) into absolute polygons."""
    data_root = Path(data_root)
    labels_dir = data_root / "labels" / split
    images_dir = data_root / "images" / split
    objects: list[ObbObject] = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        img_candidates = [images_dir / (label_file.stem + ext) for ext in (".jpg", ".png", ".jpeg")]
        img_path = next((p for p in img_candidates if p.exists()), None)
        if img_path is None:
            continue
        with Image.open(img_path) as im:  # header-only read, no full decode
            w, h = im.size
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            coords = np.array(list(map(float, parts[1:9])), dtype=np.float64).reshape(4, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            objects.append(ObbObject(image=label_file.stem, cls=cls, poly=coords))
    return objects


def inflation_stats(objects: list[ObbObject]) -> dict[str, dict]:
    """Per-class distribution of hbb_area / obb_area (>=1, higher = worse fit)."""
    per_class: dict[int, list[float]] = defaultdict(list)
    for o in objects:
        a = o.obb_area
        if a > 1:  # skip degenerate boxes
            per_class[o.cls].append(o.hbb_area / a)
    out = {}
    for cls, vals in sorted(per_class.items()):
        v = np.array(vals)
        out[DOTA_CLASSES[cls]] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "max": float(v.max()),
        }
    return out


def _hbb_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """IoU matrix for (n, 4) xyxy boxes."""
    x0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _obb_iou(a: ObbObject, b: ObbObject) -> float:
    pa, pb = Polygon(a.poly), Polygon(b.poly)
    if not (pa.is_valid and pb.is_valid):
        pa, pb = pa.buffer(0), pb.buffer(0)
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return inter / union if union > 0 else 0.0


def neighbor_overlap_stats(objects: list[ObbObject], hbb_thresholds=(0.1, 0.3, 0.5)) -> dict[str, dict]:
    """Same-class pairs within an image: HBB IoU vs OBB IoU.

    Returns per-class counts of pairs whose HBB IoU exceeds each threshold and,
    among those, how many keep OBB IoU < 0.1 ("phantom overlaps" created purely
    by the horizontal-box approximation).
    """
    groups: dict[tuple[str, int], list[ObbObject]] = defaultdict(list)
    for o in objects:
        groups[(o.image, o.cls)].append(o)

    pairs_by_class: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (_, cls), objs in groups.items():
        if len(objs) < 2:
            continue
        boxes = np.array([o.hbb for o in objs])
        iou = _hbb_iou_matrix(boxes)
        idx = np.argwhere(np.triu(iou, k=1) > 0)
        for i, j in idx:
            pairs_by_class[cls].append((float(iou[i, j]), _obb_iou(objs[i], objs[j])))

    out = {}
    for cls, pairs in sorted(pairs_by_class.items()):
        arr = np.array(pairs)  # (n, 2): hbb_iou, obb_iou
        stats = {"touching_pairs": int(arr.shape[0])}
        for t in hbb_thresholds:
            sel = arr[arr[:, 0] >= t]
            stats[f"hbb_iou>={t}"] = int(sel.shape[0])
            stats[f"hbb_iou>={t}_but_obb_iou<0.1"] = int((sel[:, 1] < 0.1).sum()) if sel.size else 0
        out[DOTA_CLASSES[cls]] = stats
    return out


def top_dense_images(objects: list[ObbObject], k: int = 10, hbb_thr: float = 0.3) -> list[tuple[str, int]]:
    """Images ranked by count of same-class pairs with HBB IoU >= thr (viz candidates)."""
    groups: dict[tuple[str, int], list[ObbObject]] = defaultdict(list)
    for o in objects:
        groups[(o.image, o.cls)].append(o)
    score: dict[str, int] = defaultdict(int)
    for (img, _), objs in groups.items():
        if len(objs) < 2:
            continue
        iou = _hbb_iou_matrix(np.array([o.hbb for o in objs]))
        score[img] += int((np.triu(iou, k=1) >= hbb_thr).sum())
    return sorted(score.items(), key=lambda kv: -kv[1])[:k]

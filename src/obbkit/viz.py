"""Side-by-side HBB vs OBB ground-truth rendering for the README comparison."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .analysis import ObbObject

HBB_COLOR = (60, 60, 230)  # BGR red-ish
OBB_COLOR = (80, 200, 60)  # BGR green
BAR_H = 44


def _imread_unicode(path: str | Path) -> np.ndarray | None:
    """cv2.imread mangles non-ASCII Windows paths silently; read bytes ourselves."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: str | Path, img: np.ndarray, quality: int = 92) -> None:
    """cv2.imwrite has the same non-ASCII path problem; encode and write bytes ourselves."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    buf.tofile(str(path))


def _crop_region(objects: list[ObbObject], img_w: int, img_h: int, max_side: int = 1400,
                 margin: int = 60) -> tuple[int, int, int, int]:
    """Tight region around the given objects, padded, clipped to image bounds."""
    polys = np.concatenate([o.poly for o in objects], axis=0)
    x0 = max(int(polys[:, 0].min()) - margin, 0)
    y0 = max(int(polys[:, 1].min()) - margin, 0)
    x1 = min(int(polys[:, 0].max()) + margin, img_w)
    y1 = min(int(polys[:, 1].max()) + margin, img_h)
    # cap the region so boxes stay visible after resize
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = max((x1 - x0), (y1 - y0), 400) // 2
    half = min(half, max_side // 2)
    x0, x1 = max(cx - half, 0), min(cx + half, img_w)
    y0, y1 = max(cy - half, 0), min(cy + half, img_h)
    return x0, y0, x1, y1


def _title_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((BAR_H, width, 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (12, BAR_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def render_hbb_vs_obb(image_path: str | Path, objects: list[ObbObject], out_path: str | Path,
                      focus: list[ObbObject] | None = None, thickness: int = 2) -> Path:
    """Write a side-by-side comparison: left = horizontal boxes, right = oriented boxes.

    Args:
        image_path: source image (original DOTA image).
        objects: all ground-truth objects of this image (absolute pixel polygons).
        out_path: output jpg path.
        focus: objects defining the crop region (defaults to all objects).
    """
    img = _imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _crop_region(focus or objects, w, h)
    crop = img[y0:y1, x0:x1]

    in_crop = [o for o in objects
               if o.poly[:, 0].min() >= x0 and o.poly[:, 0].max() <= x1
               and o.poly[:, 1].min() >= y0 and o.poly[:, 1].max() <= y1]

    hbb_panel, obb_panel = crop.copy(), crop.copy()
    for o in in_crop:
        bx0, by0, bx1, by1 = o.hbb
        cv2.rectangle(hbb_panel, (int(bx0) - x0, int(by0) - y0), (int(bx1) - x0, int(by1) - y0),
                      HBB_COLOR, thickness)
        pts = (o.poly - np.array([x0, y0])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(obb_panel, [pts], isClosed=True, color=OBB_COLOR, thickness=thickness)

    gap = np.full((hbb_panel.shape[0], 8, 3), 255, dtype=np.uint8)
    body = np.hstack([hbb_panel, gap, obb_panel])
    header = np.hstack([
        _title_bar(hbb_panel.shape[1], f"HBB (axis-aligned)  n={len(in_crop)}"),
        np.full((BAR_H, 8, 3), 255, dtype=np.uint8),
        _title_bar(obb_panel.shape[1], "OBB (oriented)"),
    ])
    out = np.vstack([header, body])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite_unicode(out_path, out)
    return out_path

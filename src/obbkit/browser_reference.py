"""CPU-only Python reference for the static browser demo's tensor and OBB contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def letterbox_geometry(width: int, height: int, size: int) -> dict[str, float | int]:
    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError("image dimensions and letterbox size must be positive")
    scale = min(size / width, size / height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    return {
        "scale": scale,
        "newWidth": new_width,
        "newHeight": new_height,
        "padX": (size - new_width) // 2,
        "padY": (size - new_height) // 2,
    }


def rgba_to_chw(values: list[int]) -> list[float]:
    rgba = np.asarray(values, dtype=np.uint8)
    if rgba.size % 4:
        raise ValueError("expected flat RGBA pixels")
    rgb = rgba.reshape(-1, 4)[:, :3].astype(np.float32) / np.float32(255.0)
    return [float(value) for value in rgb.T.reshape(-1)]


def decode_output(
    values: list[float],
    geometry: dict[str, float | int],
    confidence: float,
    class_ids: set[int],
    class_count: int,
) -> list[dict[str, float | int]]:
    output = np.asarray(values, dtype=np.float32)
    if output.size % 7:
        raise ValueError("expected flattened [N,7] output")
    scale = float(geometry["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("letterbox scale must be positive")
    pad_x = float(geometry["padX"])
    pad_y = float(geometry["padY"])
    detections: list[dict[str, float | int]] = []
    for raw_row in output.reshape(-1, 7):
        cx, cy, width, height, score, raw_class, angle = map(float, raw_row)
        if not all(math.isfinite(value) for value in map(float, raw_row)):
            raise ValueError("output values must be finite")
        if width <= 0 or height <= 0:
            raise ValueError("box width and height must be positive")
        class_id = int(raw_class)
        if raw_class != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"class id must be an integer in [0, {class_count})")
        if score < confidence or (class_ids and class_id not in class_ids):
            continue
        detections.append(
            {
                "cx": (cx - pad_x) / scale,
                "cy": (cy - pad_y) / scale,
                "w": width / scale,
                "h": height / scale,
                "conf": score,
                "cls": class_id,
                "angle": angle,
            }
        )
    return sorted(detections, key=lambda row: float(row["conf"]), reverse=True)


def rotated_corners(detection: dict[str, float | int]) -> list[list[float]]:
    cx = float(detection["cx"])
    cy = float(detection["cy"])
    half_width = float(detection["w"]) / 2
    half_height = float(detection["h"]) / 2
    angle = float(detection["angle"])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        [cx + x * cosine - y * sine, cy + x * sine + y * cosine]
        for x, y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    letterbox = fixture["letterbox"]
    geometry = letterbox_geometry(letterbox["width"], letterbox["height"], letterbox["target"])
    decode = fixture["decode"]
    detections = decode_output(
        decode["output"],
        decode["geometry"],
        decode["confidence"],
        set(decode["class_ids"]),
        decode["class_count"],
    )
    return {
        "geometry": geometry,
        "chw": rgba_to_chw(fixture["rgba"]["values"]),
        "detections": detections,
        "corners": rotated_corners(detections[0]),
    }

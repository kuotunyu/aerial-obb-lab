"""Self-hosted BYOM Gradio demo with CPU as the default device.

Set MODEL_PATH to a local .pt or .onnx file. This entry point never downloads,
exports, or falls back to a named model.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

from ultralytics import YOLO

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))
from gradio_ui import GRADIO_CSS, build_demo
from model_source import require_model_path

IMGSZ = 1024
DEVICE = os.environ.get("MODEL_DEVICE", "cpu").strip() or "cpu"
MODEL_PATH = require_model_path()
model = YOLO(str(MODEL_PATH), task="obb")
NAMES = model.names


def detect(image, conf: float, selected_classes: list[str]):
    if image is None:
        return None, []
    class_ids = [i for i, n in NAMES.items() if n in selected_classes] if selected_classes else None
    result = model.predict(
        image, conf=conf, imgsz=IMGSZ, device=DEVICE, classes=class_ids, verbose=False
    )[0]
    annotated = result.plot()[..., ::-1]
    rows = []
    if result.obb is not None:
        for cls_id, c, xywhr in zip(
            result.obb.cls.tolist(), result.obb.conf.tolist(), result.obb.xywhr.tolist()
        ):
            _, _, w, h, r = xywhr
            rows.append([NAMES[int(cls_id)], round(c, 3), round(w, 1), round(h, 1),
                         round(math.degrees(r), 1)])
    rows.sort(key=lambda x: -x[1])
    return annotated, rows


app = build_demo(
    detect_fn=detect,
    class_names=[str(name) for name in NAMES.values()],
    model_name=MODEL_PATH.name,
    device=DEVICE,
    imgsz=IMGSZ,
)
demo = app

if __name__ == "__main__":
    demo.launch(show_error=False, css=GRADIO_CSS)

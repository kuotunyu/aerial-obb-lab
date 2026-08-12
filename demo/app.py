"""Local BYOM Gradio demo: upload an image and get oriented bounding boxes.

Set MODEL_PATH to a local .pt or .onnx file before starting the app. MODEL_DEVICE
defaults to CPU and may be overridden explicitly by the user.

Run:  .venv/Scripts/python.exe demo/app.py   (Windows; .venv/bin/python on Linux/Mac)
Don't use `uv run` on a non-ASCII repo path -- see docs/DESIGN_NOTES.md T6.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

from ultralytics import YOLO

DEMO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_ROOT))
from gradio_ui import GRADIO_CSS, build_demo
from model_source import require_model_path

IMGSZ = 1024
DEVICE = os.environ.get("MODEL_DEVICE", "cpu").strip() or "cpu"
MODEL_PATH = require_model_path()
model = YOLO(str(MODEL_PATH), task="obb")
NAMES = model.names  # id -> name


def detect(image, conf: float, selected_classes: list[str]):
    if image is None:
        return None, []
    class_ids = [i for i, n in NAMES.items() if n in selected_classes] if selected_classes else None
    result = model.predict(
        image, conf=conf, imgsz=IMGSZ, device=DEVICE,
        classes=class_ids, verbose=False,
    )[0]
    annotated = result.plot()[..., ::-1]  # BGR -> RGB
    rows = []
    if result.obb is not None:
        for cls_id, c, xywhr in zip(
            result.obb.cls.tolist(), result.obb.conf.tolist(), result.obb.xywhr.tolist()
        ):
            cx, cy, w, h, r = xywhr
            rows.append([
                NAMES[int(cls_id)], round(c, 3),
                round(w, 1), round(h, 1), round(math.degrees(r), 1),
            ])
    rows.sort(key=lambda x: -x[1])
    return annotated, rows


app = build_demo(
    detect_fn=detect,
    class_names=[str(name) for name in NAMES.values()],
    model_name=MODEL_PATH.name,
    device=DEVICE,
    imgsz=IMGSZ,
)

if __name__ == "__main__":
    app.launch(show_error=False, css=GRADIO_CSS)

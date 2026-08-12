"""Self-hosted BYOM Gradio demo with CPU as the default device.

Set MODEL_PATH to a local .pt or .onnx file. This entry point never downloads,
exports, or falls back to a named model.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import gradio as gr
from ultralytics import YOLO

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))
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


with gr.Blocks(title="YOLO26-OBB local-model detection") as demo:
    gr.Markdown(
        "# YOLO26 OBB — Local Model Demo\n"
        f"Model: **local `{MODEL_PATH.name}`** | device: **{DEVICE}** | imgsz {IMGSZ}. "
        "The user supplies the model and is responsible for its dataset and license terms. "
        "No model is downloaded or bundled by this app."
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Aerial image")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence")
            classes = gr.Dropdown(choices=[str(n) for n in NAMES.values()],
                                  multiselect=True, label="Class filter (empty = all)")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column(scale=2):
            out_img = gr.Image(label="Rotated boxes")
            out_table = gr.Dataframe(headers=["class", "conf", "w(px)", "h(px)", "angle(°)"],
                                     label="Detections")
    btn.click(detect, [inp, conf, classes], [out_img, out_table])
    inp.upload(detect, [inp, conf, classes], [out_img, out_table])

if __name__ == "__main__":
    demo.launch()

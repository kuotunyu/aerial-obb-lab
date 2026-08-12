"""Local BYOM Gradio demo: upload an image and get oriented bounding boxes.

Set MODEL_PATH to a local .pt or .onnx file before starting the app. MODEL_DEVICE
defaults to CPU and may be overridden explicitly by the user.

Run:  .venv/Scripts/python.exe demo/app.py   (Windows; .venv/bin/python on Linux/Mac)
Don't use `uv run` on a non-ASCII repo path -- see docs/DESIGN_NOTES.md T6.
"""

from __future__ import annotations

import math
import os

import gradio as gr
from ultralytics import YOLO

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


with gr.Blocks(title="YOLO26-OBB aerial demo") as app:
    gr.Markdown(
        f"# YOLO26-OBB 航拍旋轉框偵測\n"
        f"model: **local `{MODEL_PATH.name}`** | device: **{DEVICE}** | imgsz {IMGSZ}"
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="上傳航拍影像")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence")
            classes = gr.Dropdown(
                choices=[str(n) for n in NAMES.values()],
                multiselect=True, label="類別過濾（留空 = 全部）",
            )
            btn = gr.Button("偵測", variant="primary")
        with gr.Column(scale=2):
            out_img = gr.Image(label="旋轉框結果")
            out_table = gr.Dataframe(
                headers=["class", "conf", "w(px)", "h(px)", "angle(°)"],
                label="偵測清單（依信心度排序）",
            )
    btn.click(detect, [inp, conf, classes], [out_img, out_table])
    inp.upload(detect, [inp, conf, classes], [out_img, out_table])

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    app.launch()

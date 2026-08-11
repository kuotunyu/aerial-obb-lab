"""HF Space (free CPU tier) demo: YOLO26n-OBB via ONNX Runtime CPU.

Self-contained: downloads/exports the ONNX model on first start, then serves a
Gradio UI. Uses the nano model + ONNX Runtime so inference stays responsive on
2 vCPUs (~1-2 s per 1024px image).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

IMGSZ = 1024
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")  # optional: repo hosting yolo26n-obb.onnx
ONNX_FILE = "yolo26n-obb.onnx"


def get_onnx() -> str:
    if HF_MODEL_REPO:
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(HF_MODEL_REPO, ONNX_FILE)
        except Exception as e:
            print(f"[space] {ONNX_FILE} not on {HF_MODEL_REPO} ({e}); exporting locally")
    if not Path(ONNX_FILE).is_file():
        YOLO("yolo26n-obb.pt").export(format="onnx", imgsz=IMGSZ, opset=19)
    return ONNX_FILE


model = YOLO(get_onnx(), task="obb")
NAMES = model.names


def detect(image, conf: float, selected_classes: list[str]):
    if image is None:
        return None, []
    class_ids = [i for i, n in NAMES.items() if n in selected_classes] if selected_classes else None
    result = model.predict(
        image, conf=conf, imgsz=IMGSZ, device="cpu", classes=class_ids, verbose=False
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


with gr.Blocks(title="YOLO26-OBB aerial detection (CPU)") as demo:
    gr.Markdown(
        "# 🛰️ YOLO26n-OBB — Oriented Object Detection on Aerial Images\n"
        "Upload an aerial/satellite image; the model draws **rotated** bounding boxes "
        "(15 DOTA classes). Runs `yolo26n-obb` with **ONNX Runtime on free CPU** — "
        "expect a few seconds per image. Trained on [DOTA](https://captain-whu.github.io/DOTA/) "
        "(academic use only; commercial use prohibited and underlying image terms may apply); "
        "code under AGPL-3.0 (Ultralytics)."
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

demo.launch()

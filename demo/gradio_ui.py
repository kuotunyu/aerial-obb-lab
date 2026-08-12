"""Shared zh-TW Gradio workbench for local OBB detection and UI preview."""

from __future__ import annotations

from html import escape
from pathlib import Path
import traceback

import gradio as gr

from ui_contract import (
    detect_enabled,
    detection_summary,
    image_status,
    safe_detection_error,
    safe_model_label,
)

GRADIO_CSS = Path(__file__).with_name("gradio.css").read_text(encoding="utf-8")


def render_header(model_name: object, device: object, imgsz: object, preview: bool) -> str:
    mode = "UI-only preview" if preview else "Model ready"
    mode_class = "obb-mode is-preview" if preview else "obb-mode"
    return (
        '<div class="obb-header-main">'
        "<div>"
        "<h1>YOLO26 OBB 航拍旋轉框偵測</h1>"
        "<p>選擇航拍影像、調整 threshold 與 class filter，再明確執行 Detect。"
        "</div>"
        f'<span class="{mode_class}">{mode}</span>'
        "</div>"
        '<div class="obb-meta" role="list" aria-label="Runtime configuration">'
        f'<span class="obb-chip" role="listitem">Model <strong>{safe_model_label(model_name)}</strong></span>'
        f'<span class="obb-chip" role="listitem">Device <strong>{escape(str(device))}</strong></span>'
        f'<span class="obb-chip" role="listitem">imgsz <strong>{escape(str(imgsz))}</strong></span>'
        "</div>"
    )


def build_demo(*, detect_fn, class_names, model_name, device, imgsz, preview=False):
    with gr.Blocks(
        title="YOLO26 OBB 航拍旋轉框偵測",
        fill_width=True,
        analytics_enabled=False,
    ) as app:
        gr.HTML(render_header(model_name, device, imgsz, preview), elem_id="app-header")
        status = gr.Markdown(image_status(False, preview=preview), elem_id="app-status")
        with gr.Row(elem_id="workbench-grid"):
            with gr.Column(scale=38, min_width=360, elem_id="input-panel"):
                inp = gr.Image(type="numpy", label="Input image", elem_id="input-image")
                conf = gr.Slider(
                    0.05,
                    0.9,
                    value=0.25,
                    step=0.05,
                    label="Confidence threshold",
                )
                classes = gr.Dropdown(
                    choices=list(class_names),
                    multiselect=True,
                    label="Class filter（留空 = 全部）",
                )
                detect_button = gr.Button(
                    "開始 Detect",
                    variant="primary",
                    interactive=False,
                    elem_id="detect-button",
                )
            with gr.Column(scale=62, min_width=0, elem_id="result-panel"):
                out_img = gr.Image(label="Detection result", elem_id="result-image")
                summary = gr.Markdown(detection_summary([]), elem_id="detection-summary")
                out_table = gr.Dataframe(
                    headers=["class", "conf", "w(px)", "h(px)", "angle(°)"],
                    label="Detection list（依 confidence 排序）",
                    interactive=False,
                )

        def set_image_state(image):
            ready = detect_enabled(image is not None, preview=preview)
            return gr.Button(value="開始 Detect", interactive=ready), image_status(
                image is not None, preview=preview
            )

        def mark_stale(image):
            return image_status(image is not None, preview=preview)

        def run_detection(image, threshold, selected):
            if image is None:
                raise gr.Error("請先選擇圖片。", print_exception=False)
            try:
                annotated, rows = detect_fn(image, threshold, selected)
            except Exception:
                traceback.print_exc()
                raise gr.Error(safe_detection_error(), print_exception=False) from None
            return (
                annotated,
                rows,
                detection_summary(rows),
                "Detect 完成；結果已依 confidence 排序。",
            )

        inp.input(set_image_state, inp, [detect_button, status], queue=False)
        inp.clear(set_image_state, inp, [detect_button, status], queue=False)
        conf.change(mark_stale, inp, status, queue=False)
        classes.change(mark_stale, inp, status, queue=False)
        detection_event = detect_button.click(
            run_detection,
            [inp, conf, classes],
            [out_img, out_table, summary, status],
            trigger_mode="once",
            concurrency_limit=1,
            show_progress="full",
        )
        detection_event.failure(
            lambda: safe_detection_error(), inputs=None, outputs=status, queue=False
        )
    return app

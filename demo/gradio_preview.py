"""Launch the real Gradio layout without loading a model or running inference."""

from __future__ import annotations

import argparse
import sys

PREVIEW_ROOT = __file__.replace("\\", "/").rsplit("/", maxsplit=1)[0]
sys.path.insert(0, PREVIEW_ROOT)
from gradio_ui import GRADIO_CSS, build_demo


DOTA_CLASSES = (
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
)


def preview_detect(*_args):
    raise RuntimeError("preview mode has no inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the model-free Gradio UI preview.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = build_demo(
        detect_fn=preview_detect,
        class_names=DOTA_CLASSES,
        model_name="preview-model.onnx",
        device="CPU",
        imgsz=1024,
        preview=True,
    )
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=False,
        inbrowser=args.open,
        css=GRADIO_CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pure copy and summary helpers for the zh-TW Gradio workbench."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape


def detection_summary(rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "偵測數量：**0** · Top confidence：**—**"
    top = max(float(row[1]) for row in rows)
    return f"偵測數量：**{len(rows)}** · Top confidence：**{top:.3f}**"


def detect_enabled(has_image: bool, *, preview: bool = False) -> bool:
    return has_image and not preview


def image_status(has_image: bool, *, preview: bool = False) -> str:
    if preview:
        return "UI-only preview；不會執行 inference。"
    if has_image:
        return "圖片已就緒；調整設定後按 Detect。"
    return "Model ready；請先選擇圖片。"


def safe_model_label(value: object) -> str:
    basename = str(value).replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return escape(basename)


def safe_detection_error() -> str:
    return "Detect failed；請檢查 local model 與 OBB output contract 後重試。"

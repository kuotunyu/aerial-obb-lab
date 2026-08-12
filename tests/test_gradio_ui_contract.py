from __future__ import annotations

from demo.ui_contract import (
    detect_enabled,
    detection_summary,
    image_status,
    safe_detection_error,
    safe_model_label,
)


def test_detection_summary_handles_empty_and_ranked_rows() -> None:
    assert detection_summary([]) == "偵測數量：**0** · Top confidence：**—**"
    assert detection_summary(
        [["ship", 0.91, 100.0, 50.0, 90.0], ["plane", 0.73, 80.0, 40.0, 12.0]]
    ) == "偵測數量：**2** · Top confidence：**0.910**"


def test_image_status_is_explicit_and_preview_is_labeled() -> None:
    assert image_status(False) == "Model ready；請先選擇圖片。"
    assert image_status(True) == "圖片已就緒；調整設定後按 Detect。"
    assert image_status(False, preview=True) == "UI-only preview；不會執行 inference。"


def test_detect_requires_an_image_and_preview_stays_disabled() -> None:
    assert detect_enabled(False) is False
    assert detect_enabled(True) is True
    assert detect_enabled(True, preview=True) is False


def test_safe_detection_error_never_echoes_exception_details() -> None:
    message = safe_detection_error()
    assert message == "Detect failed；請檢查 local model 與 OBB output contract 後重試。"
    assert "C:\\" not in message
    assert "Traceback" not in message


def test_safe_model_label_hides_paths_and_escapes_html() -> None:
    assert safe_model_label(r"C:\private\model<script>.onnx") == "model&lt;script&gt;.onnx"
    assert safe_model_label("/private/model.onnx") == "model.onnx"

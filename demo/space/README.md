---
title: YOLO26 OBB 航拍旋轉框偵測
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
short_description: 使用本機模型的 CPU 旋轉框偵測 demo
---

# YOLO26 OBB — BYOM CPU Demo

> **Reference implementation only.** Serverless browser 路徑請使用 `demo/space-static/` 的
> BYOM 實作。

將 `MODEL_PATH` 設為現有的本機 `.pt` 或 `.onnx` 檔案，上傳影像後手動按下 Detect，
取得 **oriented bounding boxes**。`MODEL_DEVICE` 預設為 `cpu`；上傳不會自動執行 inference。

<!-- claim:browser-scope -->
- Model：使用者自行提供的本機模型檔；本 app 沒有 download、export 或 named-model fallback。
- 這個 reference demo 不代表 fine-tuned `yolo26m-obb` checkpoint 的 accuracy，也不代表
  既有 T4 latency。
<!-- /claim:browser-scope -->
- 可調整 Confidence threshold 與 per-class filter；表格會顯示每筆 detection 的 rotation angle。
- 專案為完整 training-to-deployment 作品集，包含 DOTAv1 matched evaluation、
  ONNX/TensorRT benchmark 與 HBB-vs-OBB geometry analysis。

## 啟動真實 inference

```powershell
uv sync --frozen --no-install-project --group demo
$env:MODEL_PATH = "C:/models/your-model.onnx"
$env:MODEL_DEVICE = "cpu"
.venv/Scripts/python.exe demo/space/app.py
```

## 開啟 UI-only preview

```powershell
uv sync --frozen --no-install-project --group ui-preview
.venv/Scripts/python.exe demo/gradio_preview.py --open
```

Preview mode 只載入真實 Gradio layout，不載入模型、不執行 inference，也不會製造假的
detection 結果。

## Licensing 與責任邊界

- Code：**AGPL-3.0-or-later**。使用者自行提供的模型仍受其 upstream、dataset 與
  weight 條款約束；Ultralytics Enterprise 是另一條授權路徑。
- 本 release 不附帶 DOTA images／annotations。DOTA 僅限 academic use；使用者必須自行確認
  所選模型與圖片的權利。

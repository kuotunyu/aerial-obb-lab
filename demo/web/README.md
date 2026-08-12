# Aerial OBB Lab Browser Demo

這是一個使用純 HTML／CSS／JavaScript 與 ONNX Runtime Web WASM 的 Browser-native OBB
workbench。使用者選擇相容的 YOLO26 OBB ONNX model 與影像後，兩個檔案都只在 local browser
session 中處理；此 repo 不發布 model binary，也不會自動下載、export 或 fallback 到具名模型。

## 執行

在 repository root 啟動任一 static HTTP server：

```powershell
.venv/Scripts/python.exe -m http.server 8765 --directory demo/web
```

開啟 `http://localhost:8765`，依序選擇 model、影像，調整 confidence／class filter，再按下
「開始 Detect」。直接用 `file://` 開啟可能被 browser 的 WASM／resource policy 阻擋。

## Model contract 與限制

<!-- claim:browser-scope -->
- Model：user-supplied local ONNX，input 必須是 `images [1,3,1024,1024]` float32 RGB CHW，
  output 必須符合 end-to-end `output0 [1,N,7]`。
- 這個 browser integration demo **does not represent** fine-tuned `yolo26m-obb` checkpoint 的
  accuracy 或歷史 T4 latency。
<!-- /claim:browser-scope -->

- 支援 confidence threshold、15 類 class filter、rotated polygon rendering，以及依 confidence
  排序的完整數值表格。
- UI screenshot 與 browser smoke 使用 synthetic SVG 及 stubbed output，不執行模型 inference，
  也不是模型精度證據。
- 本路徑不需要 Python ML runtime、Torch、CUDA、DOTA、weights、HF token 或 secrets。

## 授權

Repository code 為 **AGPL-3.0-or-later**。使用者自行提供的 model 與影像仍受各自的 upstream
software、dataset、weight 與 image rights 約束；DOTA images／annotations 不包含在本 release。

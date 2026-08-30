# Aerial OBB Lab Browser Demo

這是一個純 HTML／CSS／JavaScript 的 Browser-native OBB workbench，提供兩種界線清楚的模式：

- **Synthetic Showcase：**按一次按鈕就載入 repository 內 authored SVG 與 fixed output，顯示
  rotated polygon、provenance 與 `N/A · no inference`。它不載入 ONNX Runtime、不執行 inference，
  也不是 accuracy、evaluation 或 latency evidence；fixture 不含 DOTA pixels。
- **BYOM inference：**使用者自行選擇相容的 YOLO26 OBB ONNX model 與影像。兩個檔案都只在 local
  browser session 中處理；repo 不發布 model binary，也不會自動下載、export 或 fallback 到具名模型。

ONNX Runtime Web 直到使用者選擇 BYOM model 才 lazy-load。首次載入或 browser cache miss 時會從
jsDelivr 取得固定版本 `onnxruntime-web@1.20.1` 的 JavaScript 與 WASM assets，因此 BYOM 是
non-zero-network。`ort.min.js` 使用 SHA-384 SRI 與 anonymous CORS；這個 SRI 只涵蓋該 JavaScript，
不涵蓋 runtime 後續取得的 WASM assets。這個 network boundary 不會上傳 model 或影像。

## 執行

在 repository root 啟動任一 static HTTP server：

```powershell
.venv/Scripts/python.exe -m http.server 8765 --directory demo/web
```

開啟 `http://localhost:8765`。可直接按「載入 Synthetic Showcase」取得 deterministic evidence；
若要執行 inference，再依序選擇 model、影像，調整 confidence／class filter，按下「開始 Detect」。
直接用 `file://` 開啟可能被 browser 的 WASM／resource policy 阻擋。

## Model contract 與限制

<!-- claim:browser-scope -->
- Model：user-supplied local ONNX，input 必須是 `images [1,3,1024,1024]` float32 RGB CHW，
  output 必須符合 end-to-end `output0 [1,N,7]`。
- 這個 browser integration demo **does not represent** fine-tuned `yolo26m-obb` checkpoint 的
  accuracy 或歷史 T4 latency。
<!-- /claim:browser-scope -->

- 支援 confidence threshold、15 類 class filter、rotated polygon rendering，以及依 confidence
  排序的完整數值表格。
- UI screenshot 與 browser smoke 使用 committed synthetic SVG 及 authored output，不執行模型
  inference；它們也不是 accuracy、evaluation 或 latency evidence。
- Showcase asset、runtime、model contract、inference、output schema、render 或 image decode 失敗時，
  UI 會清除 stale results，顯示固定且不洩漏 local filename 的錯誤，並保留安全的 retry／重新選擇路徑。
  無效或過期的新 model candidate 不會取代最後一個已驗證 session。
- 本路徑不需要 Python ML runtime、Torch、CUDA、DOTA、weights、HF token 或 secrets。

## 授權

Repository code 為 **AGPL-3.0-or-later**。使用者自行提供的 model 與影像仍受各自的 upstream
software、dataset、weight 與 image rights 約束；DOTA images／annotations 不包含在本 release。
